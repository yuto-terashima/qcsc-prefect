"""Orbital optimization utilities for RHF and UHF SQD closed loops.

Mathematical background
-----------------------
Given a Hamiltonian H (fixed) and reduced density matrices (RDMs) from a
selected-CI diagonalization, we minimize the energy expectation value over
all unitary orbital rotations U:

  RHF:
    E(U) = h0 + Tr[h1 · (U γ1 U^T)] + 0.5 · Σ h2[p,q,r,s] · g2_rot[p,r,q,s]

  UHF:
    E(Uα, Uβ) = E_nuc
      + Tr[h1_α · (Uα γ1_αα Uα^T)]
      + Tr[h1_β · (Uβ γ1_ββ Uβ^T)]
      + 0.5 · einsum('pqrs,prqs->', h2_αα, g2_αα_rot(Uα))
      +       einsum('pqrs,prqs->', h2_αβ, g2_αβ_rot(Uα,Uβ))
      + 0.5 · einsum('pqrs,prqs->', h2_ββ, g2_ββ_rot(Uβ))

Notation
--------
  h2 tensors : chemist's notation  (pq|rs)
  rdm2 tensors: physicist's notation  <p†r†sq>  (prqs axis order)
  1-RDM: γ[p,q] = <q†p>

  Correct contraction:
    E_2body = 0.5 * einsum('pqrs, prqs ->', h2_chem, rdm2_phys)

Rotation convention (consistent with orbital_optimizer_uhf.py)
----------------------------------------------------------------
  1-RDM:  γ_rot = U γ U^T
  2-RDM:  g2[p,r,q,s] = Σ_{PRQS} U[p,P] U[r,R] rdm2[P,R,Q,S] U[q,Q] U[s,S]

  This is consistent with H_new = U^T H U applied to the Hamiltonian tensors:
    h1_new = U^T h1 U
    h2_new[P,Q,R,S] = Σ_{pqrs} U[p,P] U[q,Q] h2[p,q,r,s] U[r,R] U[s,S]

Usage
-----
  from qcsc_workflow_utility.orbital_opt import optimize_orbitals, rotate_electronic_properties

  # After each SQD diagonalization iteration:
  Ua, Ub, e_opt = optimize_orbitals(elec_props, rdm1_aa, rdm1_bb, rdm2_aa, rdm2_ab, rdm2_bb)
  elec_props = rotate_electronic_properties(elec_props, Ua, Ub)
"""

from __future__ import annotations

import logging
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import scipy.linalg
import scipy.optimize

if TYPE_CHECKING:
    from qcsc_workflow_utility.chem import ElectronicProperties

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers  (skew-symmetric → unitary, RDM rotation)
# ─────────────────────────────────────────────────────────────────────────────

def _unitary_from_skew(x: np.ndarray, n: int) -> np.ndarray:
    """Build U = expm(A) where A is skew-symmetric with upper-triangle = x."""
    A = np.zeros((n, n), dtype=np.float64)
    A[np.triu_indices(n, k=1)] = x
    A -= A.T
    return scipy.linalg.expm(A)


def _rotate_rdm2(
    rdm2_prqs: np.ndarray, Ubra: np.ndarray, Uket: np.ndarray
) -> np.ndarray:
    """Rotate a physicist's 2-RDM stored as (p, r, q, s) = <p†r†sq>.

    g2[p,r,q,s] = Σ_{PRQS} Ubra[p,P] Uket[r,R] rdm2[P,R,Q,S] Ubra[q,Q] Uket[s,S]

    Split into two O(n^5) steps:
      step 1: contract creation  indices (axes 0, 1)
      step 2: contract annihilation indices (axes 2, 3)
    """
    tmp = np.einsum("pP,rR,PRQS->prQS", Ubra, Uket, rdm2_prqs, optimize="greedy")
    return np.einsum("prQS,qQ,sS->prqs", tmp, Ubra, Uket, optimize="greedy")


# ─────────────────────────────────────────────────────────────────────────────
# Energy evaluators
# ─────────────────────────────────────────────────────────────────────────────

def _uhf_energy(
    x: np.ndarray,
    norb: int,
    n_p: int,
    rdm1_aa: np.ndarray,
    rdm1_bb: np.ndarray,
    rdm2_aa: np.ndarray,
    rdm2_ab: np.ndarray,
    rdm2_bb: np.ndarray,
    h1_a: np.ndarray,
    h1_b: np.ndarray,
    h2_aa: np.ndarray,
    h2_ab: np.ndarray,
    h2_bb: np.ndarray,
    nuc: float,
) -> float:
    """UHF energy E(Uα, Uβ) as a function of the skew-symmetric parameters x.

    x[:n_p] → Aα upper-triangle → Uα = expm(Aα)
    x[n_p:] → Aβ upper-triangle → Uβ = expm(Aβ)
    """
    Ua = _unitary_from_skew(x[:n_p], norb)
    Ub = _unitary_from_skew(x[n_p:], norb)

    g1a = Ua @ rdm1_aa @ Ua.T
    g1b = Ub @ rdm1_bb @ Ub.T

    g2aa = _rotate_rdm2(rdm2_aa, Ua, Ua)
    g2ab = _rotate_rdm2(rdm2_ab, Ua, Ub)
    g2bb = _rotate_rdm2(rdm2_bb, Ub, Ub)

    return float(
        nuc
        + np.einsum("pq,pq->", h1_a, g1a)
        + np.einsum("pq,pq->", h1_b, g1b)
        + 0.5 * np.einsum("pqrs,prqs->", h2_aa, g2aa)
        +       np.einsum("pqrs,prqs->", h2_ab, g2ab)
        + 0.5 * np.einsum("pqrs,prqs->", h2_bb, g2bb)
    )


def _rhf_energy(
    x: np.ndarray,
    norb: int,
    rdm1: np.ndarray,
    rdm2: np.ndarray,
    h1: np.ndarray,
    h2: np.ndarray,
    nuc: float,
) -> float:
    """RHF energy E(U) as a function of the skew-symmetric parameter x.

    h2 in chemist's (pq|rs); rdm2 in physicist's (prqs).
    """
    U = _unitary_from_skew(x, norb)
    g1 = U @ rdm1 @ U.T
    g2 = _rotate_rdm2(rdm2, U, U)
    return float(
        nuc
        + np.einsum("pq,pq->", h1, g1)
        + 0.5 * np.einsum("pqrs,prqs->", h2, g2)
    )


# ─────────────────────────────────────────────────────────────────────────────
# JAX analytical-gradient path (optional fast path for UHF)
# ─────────────────────────────────────────────────────────────────────────────

def _make_jax_uhf_obj_and_grad(
    norb: int,
    n_p: int,
    rdm1_aa: np.ndarray,
    rdm1_bb: np.ndarray,
    rdm2_aa: np.ndarray,
    rdm2_ab: np.ndarray,
    rdm2_bb: np.ndarray,
    h1_a: np.ndarray,
    h1_b: np.ndarray,
    h2_aa: np.ndarray,
    h2_ab: np.ndarray,
    h2_bb: np.ndarray,
    nuc: float,
):
    """Build JAX-jitted (objective, gradient) for UHF orbital optimization.

    Returns (None, None) when JAX is unavailable; caller falls back to NumPy.
    """
    try:
        import jax
        import jax.numpy as jnp
        jax.config.update("jax_enable_x64", True)
    except ImportError:
        return None, None

    j = {k: jnp.array(v) for k, v in dict(
        rdm1_aa=rdm1_aa, rdm1_bb=rdm1_bb,
        rdm2_aa=rdm2_aa, rdm2_ab=rdm2_ab, rdm2_bb=rdm2_bb,
        h1_a=h1_a, h1_b=h1_b,
        h2_aa=h2_aa, h2_ab=h2_ab, h2_bb=h2_bb,
    ).items()}

    def _skew_to_U(x_half: "jnp.ndarray") -> "jnp.ndarray":
        idx = jnp.triu_indices(norb, k=1)
        A = jnp.zeros((norb, norb), dtype=x_half.dtype)
        A = A.at[idx].set(x_half)
        A = A - A.T
        return jax.scipy.linalg.expm(A)

    def _rot_rdm2_jax(rdm2_prqs, Ubra, Uket):
        tmp = jnp.einsum("pP,rR,PRQS->prQS", Ubra, Uket, rdm2_prqs, optimize=True)
        return jnp.einsum("prQS,qQ,sS->prqs", tmp, Ubra, Uket, optimize=True)

    def objective(x: "jnp.ndarray") -> "jnp.ndarray":
        Ua = _skew_to_U(x[:n_p])
        Ub = _skew_to_U(x[n_p:])
        g1a = Ua @ j["rdm1_aa"] @ Ua.T
        g1b = Ub @ j["rdm1_bb"] @ Ub.T
        g2aa = _rot_rdm2_jax(j["rdm2_aa"], Ua, Ua)
        g2ab = _rot_rdm2_jax(j["rdm2_ab"], Ua, Ub)
        g2bb = _rot_rdm2_jax(j["rdm2_bb"], Ub, Ub)
        return (
            nuc
            + jnp.einsum("pq,pq->", j["h1_a"], g1a)
            + jnp.einsum("pq,pq->", j["h1_b"], g1b)
            + 0.5 * jnp.einsum("pqrs,prqs->", j["h2_aa"], g2aa)
            +       jnp.einsum("pqrs,prqs->", j["h2_ab"], g2ab)
            + 0.5 * jnp.einsum("pqrs,prqs->", j["h2_bb"], g2bb)
        )

    obj_jit  = jax.jit(objective)
    grad_jit = jax.jit(jax.grad(objective))

    def obj_np(x):
        return float(obj_jit(jnp.array(x, dtype=jnp.float64)))

    def grad_np(x):
        return np.array(grad_jit(jnp.array(x, dtype=jnp.float64)), dtype=np.float64)

    return obj_np, grad_np


# ─────────────────────────────────────────────────────────────────────────────
# Public API — optimize_orbitals
# ─────────────────────────────────────────────────────────────────────────────

def optimize_orbitals(
    elec_props: "ElectronicProperties",
    rdm1_aa: np.ndarray,
    rdm1_bb: np.ndarray,
    rdm2_aa: np.ndarray,
    rdm2_ab: np.ndarray | None = None,
    rdm2_bb: np.ndarray | None = None,
    *,
    method: str = "L-BFGS-B",
    maxiter: int = 300,
    ftol: float = 1e-15,
    gtol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Find the orbital rotation that minimizes the SCI energy expectation value.

    Works for both RHF (``elec_props.unrestricted == False``) and UHF
    (``elec_props.unrestricted == True``).  The RHF path treats the spin-summed
    inputs as a single channel; the UHF path optimizes Uα and Uβ independently.

    Parameters
    ----------
    elec_props:
        ElectronicProperties from the current SQD iteration (carries the
        Hamiltonian integrals in chemist's notation for h2 tensors).
    rdm1_aa, rdm1_bb:
        Alpha and beta 1-RDMs in physicist's convention: γ[p,q] = <q†p>.
        For RHF, pass the spin-resolved blocks (or sum them for the spin-free path).
    rdm2_aa:
        Alpha-alpha 2-RDM in physicist's convention: <p†r†sq>, prqs axis order.
    rdm2_ab:
        Alpha-beta 2-RDM (physicist's). If None, uses rdm2_aa (spin-free approx).
    rdm2_bb:
        Beta-beta 2-RDM (physicist's). If None, uses rdm2_aa (spin-free approx).
    method:
        scipy.optimize.minimize method string (default "L-BFGS-B").
    maxiter:
        Maximum optimizer iterations.
    ftol, gtol:
        Convergence tolerances for L-BFGS-B.

    Returns
    -------
    Ua : ndarray (norb, norb)
        Optimal alpha (or RHF single-channel) rotation matrix.
    Ub : ndarray (norb, norb)
        Optimal beta rotation matrix (== Ua for RHF).
    energy : float
        Energy after orbital optimization (including nuclear repulsion).
    """
    norb = elec_props.num_orbitals
    nuc = elec_props.nuclear_repulsion_energy
    h1_a = elec_props.one_body_tensor
    h2_aa = elec_props.two_body_tensor  # chemist's (pq|rs)

    if rdm2_ab is None:
        rdm2_ab = rdm2_aa
    if rdm2_bb is None:
        rdm2_bb = rdm2_aa

    if elec_props.unrestricted:
        h1_b  = elec_props.one_body_tensor_b
        h2_ab = elec_props.two_body_tensor_ab
        h2_bb = elec_props.two_body_tensor_bb
    else:
        # RHF: treat as spin-free UHF with identical alpha/beta channels
        h1_b  = h1_a
        h2_ab = h2_aa
        h2_bb = h2_aa

    n_p = norb * (norb - 1) // 2  # parameters per spin channel
    x0 = np.zeros(2 * n_p, dtype=np.float64)

    # Try JAX analytical gradient first (UHF only — fast path); fall back to NumPy
    obj_jax, grad_jax = _make_jax_uhf_obj_and_grad(
        norb, n_p,
        rdm1_aa, rdm1_bb, rdm2_aa, rdm2_ab, rdm2_bb,
        h1_a, h1_b, h2_aa, h2_ab, h2_bb, nuc,
    )
    use_jax = (obj_jax is not None) and (method.upper() in ("L-BFGS-B", "BFGS", "CG"))

    if use_jax:
        log.info("  [OrbOpt] Using JAX analytical gradient")
        eval_obj = obj_jax
    else:
        log.info("  [OrbOpt] Using NumPy objective (no analytical gradient)")
        eval_obj = partial(
            _uhf_energy,
            norb=norb, n_p=n_p,
            rdm1_aa=rdm1_aa, rdm1_bb=rdm1_bb,
            rdm2_aa=rdm2_aa, rdm2_ab=rdm2_ab, rdm2_bb=rdm2_bb,
            h1_a=h1_a, h1_b=h1_b,
            h2_aa=h2_aa, h2_ab=h2_ab, h2_bb=h2_bb,
            nuc=nuc,
        )

    e_before = eval_obj(x0)
    log.info("  [OrbOpt] E before optimization: %.10f Ha", e_before)

    minimize_kwargs: dict = dict(
        fun=eval_obj,
        x0=x0,
        method=method,
        options={"maxiter": maxiter, "ftol": ftol, "gtol": gtol},
    )
    if use_jax:
        minimize_kwargs["jac"] = grad_jax

    res = scipy.optimize.minimize(**minimize_kwargs)

    e_after = float(res.fun)
    log.info(
        "  [OrbOpt] E after optimization: %.10f Ha  ΔE=%.4e  converged=%s  nit=%d",
        e_after, e_after - e_before, res.success, res.nit,
    )
    if not res.success:
        log.warning("  [OrbOpt] optimizer status=%d: %s", res.status, res.message)

    Ua = _unitary_from_skew(res.x[:n_p], norb)
    Ub = _unitary_from_skew(res.x[n_p:], norb)
    return Ua, Ub, e_after


# ─────────────────────────────────────────────────────────────────────────────
# Public API — rotate_electronic_properties
# ─────────────────────────────────────────────────────────────────────────────

def rotate_electronic_properties(
    elec_props: "ElectronicProperties",
    Ua: np.ndarray,
    Ub: np.ndarray | None = None,
) -> "ElectronicProperties":
    """Return a new ElectronicProperties with all integral tensors rotated by (Ua, Ub).

    Rotation formulas
    -----------------
      h1_new = U^T h1 U
      h2_new[P,Q,R,S] = Σ_{pqrs} U[p,P] U[q,Q] h2[p,q,r,s] U[r,R] U[s,S]

    For RHF (``elec_props.unrestricted == False``) a single U is applied to
    both one- and two-body tensors.  For UHF, Ua rotates alpha integrals, Ub
    rotates beta integrals, and the mixed (αβ) block uses (Ua, Ub) together.

    The ``t2`` amplitudes and ``initial_occupancy`` are intentionally NOT rotated
    because they are used only to initialize the SQD subspace; the RDMs (occupancies)
    produced by the solver are self-consistently updated each iteration.
    """
    from qcsc_workflow_utility.chem import ElectronicProperties

    if Ub is None:
        Ub = Ua

    def rot1(h1: np.ndarray, U: np.ndarray) -> np.ndarray:
        return U.T @ h1 @ U

    def rot2(h2: np.ndarray, U_bra: np.ndarray, U_ket: np.ndarray) -> np.ndarray:
        """Rotate chemist's (pq|rs) two-body tensor."""
        return np.einsum(
            "pP,qQ,pqrs,rR,sS->PQRS",
            U_bra, U_bra, h2, U_ket, U_ket,
            optimize="greedy",
        )

    h1_a_new = rot1(elec_props.one_body_tensor, Ua)
    h2_aa_new = rot2(elec_props.two_body_tensor, Ua, Ua)

    if elec_props.unrestricted:
        h1_b_new  = rot1(elec_props.one_body_tensor_b, Ub)
        h2_ab_new = rot2(elec_props.two_body_tensor_ab, Ua, Ub)
        h2_bb_new = rot2(elec_props.two_body_tensor_bb, Ub, Ub)
    else:
        h1_b_new  = None
        h2_ab_new = None
        h2_bb_new = None

    return ElectronicProperties(
        one_body_tensor=h1_a_new,
        two_body_tensor=h2_aa_new,
        t2=elec_props.t2,
        initial_occupancy=elec_props.initial_occupancy,
        nuclear_repulsion_energy=elec_props.nuclear_repulsion_energy,
        num_orbitals=elec_props.num_orbitals,
        num_electrons=elec_props.num_electrons,
        open_shell=elec_props.open_shell,
        spin_sq=elec_props.spin_sq,
        unrestricted=elec_props.unrestricted,
        one_body_tensor_b=h1_b_new,
        two_body_tensor_ab=h2_ab_new,
        two_body_tensor_bb=h2_bb_new,
        t2_ab=elec_props.t2_ab,
        t2_bb=elec_props.t2_bb,
    )
