"""SBD solver block backed by qcsc-prefect block execution."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

import numpy as np
from prefect import get_run_logger, task
from prefect.blocks.core import Block
from pydantic import Field
from pyscf.tools import fcidump
from qcsc_prefect_executor.from_blocks import run_job_from_blocks


@dataclass(frozen=True)
class SBDResult:
    """Result of an SBD calculation."""

    energy: float
    """The SCI energy."""

    orbital_occupancies: tuple[np.ndarray, np.ndarray]
    """The average orbital occupancies."""

    carryover_bitstrings: np.ndarray
    """The 2D array of bool representations of carryover bitstrings."""

    rdm1: np.ndarray | None = None
    """Alpha 1-particle reduced density matrix (L x L). Populated only when the solver is run with
    RDM output enabled (``do_rdm != 0``); None otherwise. For RHF this is the (single-spin) 1-RDM;
    for UHF it is the alpha block and ``rdm1_b`` holds beta."""

    rdm2: np.ndarray | None = None
    """Alpha-alpha 2-particle reduced density matrix (L x L x L x L). Populated only with RDM
    output. For UHF the mixed and beta-beta blocks live in ``rdm2_ab`` / ``rdm2_bb``."""

    rdm1_b: np.ndarray | None = None
    """Beta 1-RDM (UHF only)."""

    rdm2_ab: np.ndarray | None = None
    """Mixed alpha-beta 2-RDM block (UHF only)."""

    rdm2_bb: np.ndarray | None = None
    """Beta-beta 2-RDM block (UHF only)."""

    carryover_bitstrings_b: np.ndarray | None = None
    """Beta carryover bitstrings (UHF only). None for RHF; alpha lives in carryover_bitstrings."""

    @property
    def sci_state(self):
        raise NotImplementedError("SBD Prefect integration doesn't reconstruct sci_state object.")


def spin_square_from_subspace(
    ci_strings: tuple[np.ndarray, np.ndarray],
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    *,
    open_shell: bool = True,
) -> float:
    """Exact <S^2> of the selected-CI ground state over a given determinant subspace.

    Robust, RDM-convention-free spin diagnostic: it re-diagonalizes the SAME determinant
    subspace (alpha/beta CI string lists) the C++ SBD solver used, via
    ``qiskit_addon_sqd.fermion.solve_fermion``, and returns the exact <S^2> of that eigenstate.
    UHF references are not spin eigenstates (spin contamination); SQD diagonalizes the true
    Hamiltonian in the fixed-(Na, Nb) sector and restores a (near) pure spin state, so <S^2>
    should approach S(S+1) as the subspace grows. This quantifies that.

    This is intended for small/validation subspaces and the spin-contamination figure; it is an
    extra diagonalization, so it is NOT wired into the production 50q path automatically. The
    energy it computes matches the SBD solver to selected-CI precision for the same subspace.
    """
    # Imported lazily: solve_fermion pulls in the qiskit-addon-sqd fermion stack, which we only
    # need when a spin diagnostic is explicitly requested.
    from qiskit_addon_sqd.fermion import solve_fermion

    _, _, _, spin_sq = solve_fermion(
        ci_strings, one_body_tensor, two_body_tensor, open_shell=open_shell
    )
    return float(spin_sq)


def _make_job_work_dir(base_work_dir: Path) -> Path:
    base_work_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    job_dir = base_work_dir / f"job_{timestamp}_{uuid4().hex[:8]}"
    job_dir.mkdir(parents=True, exist_ok=False)
    return job_dir


def _write_dets_bin(path: Path, ci_strings: np.ndarray, norb: int, *, spin: str) -> None:
    """Write a determinant list to ``path`` in the SBD binary format.

    Format matches what the SBD ``LoadAlphaDets`` reader (used for both alpha and beta files)
    expects: ``(norb + 7) // 8`` bytes per configuration, big-endian bit order.
    """
    dets = np.asarray(ci_strings, dtype=np.int64).reshape(-1)
    if np.any(dets < 0):
        raise ValueError(f"{spin} determinants must be non-negative integers.")
    max_ci = 1 << norb
    if np.any(dets >= max_ci):
        raise ValueError(f"{spin} determinant is out of range for norb={norb}.")
    bytes_per_config = (norb + 7) // 8
    with path.open("wb") as fp:
        for ci in dets:
            fp.write(int(ci).to_bytes(bytes_per_config, byteorder="big", signed=False))


def _write_uhf_fcidump(
    path: Path,
    *,
    h1_a: np.ndarray,
    h1_b: np.ndarray,
    h2_aa: np.ndarray,
    h2_ab: np.ndarray,
    h2_bb: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    ecore: float = 0.0,
) -> None:
    """Write an interleaved-spin-orbital FCIDUMP for the SBD ``-D_UHF`` build.

    The ``_UHF`` ``SetupIntegrals`` parser does NOT use PySCF's four-block MOLPRO UHF format.
    It indexes integrals by spin-orbital with alpha = even 0-based index (``2*p``) and beta =
    odd (``2*p + 1``); the FCIDUMP records are 1-based over ``1..2*norb``. The spin block of a
    two-body record ``(i,j,k,l)`` is determined by ``S = (i-1)%2 + 2*((k-1)%2)`` and requires
    ``(i-1)%2 == (j-1)%2`` and ``(k-1)%2 == (l-1)%2`` (mixed-spin pair terms are zero):

        S=0  (alpha,alpha | alpha,alpha)  -> h2_aa
        S=1  (beta ,beta  | alpha,alpha)  -> h2_ab transposed (chem. (kl|ij) symmetry)
        S=2  (alpha,alpha | beta ,beta )  -> h2_ab
        S=3  (beta ,beta  | beta ,beta )  -> h2_bb

    Tensors are chemist-notation ``(ij|kl)`` as returned by PySCF ``ao2mo``. The reader applies
    its own 8-fold symmetrization, so we emit each unique (i<=j, k<=l, ij<=kl) record once.
    """
    nelec_a, nelec_b = nelec

    lines: list[str] = []
    lines.append(
        f" &FCI NORB={norb},NELEC={nelec_a + nelec_b},MS2={nelec_a - nelec_b},"
    )
    lines.append("  ORBSYM=" + ",".join(["1"] * norb) + ",")
    lines.append("  ISYM=1,")
    lines.append(" &END")

    def spinorb(spatial: int, spin_is_beta: bool) -> int:
        # 1-based spin-orbital index: alpha -> 2*p+1, beta -> 2*p+2 (p 0-based spatial).
        return 2 * spatial + (2 if spin_is_beta else 1)

    def emit_two_body(tensor: np.ndarray, bra_beta: bool, ket_beta: bool) -> None:
        # Chemist (ij|kl). Same-spin blocks (aa|aa, bb|bb) have full 8-fold symmetry, so emit only
        # i>=j, k>=l, (i,j)>=(k,l). The mixed block (aa|bb) only has the 4-fold permutation
        # symmetry i<->j and k<->l (the bra<->ket swap maps it to the distinct bb|aa block, which
        # the reader reconstructs by its own symmetrization), so for it we must NOT apply the
        # (i,j)>=(k,l) cut or we drop valid records like (00|11).
        same_spin = bra_beta == ket_beta
        for i in range(norb):
            for j in range(i + 1):
                ij = i * (i + 1) // 2 + j
                for k in range(norb):
                    for m in range(k + 1):
                        kl = k * (k + 1) // 2 + m
                        if same_spin and ij < kl:
                            continue
                        val = float(tensor[i, j, k, m])
                        if val == 0.0:
                            continue
                        si = spinorb(i, bra_beta)
                        sj = spinorb(j, bra_beta)
                        sk = spinorb(k, ket_beta)
                        sl = spinorb(m, ket_beta)
                        lines.append(f"{val:>28.16e} {si:4d} {sj:4d} {sk:4d} {sl:4d}")

    # Two-body blocks: S=0 aa|aa, S=2 aa|bb, S=3 bb|bb. The S=1 bb|aa block is the (kl|ij)
    # transpose of aa|bb and is recovered by the reader's symmetrization, so we emit aa|bb only.
    emit_two_body(h2_aa, bra_beta=False, ket_beta=False)
    emit_two_body(h2_ab, bra_beta=False, ket_beta=True)
    emit_two_body(h2_bb, bra_beta=True, ket_beta=True)

    # One-body: alpha block at even spin-orbitals, beta at odd. k=l=0 marks a one-body record.
    def emit_one_body(h1: np.ndarray, beta: bool) -> None:
        for i in range(norb):
            for j in range(i + 1):
                val = float(h1[i, j])
                if val == 0.0:
                    continue
                si = spinorb(i, beta)
                sj = spinorb(j, beta)
                lines.append(f"{val:>28.16e} {si:4d} {sj:4d} {0:4d} {0:4d}")

    emit_one_body(h1_a, beta=False)
    emit_one_body(h1_b, beta=True)

    # Core energy record (i=j=k=l=0).
    lines.append(f"{float(ecore):>28.16e} {0:4d} {0:4d} {0:4d} {0:4d}")

    path.write_text("\n".join(lines) + "\n")


def _build_solver_args(solver: "SBDSolverJob") -> list[str]:
    args = [
        "--task_comm_size",
        str(solver.task_comm_size),
        "--adet_comm_size",
        str(solver.adet_comm_size),
        "--bdet_comm_size",
        str(solver.bdet_comm_size),
        "--block",
        str(solver.block),
        "--iteration",
        str(solver.iteration),
        "--tolerance",
        str(solver.tolerance),
        "--carryover_ratio",
        str(solver.carryover_ratio),
        "--carryover_type",
        str(solver.carryover_type),
        "--rdm",
        str(solver.do_rdm),
    ]
    # NOTE: --dump_matrix_form_wf (which wrote matrixformwf.txt) is intentionally NOT passed. That
    # dump serializes the full CI wavefunction as text and is never consumed by the pipeline; at
    # large sqd_dim it is catastrophic (e.g. ~31 GB per walker at sqd_dim=5e8, which stalled the
    # solver on Lustre I/O until it hit the wall-time limit before finishing the diagonalization).
    # The solver skips the dump when the flag is absent (sbdiag.h: only writes if the path != "").
    if solver.solver_mode == "gpu":
        args.extend(["--adetfile", "AlphaDets.bin", "--carryoverfile", "carryover.txt"])
    if solver.method == "uhf":
        # The UHF binary reads a second determinant file; both flag names are defined in our
        # patched main.cc (upstream hardcodes AlphaDets.bin and has no --bdetfile).
        args.extend(["--adetfile", "AlphaDets.bin", "--bdetfile", "BetaDets.bin"])
    if solver.user_args:
        args.extend(list(solver.user_args))
    return args


def _executable_key(solver: "SBDSolverJob") -> str:
    """Derive the HPCProfileBlock executable_map key from backend x method.

    sbd_diag (cpu/fugaku + rhf), sbd_diag_gpu, sbd_diag_uhf, sbd_diag_gpu_uhf. Lets RHF/UHF and
    any backend coexist as separate binaries without a schema change.
    """
    key = "sbd_diag"
    if solver.solver_mode == "gpu":
        key += "_gpu"
    if solver.method == "uhf":
        key += "_uhf"
    return key


def _prep_files(
    *,
    work_dir: Path,
    ci_strings: tuple[np.ndarray, np.ndarray],
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    method: str = "rhf",
    one_body_tensor_b: np.ndarray | None = None,
    two_body_tensor_ab: np.ndarray | None = None,
    two_body_tensor_bb: np.ndarray | None = None,
) -> None:
    logger = get_run_logger()

    if method == "uhf":
        if (
            one_body_tensor_b is None
            or two_body_tensor_ab is None
            or two_body_tensor_bb is None
        ):
            raise ValueError(
                "UHF solver requires one_body_tensor_b, two_body_tensor_ab, and "
                "two_body_tensor_bb (the beta / mixed / beta-beta integral blocks)."
            )
        # Interleaved-spin-orbital FCIDUMP for the -D_UHF binary (NOT PySCF four-block format).
        logger.debug("Writing UHF fcidump.txt file.")
        _write_uhf_fcidump(
            work_dir / "fcidump.txt",
            h1_a=one_body_tensor,
            h1_b=one_body_tensor_b,
            h2_aa=two_body_tensor,
            h2_ab=two_body_tensor_ab,
            h2_bb=two_body_tensor_bb,
            norb=norb,
            nelec=nelec,
        )
        logger.debug("Writing AlphaDets.bin and BetaDets.bin files.")
        _write_dets_bin(work_dir / "AlphaDets.bin", ci_strings[0], norb, spin="alpha")
        _write_dets_bin(work_dir / "BetaDets.bin", ci_strings[1], norb, spin="beta")
        return

    # Write PySCF FCI dump file (RHF).
    logger.debug("Writing fcidump.txt file.")
    fcidump.from_integrals(
        str(work_dir / "fcidump.txt"),
        one_body_tensor,
        two_body_tensor,
        norb,
        nelec,
    )

    # Write alpha determinant list consumed by SBD binary.
    logger.debug("Writing AlphaDets.bin file.")
    _write_dets_bin(work_dir / "AlphaDets.bin", ci_strings[0], norb, spin="alpha")


def _read_carryover_bin(path: Path, norb: int) -> np.ndarray:
    """Decode an SBD carryover.bin file into a (n, norb) bool array."""
    bytes_per_config = (norb + 7) // 8
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return np.empty((0, norb), dtype=bool)
    if data.size % bytes_per_config != 0:
        raise ValueError(
            f"{path.name} size is not aligned with expected bytes-per-config; "
            f"norb={norb}, bytes_per_config={bytes_per_config}, raw_size={data.size}"
        )
    bits = np.unpackbits(data, bitorder="big").reshape(-1, bytes_per_config * 8)[:, :norb]
    return bits.astype(bool)


def _read_rdm_file(path: Path, norb: int, rank: int) -> np.ndarray | None:
    """Read a plain-text RDM file written by the SBD solver (--rdm != 0).

    The solver writes one value per line (row-major, physicist's convention).
    Returns the reshaped array, or None when the file is absent.
    """
    if not path.exists():
        return None
    data = np.loadtxt(path, dtype=np.float64)
    data = np.asarray(data).reshape(-1)
    expected = norb ** rank
    if data.size != expected:
        raise ValueError(
            f"{path.name}: expected {expected} values for norb={norb} rank-{rank} RDM, "
            f"got {data.size}"
        )
    return data.reshape([norb] * rank)


def _read_files(
    *,
    work_dir: Path,
    norb: int,
    method: str = "rhf",
    do_rdm: int = 0,
) -> SBDResult:
    logger = get_run_logger()
    logger.debug("Reading occ_a.txt and occ_b.txt file.")
    occa = np.atleast_1d(np.loadtxt(work_dir / "occ_a.txt", dtype=np.float64))
    occb = np.atleast_1d(np.loadtxt(work_dir / "occ_b.txt", dtype=np.float64))

    logger.debug("Reading carryover.bin file.")
    carryover = _read_carryover_bin(work_dir / "carryover.bin", norb)

    # UHF writes a separate beta carryover; RHF leaves it absent (alpha == beta).
    carryover_b = None
    if method == "uhf":
        carryover_b_path = work_dir / "carryover_b.bin"
        if carryover_b_path.exists():
            logger.debug("Reading carryover_b.bin file.")
            carryover_b = _read_carryover_bin(carryover_b_path, norb)

    logger.debug("Reading davidson_energy.txt file.")
    energy = float(np.loadtxt(work_dir / "davidson_energy.txt").item())

    # RDM files (written only when do_rdm != 0).
    # Convention: rdm1_a.txt / rdm1_b.txt  →  physicist's γ[p,q] = <q†p>
    #             rdm2_aa.txt / rdm2_ab.txt / rdm2_bb.txt  →  physicist's prqs
    rdm1 = rdm2 = rdm1_b = rdm2_ab = rdm2_bb = None
    if do_rdm != 0:
        logger.debug("Reading RDM files (do_rdm=%d).", do_rdm)
        rdm1   = _read_rdm_file(work_dir / "rdm1_a.txt",  norb, rank=2)
        rdm2   = _read_rdm_file(work_dir / "rdm2_aa.txt", norb, rank=4)
        rdm1_b = _read_rdm_file(work_dir / "rdm1_b.txt",  norb, rank=2)
        rdm2_ab = _read_rdm_file(work_dir / "rdm2_ab.txt", norb, rank=4)
        rdm2_bb = _read_rdm_file(work_dir / "rdm2_bb.txt", norb, rank=4)
        if rdm1 is not None:
            logger.debug("Loaded rdm1 (alpha) shape=%s", rdm1.shape)
        if rdm2 is not None:
            logger.debug("Loaded rdm2 (aa) shape=%s", rdm2.shape)

    return SBDResult(
        energy=energy,
        orbital_occupancies=(occa, occb),
        carryover_bitstrings=carryover,
        carryover_bitstrings_b=carryover_b,
        rdm1=rdm1,
        rdm2=rdm2,
        rdm1_b=rdm1_b,
        rdm2_ab=rdm2_ab,
        rdm2_bb=rdm2_bb,
    )


class SBDSolverJob(Block):
    """Prefect block facade for SBD execution through qcsc-prefect blocks."""

    _block_type_name = "SBD Solver Job"
    _block_type_slug = "sbd_solver_job"

    root_dir: str = Field(
        title="Root Directory",
        description="Root directory where per-job work directories are created.",
    )
    command_block_name: str = Field(
        default="cmd-sbd-diag",
        title="Command Block Name",
        description="Prefect CommandBlock document name.",
    )
    execution_profile_block_name: str = Field(
        default="exec-sbd-mpi",
        title="Execution Profile Block Name",
        description="Prefect ExecutionProfileBlock document name.",
    )
    hpc_profile_block_name: str = Field(
        default="hpc-miyabi-sbd",
        title="HPC Profile Block Name",
        description="Prefect HPCProfileBlock document name.",
    )
    script_filename: str = Field(default="sbd_solver.pbs", title="Script Filename")
    metrics_artifact_key: str = Field(default="miyabi-sbd-metrics", title="Metrics Artifact Key")
    timeout_seconds: float = Field(default=7200.0, title="Timeout Seconds")
    user_args: list[str] = Field(default_factory=list, title="Additional User Args")

    task_comm_size: int = Field(
        default=1,
        gt=0,
        title="Task Comm Size",
        description=(
            "Size of task communicator. Controls distribution of Hamiltonian column operations."
        ),
    )
    adet_comm_size: int = Field(
        default=1,
        gt=0,
        title="Adet Comm Size",
        description="Number of alpha-determinant partitions.",
    )
    bdet_comm_size: int = Field(
        default=1,
        gt=0,
        title="Bdet Comm Size",
        description="Number of beta-determinant partitions.",
    )
    block: int = Field(
        default=10,
        gt=0,
        title="Block",
        description="Maximum Davidson subspace size.",
    )
    iteration: int = Field(
        default=2,
        gt=0,
        title="Iteration",
        description="Number of Davidson restarts.",
    )
    tolerance: float = Field(
        default=1e-4,
        gt=0.0,
        title="Tolerance",
        description="Convergence threshold for Davidson residual norm.",
    )
    carryover_ratio: float = Field(
        default=0.5,
        gt=0.0,
        le=1.0,
        title="Carryover Ratio",
        description=(
            "Ratio of bitstrings retained as carryover candidates (used when carryover_type>0)."
        ),
    )
    carryover_type: int = Field(
        default=0,
        ge=0,
        le=3,
        title="Carryover Type",
        description=(
            "Passed to the solver as --carryover_type. The solver only computes and writes the "
            "carryover determinant lists (carryover.bin / carryover_b.bin) when this is nonzero; "
            "0 (default) disables carryover entirely, matching all prior runs where the files came "
            "out empty. 1 = keep the top carryover_ratio fraction of each spin's determinants by "
            "weight; 2 = type 1 plus a singles extension of the kept set; 3 = singles extension of "
            "the whole wavefunction by carryover_threshold. Use >=1 (with n_recovery_steps>=2) to "
            "actually carry dominant determinants across recovery passes."
        ),
    )
    do_rdm: int = Field(
        default=0,
        ge=0,
        title="RDM Output",
        description=(
            "Passed to the solver as --rdm. 0 (default) computes only the diagonal 1-RDM used for "
            "configuration recovery -- byte-identical to prior behavior. Nonzero makes the solver "
            "build the full one- and two-particle RDMs (alpha/beta one-body, aa/ab/ba/bb two-body) "
            "and write them to disk for downstream orbital optimization."
        ),
    )
    solver_mode: Literal["cpu", "gpu", "fugaku"] = Field(
        default="cpu",
        title="Solver Mode",
        description="SBD execution backend (cpu, gpu, or fugaku).",
    )
    method: Literal["rhf", "uhf"] = Field(
        default="rhf",
        title="Method",
        description=(
            "Electronic-structure method. 'rhf' (restricted, default) or 'uhf' (unrestricted / "
            "open-shell). UHF runs the -D_UHF binary with separate alpha/beta determinants and a "
            "spin-resolved FCIDUMP."
        ),
    )

    async def run(
        self,
        ci_strings: tuple[np.ndarray, np.ndarray],
        one_body_tensor: np.ndarray,
        two_body_tensor: np.ndarray,
        norb: int,
        nelec: tuple[int, int],
        one_body_tensor_b: np.ndarray | None = None,
        two_body_tensor_ab: np.ndarray | None = None,
        two_body_tensor_bb: np.ndarray | None = None,
    ) -> SBDResult:
        """Run SBD solver job and return parsed outputs.

        For ``method == "uhf"`` the beta / mixed / beta-beta integral blocks must be supplied via
        ``one_body_tensor_b`` / ``two_body_tensor_ab`` / ``two_body_tensor_bb`` and ``ci_strings``
        holds genuinely distinct alpha and beta determinant lists.
        """
        return await _run_sbd_inner(
            solver=self,
            ci_strings=ci_strings,
            one_body_tensor=one_body_tensor,
            two_body_tensor=two_body_tensor,
            norb=norb,
            nelec=nelec,
            one_body_tensor_b=one_body_tensor_b,
            two_body_tensor_ab=two_body_tensor_ab,
            two_body_tensor_bb=two_body_tensor_bb,
        )


@task(name="solve_eigenstate")
async def _run_sbd_inner(
    *,
    solver: SBDSolverJob,
    ci_strings: tuple[np.ndarray, np.ndarray],
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    one_body_tensor_b: np.ndarray | None = None,
    two_body_tensor_ab: np.ndarray | None = None,
    two_body_tensor_bb: np.ndarray | None = None,
) -> SBDResult:
    base_work_dir = Path(solver.root_dir).expanduser().resolve()
    job_work_dir = _make_job_work_dir(base_work_dir)

    _prep_files(
        work_dir=job_work_dir,
        ci_strings=ci_strings,
        one_body_tensor=one_body_tensor,
        two_body_tensor=two_body_tensor,
        norb=norb,
        nelec=nelec,
        method=solver.method,
        one_body_tensor_b=one_body_tensor_b,
        two_body_tensor_ab=two_body_tensor_ab,
        two_body_tensor_bb=two_body_tensor_bb,
    )

    result = await run_job_from_blocks(
        command_block_name=solver.command_block_name,
        execution_profile_block_name=solver.execution_profile_block_name,
        hpc_profile_block_name=solver.hpc_profile_block_name,
        work_dir=job_work_dir,
        script_filename=solver.script_filename,
        user_args=_build_solver_args(solver),
        watch_poll_interval=5.0,
        timeout_seconds=solver.timeout_seconds,
        metrics_artifact_key=solver.metrics_artifact_key,
    )
    if result.exit_status != 0:
        raise RuntimeError(f"SBDSolverJob failed: exit_status={result.exit_status}")

    return _read_files(work_dir=job_work_dir, norb=norb, method=solver.method, do_rdm=solver.do_rdm)
