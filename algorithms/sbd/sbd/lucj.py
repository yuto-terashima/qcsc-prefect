# Workflow for observability demo on Miyabi

import ffsim
import numpy as np
from prefect import task
from qcsc_workflow_utility.chem import (
    ElectronicProperties,
    NpStrict1DArrayF64,
    NpStrict2DArrayF64,
    NpStrict4DArrayF64,
)
from qiskit.circuit import ClassicalRegister, QuantumCircuit, QuantumRegister

MODULE_RNG = np.random.default_rng(seed=4520)


def _default_bb_indices(
    aa_indices: list[tuple[int, int]],
    bb_indices: list[tuple[int, int]] | None,
) -> list[tuple[int, int]]:
    """Beta-beta interaction pairs default to the same orbital topology as alpha-alpha."""
    return aa_indices if bb_indices is None else bb_indices


@task
def initialize_ucj_parameters(
    elec_props: ElectronicProperties,
    aa_indices: list[tuple[int, int]],
    ab_indices: list[tuple[int, int]],
    num_walkers: int,
    randomization_factor: float,
    n_lucj_layers: int,
    bb_indices: list[tuple[int, int]] | None = None,
) -> NpStrict2DArrayF64:
    global MODULE_RNG

    if elec_props.unrestricted:
        return _initialize_ucj_parameters_uhf(
            elec_props=elec_props,
            aa_indices=aa_indices,
            ab_indices=ab_indices,
            bb_indices=_default_bb_indices(aa_indices, bb_indices),
            num_walkers=num_walkers,
            randomization_factor=randomization_factor,
            n_lucj_layers=n_lucj_layers,
        )

    def _t2_to_ucj_parameters(t2: NpStrict4DArrayF64) -> NpStrict1DArrayF64:
        nonlocal aa_indices
        nonlocal ab_indices
        nonlocal n_lucj_layers

        # optimize=True performs the "compressed double factorization": the kept DF terms are
        # re-optimized to best approximate the full UCCSD generator. This can raise the prepared
        # state's <H> (Trotter error) but for SQD that is beneficial — it spreads the wavefunction
        # over many more configurations, producing a far more diverse sample set to diagonalize
        # over (measured ~6x participation ratio vs naive truncation, which is near single-HF).
        tmp_operator = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
            t2=t2,
            n_reps=n_lucj_layers + 1,
            interaction_pairs=(aa_indices, ab_indices),
            optimize=True,
            options={"maxiter": 50},
        )
        truncated_ucj_op = ffsim.UCJOpSpinBalanced(
            diag_coulomb_mats=tmp_operator.diag_coulomb_mats[:-1],
            orbital_rotations=tmp_operator.orbital_rotations[:-1],
            final_orbital_rotation=tmp_operator.orbital_rotations[-1],
        )
        return truncated_ucj_op.to_parameters(interaction_pairs=(aa_indices, ab_indices))

    # First walker is the bare CCSD parameters
    initial_params = [_t2_to_ucj_parameters(t2=elec_props.t2)]

    # The rest of walkers are randomized parameters
    for _ in range(num_walkers - 1):
        rand_values = randomization_factor * (MODULE_RNG.random(elec_props.t2.shape) - 0.5)
        drifted_params = _t2_to_ucj_parameters(t2=elec_props.t2 + rand_values)
        initial_params.append(drifted_params)
    return initial_params


def _initialize_ucj_parameters_uhf(
    elec_props: ElectronicProperties,
    aa_indices: list[tuple[int, int]],
    ab_indices: list[tuple[int, int]],
    bb_indices: list[tuple[int, int]],
    num_walkers: int,
    randomization_factor: float,
    n_lucj_layers: int,
) -> NpStrict2DArrayF64:
    """Spin-unbalanced (UHF) counterpart of :func:`initialize_ucj_parameters`.

    The UCCSD t2 tuple ``(t2aa, t2ab, t2bb)`` lives across ``elec_props.t2`` / ``.t2_ab`` /
    ``.t2_bb``; interaction pairs become a 3-tuple ``(aa, ab, bb)``. Randomization perturbs all
    three blocks independently (they have different shapes).
    """
    global MODULE_RNG
    interaction_pairs = (aa_indices, ab_indices, bb_indices)

    def _t2_to_ucj_parameters(
        t2: tuple[NpStrict4DArrayF64, NpStrict4DArrayF64, NpStrict4DArrayF64],
    ) -> NpStrict1DArrayF64:
        tmp_operator = ffsim.UCJOpSpinUnbalanced.from_t_amplitudes(
            t2=t2,
            n_reps=n_lucj_layers + 1,
            interaction_pairs=interaction_pairs,
        )
        truncated_ucj_op = ffsim.UCJOpSpinUnbalanced(
            diag_coulomb_mats=tmp_operator.diag_coulomb_mats[:-1],
            orbital_rotations=tmp_operator.orbital_rotations[:-1],
            final_orbital_rotation=tmp_operator.orbital_rotations[-1],
        )
        return truncated_ucj_op.to_parameters(interaction_pairs=interaction_pairs)

    t2_tuple = (elec_props.t2, elec_props.t2_ab, elec_props.t2_bb)

    # First walker is the bare UCCSD parameters
    initial_params = [_t2_to_ucj_parameters(t2=t2_tuple)]

    # The rest of walkers are randomized parameters; perturb each spin block independently.
    for _ in range(num_walkers - 1):
        drifted = tuple(
            block + randomization_factor * (MODULE_RNG.random(block.shape) - 0.5)
            for block in t2_tuple
        )
        initial_params.append(_t2_to_ucj_parameters(t2=drifted))
    return initial_params


@task
def create_lucj_circuit(
    ucj_parameter: NpStrict1DArrayF64,
    elec_props: ElectronicProperties,
    aa_indices: list[tuple[int, int]],
    ab_indices: list[tuple[int, int]],
    n_lucj_layers: int,
    use_reset_mitigation: bool,
    bb_indices: list[tuple[int, int]] | None = None,
) -> QuantumCircuit:
    qreg = QuantumRegister(2 * elec_props.num_orbitals, name="q")
    creg_test = ClassicalRegister(2 * elec_props.num_orbitals, name="test")
    creg_meas = ClassicalRegister(2 * elec_props.num_orbitals, name="meas")

    regs = [qreg, creg_meas]
    if use_reset_mitigation:
        regs.append(creg_test)

    circ = QuantumCircuit(*regs)
    if use_reset_mitigation:
        circ.measure(qreg, creg_test)
        circ.barrier()
    circ.append(
        ffsim.qiskit.PrepareHartreeFockJW(
            norb=elec_props.num_orbitals,
            nelec=elec_props.num_electrons,
        ),
        qargs=qreg,
    )
    if elec_props.unrestricted:
        interaction_pairs = (aa_indices, ab_indices, _default_bb_indices(aa_indices, bb_indices))
        ucj_op = ffsim.UCJOpSpinUnbalanced.from_parameters(
            params=ucj_parameter,
            norb=elec_props.num_orbitals,
            n_reps=n_lucj_layers,
            interaction_pairs=interaction_pairs,
            with_final_orbital_rotation=True,
        )
        circ.append(
            ffsim.qiskit.UCJOpSpinUnbalancedJW(ucj_op=ucj_op),
            qargs=qreg,
        )
    else:
        ucj_op = ffsim.UCJOpSpinBalanced.from_parameters(
            params=ucj_parameter,
            norb=elec_props.num_orbitals,
            n_reps=n_lucj_layers,
            interaction_pairs=(aa_indices, ab_indices),
            with_final_orbital_rotation=True,
        )
        circ.append(
            ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op=ucj_op),
            qargs=qreg,
        )
    circ.measure(qreg, creg_meas)
    return circ
