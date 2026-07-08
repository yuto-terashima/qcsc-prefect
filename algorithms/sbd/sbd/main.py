# Workflow for observability demo on Miyabi

import os
import pathlib
from typing import Self

import numpy as np
from prefect import flow, get_run_logger, task
from prefect.artifacts import create_table_artifact
from prefect.cache_policies import RUN_ID, Inputs
from prefect.futures import PrefectFutureList
from prefect.task_runners import ConcurrentTaskRunner
from prefect_ray import RayTaskRunner
from pydantic import BaseModel, Field
from qcsc_workflow_utility.chem import (
    ElectronicProperties,
    NpStrict1DArrayF64,
    NpStrict2DArrayF64,
    compute_molecular_integrals_from_fcidump,
)
from qcsc_workflow_utility.orbital_opt import optimize_orbitals, rotate_electronic_properties

from .data_io import extend_table_artifact
from .flow_params import FlowParameters
from .lucj import initialize_ucj_parameters
from .np_type_extension import NpStrict2DArrayBool
from .solver_job import SBDSolverJob
from .sqd import walker_sqd

MODULE_RNG = np.random.default_rng(seed=4574)
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def _build_task_runner():
    mode = os.getenv("SBD_TASK_RUNNER", "ray").strip().lower()
    if mode == "concurrent":
        return ConcurrentTaskRunner()

    # Keep legacy behavior when PREFECT_RAY_NUM_CPUS is not explicitly set.
    ray_cpus_raw = os.getenv("PREFECT_RAY_NUM_CPUS", "").strip()
    if not ray_cpus_raw:
        return RayTaskRunner

    for key, value in THREAD_ENV.items():
        os.environ.setdefault(key, value)

    ray_cpus = int(ray_cpus_raw)
    return RayTaskRunner(
        init_kwargs={
            "num_cpus": ray_cpus,
            "runtime_env": {"env_vars": THREAD_ENV},
        }
    )


class OptimizerState(BaseModel):
    """Intermediate data for optimization."""

    energies: NpStrict1DArrayF64
    populations: NpStrict2DArrayF64
    carryover: NpStrict2DArrayBool
    best_index: int | None = Field(
        default=None,
        ge=0,
    )

    def best_energy(self) -> float | None:
        if self.best_index is None:
            return None
        return float(self.energies[self.best_index])

    def copy(self) -> Self:
        return OptimizerState(
            energies=self.energies.copy(),
            populations=self.populations.copy(),
            carryover=self.carryover.copy(),
            best_index=self.best_index,
        )

    @classmethod
    def from_parameters(
        cls,
        num_walkers: int,
        norb: int,
        n_aa_params: int,
        n_ab_params: int,
        n_reps: int,
    ) -> "OptimizerState":
        num_lucj_params = n_reps * (n_aa_params + n_ab_params + norb**2) + norb**2
        return OptimizerState(
            energies=np.zeros(num_walkers, dtype=np.float64),
            populations=np.full((num_walkers, num_lucj_params), np.nan, dtype=np.float64),
            carryover=np.full((0, norb), np.nan, dtype=bool),
        )


@flow(
    task_runner=_build_task_runner(),
)
def riken_sqd_de(
    parameters: FlowParameters,
):
    logger = get_run_logger()
    logger.info("Task runner mode: %s", os.getenv("SBD_TASK_RUNNER", "ray").strip().lower())

    # ★ fail-fast: solver block existence & sanity check
    slug, name = parse_block_ref(parameters.solver_block_ref)
    if slug != "sbd_solver_job":
        raise ValueError(
            f"solver_block_ref must be 'sbd_solver_job/<name>'. got: {parameters.solver_block_ref}"
        )

    try:
        solver = SBDSolverJob.load(name)
    except Exception:
        logger.exception("Failed to load solver block: %s", parameters.solver_block_ref)
        raise

    logger.info(
        "Solver OK: ref=%s mode=%s",
        parameters.solver_block_ref,
        getattr(solver, "solver_mode", "unknown"),
    )

    telemetry_data = []
    create_table_artifact(
        table=telemetry_data,
        key="sqd-telemetry",
        description="SQD intermediate data.",
    )

    # The solver block is the single source of truth for RHF vs UHF; drive the classical
    # integral computation (and the rest of the open-shell pipeline) from solver.method.
    unrestricted = getattr(solver, "method", "rhf") == "uhf"
    logger.info("Electronic-structure method: %s", "uhf" if unrestricted else "rhf")

    # Orbital optimization is enabled when the solver is configured to compute RDMs (do_rdm != 0).
    # When active, the best-walker RDMs from each trial are used to rotate the Hamiltonian integrals
    # before the next trial, progressively lowering the variational energy across trials.
    do_orbital_opt = getattr(solver, "do_rdm", 0) != 0
    logger.info("Orbital optimization: %s (do_rdm=%d)", "enabled" if do_orbital_opt else "disabled",
                getattr(solver, "do_rdm", 0))

    elec_props = compute_molecular_integrals_from_fcidump(
        fcidump_file=parameters.fcidump,
        unrestricted=unrestricted,
    )

    # We assume heavy-hex topology
    # Orbitals for different spins have connections between every 4th orbital.
    aa_indices = [(p, p + 1) for p in range(elec_props.num_orbitals - 1)]
    ab_indices = [(p, p) for p in range(0, elec_props.num_orbitals, 4)]

    state = OptimizerState.from_parameters(
        num_walkers=parameters.de_params.num_walkers,
        norb=elec_props.num_orbitals,
        n_aa_params=len(aa_indices),
        n_ab_params=len(ab_indices),
        n_reps=parameters.circ_params.n_lucj_layers,
    )

    # Start differential evolution
    for i in range(parameters.de_params.iterations):
        logger.info(f"Running differential evolution trial {i}")

        state, best_sbd_result = differential_evolution_trial(
            trial_index=i,
            parameters=parameters,
            elec_props=elec_props,
            aa_indices=aa_indices,
            ab_indices=ab_indices,
            state=state,
        )

        logger.info(f"Current best energy = {state.best_energy()} (walker {state.best_index})")

        # ── Orbital optimization ────────────────────────────────────────────
        # When the solver writes full RDMs (do_rdm != 0) and the best walker has
        # populated rdm1 / rdm2 fields, rotate the Hamiltonian integrals so that
        # subsequent trials start from an improved orbital basis.
        if do_orbital_opt and best_sbd_result is not None:
            rdm1_aa = best_sbd_result.rdm1
            rdm2_aa = best_sbd_result.rdm2
            if rdm1_aa is not None and rdm2_aa is not None:
                rdm1_bb = best_sbd_result.rdm1_b if best_sbd_result.rdm1_b is not None else rdm1_aa
                rdm2_ab = best_sbd_result.rdm2_ab if best_sbd_result.rdm2_ab is not None else rdm2_aa
                rdm2_bb = best_sbd_result.rdm2_bb if best_sbd_result.rdm2_bb is not None else rdm2_aa
                logger.info(
                    "Trial %d: running orbital optimization (norb=%d, unrestricted=%s) ...",
                    i, elec_props.num_orbitals, unrestricted,
                )
                try:
                    # solver_job writes rdm2 files in prqs-storage (physicist's, prqs axis order).
                    # orbital_opt.optimize_orbitals expects rdm2_notation to be declared explicitly.
                    Ua, Ub, e_opt = optimize_orbitals(
                        elec_props=elec_props,
                        rdm1_aa=rdm1_aa,
                        rdm1_bb=rdm1_bb,
                        rdm2_aa=rdm2_aa,
                        rdm2_ab=rdm2_ab,
                        rdm2_bb=rdm2_bb,
                        rdm2_notation="prqs",
                    )
                    logger.info(
                        "Trial %d: orbital optimization energy = %.10f Ha", i, e_opt
                    )
                    elec_props = rotate_electronic_properties(elec_props, Ua, Ub)
                    logger.info("Trial %d: Hamiltonian rotated for next trial.", i)
                except Exception:
                    logger.exception(
                        "Trial %d: orbital optimization failed; keeping current integrals.", i
                    )
            else:
                logger.info(
                    "Trial %d: RDMs not available in SBDResult (rdm1=%s); "
                    "skipping orbital optimization.",
                    i, "None" if rdm1_aa is None else "present",
                )

    return state.best_energy()


@task(
    task_run_name="de_trial#{trial_index:02d}",
    # Cache on the flow run ID and trial_index.
    # This is roughly identical with the conventional checkpoint mechanism.
    cache_policy=Inputs(
        exclude=[
            "parameters",
            "elec_props",
            "aa_indices",
            "ab_indices",
            "state",
        ]
    )
    + RUN_ID,
)
def differential_evolution_trial(
    trial_index: int,
    parameters: FlowParameters,
    elec_props: ElectronicProperties,
    aa_indices: list[tuple[int, int]],
    ab_indices: list[tuple[int, int]],
    state: OptimizerState,
) -> tuple[OptimizerState, "SBDResult | None"]:
    """Run one trial of differential evolution.

    Returns
    -------
    new_state : OptimizerState
        Updated optimizer state after selection.
    best_sbd_result : SBDResult or None
        The SBDResult of the best-energy walker in this trial (for orbital
        optimization).  None when no walker produced a valid result.
    """
    from .solver_job import SBDResult

    logger = get_run_logger()

    if state.best_index is not None:
        # Create next generation
        trial_populations = mutation_and_crossover(
            current_populations=state.populations,
            best_index=state.best_index,
            scaling_factor=parameters.de_params.fxc,
            crossover_rate=parameters.de_params.cr_prob,
        )
    else:
        # Initialize populations
        trial_populations = initialize_ucj_parameters(
            elec_props=elec_props,
            aa_indices=aa_indices,
            ab_indices=ab_indices,
            num_walkers=parameters.de_params.num_walkers,
            randomization_factor=parameters.de_params.randomization_factor,
            n_lucj_layers=parameters.circ_params.n_lucj_layers,
        )

    _, solver_block_name = parse_block_ref(parameters.solver_block_ref)

    futs = PrefectFutureList()
    for walker_index, ucj_parameter in enumerate(trial_populations):
        prefect_fut = walker_sqd.submit(
            trial_index=trial_index,
            walker_index=walker_index,
            ucj_parameter=ucj_parameter,
            circuit_params=parameters.circ_params,
            elec_props=elec_props,
            aa_indices=aa_indices,
            ab_indices=ab_indices,
            carryover=state.carryover,
            sqd_dim=parameters.sqd_dim,
            solver_block_name=solver_block_name,
            quantum_source=parameters.quantum_source,
            random_seed=parameters.random_seed,
            n_recovery_steps=parameters.n_recovery_steps,
            n_batches=parameters.n_batches,
        )
        futs.append(prefect_fut)

    # Collect results
    result_energies = np.full(parameters.de_params.num_walkers, np.nan, dtype=np.float64)
    result_carryovers: list[NpStrict2DArrayBool] = [None] * parameters.de_params.num_walkers
    result_sbd_results: list[SBDResult | None] = [None] * parameters.de_params.num_walkers
    records: list[dict] = [None] * parameters.de_params.num_walkers
    for walker_index, ((energy, carryover, sbd_result), telemery) in enumerate(futs.result()):
        result_energies[walker_index] = energy
        result_carryovers[walker_index] = carryover
        result_sbd_results[walker_index] = sbd_result
        records[walker_index] = telemery

    # Update artifact
    artifact_id = extend_table_artifact(
        artifact_key="sqd-telemetry",
        new_table=records,
    )
    logger.debug(f"Updated sqd-telemetry artifact {str(artifact_id)}")

    new_state = selection(
        trial_populations=trial_populations,
        trial_energies=result_energies,
        trial_carryovers=result_carryovers,
        current_state=state,
    )

    # Return the SBDResult of the best-energy walker for orbital optimization.
    best_index = int(np.nanargmin(result_energies)) if not np.all(np.isnan(result_energies)) else None
    best_sbd_result: SBDResult | None = (
        result_sbd_results[best_index] if best_index is not None else None
    )

    return new_state, best_sbd_result


@task
def mutation_and_crossover(
    current_populations: NpStrict2DArrayF64,
    best_index: int,
    scaling_factor: float,
    crossover_rate: float,
) -> NpStrict2DArrayF64:
    global MODULE_RNG
    num_walkers, num_params = current_populations.shape

    if num_walkers < 4:
        # Each mutant draws 4 distinct other walkers (a - b + c - d); this is undefined below 4.
        # num_walkers < 4 is only allowed for a single evaluation pass (iterations = 1), which
        # never reaches this function, so reaching here with < 4 is a misconfiguration.
        raise ValueError(
            "Differential-evolution mutation requires num_walkers >= 4 "
            f"(got {num_walkers}); use iterations = 1 for a single-walker evaluation pass."
        )

    mutant = np.zeros_like(current_populations, dtype=np.float64)
    for i in range(num_walkers):
        r1, r2, r3, r4 = MODULE_RNG.choice(
            num_walkers,
            size=4,
            replace=False,
            shuffle=True,
        )
        drift_vec = (
            current_populations[r1]
            - current_populations[r2]
            + current_populations[r3]
            - current_populations[r4]
        )
        mutant[i] = current_populations[best_index] + scaling_factor * drift_vec

    for i in range(num_walkers):
        crossover_weights = MODULE_RNG.random(num_params)
        mask = crossover_weights > crossover_rate
        # Mutate at least one dimension
        index_to_keep = MODULE_RNG.choice(num_params, size=1)
        mask[index_to_keep] = False
        mutant[i, mask] = current_populations[i, mask]

    return mutant


@task
def selection(
    trial_populations: list[NpStrict2DArrayF64],
    trial_energies: NpStrict1DArrayF64,
    trial_carryovers: list[NpStrict2DArrayBool],
    current_state: OptimizerState,
) -> OptimizerState:
    logger = get_run_logger()

    new_state = current_state.copy()

    # The population array is pre-allocated from the spin-balanced (RHF) parameter count in
    # OptimizerState.from_parameters. The spin-unbalanced (UHF) UCJ operator has more parameters
    # (separate alpha/beta orbital rotations plus a beta-beta block), so the actual per-walker
    # vector is longer. Re-size the (still-uninitialized, all-NaN) population array to match the
    # real trial-parameter width the first time we see it; for RHF the widths already agree so
    # this is a no-op, and on later iterations the array holds real data and is left untouched.
    trial_width = int(np.asarray(trial_populations[0]).shape[0])
    if (
        new_state.populations.shape[1] != trial_width
        and np.all(np.isnan(new_state.populations))
    ):
        new_state.populations = np.full(
            (new_state.populations.shape[0], trial_width), np.nan, dtype=np.float64
        )

    best_index = int(np.nanargmin(trial_energies))
    if (
        current_state.best_energy() is None
        or trial_energies[best_index] < current_state.best_energy()
    ):
        # Update carryover when the best energy is updated
        logger.info(f"walker {best_index}: Update the best energy and carryover")
        new_state.best_index = best_index
        new_state.carryover = trial_carryovers[best_index]
    for walker_idx in range(len(trial_energies)):
        if np.isnan(trial_energies[walker_idx]):
            continue
        delta_e = trial_energies[walker_idx] - current_state.energies[walker_idx]
        logger.info(
            f"walker {walker_idx}: Davidson final energy = "
            f"{trial_energies[walker_idx]} (ΔE = {delta_e})"
        )
        if delta_e < 0:
            # Update reference energy and population when the trial gets lower energy
            new_state.energies[walker_idx] = trial_energies[walker_idx]
            new_state.populations[walker_idx] = trial_populations[walker_idx]
    return new_state


def parse_block_ref(ref: str) -> tuple[str, str]:
    parts = ref.split("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Invalid solver_block_ref: {ref}")
    return parts[0], parts[1]


def deploy():
    """Deploy workflow with a local worker."""
    # Prefect deploys with relative path.
    # Workflow is now installed in site-packages.
    os.chdir(pathlib.Path(__file__).parent)

    riken_sqd_de.serve(
        name="riken_sqd_de",
        description="SQD with LUCJ parameter optimization with differential evoluation.",
    )
