# Workflow for observability demo on Miyabi

import asyncio
from datetime import datetime, timezone
from time import perf_counter
from typing import Any

import numpy as np
from prefect import get_run_logger, task
from prefect.variables import Variable
from prefect_qiskit import QuantumRuntime
from qcsc_workflow_utility.chem import ElectronicProperties, NpStrict1DArrayF64
from qiskit.primitives.containers import BitArray
from qiskit_addon_sqd.configuration_recovery import (
    post_select_by_hamming_weight,
)
from qiskit_addon_sqd.configuration_recovery import (
    recover_configurations as _recover_configurations,
)
from qiskit_addon_sqd.counts import bit_array_to_arrays, generate_bit_array_uniform

from .data_io import load_ndarray, save_ndarray
from .flow_params import CircuitParameters
from .lucj import create_lucj_circuit
from .np_type_extension import (
    NpStrict1DArrayLL,
    NpStrict2DArrayBool,
)
from .solver_job import SBDResult, SBDSolverJob
from .transpile_custom import find_optimal_layout, transpile_circuit

# Convert Addon function into Prefect Task
recover_configurations = task(_recover_configurations)

MODULE_RNG = np.random.default_rng(seed=1333)


def _spin_halves_as_ints(
    bitstrings: np.ndarray, norb: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split (beta-left, alpha-right) bool bitstrings into per-spin integer strings.

    Layout matches subsample_open_shell / recover_configurations: bits [:norb] are beta,
    bits [norb:] are alpha; bit 0 is the most-significant in the printed string.
    """
    weights = (1 << np.arange(norb - 1, -1, -1)).astype(np.int64)
    beta = (bitstrings[:, :norb].astype(np.int64) * weights).sum(axis=1)
    alpha = (bitstrings[:, norb:].astype(np.int64) * weights).sum(axis=1)
    return alpha, beta


def _comprehensive_summary(
    logger,
    *,
    raw_bitstrings: np.ndarray,
    raw_probs: np.ndarray,
    norb: int,
    num_elec_a: int,
    num_elec_b: int,
    sqd_dim: int,
    n_batches: int,
    n_recovery_steps: int,
    best_energy: float,
    best_net_dim: int,
    work_dir: str | None = None,
    walker_tag: str = "",
) -> None:
    """Emit a comprehensive end-of-walker diagnostic: distinct-string counts, the sqd_dim mapping,
    per-spin probability-weighted frequencies, a top-20 table, and (if matplotlib is available) a
    saved bar plot of the top-20 string probabilities.

    All quantities are derived from the raw measured pool (raw_bitstrings / raw_probs), so this
    documents exactly what the device produced and how it maps into the diagonalized subspace.
    """
    n_total = int(raw_bitstrings.shape[0])
    if n_total == 0:
        logger.info("[summary]%s empty raw pool; nothing to summarize.", walker_tag)
        return

    probs = np.asarray(raw_probs, dtype=np.float64)
    alpha_ints, beta_ints = _spin_halves_as_ints(raw_bitstrings, norb)
    hf_a = (1 << num_elec_a) - 1

    # Aggregate probability per distinct spin-half string (a half repeats across many full configs).
    def _agg(ints: np.ndarray):
        uniq, inv = np.unique(ints, return_inverse=True)
        pw = np.zeros(uniq.size, dtype=np.float64)
        np.add.at(pw, inv, probs)
        order = np.argsort(pw)[::-1]
        return uniq[order], pw[order]

    a_uniq, a_pw = _agg(alpha_ints)
    b_uniq, b_pw = _agg(beta_ints)

    import math as _math

    full_a = _math.comb(norb, num_elec_a)
    full_b = _math.comb(norb, num_elec_b)
    dets_per_spin = int(sqd_dim**0.5)

    def _exc(x: int, ne: int) -> int:
        hf = (1 << ne) - 1
        return bin(int(x) ^ hf).count("1") // 2

    logger.info("[summary]%s ================= COMPREHENSIVE WALKER SUMMARY =================",
                walker_tag)
    logger.info("[summary]%s system: norb=%d nelec=(%d,%d)  |  best_energy=%.6f",
                walker_tag, norb, num_elec_a, num_elec_b, best_energy)
    logger.info("[summary]%s raw pool: %d measured configs, %d distinct full configs",
                walker_tag, n_total, int(np.unique(raw_bitstrings, axis=0).shape[0]))
    logger.info(
        "[summary]%s distinct ALPHA strings seen: %d / %d possible (%.3f%%)  |  "
        "distinct BETA strings seen: %d / %d possible (%.3f%%)",
        walker_tag, a_uniq.size, full_a, 100.0 * a_uniq.size / full_a,
        b_uniq.size, full_b, 100.0 * b_uniq.size / full_b,
    )
    logger.info(
        "[summary]%s sqd_dim=%d -> sqrt=%d dets/spin -> net CI matrix = %d x %d = %d "
        "(kept top %d of %d distinct alpha, top %d of %d distinct beta per batch); "
        "n_batches=%d, n_recovery_steps=%d, achieved net_dim=%d",
        walker_tag, sqd_dim, dets_per_spin, dets_per_spin, dets_per_spin,
        dets_per_spin * dets_per_spin,
        min(dets_per_spin, a_uniq.size), a_uniq.size,
        min(dets_per_spin, b_uniq.size), b_uniq.size,
        n_batches, n_recovery_steps, best_net_dim,
    )
    # Probability mass captured by the top dets_per_spin strings (what one batch can hold).
    a_top_mass = float(a_pw[:dets_per_spin].sum())
    b_top_mass = float(b_pw[:dets_per_spin].sum())
    logger.info(
        "[summary]%s probability mass in top-%d strings: alpha=%.4f beta=%.4f  "
        "(alpha p: max=%.2e median=%.2e min=%.2e)",
        walker_tag, dets_per_spin, a_top_mass, b_top_mass,
        float(a_pw[0]), float(np.median(a_pw)), float(a_pw[-1]),
    )

    # Top-20 ALPHA strings table.
    top = min(20, a_uniq.size)
    logger.info("[summary]%s top-%d ALPHA strings by probability:", walker_tag, top)
    logger.info("[summary]%s   rank  bitstring%s  prob       exc  (HF=%0*d)",
                walker_tag, " " * max(0, norb - 9), norb, hf_a)
    for i in range(top):
        s = format(int(a_uniq[i]), f"0{norb}b")
        logger.info("[summary]%s   %4d  %s  %.4e  %3d", walker_tag, i + 1, s,
                    float(a_pw[i]), _exc(int(a_uniq[i]), num_elec_a))
    # Top-20 BETA strings table.
    topb = min(20, b_uniq.size)
    logger.info("[summary]%s top-%d BETA strings by probability:", walker_tag, topb)
    for i in range(topb):
        s = format(int(b_uniq[i]), f"0{norb}b")
        logger.info("[summary]%s   %4d  %s  %.4e  %3d", walker_tag, i + 1, s,
                    float(b_pw[i]), _exc(int(b_uniq[i]), num_elec_b))

    # Optional plot of the top-20 alpha/beta probabilities, saved next to the work dir.
    try:
        import os

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, uniq, pw, label, ne in (
            (axes[0], a_uniq, a_pw, "alpha", num_elec_a),
            (axes[1], b_uniq, b_pw, "beta", num_elec_b),
        ):
            k = min(20, uniq.size)
            labels = [format(int(uniq[i]), f"0{norb}b") for i in range(k)]
            colors = [
                "#2c3e50" if _exc(int(uniq[i]), ne) == 0
                else ("#27ae60" if _exc(int(uniq[i]), ne) <= 2 else "#c0392b")
                for i in range(k)
            ]
            ax.bar(range(k), pw[:k], color=colors)
            ax.set_xticks(range(k))
            ax.set_xticklabels(labels, rotation=90, fontsize=5, family="monospace")
            ax.set_ylabel("aggregated probability")
            ax.set_title(f"top-{k} {label} strings (HF=navy, S/D=green, >2exc=red)")
        fig.tight_layout()
        out_dir = work_dir if work_dir and os.path.isdir(work_dir) else "."
        out = os.path.join(out_dir, f"summary_top20{walker_tag.replace(' ', '_') or ''}.png")
        fig.savefig(out, dpi=130)
        plt.close(fig)
        logger.info("[summary]%s top-20 probability plot saved: %s", walker_tag, out)
    except Exception as exc:  # plotting is best-effort; never fail the run on it
        logger.info("[summary]%s plot skipped (%s: %s)", walker_tag,
                    type(exc).__name__, str(exc)[:120])

    logger.info("[summary]%s ===============================================================",
                walker_tag)


def _excitation_summary(ci_strings: np.ndarray, num_elec: int) -> str:
    """Summarize a determinant list by excitation level from Hartree-Fock.

    The Hamiltonian only couples HF to single/double excitations (Slater-Condon), so a subspace
    dominated by high (>2) excitations cannot lower the energy below HF. This is the key diagnostic
    for why a large subspace can still collapse to the SCF energy. Returns a compact histogram.
    """
    ci = np.asarray(ci_strings, dtype=np.int64).reshape(-1)
    if ci.size == 0:
        return "empty"
    hf = (1 << num_elec) - 1  # lowest num_elec orbitals occupied
    # excitation level = (number of differing occupied orbitals) / 2
    exc = np.array([bin(int(x) ^ hf).count("1") // 2 for x in ci])
    n_hf = int(np.sum(exc == 0))
    n_sd = int(np.sum((exc >= 1) & (exc <= 2)))  # singles+doubles (couple to HF)
    n_high = int(np.sum(exc >= 3))
    return (
        f"n={ci.size} unique={len(set(ci.tolist()))} HF={n_hf} "
        f"singles+doubles={n_sd} higher(>2)={n_high} max_exc={int(exc.max())}"
    )


@task(
    task_run_name="run_sqd_#{trial_index:02d}-{walker_index}",
)
def walker_sqd(
    trial_index: int,
    walker_index: int,
    ucj_parameter: NpStrict1DArrayF64,
    circuit_params: CircuitParameters,
    elec_props: ElectronicProperties,
    aa_indices: list[tuple[int, int]],
    ab_indices: list[tuple[int, int]],
    carryover: NpStrict2DArrayBool,
    sqd_dim: int,
    solver_block_name: str,
    quantum_source: str,
    random_seed: int,
    n_recovery_steps: int = 1,
    n_batches: int = 1,
) -> tuple[tuple[float, NpStrict2DArrayBool, "SBDResult | None"], dict[str, Any]]:
    logger = get_run_logger()
    davidson_solver = SBDSolverJob.load(solver_block_name)

    telemetry = {
        "trial_index": trial_index,
        "walker_index": walker_index,
    }

    options = Variable.get("sqd_options", default={"params": {"shots": 100_000}})
    runtime = None

    if quantum_source == "real-device":
        try:
            runtime = QuantumRuntime.load("ibm-runner")
        except ValueError as exc:
            raise RuntimeError(
                "Quantum source 'real-device' requested but QuantumRuntime block "
                "'ibm-runner' is not defined. Set quantum_source='random' to use "
                "deterministic random sampling instead."
            ) from exc

    if runtime is not None:
        logger.info("Preparing quantum sampling on backend %s.", runtime.resource_name)
        target_start = perf_counter()
        target = runtime.get_target()
        logger.info(
            "Loaded backend target for %s in %.2fs.",
            runtime.resource_name,
            perf_counter() - target_start,
        )
        vir_circuit = create_lucj_circuit(
            ucj_parameter=ucj_parameter,
            elec_props=elec_props,
            aa_indices=aa_indices,
            ab_indices=ab_indices,
            n_lucj_layers=circuit_params.n_lucj_layers,
            use_reset_mitigation=circuit_params.use_reset_mitigation,
        )
        logger.info(
            "Searching ISA layout for backend %s "
            "(max_iterations=%s, swap_trials=%s, layout_trials=%s).",
            runtime.resource_name,
            circuit_params.sabre_max_iterations,
            circuit_params.sabre_swap_trials,
            circuit_params.sabre_layout_trials,
        )
        layout = find_optimal_layout(
            test_circuit=vir_circuit,
            target=target,
            optimization_level=circuit_params.optimization_level,
            max_iterations=circuit_params.sabre_max_iterations,
            swap_trials=circuit_params.sabre_swap_trials,
            layout_trials=circuit_params.sabre_layout_trials,
        )
        logger.info("Transpiling ISA circuit for backend %s.", runtime.resource_name)
        transpile_start = perf_counter()
        isa_circuit = transpile_circuit(
            circuit=vir_circuit,
            target=target,
            layout=layout,
            optimization_level=circuit_params.optimization_level,
        )
        logger.info(
            "Completed ISA transpilation for %s in %.2fs.",
            runtime.resource_name,
            perf_counter() - transpile_start,
        )
        # Shot batching: reach a large effective shot count by submitting K smaller sampler jobs
        # and mixing them, instead of one huge job (which times out / can exceed IBM's per-job
        # cap). n_shot_batches lives at the TOP LEVEL of the options dict (not inside params, which
        # hits the IBM REST schema). K=1 (default) is byte-for-byte the original single submission.
        total_shots = int(options.get("params", {}).get("shots", 100_000))
        n_shot_batches = max(1, int(options.get("n_shot_batches", 1)))
        per_batch_shots = max(1, total_shots // n_shot_batches)

        # Per-call options must carry only the IBM params; strip the harness-only n_shot_batches.
        batch_options = {k: v for k, v in options.items() if k != "n_shot_batches"}
        batch_options = {**batch_options, "params": {**batch_options.get("params", {})}}
        batch_options["params"]["shots"] = per_batch_shots

        logger.info(
            "Submitting sampler workload to %s (shots=%s = %d batch(es) x %d).",
            runtime.resource_name,
            total_shots,
            n_shot_batches,
            per_batch_shots,
        )
        sampling_start = perf_counter()
        batch_arrays = []
        raw_shot_total = 0
        kept_shot_total = 0
        for shot_batch in range(n_shot_batches):
            try:
                pub_result = runtime.sampler(
                    sampler_pubs=[(isa_circuit,)],
                    options=batch_options,
                    tags=["res: quantum"],
                )
            except Exception:
                logger.exception(
                    "Sampler submission or execution failed for backend %s (shot batch %d/%d).",
                    runtime.resource_name,
                    shot_batch + 1,
                    n_shot_batches,
                )
                raise
            # Reset mitigation is applied per batch (the test register is per submission).
            meas_bits = pub_result[0].data.meas
            if circuit_params.use_reset_mitigation:
                test_bits = pub_result[0].data.test
                kept = meas_bits.get_bitstrings(test_bits.bitcount() == 0)
                if len(kept) == 0:
                    # The reset-test register rejected every shot in this batch. This is expected on
                    # mock/simulator backends that do not model the reset ancilla; on real hardware
                    # it would mean a wholly failed batch. Either way, an empty batch must not crash
                    # the flow (BitArray.from_samples([]) raises), so fall back to the unmitigated
                    # shots for this batch and warn.
                    logger.warning(
                        "Reset mitigation kept 0/%d shots on %s (shot batch %d/%d); "
                        "falling back to unmitigated shots for this batch.",
                        meas_bits.num_shots,
                        runtime.resource_name,
                        shot_batch + 1,
                        n_shot_batches,
                    )
                    batch_array = meas_bits
                else:
                    batch_array = BitArray.from_samples(kept, num_bits=meas_bits.num_bits)
            else:
                batch_array = meas_bits
            batch_arrays.append(batch_array)
            raw_shot_total += int(meas_bits.num_shots)
            kept_shot_total += int(batch_array.num_shots)
            if n_shot_batches > 1:
                logger.info(
                    "[diag] shot batch %d/%d: %d kept / %d shots",
                    shot_batch + 1, n_shot_batches, batch_array.num_shots, meas_bits.num_shots,
                )

        # Mix the batches into a single pool (sum of shots); K=1 just uses the lone array.
        bit_array = (
            batch_arrays[0] if n_shot_batches == 1 else BitArray.concatenate_shots(batch_arrays)
        )
        logger.info(
            "Completed %d sampler batch(es) on %s in %.2fs (merged %d kept shots).",
            n_shot_batches,
            runtime.resource_name,
            perf_counter() - sampling_start,
            kept_shot_total,
        )
        # Update application telemetry (aggregate retention over all batches).
        telemetry.update(
            shot_retention_rate=float(kept_shot_total / raw_shot_total) if raw_shot_total else 0.0,
        )

        # Persist the merged sample pool ONCE, before any diagonalization, so this (expensive,
        # real-hardware) sample can be re-diagonalized later without re-sampling. Best-effort:
        # a failed save must never abort a hardware run. Reload with quantum_source="saved".
        try:
            pool_path = save_ndarray(
                file_prefix=f"raw_samples_t{trial_index:02d}_w{walker_index}",
                packed_bits=bit_array.array,
                num_bits=np.array([bit_array.num_bits], dtype=np.int64),
                num_shots=np.array([bit_array.num_shots], dtype=np.int64),
            )
            telemetry["raw_samples_path"] = pool_path
            logger.info(
                "[persist] merged %d-shot pool (trial %d walker %d) saved: %s",
                int(bit_array.num_shots), trial_index, walker_index, pool_path,
            )
        except Exception:
            logger.exception("[persist] failed to save merged sample pool (continuing).")
    elif quantum_source == "saved":
        # Reload a previously persisted merged pool and diagonalize from it — no IBM call, no
        # credentials. options["saved_samples"] is a per-walker list of saved-artifact paths (the
        # file://... or S3 keys returned by the persist step above).
        saved_paths = options.get("saved_samples")
        if not saved_paths or walker_index >= len(saved_paths):
            raise RuntimeError(
                "quantum_source='saved' but no saved_samples path for walker "
                f"{walker_index} (got {saved_paths!r}). Set --saved-samples-dir / "
                "SBD_SAVED_SAMPLES_DIR to the persisted pool(s)."
            )
        pool_path = saved_paths[walker_index]
        logger.info("Loading saved sample pool for walker %d: %s", walker_index, pool_path)
        packed = load_ndarray(pool_path, "packed_bits")
        num_bits = int(load_ndarray(pool_path, "num_bits")[0])
        bit_array = BitArray(packed, num_bits)
        logger.info(
            "[saved] loaded %d-shot pool (%d bits) from %s",
            int(bit_array.num_shots), num_bits, pool_path,
        )
    else:
        # Random sampling
        # Isolate bitstring seed from the module seed for equivalent control with real device path.
        seed = int(
            random_seed
            + ((trial_index + walker_index) * (trial_index + walker_index + 1) // 2 + walker_index)
        )
        logger.info("Sampling bitstrings with RNG seed %s", seed)
        bit_array = generate_bit_array_uniform(
            num_samples=options.get("params", {}).get("shots", 100_000),
            num_bits=elec_props.num_orbitals * 2,
            rand_seed=seed,
        )

    logger.debug("Starting configuration recovery and diagonalization.")
    raw_bitstrings, raw_probs = bit_array_to_arrays(bit_array)
    norb = elec_props.num_orbitals
    num_elec_a, num_elec_b = elec_props.num_electrons

    # SQD self-consistency configuration-recovery loop.
    #
    # Canonical SQD iterates: recover configurations using the current average orbital
    # occupancies -> postselect -> subsample -> exactly diagonalize the subspace -> read updated
    # occupancies from the solver -> recover again. Recovery always re-uses the SAME raw quantum
    # samples (no re-sampling of the device); only the occupancies, and hence the recovered/
    # subsampled subspace, evolve. We keep the best (lowest-energy) pass.
    #
    # avg_occ starts from the CCSD natural-orbital occupancies (per-spin for UHF) and is refreshed
    # each pass from the SBD solver's orbital_occupancies. n_recovery_steps=1 reproduces the
    # previous single-pass behavior exactly.
    avg_occ = elec_props.initial_occupancy

    # Per-walker random generator. recover_configurations and the subsample routines must draw
    # INDEPENDENT subspaces for each walker, otherwise differential evolution / multi-walker
    # averaging sees correlated draws and loses its diversity. A module-global Generator also races
    # under the threaded (concurrent) task runner. Derive a deterministic, per-(trial, walker)
    # seed -- reusing the same Cantor-pairing offset as the random-source path above -- so runs stay
    # reproducible while every walker is independent.
    walker_seed = int(
        random_seed
        + ((trial_index + walker_index) * (trial_index + walker_index + 1) // 2 + walker_index)
    )
    walker_rng = np.random.default_rng(seed=walker_seed)

    best_energy: float | None = None
    best_carryover: NpStrict2DArrayBool | None = None
    best_sbd_result: "SBDResult | None" = None
    best_report_s3: str | None = None
    best_num_post = 0
    best_net_dim = 0

    # One-time diagnostics on the raw quantum samples (before any recovery).
    logger.info(
        "[diag] raw samples: %d bitstrings, %d unique (norb=%d, nelec=(%d,%d))",
        raw_bitstrings.shape[0],
        len({row.tobytes() for row in raw_bitstrings}),
        norb,
        num_elec_a,
        num_elec_b,
    )

    for recovery_step in range(n_recovery_steps):
        # [diag] occupancies feeding this recovery pass (per-spin); fractional values are what
        # configuration recovery needs to bias toward correlated configs.
        occ_a_arr, occ_b_arr = np.asarray(avg_occ[0]), np.asarray(avg_occ[1])
        n_frac = int(np.sum((occ_a_arr > 0.01) & (occ_a_arr < 0.99))) + int(
            np.sum((occ_b_arr > 0.01) & (occ_b_arr < 0.99))
        )
        logger.info(
            "[diag] recovery %d/%d input occ: sum_a=%.3f sum_b=%.3f fractional=%d",
            recovery_step + 1,
            n_recovery_steps,
            float(occ_a_arr.sum()),
            float(occ_b_arr.sum()),
            n_frac,
        )

        bitstrings, probs = recover_configurations(
            bitstring_matrix=raw_bitstrings,
            probabilities=raw_probs,
            avg_occupancies=avg_occ,
            num_elec_a=num_elec_a,
            num_elec_b=num_elec_b,
            rand_seed=walker_rng,
        )
        bitstrings_post, probs_post = postselect_bitstrings(
            bitstring_matrix=bitstrings,
            probabilities=probs,
            hamming_right=num_elec_a,
            hamming_left=num_elec_b,
        )
        # [diag] how many configs survive Hamming-weight post-selection (the usable subspace seed).
        logger.info(
            "[diag] recovery %d/%d post-select: %d / %d configs survived (%d unique)",
            recovery_step + 1,
            n_recovery_steps,
            bitstrings_post.shape[0],
            bitstrings.shape[0],
            len({row.tobytes() for row in bitstrings_post}) if bitstrings_post.shape[0] else 0,
        )

        # [diag] chemistry sanity: every post-selected config must sit in exactly the
        # (num_elec_a alpha, num_elec_b beta) sector, which fixes Sz = (na - nb)/2 exactly. The
        # bitstring layout is (beta-left [:norb], alpha-right [norb:]); see subsample_open_shell.
        if bitstrings_post.shape[0]:
            beta_pop = bitstrings_post[:, :norb].sum(axis=1)
            alpha_pop = bitstrings_post[:, norb:].sum(axis=1)
            sector_ok = bool(
                np.all(alpha_pop == num_elec_a) and np.all(beta_pop == num_elec_b)
            )
            if not sector_ok:
                logger.warning(
                    "[diag] recovery %d/%d post-select LEAKED wrong (Na,Nb) sector: "
                    "alpha in %s, beta in %s (expected %d, %d)",
                    recovery_step + 1, n_recovery_steps,
                    sorted(set(alpha_pop.tolist())), sorted(set(beta_pop.tolist())),
                    num_elec_a, num_elec_b,
                )

        # K-batch SQD (arXiv:2405.05068): draw n_batches independent subspaces from the SAME
        # recovered distribution, diagonalize each, then take the MINIMUM energy as the variational
        # estimate and the MEAN occupancy across batches to feed the next recovery pass. A single
        # batch's occupancy is too noisy to steer recovery; averaging over batches stabilizes it.
        if elec_props.unrestricted:
            carryover_b = carryover[:, :norb]
            carryover_a = (
                carryover[:, norb:] if carryover.shape[1] >= 2 * norb else carryover[:, :norb]
            )

        batch_energy: float | None = None  # min over batches (the variational estimate)
        batch_carryover = None  # carryover of the best (lowest-energy) batch
        batch_sbd_result = None  # SBDResult of the best (lowest-energy) batch
        batch_net_dim = 0
        batch_save_kwargs: dict = {}
        batch_energies: list[float] = []  # every batch's total energy, for a min/mean/std summary
        occ_a_accum = np.zeros(norb, dtype=np.float64)
        occ_b_accum = np.zeros(norb, dtype=np.float64)

        for batch in range(n_batches):
            if elec_props.unrestricted:
                ci_a, ci_b = subsample_open_shell(
                    bitstring_matrix=bitstrings_post,
                    probabilities=probs_post,
                    carryover_a=carryover_a,
                    carryover_b=carryover_b,
                    subspace_dim=sqd_dim,
                    norb=norb,
                    num_elec_a=num_elec_a,
                    num_elec_b=num_elec_b,
                    rng=walker_rng,
                )
                if batch == 0:
                    # [diag] excitation profile of batch 0 (make-or-break: a subspace dominated by
                    # >2 excitations cannot couple to HF and the energy stays at SCF).
                    logger.info("[diag] recovery %d/%d alpha dets: %s", recovery_step + 1,
                                n_recovery_steps, _excitation_summary(ci_a, num_elec_a))
                    logger.info("[diag] recovery %d/%d beta  dets: %s", recovery_step + 1,
                                n_recovery_steps, _excitation_summary(ci_b, num_elec_b))
                sbd_result = asyncio.run(
                    davidson_solver.run(
                        ci_strings=(ci_a, ci_b),
                        one_body_tensor=elec_props.one_body_tensor,
                        two_body_tensor=elec_props.two_body_tensor,
                        norb=norb,
                        nelec=elec_props.num_electrons,
                        one_body_tensor_b=elec_props.one_body_tensor_b,
                        two_body_tensor_ab=elec_props.two_body_tensor_ab,
                        two_body_tensor_bb=elec_props.two_body_tensor_bb,
                    )
                )
                this_carryover = _stack_spin_carryover(
                    carryover_a=sbd_result.carryover_bitstrings,
                    carryover_b=sbd_result.carryover_bitstrings_b,
                    norb=norb,
                )
                this_net_dim = int(len(ci_a) * len(ci_b))
                this_save_kwargs = dict(alphadets=ci_a, betadets=ci_b)
            else:
                ci_strings = subsample_close_shell(
                    bitstring_matrix=bitstrings_post,
                    probabilities=probs_post,
                    carryover=carryover,
                    subspace_dim=sqd_dim,
                    norb=norb,
                    num_elec_a=num_elec_a,
                    rng=walker_rng,
                )
                if batch == 0:
                    logger.info("[diag] recovery %d/%d dets: %s", recovery_step + 1,
                                n_recovery_steps, _excitation_summary(ci_strings, num_elec_a))
                sbd_result = asyncio.run(
                    davidson_solver.run(
                        ci_strings=(ci_strings, ci_strings),
                        one_body_tensor=elec_props.one_body_tensor,
                        two_body_tensor=elec_props.two_body_tensor,
                        norb=norb,
                        nelec=elec_props.num_electrons,
                    )
                )
                this_carryover = sbd_result.carryover_bitstrings
                this_net_dim = int(len(ci_strings) ** 2)
                this_save_kwargs = dict(alphadets=ci_strings)

            this_energy = sbd_result.energy + elec_props.nuclear_repulsion_energy
            batch_energies.append(float(this_energy))
            # Accumulate occupancies (averaged over batches -> next recovery pass).
            occ_a_accum += np.asarray(sbd_result.orbital_occupancies[0], dtype=np.float64)
            occ_b_accum += np.asarray(sbd_result.orbital_occupancies[1], dtype=np.float64)
            logger.info(
                "[diag] recovery %d/%d batch %d/%d: energy = %.6f (net dim = %d)",
                recovery_step + 1, n_recovery_steps, batch + 1, n_batches,
                this_energy, this_net_dim,
            )
            # Keep the lowest-energy batch as this step's variational estimate.
            if batch_energy is None or this_energy < batch_energy:
                batch_energy = this_energy
                batch_carryover = this_carryover
                batch_sbd_result = sbd_result  # track best-batch SBDResult for orbital opt
                batch_net_dim = this_net_dim
                batch_save_kwargs = this_save_kwargs

        # [diag] report BOTH the reported (min) energy and the across-batch spread. The min is the
        # variational estimate we keep; the spread shows how lucky the best draw was and the K-batch
        # variance (large std => more batches / larger sqd_dim would likely help).
        if batch_energies:
            _be = np.asarray(batch_energies, dtype=np.float64)
            logger.info(
                "[diag] recovery %d/%d batch energies: min=%.6f mean=%.6f max=%.6f std=%.6f "
                "spread=%.1f mHa (n=%d)",
                recovery_step + 1, n_recovery_steps,
                float(_be.min()), float(_be.mean()), float(_be.max()), float(_be.std()),
                float((_be.max() - _be.min()) * 1000.0), _be.size,
            )

        energy = batch_energy
        step_carryover = batch_carryover
        step_sbd_result = batch_sbd_result  # SBDResult of the best batch in this recovery step
        net_dim = batch_net_dim
        save_kwargs = batch_save_kwargs
        # Mean occupancy across batches -> the occupancies that steer the NEXT recovery pass.
        step_occ_a = occ_a_accum / n_batches
        step_occ_b = occ_b_accum / n_batches

        # [diag] per-orbital spin density (n_alpha - n_beta) of the diagonalized state. Large
        # localized values indicate the spin polarization UHF is meant to capture; the sum equals
        # 2*Sz = (Na - Nb). This is the cheap, RDM-free spin check (full <S^2> needs the 2-RDM).
        spin_density = step_occ_a - step_occ_b
        logger.info(
            "[diag] recovery %d/%d spin density (n_a-n_b): sum=%.3f (=2Sz=%d) per-orb=%s",
            recovery_step + 1,
            n_recovery_steps,
            float(spin_density.sum()),
            num_elec_a - num_elec_b,
            np.array2string(spin_density, precision=3, max_line_width=200),
        )

        logger.info(
            "Recovery step %d/%d: best energy = %.6f over %d batch(es) (net subspace dim = %d)",
            recovery_step + 1,
            n_recovery_steps,
            energy,
            n_batches,
            net_dim,
        )

        report_s3 = save_ndarray(
            file_prefix="sqd_data",
            ucj_parameter=ucj_parameter,
            raw_bitstrings=raw_bitstrings,
            recovered_bitstrings=bitstrings,
            avg_occupancy=step_occ_a,
            avg_occupancy_b=step_occ_b,
            carryover=step_carryover,
            **save_kwargs,
        )

        # Keep the best (lowest-energy) pass.
        if best_energy is None or energy < best_energy:
            best_energy = energy
            best_carryover = step_carryover
            best_sbd_result = step_sbd_result  # carry best SBDResult for orbital optimization
            best_report_s3 = report_s3
            best_num_post = len(bitstrings_post)
            best_net_dim = net_dim

        # Feed the batch-averaged occupancies into the next recovery pass (self-consistency).
        avg_occ = (step_occ_a, step_occ_b)

    logger.debug("Completed configuration recovery loop.")

    # Comprehensive end-of-walker summary: distinct-string counts, sqd_dim mapping, per-spin
    # probability frequencies, a top-20 table, and a saved top-20 plot. Best-effort; never fails.
    try:
        _comprehensive_summary(
            logger,
            raw_bitstrings=raw_bitstrings,
            raw_probs=raw_probs,
            norb=norb,
            num_elec_a=num_elec_a,
            num_elec_b=num_elec_b,
            sqd_dim=sqd_dim,
            n_batches=n_batches,
            n_recovery_steps=n_recovery_steps,
            best_energy=float(best_energy),
            best_net_dim=best_net_dim,
            work_dir=getattr(davidson_solver, "root_dir", None),
            walker_tag=f" #{trial_index:02d}-{walker_index}",
        )
    except Exception as exc:  # diagnostics must never break the actual calculation
        logger.info("[summary] skipped (%s: %s)", type(exc).__name__, str(exc)[:160])
    telemetry.update(
        num_post_determinants=best_num_post,
        net_subspace_dim=best_net_dim,
        energy=float(best_energy),
        n_recovery_steps=n_recovery_steps,
        n_batches=n_batches,
        sqd_data=str(best_report_s3),
        last_updated=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )

    return ((best_energy, best_carryover, best_sbd_result), telemetry)


def _stack_spin_carryover(
    carryover_a: NpStrict2DArrayBool,
    carryover_b: NpStrict2DArrayBool | None,
    norb: int,
) -> NpStrict2DArrayBool:
    """Combine per-spin carryover into one (beta-left, alpha-right) 2*norb-wide bool array.

    Alpha and beta carryover sets can differ in length; the shorter one is zero-padded so the two
    halves line up row-wise. This mirrors the sampled-bitstring layout consumed on the next pass.
    """
    if carryover_b is None:
        carryover_b = np.empty((0, norb), dtype=bool)
    n = max(carryover_a.shape[0], carryover_b.shape[0])
    out = np.zeros((n, 2 * norb), dtype=bool)
    out[: carryover_b.shape[0], :norb] = carryover_b
    out[: carryover_a.shape[0], norb:] = carryover_a
    return out


@task
def postselect_bitstrings(
    bitstring_matrix: NpStrict2DArrayBool,
    probabilities: NpStrict1DArrayF64,
    *,
    hamming_right: int,
    hamming_left: int,
) -> tuple[NpStrict2DArrayBool, NpStrict1DArrayF64]:
    mask_postsel = post_select_by_hamming_weight(
        bitstring_matrix,
        hamming_right=hamming_right,
        hamming_left=hamming_left,
    )
    bs_mat_postsel = bitstring_matrix[mask_postsel]
    probs_postsel = probabilities[mask_postsel]
    probs_postsel = np.abs(probs_postsel) / np.sum(np.abs(probs_postsel))

    return bs_mat_postsel, probs_postsel


@task
def subsample_close_shell(
    bitstring_matrix: NpStrict2DArrayBool,
    probabilities: NpStrict1DArrayF64,
    carryover: NpStrict2DArrayBool,
    subspace_dim: int,
    norb: int,
    num_elec_a: int,
    rng: np.random.Generator | None = None,
) -> NpStrict1DArrayLL:
    # Use an explicit per-walker Generator when provided so concurrent walkers draw INDEPENDENT
    # subspaces (and there is no thread race on a shared global). Falls back to MODULE_RNG only for
    # legacy callers that pass nothing.
    if rng is None:
        rng = MODULE_RNG

    num_configs = bitstring_matrix.shape[0]
    num_carryover = carryover.shape[0]

    # Make sure the Hartree Fock is included at index 0 of determinants.
    # This is requirement of the SBD solver.
    # The Hartree Fock bitstring is something like '0000011111'
    hartreefock = (1 << num_elec_a) - 1

    # Assume longlong is 64 bit integer.
    # Bit at index > 64 overflows.
    assert norb < 64

    ci_strs_a = np.zeros(num_configs, dtype=np.longlong)
    ci_strs_b = np.zeros(num_configs, dtype=np.longlong)
    ci_strs_carryover = np.zeros(num_carryover, dtype=np.longlong)

    # For performance, we accumulate the left and right CI strings together, column-wise,
    # across the two halves of the input bitstring matrix.
    for i in range(norb):
        ci_strs_b[:] += bitstring_matrix[:, i] * 2 ** (norb - 1 - i)
        ci_strs_a[:] += bitstring_matrix[:, norb + i] * 2 ** (norb - 1 - i)
        ci_strs_carryover[:] += carryover[:, i] * 2 ** (norb - 1 - i)
    mixed_ci_strigs = np.concatenate((ci_strs_a, ci_strs_b))

    # Reduce duplicated elements from CI strings and accumurate probabilities.
    ci_strs_unique, ci_probs_unique = _deduplicate_and_accumurate_probs(
        ci_strings=mixed_ci_strigs,
        probabilities=np.tile(probabilities, 2) / 2.0,
    )

    # Remove HF string to make sure it appears at index 0
    non_hf_mask = ci_strs_unique != hartreefock
    ci_strs_carryover = ci_strs_carryover[ci_strs_carryover != hartreefock]

    num_new_samples = int(np.sqrt(subspace_dim)) - len(ci_strs_carryover) - 1
    if len(ci_strs_unique) > num_new_samples:
        # Choose bitstrings not included in carryover bitstrings
        # Subspace dimension must be preserved
        non_co_mask = ~np.isin(ci_strs_unique, ci_strs_carryover)
        mask = non_hf_mask & non_co_mask
        ci_strs_unique = ci_strs_unique[mask]
        ci_probs_unique = ci_probs_unique[mask]
        new_strings = rng.choice(
            ci_strs_unique,
            size=num_new_samples,
            replace=False,
            p=ci_probs_unique / ci_probs_unique.sum(),
        )
    else:
        new_strings = ci_strs_unique[non_hf_mask]

    # Carryover bitstrings are always included
    return np.concatenate(([hartreefock], ci_strs_carryover, new_strings), dtype=np.longlong)


@task
def subsample_open_shell(
    bitstring_matrix: NpStrict2DArrayBool,
    probabilities: NpStrict1DArrayF64,
    carryover_a: NpStrict2DArrayBool,
    carryover_b: NpStrict2DArrayBool,
    subspace_dim: int,
    norb: int,
    num_elec_a: int,
    num_elec_b: int,
    rng: np.random.Generator | None = None,
) -> tuple[NpStrict1DArrayLL, NpStrict1DArrayLL]:
    """Build separate alpha and beta CI string lists from bitstrings (open-shell / UHF).

    Unlike :func:`subsample_close_shell` (which averages and dedups across both spin halves of a
    single bitstring matrix and returns one alpha list), this keeps the alpha and beta pools fully
    independent: each spin gets its own deduplication, its own Hartree-Fock string at index 0, and
    its own carryover determinants. Returns the pair ``(ci_strs_a, ci_strs_b)`` consumed by the
    UHF SBD solver as distinct AlphaDets/BetaDets.

    The input bitstring layout matches the close-shell routine: ``bitstring_matrix[:, :norb]`` is
    the beta (left) half and ``bitstring_matrix[:, norb:]`` is the alpha (right) half.
    """
    # Assume longlong is 64 bit integer. Bit at index > 64 overflows.
    assert norb < 64

    num_configs = bitstring_matrix.shape[0]

    ci_strs_a = np.zeros(num_configs, dtype=np.longlong)
    ci_strs_b = np.zeros(num_configs, dtype=np.longlong)
    for i in range(norb):
        ci_strs_b[:] += bitstring_matrix[:, i] * 2 ** (norb - 1 - i)
        ci_strs_a[:] += bitstring_matrix[:, norb + i] * 2 ** (norb - 1 - i)

    ci_a = _subsample_one_spin(
        ci_strs=ci_strs_a,
        probabilities=probabilities,
        carryover=carryover_a,
        subspace_dim=subspace_dim,
        norb=norb,
        num_elec=num_elec_a,
        rng=rng,
    )
    ci_b = _subsample_one_spin(
        ci_strs=ci_strs_b,
        probabilities=probabilities,
        carryover=carryover_b,
        subspace_dim=subspace_dim,
        norb=norb,
        num_elec=num_elec_b,
        rng=rng,
    )
    return ci_a, ci_b


def _subsample_one_spin(
    ci_strs: NpStrict1DArrayLL,
    probabilities: NpStrict1DArrayF64,
    carryover: NpStrict2DArrayBool,
    subspace_dim: int,
    norb: int,
    num_elec: int,
    rng: np.random.Generator | None = None,
) -> NpStrict1DArrayLL:
    """Dedup + carryover + HF-at-index-0 for a single spin channel.

    Mirrors the per-spin logic of :func:`subsample_close_shell` but operates on one spin's CI
    string integers and its own electron count and carryover, without averaging across spins.
    """
    if rng is None:
        rng = MODULE_RNG

    num_carryover = carryover.shape[0]

    # Hartree-Fock string for this spin, e.g. '0000011111' for num_elec=5.
    hartreefock = (1 << num_elec) - 1

    ci_strs_carryover = np.zeros(num_carryover, dtype=np.longlong)
    for i in range(norb):
        ci_strs_carryover[:] += carryover[:, i] * 2 ** (norb - 1 - i)

    ci_strs_unique, ci_probs_unique = _deduplicate_and_accumurate_probs(
        ci_strings=ci_strs,
        probabilities=probabilities,
    )

    # Remove HF string to make sure it appears at index 0.
    non_hf_mask = ci_strs_unique != hartreefock
    ci_strs_carryover = ci_strs_carryover[ci_strs_carryover != hartreefock]

    num_new_samples = int(np.sqrt(subspace_dim)) - len(ci_strs_carryover) - 1
    if len(ci_strs_unique) > num_new_samples:
        # Choose bitstrings not included in carryover bitstrings; preserve subspace dimension.
        non_co_mask = ~np.isin(ci_strs_unique, ci_strs_carryover)
        mask = non_hf_mask & non_co_mask
        ci_strs_unique = ci_strs_unique[mask]
        ci_probs_unique = ci_probs_unique[mask]
        new_strings = rng.choice(
            ci_strs_unique,
            size=num_new_samples,
            replace=False,
            p=ci_probs_unique / ci_probs_unique.sum(),
        )
    else:
        new_strings = ci_strs_unique[non_hf_mask]

    # Carryover bitstrings are always included; HF is forced to index 0.
    return np.concatenate(([hartreefock], ci_strs_carryover, new_strings), dtype=np.longlong)


def _deduplicate_and_accumurate_probs(
    ci_strings: NpStrict1DArrayLL,
    probabilities: NpStrict1DArrayF64,
) -> tuple[NpStrict1DArrayLL, NpStrict1DArrayF64]:
    ci_strs_unique, ci_strs_inv = np.unique(
        ci_strings,
        return_inverse=True,
    )
    ci_probs_unique = np.bincount(
        ci_strs_inv,
        weights=probabilities,
        minlength=len(ci_strs_unique),
    )
    return ci_strs_unique, ci_probs_unique
