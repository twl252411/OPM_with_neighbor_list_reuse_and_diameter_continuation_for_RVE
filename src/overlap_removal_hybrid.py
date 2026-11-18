"""Hybrid periodic overlap removal with automatic original/fast selection.

Low target collision fractions use the original strict algorithm to avoid
neighbor-list and continuation overhead. Higher collision fractions use the
fast continuation solver from ``overlap_removal_fast.py``.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from overlap_removal_fast import (
    FastSolverConfig,
    SolverResult,
    compute_actual_fractions,
    compute_gradients_and_metrics,
    compute_particle_number,
    generate_initial_points,
    generate_periodic_images,
    query_pairs,
    solve_fast,
    validate_configuration,
    validate_solver_inputs,
    theoretical_packing_limit,
    warm_up_numba,
)


@dataclass
class HybridSolverConfig:
    """Configuration for automatic original/fast solver selection.

    Parameters
    ----------
    phi_switch_2d, phi_switch_3d : float
        Target collision-fraction thresholds. Cases at or below the threshold
        use the original strict algorithm.
    original_alpha : float
        Fixed gradient-descent step for the original solver.
    original_max_iterations : int
        Maximum original-solver iteration count.
    strict_overlap_ratio : float
        Maximum-overlap tolerance divided by target collision diameter.
    norm_eps : float
        Coincidence threshold.
    print_interval : int
        Original-solver progress print interval.
    verbose : bool
        Print solver selection and iteration progress.
    fast : FastSolverConfig
        Configuration passed to the fast solver.
    """

    phi_switch_2d: float = 0.50
    phi_switch_3d: float = 0.40
    original_alpha: float = 0.4
    original_max_iterations: int = 100_000
    strict_overlap_ratio: float = 1.0e-8
    norm_eps: float = 1.0e-12
    print_interval: int = 200
    verbose: bool = True
    fast: FastSolverConfig = field(default_factory=FastSolverConfig)


def solve_original_strict(
    initial_points: np.ndarray,
    box_size: np.ndarray,
    diameter_physical: float,
    enlargement_factor: float,
    seed: int = 0,
    config: HybridSolverConfig | None = None,
) -> SolverResult:
    """Run the original fixed-diameter algorithm with strict termination.

    The periodic KDTree is rebuilt at every gradient evaluation. The only
    deliberate difference from the historical script is the final stopping
    condition, which uses maximum pair overlap rather than total potential.
    This makes comparison with the fast solver geometrically fair.
    """

    if config is None:
        config = HybridSolverConfig()

    points, box_size, diameter_target = validate_solver_inputs(
        initial_points,
        box_size,
        diameter_physical,
        enlargement_factor,
        skin_ratio=None,
    )
    if config.original_alpha <= 0.0 or not math.isfinite(config.original_alpha):
        raise ValueError("original_alpha must be finite and positive.")
    if config.strict_overlap_ratio <= 0.0:
        raise ValueError("strict_overlap_ratio must be positive.")

    dim = points.shape[1]
    particle_number = points.shape[0]
    v1_actual, collision_vf = compute_actual_fractions(
        particle_number,
        box_size,
        diameter_physical,
        diameter_target,
    )
    if collision_vf >= theoretical_packing_limit(dim):
        raise ValueError(
            f"Target collision fraction {collision_vf:.8f} is at or above "
            f"the theoretical packing limit {theoretical_packing_limit(dim):.8f}."
        )
    strict_tolerance = config.strict_overlap_ratio * diameter_target

    warm_up_numba(dim)
    gradients = np.zeros_like(points)
    tree_builds = 0
    pair_evaluations = 0
    last_potential = math.inf
    last_max_overlap = math.inf
    last_overlap_count = -1

    start_total = time.perf_counter()
    success = False
    status = "max_iterations"
    iterations = 0
    gradient_evaluations = 0

    for iteration in range(config.original_max_iterations + 1):
        tree = cKDTree(points, boxsize=box_size)
        pairs = query_pairs(tree, diameter_target)
        tree_builds += 1
        pair_evaluations += int(pairs.shape[0])

        potential, max_overlap, overlap_count = compute_gradients_and_metrics(
            points,
            pairs,
            box_size,
            diameter_target,
            gradients,
            config.norm_eps,
            seed,
        )
        gradient_evaluations += 1
        last_potential = float(potential)
        last_max_overlap = float(max_overlap)
        last_overlap_count = int(overlap_count)

        if config.verbose and (
            iteration % config.print_interval == 0
            or max_overlap <= strict_tolerance
        ):
            print(
                "Original strict "
                f"iteration {iteration}: potential={potential:.6e}, "
                f"max_overlap={max_overlap:.6e}, pairs={overlap_count}"
            )

        if max_overlap <= strict_tolerance:
            success = True
            status = "converged"
            iterations = iteration
            break

        if iteration == config.original_max_iterations:
            iterations = iteration
            break

        # Preserve the original fixed-step update without a displacement cap.
        points = (points - config.original_alpha * gradients) % box_size

        if not np.all(np.isfinite(points)):
            status = "invalid_coordinates"
            iterations = iteration + 1
            break

    solve_runtime = time.perf_counter() - start_total

    validation_start = time.perf_counter()
    validation = validate_configuration(
        points,
        box_size,
        diameter_target,
        config.norm_eps,
        seed,
        overlap_tolerance=config.strict_overlap_ratio * diameter_target,
    )
    validation_runtime = time.perf_counter() - validation_start

    validation_success = (
        validation.max_overlap <= strict_tolerance
        and validation.nan_count == 0
        and validation.inf_count == 0
        and validation.coordinates_inside_box
    )
    success = bool(success and validation_success)
    if not success and status == "converged":
        status = "validation_failed"

    runtime_total = solve_runtime + validation_runtime

    return SolverResult(
        points=points,
        success=success,
        status=status,
        solver_name="original_strict",
        solver_path="original_fallback",
        runtime_total=runtime_total,
        runtime_continuation=0.0,
        runtime_cleanup=solve_runtime,
        runtime_validation=validation_runtime,
        iterations_total=iterations,
        iterations_continuation=0,
        iterations_cleanup=iterations,
        gradient_evaluations=gradient_evaluations,
        tree_builds_total=tree_builds + 1,
        tree_builds_continuation=0,
        tree_builds_cleanup=tree_builds,
        neighbor_list_reuses=0,
        reuse_ratio=0.0,
        pair_evaluations=pair_evaluations + validation.candidate_pair_count,
        diameter_physical=float(diameter_physical),
        diameter_target=float(diameter_target),
        particle_number=int(particle_number),
        v1_actual_physical=float(v1_actual),
        collision_vf_actual=float(collision_vf),
        final_potential=validation.potential,
        final_max_overlap=validation.max_overlap,
        final_relative_max_overlap=validation.relative_max_overlap,
        final_overlap_pair_count=validation.overlap_pair_count,
        final_violating_overlap_pair_count=(
            validation.violating_overlap_pair_count
        ),
        minimum_center_distance=validation.minimum_center_distance,
        nan_count=validation.nan_count,
        inf_count=validation.inf_count,
        continuation_stages=[],
    )


def solve_original_legacy(
    initial_points: np.ndarray,
    box_size: np.ndarray,
    diameter_physical: float,
    enlargement_factor: float,
    seed: int = 0,
    config: HybridSolverConfig | None = None,
) -> SolverResult:
    """Run the historical total-potential stopping condition.

    This function is provided only for reproducing the supplied original
    script. It should not be used as the main baseline against a strict fast
    solver.
    """

    if config is None:
        config = HybridSolverConfig()

    points, box_size, diameter_target = validate_solver_inputs(
        initial_points,
        box_size,
        diameter_physical,
        enlargement_factor,
        skin_ratio=None,
    )
    if config.original_alpha <= 0.0 or not math.isfinite(config.original_alpha):
        raise ValueError("original_alpha must be finite and positive.")
    dim = points.shape[1]
    particle_number = points.shape[0]
    v1_actual, collision_vf = compute_actual_fractions(
        particle_number,
        box_size,
        diameter_physical,
        diameter_target,
    )
    if collision_vf >= theoretical_packing_limit(dim):
        raise ValueError(
            f"Target collision fraction {collision_vf:.8f} is at or above "
            f"the theoretical packing limit {theoretical_packing_limit(dim):.8f}."
        )
    potential_tolerance = (diameter_target * 1.0e-4) ** 2

    warm_up_numba(dim)
    gradients = np.zeros_like(points)
    tree_builds = 0
    pair_evaluations = 0
    last_potential = math.inf
    iterations = 0
    gradient_evaluations = 0
    success = False
    status = "max_iterations"

    start_total = time.perf_counter()
    for iteration in range(config.original_max_iterations + 1):
        tree = cKDTree(points, boxsize=box_size)
        pairs = query_pairs(tree, diameter_target)
        tree_builds += 1
        pair_evaluations += int(pairs.shape[0])
        potential, max_overlap, overlap_count = compute_gradients_and_metrics(
            points,
            pairs,
            box_size,
            diameter_target,
            gradients,
            config.norm_eps,
            seed,
        )
        gradient_evaluations += 1
        last_potential = float(potential)

        if potential < potential_tolerance:
            success = True
            status = "legacy_potential_converged"
            iterations = iteration
            break

        if iteration == config.original_max_iterations:
            iterations = iteration
            break

        points = (points - config.original_alpha * gradients) % box_size
        if not np.all(np.isfinite(points)):
            status = "invalid_coordinates"
            iterations = iteration + 1
            break

    solve_runtime = time.perf_counter() - start_total
    validation_start = time.perf_counter()
    validation = validate_configuration(
        points,
        box_size,
        diameter_target,
        config.norm_eps,
        seed,
        overlap_tolerance=0.0,
    )
    validation_runtime = time.perf_counter() - validation_start

    return SolverResult(
        points=points,
        success=success,
        status=status,
        solver_name="original_legacy",
        solver_path="original_legacy",
        runtime_total=solve_runtime + validation_runtime,
        runtime_continuation=0.0,
        runtime_cleanup=solve_runtime,
        runtime_validation=validation_runtime,
        iterations_total=iterations,
        iterations_continuation=0,
        iterations_cleanup=iterations,
        gradient_evaluations=gradient_evaluations,
        tree_builds_total=tree_builds + 1,
        tree_builds_continuation=0,
        tree_builds_cleanup=tree_builds,
        neighbor_list_reuses=0,
        reuse_ratio=0.0,
        pair_evaluations=pair_evaluations + validation.candidate_pair_count,
        diameter_physical=float(diameter_physical),
        diameter_target=float(diameter_target),
        particle_number=int(particle_number),
        v1_actual_physical=float(v1_actual),
        collision_vf_actual=float(collision_vf),
        final_potential=validation.potential,
        final_max_overlap=validation.max_overlap,
        final_relative_max_overlap=validation.relative_max_overlap,
        final_overlap_pair_count=validation.overlap_pair_count,
        final_violating_overlap_pair_count=(
            validation.violating_overlap_pair_count
        ),
        minimum_center_distance=validation.minimum_center_distance,
        nan_count=validation.nan_count,
        inf_count=validation.inf_count,
        continuation_stages=[],
    )


def solve_hybrid(
    initial_points: np.ndarray,
    box_size: np.ndarray,
    diameter_physical: float,
    enlargement_factor: float,
    seed: int = 0,
    config: HybridSolverConfig | None = None,
) -> SolverResult:
    """Automatically select original strict or fast continuation mode.

    Selection is based on the actual target collision fraction, not the
    nominal physical volume fraction.
    """

    if config is None:
        config = HybridSolverConfig()

    points, box_size, diameter_target = validate_solver_inputs(
        initial_points,
        box_size,
        diameter_physical,
        enlargement_factor,
        skin_ratio=None,
    )
    dim = points.shape[1]
    _, collision_vf = compute_actual_fractions(
        points.shape[0],
        box_size,
        diameter_physical,
        diameter_target,
    )
    if collision_vf >= theoretical_packing_limit(dim):
        raise ValueError(
            f"Target collision fraction {collision_vf:.8f} is at or above "
            f"the theoretical packing limit {theoretical_packing_limit(dim):.8f}."
        )
    phi_switch = config.phi_switch_2d if dim == 2 else config.phi_switch_3d

    if collision_vf <= phi_switch:
        if config.verbose:
            print(
                "Hybrid selection: original_strict "
                f"(collision_vf={collision_vf:.6f} <= switch={phi_switch:.6f})"
            )
        result = solve_original_strict(
            initial_points=points.copy(),
            box_size=box_size,
            diameter_physical=diameter_physical,
            enlargement_factor=enlargement_factor,
            seed=seed,
            config=config,
        )
        result.solver_name = "hybrid"
        result.solver_path = "original_fallback"
        return result

    if config.verbose:
        print(
            "Hybrid selection: fast_continuation "
            f"(collision_vf={collision_vf:.6f} > switch={phi_switch:.6f})"
        )

    # Keep branch settings consistent without mutating the caller's config.
    fast_config = replace(
        config.fast,
        strict_overlap_ratio=config.strict_overlap_ratio,
        norm_eps=config.norm_eps,
        verbose=config.verbose,
        print_interval=config.print_interval,
    )

    result = solve_fast(
        initial_points=points.copy(),
        box_size=box_size,
        diameter_physical=diameter_physical,
        enlargement_factor=enlargement_factor,
        seed=seed,
        config=fast_config,
    )
    result.solver_name = "hybrid"
    return result


def save_hybrid_outputs(
    result: SolverResult,
    box_size: np.ndarray,
    output_directory: str | Path,
) -> None:
    """Save hybrid primary points, physical periodic images, and diagnostics."""

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    dim = result.points.shape[1]
    periodic_points = generate_periodic_images(
        result.points,
        np.asarray(box_size, dtype=np.float64),
        result.diameter_physical,
    )

    np.savetxt(output_directory / f"points_primary_{dim}d.txt", result.points)
    np.savetxt(output_directory / f"points_periodic_{dim}d.txt", periodic_points)
    with (output_directory / f"hybrid_result_{dim}d.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(result.summary_dict(), file, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # --------------------------- Parameters ---------------------------
    dim = 3
    rve_size = np.ones(dim, dtype=np.float64) * 100.0
    diameter_physical = 20.0
    v1_nominal = 0.55
    enlargement_factor = 1.025
    seed = 0

    hybrid_config = HybridSolverConfig(
        phi_switch_2d=0.50,
        phi_switch_3d=0.40,
        original_alpha=0.4,
        verbose=True,
        print_interval=200,
    )

    # ------------------------- Initialization -------------------------
    particle_number = compute_particle_number(
        rve_size,
        diameter_physical,
        v1_nominal,
    )
    initial_points = generate_initial_points(
        particle_number,
        rve_size,
        seed,
    )

    # ---------------------------- Solve -------------------------------
    result = solve_hybrid(
        initial_points=initial_points,
        box_size=rve_size,
        diameter_physical=diameter_physical,
        enlargement_factor=enlargement_factor,
        seed=seed,
        config=hybrid_config,
    )

    # ---------------------------- Output ------------------------------
    save_hybrid_outputs(result, rve_size, output_directory="hybrid_output")

    print("\n=============== Hybrid solver summary ================")
    for key, value in result.summary_dict().items():
        if key != "continuation_stages":
            print(f"{key}: {value}")
