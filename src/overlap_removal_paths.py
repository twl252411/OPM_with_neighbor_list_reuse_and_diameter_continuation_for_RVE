"""Paper-aligned four-path overlap-removal wrappers.

This module provides one explicit implementation layer for the four
construction paths used in the manuscript:

``R``
    Non-accelerated reference path. The target diameter is used from the
    beginning, and the exact periodic KDTree is rebuilt at every iteration.

``A_Dt``
    Direct accelerated path at the target collision diameter. Safe
    protective-skin neighbor-list reuse is enabled, but adaptive radius
    continuation is disabled.

``A_cont``
    Accelerated continuation path. Safe neighbor-list reuse is enabled and
    adaptive collision-diameter continuation is forced on.

``hybrid``
    One-time collision-fraction dispatch. The reference path is selected
    below the hybrid threshold; otherwise the selected accelerated policy is
    used, with the continuation threshold deciding between ``A_Dt`` and
    ``A_cont``.

The numerical parameters are the frozen manuscript settings. The functions
reuse the reviewed solver implementations in ``overlap_removal_fast.py`` and
``overlap_removal_hybrid.py`` so that this file is a transparent path map
rather than a fork of the algorithms.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Iterable, Literal

import numpy as np

from overlap_removal_fast import (
    FastSolverConfig,
    SolverResult,
    compute_actual_fractions,
    compute_particle_number,
    generate_initial_points,
    generate_initial_points_legacy,
    solve_fast,
)
from overlap_removal_hybrid import (
    HybridSolverConfig,
    solve_hybrid,
    solve_original_strict,
)


PaperPath = Literal["R", "A_Dt", "A_cont", "hybrid"]
PAPER_FOUR_PATHS: tuple[PaperPath, ...] = ("R", "A_Dt", "A_cont", "hybrid")


@dataclass(frozen=True)
class PaperPathSettings:
    """Frozen numerical settings used by the manuscript path comparison."""

    # Safe neighbor-list reuse
    skin_ratio_2d: float = 0.125
    skin_ratio_3d: float = 0.100
    neighbor_safety_factor: float = 0.80

    # Reference path and final acceptance
    original_alpha: float = 0.40
    strict_overlap_ratio: float = 1.0e-8
    norm_eps: float = 1.0e-12

    # Continuation initialization and increments
    phi_easy_2d: float = 0.55
    phi_easy_3d: float = 0.42
    phi_continuation_2d: float = 0.70
    phi_continuation_3d: float = 0.62
    lambda_min: float = 0.75
    delta_lambda_initial: float = 0.02
    delta_lambda_min: float = 0.0025
    delta_lambda_max: float = 0.08

    # Continuation-stage, rescue, cleanup, and certification tolerances
    stage_overlap_ratio: float = 1.0e-3
    stage_growth_fraction: float = 0.10
    precleanup_overlap_ratio: float = 3.0e-8
    max_move_ratio: float = 0.05
    max_move_skin_ratio: float = 0.20
    cleanup_max_move_ratio: float = 0.02

    # Iteration controls
    stage_fast_iterations: int = 100
    stage_slow_iterations: int = 500
    max_iterations_stage: int = 10_000
    max_iterations_stage_rescue: int = 3_000
    max_iterations_precleanup: int = 50_000
    max_iterations_cleanup: int = 100_000
    original_max_iterations: int = 100_000
    stagnation_window: int = 100
    stagnation_relative_drop: float = 1.0e-4

    # Hybrid dispatch thresholds
    phi_switch_2d: float = 0.26
    phi_switch_3d: float = 0.20


def paper_l_over_d0(dimension: int) -> float:
    """Return the manuscript RVE size ratio for efficiency comparisons."""

    if dimension == 2:
        return 20.0
    if dimension == 3:
        return 8.0
    raise ValueError("dimension must be 2 or 3.")


def make_box_size(
    dimension: int,
    l_over_d0: float | None = None,
    diameter_physical: float = 1.0,
) -> np.ndarray:
    """Create the square/cubic periodic RVE used by the paper scans."""

    if dimension not in (2, 3):
        raise ValueError("dimension must be 2 or 3.")
    ratio = paper_l_over_d0(dimension) if l_over_d0 is None else float(l_over_d0)
    return np.ones(dimension, dtype=np.float64) * ratio * float(diameter_physical)


def skin_ratio_for_dimension(settings: PaperPathSettings, dimension: int) -> float:
    """Return the calibrated skin ratio for 2D circles or 3D spheres."""

    if dimension == 2:
        return settings.skin_ratio_2d
    if dimension == 3:
        return settings.skin_ratio_3d
    raise ValueError("dimension must be 2 or 3.")


def collision_fractions_for_case(
    box_size: Iterable[float],
    diameter_physical: float,
    enlargement_factor: float,
    v1_nominal: float,
) -> tuple[int, float, float]:
    """Return ``N``, actual physical fraction, and target collision fraction."""

    box = np.asarray(box_size, dtype=np.float64)
    particle_number = compute_particle_number(box, diameter_physical, v1_nominal)
    v1_actual, v1_target = compute_actual_fractions(
        particle_number,
        box,
        diameter_physical,
        diameter_physical * enlargement_factor,
    )
    return int(particle_number), float(v1_actual), float(v1_target)


def generate_matched_initial_points(
    dimension: int,
    v1_nominal: float,
    seed: int,
    enlargement_factor: float = 1.0,
    diameter_physical: float = 1.0,
    l_over_d0: float | None = None,
    point_generation: Literal["modern", "legacy"] = "modern",
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    """Generate the matched initial centers used by all four paths.

    The particle count follows the manuscript definition based on the nominal
    physical fraction ``v1_nominal`` and the physical diameter ``D0``. The same
    returned coordinate array should be copied into each solver call.
    """

    box = make_box_size(dimension, l_over_d0, diameter_physical)
    particle_number, v1_actual, v1_target = collision_fractions_for_case(
        box,
        diameter_physical,
        enlargement_factor,
        v1_nominal,
    )
    if point_generation == "legacy":
        points = generate_initial_points_legacy(particle_number, box, seed)
    elif point_generation == "modern":
        points = generate_initial_points(particle_number, box, seed)
    else:
        raise ValueError("point_generation must be 'modern' or 'legacy'.")
    metadata = {
        "particle_number": particle_number,
        "v1_nominal": float(v1_nominal),
        "v1_actual_physical": v1_actual,
        "collision_vf_actual": v1_target,
        "enlargement_factor": float(enlargement_factor),
        "diameter_physical": float(diameter_physical),
        "box_to_diameter_ratio": float(box[0] / diameter_physical),
        "seed": int(seed),
    }
    return points, box, metadata


def make_fast_config(
    dimension: int,
    settings: PaperPathSettings | None = None,
    *,
    force_continuation: bool | None,
    verbose: bool = False,
) -> FastSolverConfig:
    """Create a manuscript-aligned fast-solver configuration."""

    settings = settings or PaperPathSettings()
    config = FastSolverConfig(verbose=verbose)
    config.alpha = settings.original_alpha
    config.norm_eps = settings.norm_eps
    config.skin_ratio = skin_ratio_for_dimension(settings, dimension)
    config.neighbor_safety_factor = settings.neighbor_safety_factor
    config.phi_easy_2d = settings.phi_easy_2d
    config.phi_easy_3d = settings.phi_easy_3d
    config.phi_continuation_2d = settings.phi_continuation_2d
    config.phi_continuation_3d = settings.phi_continuation_3d
    config.lambda_min = settings.lambda_min
    config.delta_lambda_initial = settings.delta_lambda_initial
    config.delta_lambda_min = settings.delta_lambda_min
    config.delta_lambda_max = settings.delta_lambda_max
    config.stage_overlap_ratio = settings.stage_overlap_ratio
    config.stage_growth_fraction = settings.stage_growth_fraction
    config.stage_fast_iterations = settings.stage_fast_iterations
    config.stage_slow_iterations = settings.stage_slow_iterations
    config.max_iterations_stage = settings.max_iterations_stage
    config.max_iterations_stage_rescue = settings.max_iterations_stage_rescue
    config.max_iterations_precleanup = settings.max_iterations_precleanup
    config.precleanup_overlap_ratio = settings.precleanup_overlap_ratio
    config.max_iterations_cleanup = settings.max_iterations_cleanup
    config.strict_overlap_ratio = settings.strict_overlap_ratio
    config.max_move_ratio = settings.max_move_ratio
    config.max_move_skin_ratio = settings.max_move_skin_ratio
    config.cleanup_max_move_ratio = settings.cleanup_max_move_ratio
    config.reuse_neighbor_lists = True
    config.limit_displacement = True
    config.force_continuation = force_continuation
    config.stagnation_window = settings.stagnation_window
    config.stagnation_relative_drop = settings.stagnation_relative_drop
    config.verbose = verbose
    return config


def make_hybrid_config(
    dimension: int,
    settings: PaperPathSettings | None = None,
    *,
    verbose: bool = False,
) -> HybridSolverConfig:
    """Create a manuscript-aligned hybrid/reference configuration."""

    settings = settings or PaperPathSettings()
    config = HybridSolverConfig(verbose=verbose)
    config.phi_switch_2d = settings.phi_switch_2d
    config.phi_switch_3d = settings.phi_switch_3d
    config.original_alpha = settings.original_alpha
    config.original_max_iterations = settings.original_max_iterations
    config.strict_overlap_ratio = settings.strict_overlap_ratio
    config.norm_eps = settings.norm_eps
    config.fast = make_fast_config(
        dimension,
        settings,
        force_continuation=None,
        verbose=verbose,
    )
    return config


def _dimension_from_points(initial_points: np.ndarray) -> int:
    points = np.asarray(initial_points)
    if points.ndim != 2 or points.shape[1] not in (2, 3):
        raise ValueError("initial_points must have shape (N, 2) or (N, 3).")
    return int(points.shape[1])


def solve_R(
    initial_points: np.ndarray,
    box_size: Iterable[float],
    diameter_physical: float,
    enlargement_factor: float,
    seed: int = 0,
    settings: PaperPathSettings | None = None,
    *,
    verbose: bool = False,
) -> SolverResult:
    """Run the manuscript reference path ``R``."""

    dimension = _dimension_from_points(initial_points)
    config = make_hybrid_config(dimension, settings, verbose=verbose)
    result = solve_original_strict(
        initial_points=np.asarray(initial_points, dtype=np.float64).copy(),
        box_size=np.asarray(box_size, dtype=np.float64),
        diameter_physical=diameter_physical,
        enlargement_factor=enlargement_factor,
        seed=seed,
        config=config,
    )
    result.solver_name = "R"
    result.solver_path = "R"
    return result


def solve_A_Dt(
    initial_points: np.ndarray,
    box_size: Iterable[float],
    diameter_physical: float,
    enlargement_factor: float,
    seed: int = 0,
    settings: PaperPathSettings | None = None,
    *,
    verbose: bool = False,
) -> SolverResult:
    """Run the direct accelerated path ``A_Dt``."""

    dimension = _dimension_from_points(initial_points)
    config = make_fast_config(
        dimension,
        settings,
        force_continuation=False,
        verbose=verbose,
    )
    result = solve_fast(
        initial_points=np.asarray(initial_points, dtype=np.float64).copy(),
        box_size=np.asarray(box_size, dtype=np.float64),
        diameter_physical=diameter_physical,
        enlargement_factor=enlargement_factor,
        seed=seed,
        config=config,
    )
    result.solver_name = "A_Dt"
    result.solver_path = "A_Dt"
    return result


def solve_A_cont(
    initial_points: np.ndarray,
    box_size: Iterable[float],
    diameter_physical: float,
    enlargement_factor: float,
    seed: int = 0,
    settings: PaperPathSettings | None = None,
    *,
    verbose: bool = False,
) -> SolverResult:
    """Run the continuation accelerated path ``A_cont``."""

    dimension = _dimension_from_points(initial_points)
    config = make_fast_config(
        dimension,
        settings,
        force_continuation=True,
        verbose=verbose,
    )
    result = solve_fast(
        initial_points=np.asarray(initial_points, dtype=np.float64).copy(),
        box_size=np.asarray(box_size, dtype=np.float64),
        diameter_physical=diameter_physical,
        enlargement_factor=enlargement_factor,
        seed=seed,
        config=config,
    )
    result.solver_name = "A_cont"
    result.solver_path = "A_cont"
    return result


def solve_hybrid_policy(
    initial_points: np.ndarray,
    box_size: Iterable[float],
    diameter_physical: float,
    enlargement_factor: float,
    seed: int = 0,
    settings: PaperPathSettings | None = None,
    *,
    verbose: bool = False,
) -> SolverResult:
    """Run the manuscript hybrid dispatch path."""

    dimension = _dimension_from_points(initial_points)
    config = make_hybrid_config(dimension, settings, verbose=verbose)
    result = solve_hybrid(
        initial_points=np.asarray(initial_points, dtype=np.float64).copy(),
        box_size=np.asarray(box_size, dtype=np.float64),
        diameter_physical=diameter_physical,
        enlargement_factor=enlargement_factor,
        seed=seed,
        config=config,
    )
    result.solver_name = "hybrid"
    result.solver_path = f"hybrid:{result.solver_path}"
    return result


def solve_paper_path(
    path: PaperPath | str,
    initial_points: np.ndarray,
    box_size: Iterable[float],
    diameter_physical: float,
    enlargement_factor: float,
    seed: int = 0,
    settings: PaperPathSettings | None = None,
    *,
    verbose: bool = False,
) -> SolverResult:
    """Run one named manuscript path."""

    normalized = path.strip()
    aliases = {
        "R": "R",
        "reference": "R",
        "ref": "R",
        "A_Dt": "A_Dt",
        "A-Dt": "A_Dt",
        "A_dt": "A_Dt",
        "direct": "A_Dt",
        "A_cont": "A_cont",
        "A-cont": "A_cont",
        "continuation": "A_cont",
        "hybrid": "hybrid",
        "P_hyb": "hybrid",
    }
    try:
        canonical = aliases[normalized]
    except KeyError as exc:
        raise ValueError(
            "path must be one of R, A_Dt, A_cont, or hybrid."
        ) from exc

    if canonical == "R":
        return solve_R(
            initial_points,
            box_size,
            diameter_physical,
            enlargement_factor,
            seed,
            settings,
            verbose=verbose,
        )
    if canonical == "A_Dt":
        return solve_A_Dt(
            initial_points,
            box_size,
            diameter_physical,
            enlargement_factor,
            seed,
            settings,
            verbose=verbose,
        )
    if canonical == "A_cont":
        return solve_A_cont(
            initial_points,
            box_size,
            diameter_physical,
            enlargement_factor,
            seed,
            settings,
            verbose=verbose,
        )
    return solve_hybrid_policy(
        initial_points,
        box_size,
        diameter_physical,
        enlargement_factor,
        seed,
        settings,
        verbose=verbose,
    )


def solve_all_paper_paths(
    initial_points: np.ndarray,
    box_size: Iterable[float],
    diameter_physical: float,
    enlargement_factor: float,
    seed: int = 0,
    settings: PaperPathSettings | None = None,
    *,
    paths: Iterable[PaperPath | str] = PAPER_FOUR_PATHS,
    verbose: bool = False,
) -> dict[str, SolverResult]:
    """Run the requested manuscript paths from identical initial centers."""

    results: dict[str, SolverResult] = {}
    for path in paths:
        result = solve_paper_path(
            path,
            initial_points,
            box_size,
            diameter_physical,
            enlargement_factor,
            seed,
            settings,
            verbose=verbose,
        )
        results[str(path)] = result
    return results


def solve_nominal_paper_case(
    dimension: int,
    v1_nominal: float,
    enlargement_factor: float,
    seed: int,
    *,
    diameter_physical: float = 1.0,
    l_over_d0: float | None = None,
    point_generation: Literal["modern", "legacy"] = "modern",
    settings: PaperPathSettings | None = None,
    paths: Iterable[PaperPath | str] = PAPER_FOUR_PATHS,
    verbose: bool = False,
) -> tuple[dict[str, SolverResult], dict[str, float | int]]:
    """Generate one matched paper case and run the selected paths."""

    points, box, metadata = generate_matched_initial_points(
        dimension=dimension,
        v1_nominal=v1_nominal,
        seed=seed,
        enlargement_factor=enlargement_factor,
        diameter_physical=diameter_physical,
        l_over_d0=l_over_d0,
        point_generation=point_generation,
    )
    results = solve_all_paper_paths(
        initial_points=points,
        box_size=box,
        diameter_physical=diameter_physical,
        enlargement_factor=enlargement_factor,
        seed=seed,
        settings=settings,
        paths=paths,
        verbose=verbose,
    )
    return results, metadata


def summarize_results(
    results: dict[str, SolverResult],
    metadata: dict[str, float | int] | None = None,
    settings: PaperPathSettings | None = None,
) -> dict[str, object]:
    """Return a compact JSON-serializable summary for scripts or notebooks."""

    settings = settings or PaperPathSettings()
    return {
        "metadata": metadata or {},
        "paper_settings": asdict(settings),
        "paths": {
            path: {
                "success": result.success,
                "status": result.status,
                "solver_name": result.solver_name,
                "solver_path": result.solver_path,
                "runtime_total": result.runtime_total,
                "runtime_continuation": result.runtime_continuation,
                "runtime_cleanup": result.runtime_cleanup,
                "runtime_validation": result.runtime_validation,
                "iterations_total": result.iterations_total,
                "tree_builds_total": result.tree_builds_total,
                "tree_builds_continuation": result.tree_builds_continuation,
                "tree_builds_cleanup": result.tree_builds_cleanup,
                "neighbor_list_reuses": result.neighbor_list_reuses,
                "reuse_ratio": result.reuse_ratio,
                "final_relative_max_overlap": result.final_relative_max_overlap,
                "minimum_center_distance": result.minimum_center_distance,
                "particle_number": result.particle_number,
                "v1_actual_physical": result.v1_actual_physical,
                "collision_vf_actual": result.collision_vf_actual,
            }
            for path, result in results.items()
        },
    }


def _parse_paths(raw: str) -> tuple[str, ...]:
    if raw.strip().lower() == "all":
        return PAPER_FOUR_PATHS
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run manuscript-aligned R, A_Dt, A_cont, and hybrid paths."
    )
    parser.add_argument("--dimension", type=int, choices=(2, 3), required=True)
    parser.add_argument("--v1", type=float, required=True, help="Nominal v1.")
    parser.add_argument("--enlargement", "--e", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--diameter", type=float, default=1.0)
    parser.add_argument(
        "--l-over-d0",
        type=float,
        default=None,
        help="RVE size ratio. Defaults to 20 in 2D and 8 in 3D.",
    )
    parser.add_argument(
        "--point-generation",
        choices=("modern", "legacy"),
        default="modern",
    )
    parser.add_argument(
        "--paths",
        default="all",
        help="Comma-separated subset: R,A_Dt,A_cont,hybrid; default all.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    selected_paths = _parse_paths(args.paths)
    results, metadata = solve_nominal_paper_case(
        dimension=args.dimension,
        v1_nominal=args.v1,
        enlargement_factor=args.enlargement,
        seed=args.seed,
        diameter_physical=args.diameter,
        l_over_d0=args.l_over_d0,
        point_generation=args.point_generation,
        paths=selected_paths,
        verbose=args.verbose,
    )
    print(json.dumps(summarize_results(results, metadata), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
