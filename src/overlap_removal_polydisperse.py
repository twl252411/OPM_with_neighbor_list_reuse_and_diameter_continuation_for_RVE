"""Safe periodic overlap removal for polydisperse circles and spheres.

The physical diameter of particle ``i`` is ``D_i``.  With enlargement factor
``e`` and continuation scale ``lambda``, the active pair exclusion distance is

    a_ij(lambda) = lambda * e * (D_i + D_j) / 2.

Candidate construction uses a global periodic KDTree query followed by a
pair-specific cutoff filter.  A reused list remains complete while

    2*u_max + (lambda-lambda_build)*a_max < eta_s*s,

where ``a_max`` is the largest target pair exclusion distance.  The bound is
conservative but keeps the monodisperse proof valid for pair-dependent ranges.
Final acceptance is based on the maximum pair-relative overlap and an
independent fresh periodic query.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numba import njit
from scipy.spatial import cKDTree

from overlap_removal_fast import (
    FastSolverConfig,
    _deterministic_unit_direction,
    _limited_update,
    _stage_has_stagnated,
    particle_measure,
    periodic_displacement_statistics,
    query_pairs,
)


@dataclass
class PolydisperseNeighborListState:
    """Reusable-list state for pair-dependent exclusion distances."""

    pairs: np.ndarray | None = None
    reference_points: np.ndarray | None = None
    scale_at_build: float = 0.0
    tree_builds: int = 0
    reuses: int = 0
    reuse_lengths: list[int] = field(default_factory=list)
    current_reuse_length: int = 0
    rebuild_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    pending_rebuild_reason: str = "initialization"

    def invalidate(self, reason: str = "invalidated") -> None:
        if self.current_reuse_length > 0:
            self.reuse_lengths.append(self.current_reuse_length)
        self.pairs = None
        self.reference_points = None
        self.scale_at_build = 0.0
        self.current_reuse_length = 0
        self.pending_rebuild_reason = reason

    def finalize(self) -> None:
        if self.current_reuse_length > 0:
            self.reuse_lengths.append(self.current_reuse_length)
            self.current_reuse_length = 0


@dataclass
class PolydisperseValidationResult:
    """Independent endpoint diagnostics for a polydisperse configuration."""

    candidate_pair_count: int
    potential: float
    max_overlap: float
    max_relative_overlap: float
    overlap_pair_count: int
    violating_overlap_pair_count: int
    minimum_center_distance: float
    minimum_surface_gap: float
    minimum_normalized_separation: float
    nan_count: int
    inf_count: int
    coordinates_inside_box: bool


@dataclass
class PolydisperseSolverResult:
    """Result and diagnostics returned by :func:`solve_polydisperse_fast`."""

    points: np.ndarray
    diameters_physical: np.ndarray
    diameters_target: np.ndarray
    success: bool
    status: str
    solver_name: str
    solver_path: str
    runtime_total: float
    runtime_continuation: float
    runtime_cleanup: float
    runtime_validation: float
    iterations_total: int
    iterations_continuation: int
    iterations_cleanup: int
    gradient_evaluations: int
    tree_builds_total: int
    tree_builds_continuation: int
    tree_builds_cleanup: int
    neighbor_list_reuses: int
    reuse_ratio: float
    pair_evaluations: int
    particle_number: int
    v1_actual_physical: float
    collision_vf_actual: float
    diameter_mean: float
    diameter_cv: float
    diameter_min: float
    diameter_max: float
    final_potential: float
    final_max_overlap: float
    final_relative_max_overlap: float
    final_overlap_pair_count: int
    final_violating_overlap_pair_count: int
    minimum_center_distance: float
    minimum_surface_gap: float
    minimum_normalized_separation: float
    nan_count: int
    inf_count: int
    continuation_attempted_stages: int = 0
    continuation_accepted_stages: int = 0
    continuation_rescue_calls: int = 0
    continuation_successful_rescues: int = 0
    continuation_rollbacks: int = 0
    continuation_stages: list[dict[str, Any]] = field(default_factory=list)
    rebuild_diagnostics: list[dict[str, Any]] = field(default_factory=list)

    def summary_dict(self) -> dict[str, Any]:
        data = dict(vars(self))
        data.pop("points", None)
        data["diameters_physical"] = self.diameters_physical.tolist()
        data["diameters_target"] = self.diameters_target.tolist()
        return data


def pair_exclusion_distances(
    pairs: np.ndarray,
    particle_diameters: np.ndarray,
) -> np.ndarray:
    """Return ``(D_i + D_j)/2`` for every pair."""

    if pairs.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    return 0.5 * (
        particle_diameters[pairs[:, 0]] + particle_diameters[pairs[:, 1]]
    )


def _filter_pair_specific_cutoff(
    points: np.ndarray,
    pairs: np.ndarray,
    box_size: np.ndarray,
    particle_diameters: np.ndarray,
    skin: float = 0.0,
) -> np.ndarray:
    """Filter a global radius-query result by pair-specific cutoffs."""

    if pairs.shape[0] == 0:
        return pairs
    delta = points[pairs[:, 0]] - points[pairs[:, 1]]
    delta -= box_size * np.rint(delta / box_size)
    distances = np.linalg.norm(delta, axis=1)
    cutoffs = pair_exclusion_distances(pairs, particle_diameters) + skin
    return pairs[distances <= cutoffs]


@njit(cache=True)
def compute_polydisperse_gradients_and_metrics(
    points: np.ndarray,
    pairs: np.ndarray,
    box_size: np.ndarray,
    particle_diameters: np.ndarray,
    gradients: np.ndarray,
    norm_eps: float,
    seed: int,
) -> tuple[float, float, float, int]:
    """Evaluate the pair-specific quadratic overlap potential and gradient."""

    gradients[:, :] = 0.0
    potential = 0.0
    max_overlap = 0.0
    max_relative_overlap = 0.0
    overlap_pair_count = 0
    dim = points.shape[1]

    for pair_index in range(pairs.shape[0]):
        particle_i = pairs[pair_index, 0]
        particle_j = pairs[pair_index, 1]
        exclusion = 0.5 * (
            particle_diameters[particle_i] + particle_diameters[particle_j]
        )

        delta = np.empty(dim, dtype=np.float64)
        norm_sq = 0.0
        for axis in range(dim):
            value = points[particle_i, axis] - points[particle_j, axis]
            value -= box_size[axis] * round(value / box_size[axis])
            delta[axis] = value
            norm_sq += value * value

        distance = math.sqrt(norm_sq)
        overlap = exclusion - distance
        if overlap <= 0.0:
            continue

        overlap_pair_count += 1
        potential += 0.5 * overlap * overlap
        if overlap > max_overlap:
            max_overlap = overlap
        relative_overlap = overlap / exclusion
        if relative_overlap > max_relative_overlap:
            max_relative_overlap = relative_overlap

        if distance > norm_eps:
            inv_distance = 1.0 / distance
            for axis in range(dim):
                component = -overlap * delta[axis] * inv_distance
                gradients[particle_i, axis] += component
                gradients[particle_j, axis] -= component
        else:
            direction = _deterministic_unit_direction(
                particle_i, particle_j, seed, dim
            )
            for axis in range(dim):
                component = -overlap * direction[axis]
                gradients[particle_i, axis] += component
                gradients[particle_j, axis] -= component

    return potential, max_overlap, max_relative_overlap, overlap_pair_count


def generate_polydisperse_diameters(
    particle_number: int,
    mean_diameter: float,
    coefficient_of_variation: float,
    dim: int,
    seed: int,
    preserve_total_measure: bool = True,
) -> np.ndarray:
    """Generate lognormal particle diameters with a reproducible seed.

    When ``preserve_total_measure`` is true, the sample is rescaled so that
    ``sum(D_i**dim) == particle_number * mean_diameter**dim``.  This makes the
    physical fraction identical to the corresponding monodisperse case while
    retaining the sampled diameter ratios.
    """

    if particle_number < 1:
        raise ValueError("particle_number must be positive.")
    if dim not in (2, 3):
        raise ValueError("dim must be 2 or 3.")
    if not math.isfinite(mean_diameter) or mean_diameter <= 0.0:
        raise ValueError("mean_diameter must be finite and positive.")
    if (
        not math.isfinite(coefficient_of_variation)
        or coefficient_of_variation < 0.0
    ):
        raise ValueError("coefficient_of_variation must be finite and nonnegative.")
    if coefficient_of_variation == 0.0:
        return np.full(particle_number, mean_diameter, dtype=np.float64)

    sigma = math.sqrt(math.log1p(coefficient_of_variation**2))
    mu = math.log(mean_diameter) - 0.5 * sigma * sigma
    rng = np.random.default_rng(seed)
    diameters = rng.lognormal(mu, sigma, particle_number).astype(np.float64)
    if preserve_total_measure:
        scale = (
            particle_number * mean_diameter**dim / float(np.sum(diameters**dim))
        ) ** (1.0 / dim)
        diameters *= scale
    return diameters


def compute_polydisperse_fractions(
    particle_diameters: np.ndarray,
    box_size: np.ndarray,
    enlargement_factor: float,
) -> tuple[float, float]:
    """Return actual physical and target collision fractions."""

    dim = int(box_size.size)
    measure = float(
        np.sum([particle_measure(float(value), dim) for value in particle_diameters])
    )
    physical = measure / float(np.prod(box_size))
    return physical, physical * enlargement_factor**dim


def validate_polydisperse_inputs(
    initial_points: np.ndarray,
    box_size: np.ndarray,
    particle_diameters: np.ndarray,
    enlargement_factor: float,
    skin_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate inputs and return points, box, physical and target diameters."""

    points = np.asarray(initial_points, dtype=np.float64).copy()
    box = np.asarray(box_size, dtype=np.float64).copy()
    diameters = np.asarray(particle_diameters, dtype=np.float64).copy()
    if points.ndim != 2 or points.shape[1] not in (2, 3):
        raise ValueError("initial_points must have shape (N, 2) or (N, 3).")
    if diameters.shape != (points.shape[0],):
        raise ValueError("particle_diameters must contain one value per particle.")
    if not np.all(np.isfinite(points)):
        raise ValueError("initial_points contains NaN or Inf values.")
    if not np.all(np.isfinite(box)) or np.any(box <= 0.0):
        raise ValueError("All box lengths must be finite and positive.")
    if not np.all(np.isfinite(diameters)) or np.any(diameters <= 0.0):
        raise ValueError("All particle diameters must be finite and positive.")
    if not math.isfinite(enlargement_factor) or enlargement_factor <= 0.0:
        raise ValueError("enlargement_factor must be finite and positive.")
    if not math.isfinite(skin_ratio) or skin_ratio <= 0.0:
        raise ValueError("skin_ratio must be finite and positive.")

    target = enlargement_factor * diameters
    maximum_cutoff = float(np.max(target)) * (1.0 + skin_ratio)
    if maximum_cutoff >= 0.5 * float(np.min(box)):
        raise ValueError(
            "The largest target pair cutoff plus skin must be smaller than "
            "half the minimum box length."
        )
    points %= box
    return points, box, diameters, target


def ensure_safe_polydisperse_neighbor_list(
    points: np.ndarray,
    box_size: np.ndarray,
    target_particle_diameters: np.ndarray,
    current_scale: float,
    skin: float,
    safety_factor: float,
    state: PolydisperseNeighborListState,
) -> tuple[np.ndarray, bool, float]:
    """Return a complete pair-specific candidate list, rebuilding if needed."""

    maximum_target_exclusion = float(np.max(target_particle_diameters))
    must_rebuild = state.pairs is None or state.reference_points is None
    safety_measure = math.inf
    displacement_stats: dict[str, float | None] | None = None
    displacement_contribution: float | None = None
    growth_contribution: float | None = None
    previous_scale = (
        float(state.scale_at_build) if state.reference_points is not None else None
    )
    rebuild_reason = state.pending_rebuild_reason
    if not must_rebuild:
        displacement_stats = periodic_displacement_statistics(
            points, state.reference_points, box_size
        )
        max_displacement = float(displacement_stats["u_max"])
        growth = max(0.0, current_scale - state.scale_at_build)
        displacement_contribution = 2.0 * max_displacement
        growth_contribution = growth * maximum_target_exclusion
        safety_measure = displacement_contribution + growth_contribution
        threshold = safety_factor * skin
        must_rebuild = safety_measure >= threshold
        if must_rebuild:
            displacement_trigger = displacement_contribution >= threshold
            growth_trigger = growth_contribution >= threshold
            if displacement_trigger and growth_trigger:
                rebuild_reason = "displacement_and_diameter_growth"
            elif displacement_trigger:
                rebuild_reason = "displacement"
            elif growth_trigger:
                rebuild_reason = "diameter_growth"
            else:
                rebuild_reason = "combined_budget"

    if must_rebuild:
        if state.current_reuse_length > 0:
            state.reuse_lengths.append(state.current_reuse_length)
        active_diameters = current_scale * target_particle_diameters
        global_cutoff = float(np.max(active_diameters)) + skin
        raw_pairs = query_pairs(cKDTree(points, boxsize=box_size), global_cutoff)
        pairs = _filter_pair_specific_cutoff(
            points, raw_pairs, box_size, active_diameters, skin
        )
        state.pairs = pairs
        state.reference_points = points.copy()
        state.scale_at_build = current_scale
        state.tree_builds += 1
        state.current_reuse_length = 0
        diagnostic: dict[str, Any] = {
            "tree_build_index": state.tree_builds,
            "reason": rebuild_reason,
            "scale_current": float(current_scale),
            "scale_at_previous_build": previous_scale,
            "maximum_target_exclusion": maximum_target_exclusion,
            "skin": float(skin),
            "safety_threshold": float(safety_factor * skin),
            "displacement_contribution": displacement_contribution,
            "diameter_growth_contribution": growth_contribution,
            "safety_budget": (
                float(safety_measure) if math.isfinite(safety_measure) else None
            ),
        }
        if displacement_stats is None:
            diagnostic.update({
                "u_max": None,
                "u_95": None,
                "u_99": None,
                "u_max_over_u_95": None,
                "u_max_over_u_99": None,
            })
        else:
            diagnostic.update(displacement_stats)
        state.rebuild_diagnostics.append(diagnostic)
        state.pending_rebuild_reason = "safety_budget"
        return pairs, True, safety_measure

    state.reuses += 1
    state.current_reuse_length += 1
    return state.pairs, False, safety_measure


def validate_polydisperse_configuration(
    points: np.ndarray,
    box_size: np.ndarray,
    target_particle_diameters: np.ndarray,
    norm_eps: float,
    seed: int,
    relative_overlap_tolerance: float = 0.0,
) -> PolydisperseValidationResult:
    """Independently validate all pair-dependent target exclusions."""

    nan_count = int(np.isnan(points).sum())
    inf_count = int(np.isinf(points).sum())
    inside = bool(
        nan_count == 0
        and inf_count == 0
        and np.all(points >= 0.0)
        and np.all(points < box_size)
    )
    if nan_count or inf_count:
        return PolydisperseValidationResult(
            0, math.inf, math.inf, math.inf, -1, -1, math.nan,
            math.nan, math.nan, nan_count, inf_count, False
        )

    tree = cKDTree(points, boxsize=box_size)
    raw_pairs = query_pairs(tree, float(np.max(target_particle_diameters)))
    pairs = _filter_pair_specific_cutoff(
        points, raw_pairs, box_size, target_particle_diameters
    )
    gradients = np.zeros_like(points)
    potential, max_overlap, max_relative, overlap_count = (
        compute_polydisperse_gradients_and_metrics(
            points, pairs, box_size, target_particle_diameters,
            gradients, norm_eps, seed
        )
    )

    if points.shape[0] < 2:
        minimum_distance = math.inf
    else:
        nearest_distances, _ = tree.query(points, k=2)
        minimum_distance = float(np.min(nearest_distances[:, 1]))

    if raw_pairs.shape[0]:
        delta = points[raw_pairs[:, 0]] - points[raw_pairs[:, 1]]
        delta -= box_size * np.rint(delta / box_size)
        distances = np.linalg.norm(delta, axis=1)
        exclusions = pair_exclusion_distances(raw_pairs, target_particle_diameters)
        overlaps = exclusions - distances
        relative = overlaps / exclusions
        violating = int(np.count_nonzero(relative > relative_overlap_tolerance))
        minimum_gap = float(np.min(distances - exclusions))
        minimum_normalized = float(np.min(distances / exclusions))
    else:
        violating = 0
        minimum_distance = math.inf
        minimum_gap = math.inf
        minimum_normalized = math.inf

    return PolydisperseValidationResult(
        candidate_pair_count=int(pairs.shape[0]),
        potential=float(potential),
        max_overlap=float(max_overlap),
        max_relative_overlap=float(max_relative),
        overlap_pair_count=int(overlap_count),
        violating_overlap_pair_count=violating,
        minimum_center_distance=minimum_distance,
        minimum_surface_gap=minimum_gap,
        minimum_normalized_separation=minimum_normalized,
        nan_count=nan_count,
        inf_count=inf_count,
        coordinates_inside_box=inside,
    )


def audit_polydisperse_all_pairs(
    points: np.ndarray,
    box_size: np.ndarray,
    target_particle_diameters: np.ndarray,
    relative_overlap_tolerance: float = 0.0,
    block_size: int = 512,
) -> PolydisperseValidationResult:
    """Certify a configuration by explicitly evaluating every unordered pair.

    This audit is independent of KDTree candidate construction and is intended
    for representative endpoints and regression tests rather than every inner
    solver iteration.
    """

    points = np.asarray(points, dtype=np.float64)
    box = np.asarray(box_size, dtype=np.float64)
    diameters = np.asarray(target_particle_diameters, dtype=np.float64)
    if block_size < 1:
        raise ValueError("block_size must be positive.")
    if points.ndim != 2 or diameters.shape != (points.shape[0],):
        raise ValueError("Incompatible points and particle diameters.")
    nan_count = int(np.isnan(points).sum())
    inf_count = int(np.isinf(points).sum())
    inside = bool(
        nan_count == 0
        and inf_count == 0
        and np.all(points >= 0.0)
        and np.all(points < box)
    )
    if nan_count or inf_count:
        return PolydisperseValidationResult(
            0, math.inf, math.inf, math.inf, -1, -1, math.nan,
            math.nan, math.nan, nan_count, inf_count, False
        )

    particle_number = points.shape[0]
    pair_count = particle_number * (particle_number - 1) // 2
    potential = 0.0
    max_overlap = 0.0
    max_relative = 0.0
    overlap_count = 0
    violating_count = 0
    minimum_distance = math.inf
    minimum_gap = math.inf
    minimum_normalized = math.inf

    for i_start in range(0, particle_number, block_size):
        i_stop = min(particle_number, i_start + block_size)
        for j_start in range(i_start, particle_number, block_size):
            j_stop = min(particle_number, j_start + block_size)
            i_indices = np.arange(i_start, i_stop)[:, None]
            j_indices = np.arange(j_start, j_stop)[None, :]
            valid = i_indices < j_indices
            if not np.any(valid):
                continue
            delta = (
                points[i_start:i_stop, None, :]
                - points[None, j_start:j_stop, :]
            )
            delta -= box * np.rint(delta / box)
            distances = np.linalg.norm(delta, axis=2)[valid]
            exclusions = (
                0.5
                * (
                    diameters[i_start:i_stop, None]
                    + diameters[None, j_start:j_stop]
                )[valid]
            )
            overlaps = exclusions - distances
            relative = overlaps / exclusions
            positive = overlaps > 0.0
            potential += float(0.5 * np.sum(overlaps[positive] ** 2))
            overlap_count += int(np.count_nonzero(positive))
            violating_count += int(
                np.count_nonzero(relative > relative_overlap_tolerance)
            )
            max_overlap = max(max_overlap, float(np.max(overlaps, initial=0.0)))
            max_relative = max(
                max_relative, float(np.max(relative, initial=0.0))
            )
            minimum_distance = min(minimum_distance, float(np.min(distances)))
            minimum_gap = min(
                minimum_gap, float(np.min(distances - exclusions))
            )
            minimum_normalized = min(
                minimum_normalized, float(np.min(distances / exclusions))
            )

    return PolydisperseValidationResult(
        candidate_pair_count=pair_count,
        potential=potential,
        max_overlap=max_overlap,
        max_relative_overlap=max_relative,
        overlap_pair_count=overlap_count,
        violating_overlap_pair_count=violating_count,
        minimum_center_distance=minimum_distance,
        minimum_surface_gap=minimum_gap,
        minimum_normalized_separation=minimum_normalized,
        nan_count=nan_count,
        inf_count=inf_count,
        coordinates_inside_box=inside,
    )


def _relax_polydisperse_stage(
    points: np.ndarray,
    box_size: np.ndarray,
    target_particle_diameters: np.ndarray,
    current_scale: float,
    absolute_tolerance: float,
    seed: int,
    config: FastSolverConfig,
    state: PolydisperseNeighborListState,
    max_iterations: int,
) -> tuple[np.ndarray, bool, str, dict[str, Any]]:
    points = points.copy()
    gradients = np.zeros_like(points)
    maximum_target_exclusion = float(np.max(target_particle_diameters))
    skin = config.skin_ratio * maximum_target_exclusion
    potential_history: list[float] = []
    overlap_history: list[float] = []
    starting_builds = state.tree_builds
    starting_reuses = state.reuses
    pair_evaluations = 0
    last = (math.inf, math.inf, math.inf, -1)

    for iteration in range(max_iterations + 1):
        pairs, rebuilt, safety_measure = ensure_safe_polydisperse_neighbor_list(
            points, box_size, target_particle_diameters, current_scale,
            skin, config.neighbor_safety_factor, state
        )
        pair_evaluations += int(pairs.shape[0])
        active_diameters = current_scale * target_particle_diameters
        last = compute_polydisperse_gradients_and_metrics(
            points, pairs, box_size, active_diameters,
            gradients, config.norm_eps, seed
        )
        potential, max_overlap, max_relative, overlap_count = last
        potential_history.append(float(potential))
        overlap_history.append(float(max_overlap))
        if config.verbose and (
            iteration % config.print_interval == 0
            or max_overlap <= absolute_tolerance
        ):
            print(
                f"  Polydisperse stage iteration {iteration}: "
                f"lambda={current_scale:.6f}, potential={potential:.6e}, "
                f"max_overlap={max_overlap:.6e}, pairs={overlap_count}, "
                f"rebuilt={rebuilt}"
            )
        if max_overlap <= absolute_tolerance:
            return points, True, "converged", {
                "iterations": iteration,
                "gradient_evaluations": iteration + 1,
                "tree_builds": state.tree_builds - starting_builds,
                "neighbor_reuses": state.reuses - starting_reuses,
                "pair_evaluations": pair_evaluations,
                "potential": float(potential),
                "max_overlap": float(max_overlap),
                "max_relative_overlap": float(max_relative),
                "overlap_count": int(overlap_count),
                "safety_measure": float(safety_measure),
            }
        if iteration == max_iterations:
            break
        if _stage_has_stagnated(
            potential_history, overlap_history,
            config.stagnation_window, config.stagnation_relative_drop
        ):
            status = "stagnated"
            break
        max_move = min(
            config.max_move_ratio * current_scale * maximum_target_exclusion,
            config.max_move_skin_ratio * skin,
        )
        points = _limited_update(
            points, gradients, box_size, config.alpha, max_move
        )
        if not np.all(np.isfinite(points)):
            status = "invalid_coordinates"
            break
    else:
        status = "max_iterations"
    if iteration >= max_iterations:
        status = "max_iterations"
    potential, max_overlap, max_relative, overlap_count = last
    return points, False, status, {
        "iterations": int(iteration),
        "gradient_evaluations": int(iteration + 1),
        "tree_builds": state.tree_builds - starting_builds,
        "neighbor_reuses": state.reuses - starting_reuses,
        "pair_evaluations": pair_evaluations,
        "potential": float(potential),
        "max_overlap": float(max_overlap),
        "max_relative_overlap": float(max_relative),
        "overlap_count": int(overlap_count),
        "safety_measure": math.nan,
    }


def _exact_polydisperse_relaxation(
    points: np.ndarray,
    box_size: np.ndarray,
    target_particle_diameters: np.ndarray,
    current_scale: float,
    absolute_tolerance: float,
    seed: int,
    config: FastSolverConfig,
    max_iterations: int,
    alpha: float,
) -> tuple[np.ndarray, bool, str, dict[str, Any]]:
    points = points.copy()
    gradients = np.zeros_like(points)
    active_diameters = current_scale * target_particle_diameters
    maximum_exclusion = float(np.max(active_diameters))
    pair_evaluations = 0
    last = (math.inf, math.inf, math.inf, -1)
    for iteration in range(max_iterations + 1):
        raw_pairs = query_pairs(
            cKDTree(points, boxsize=box_size), maximum_exclusion
        )
        pairs = _filter_pair_specific_cutoff(
            points, raw_pairs, box_size, active_diameters
        )
        pair_evaluations += int(pairs.shape[0])
        last = compute_polydisperse_gradients_and_metrics(
            points, pairs, box_size, active_diameters,
            gradients, config.norm_eps, seed
        )
        potential, max_overlap, max_relative, overlap_count = last
        if max_overlap <= absolute_tolerance:
            return points, True, "converged", {
                "iterations": iteration,
                "gradient_evaluations": iteration + 1,
                "tree_builds": iteration + 1,
                "pair_evaluations": pair_evaluations,
                "potential": float(potential),
                "max_overlap": float(max_overlap),
                "max_relative_overlap": float(max_relative),
                "overlap_count": int(overlap_count),
            }
        if iteration == max_iterations:
            break
        max_move = config.cleanup_max_move_ratio * maximum_exclusion
        points = _limited_update(points, gradients, box_size, alpha, max_move)
        if not np.all(np.isfinite(points)):
            return points, False, "invalid_coordinates", {
                "iterations": iteration + 1,
                "gradient_evaluations": iteration + 1,
                "tree_builds": iteration + 1,
                "pair_evaluations": pair_evaluations,
                "potential": float(potential),
                "max_overlap": float(max_overlap),
                "max_relative_overlap": float(max_relative),
                "overlap_count": int(overlap_count),
            }
    potential, max_overlap, max_relative, overlap_count = last
    return points, False, "max_iterations", {
        "iterations": max_iterations,
        "gradient_evaluations": max_iterations + 1,
        "tree_builds": max_iterations + 1,
        "pair_evaluations": pair_evaluations,
        "potential": float(potential),
        "max_overlap": float(max_overlap),
        "max_relative_overlap": float(max_relative),
        "overlap_count": int(overlap_count),
    }


def solve_polydisperse_fast(
    initial_points: np.ndarray,
    box_size: np.ndarray,
    particle_diameters: np.ndarray,
    enlargement_factor: float,
    seed: int = 0,
    config: FastSolverConfig | None = None,
) -> PolydisperseSolverResult:
    """Run safe neighbor-list reuse and continuation for polydisperse particles."""

    if config is None:
        config = FastSolverConfig()
    points, box, physical_diameters, target_diameters = (
        validate_polydisperse_inputs(
            initial_points, box_size, particle_diameters,
            enlargement_factor, config.skin_ratio
        )
    )
    if not (0.0 < config.neighbor_safety_factor <= 1.0):
        raise ValueError("neighbor_safety_factor must lie in (0, 1].")
    if config.strict_overlap_ratio <= 0.0:
        raise ValueError("strict_overlap_ratio must be positive.")

    dim = points.shape[1]
    v1_actual, collision_vf = compute_polydisperse_fractions(
        physical_diameters, box, enlargement_factor
    )
    if collision_vf >= 1.0:
        raise ValueError("Target collision fraction must be smaller than one.")
    phi_easy = config.phi_easy_2d if dim == 2 else config.phi_easy_3d
    phi_continuation = (
        config.phi_continuation_2d if dim == 2
        else config.phi_continuation_3d
    )
    use_continuation = collision_vf >= phi_continuation
    if use_continuation:
        lambda_initial = max(
            config.lambda_min,
            min(1.0, (phi_easy / collision_vf) ** (1.0 / dim)),
        )
    else:
        lambda_initial = 1.0

    # Compile the pair-specific kernel before the timed interval.
    dummy_points = np.zeros((2, dim), dtype=np.float64)
    compute_polydisperse_gradients_and_metrics(
        dummy_points, np.array([[0, 1]], dtype=np.int64),
        np.ones(dim), np.ones(2), np.zeros_like(dummy_points),
        config.norm_eps, seed
    )
    start_total = time.perf_counter()
    continuation_start = time.perf_counter()
    state = PolydisperseNeighborListState()
    stages: list[dict[str, Any]] = []
    continuation_iterations = 0
    continuation_gradients = 0
    continuation_pairs = 0
    rescue_tree_builds = 0
    continuation_rollbacks = 0
    delta_lambda = config.delta_lambda_initial
    accepted_points = points.copy()
    accepted_lambda: float | None = None
    target_lambda = lambda_initial
    continuation_success = True
    continuation_status = "converged"
    maximum_target_exclusion = float(np.max(target_diameters))

    while use_continuation and target_lambda < 1.0 - 1.0e-15:
        growth = delta_lambda if accepted_lambda is None else target_lambda - accepted_lambda
        tolerance = max(
            config.stage_overlap_ratio * maximum_target_exclusion,
            config.stage_growth_fraction * growth * maximum_target_exclusion,
        )
        trial, ok, status, stats = _relax_polydisperse_stage(
            accepted_points, box, target_diameters, target_lambda,
            tolerance, seed, config, state, config.max_iterations_stage
        )
        continuation_iterations += int(stats["iterations"])
        continuation_gradients += int(stats["gradient_evaluations"])
        continuation_pairs += int(stats["pair_evaluations"])
        neighbor_stats = dict(stats)
        rescue_stats: dict[str, Any] | None = None
        used_rescue = False
        if not ok:
            rescue, rescue_ok, rescue_status, rescue_stats = (
                _exact_polydisperse_relaxation(
                    trial, box, target_diameters, target_lambda,
                    tolerance, seed, config,
                    config.max_iterations_stage_rescue, 0.5 * config.alpha
                )
            )
            used_rescue = True
            continuation_iterations += int(rescue_stats["iterations"])
            continuation_gradients += int(rescue_stats["gradient_evaluations"])
            continuation_pairs += int(rescue_stats["pair_evaluations"])
            rescue_tree_builds += int(rescue_stats["tree_builds"])
            state.invalidate("post_rescue")
            if rescue_ok:
                trial, ok, status = rescue, True, rescue_status
            else:
                status = rescue_status
        total_stage_iterations = int(neighbor_stats["iterations"]) + (
            int(rescue_stats["iterations"]) if rescue_stats else 0
        )
        stages.append({
            "stage_type": "continuation",
            "lambda": float(target_lambda),
            "maximum_active_exclusion": float(
                target_lambda * maximum_target_exclusion
            ),
            "tolerance": float(tolerance),
            "success": bool(ok),
            "status": status,
            "used_rescue": used_rescue,
            "iterations": total_stage_iterations,
            "neighbor_stage": neighbor_stats,
            "rescue_stage": rescue_stats,
        })
        if not ok:
            if accepted_lambda is None:
                lower = max(config.lambda_min, target_lambda - delta_lambda)
                if lower < target_lambda - 1.0e-15:
                    state.invalidate("rollback_lower_initial_scale")
                    continuation_rollbacks += 1
                    target_lambda = lower
                    delta_lambda = max(config.delta_lambda_min, 0.5 * delta_lambda)
                    continue
                continuation_success = False
                continuation_status = "continuation_failed"
                break
            delta_lambda *= 0.5
            if delta_lambda < config.delta_lambda_min:
                continuation_success = False
                continuation_status = "continuation_failed"
                break
            state.invalidate("rollback_smaller_growth")
            continuation_rollbacks += 1
            target_lambda = min(1.0, accepted_lambda + delta_lambda)
            continue
        accepted_points = trial
        accepted_lambda = target_lambda
        if total_stage_iterations < config.stage_fast_iterations:
            delta_lambda = min(config.delta_lambda_max, 1.4 * delta_lambda)
        elif total_stage_iterations > config.stage_slow_iterations:
            delta_lambda = max(config.delta_lambda_min, 0.5 * delta_lambda)
        target_lambda = min(1.0, accepted_lambda + delta_lambda)

    points = accepted_points
    if continuation_success:
        pre_tolerance = config.precleanup_overlap_ratio * maximum_target_exclusion
        pre_points, pre_ok, pre_status, pre_stats = _relax_polydisperse_stage(
            points, box, target_diameters, 1.0, pre_tolerance,
            seed, config, state, config.max_iterations_precleanup
        )
        continuation_iterations += int(pre_stats["iterations"])
        continuation_gradients += int(pre_stats["gradient_evaluations"])
        continuation_pairs += int(pre_stats["pair_evaluations"])
        stages.append({
            "stage_type": "target_precleanup",
            "lambda": 1.0,
            "maximum_active_exclusion": maximum_target_exclusion,
            "tolerance": pre_tolerance,
            "success": bool(pre_ok),
            "status": pre_status,
            "used_rescue": False,
            **pre_stats,
        })
        if np.all(np.isfinite(pre_points)):
            points = pre_points
        else:
            continuation_success = False
            continuation_status = "invalid_coordinates"
    state.finalize()
    runtime_continuation = time.perf_counter() - continuation_start

    cleanup_start = time.perf_counter()
    if continuation_success:
        absolute_strict_tolerance = (
            config.strict_overlap_ratio * float(np.min(target_diameters))
        )
        points, cleanup_ok, cleanup_status, cleanup_stats = (
            _exact_polydisperse_relaxation(
                points, box, target_diameters, 1.0,
                absolute_strict_tolerance, seed, config,
                config.max_iterations_cleanup, config.alpha
            )
        )
    else:
        cleanup_ok = False
        cleanup_status = continuation_status
        cleanup_stats = {
            "iterations": 0, "gradient_evaluations": 0,
            "tree_builds": 0, "pair_evaluations": 0,
        }
    runtime_cleanup = time.perf_counter() - cleanup_start

    validation_start = time.perf_counter()
    validation = validate_polydisperse_configuration(
        points, box, target_diameters, config.norm_eps, seed,
        relative_overlap_tolerance=config.strict_overlap_ratio
    )
    runtime_validation = time.perf_counter() - validation_start
    validation_ok = bool(
        validation.max_relative_overlap <= config.strict_overlap_ratio
        and validation.violating_overlap_pair_count == 0
        and validation.coordinates_inside_box
    )
    success = bool(continuation_success and cleanup_ok and validation_ok)
    if success:
        final_status = "converged"
    elif not continuation_success:
        final_status = continuation_status
    elif not cleanup_ok:
        final_status = cleanup_status
    else:
        final_status = "validation_failed"

    continuation_builds = state.tree_builds + rescue_tree_builds
    cleanup_builds = int(cleanup_stats["tree_builds"])
    list_calls = state.tree_builds + state.reuses
    diameter_mean = float(np.mean(physical_diameters))
    diameter_cv = float(np.std(physical_diameters, ddof=1) / diameter_mean)
    continuation_only_stages = [
        stage for stage in stages if stage.get("stage_type") == "continuation"
    ]
    attempted_stages = len(continuation_only_stages)
    accepted_stages = sum(
        bool(stage.get("success")) for stage in continuation_only_stages
    )
    rescue_calls = sum(
        bool(stage.get("used_rescue")) for stage in continuation_only_stages
    )
    successful_rescues = sum(
        bool(stage.get("used_rescue")) and bool(stage.get("success"))
        for stage in continuation_only_stages
    )
    runtime_total = time.perf_counter() - start_total
    return PolydisperseSolverResult(
        points=points,
        diameters_physical=physical_diameters,
        diameters_target=target_diameters,
        success=success,
        status=final_status,
        solver_name="polydisperse_fast_strict",
        solver_path=(
            "polydisperse_fast_continuation" if use_continuation
            else "polydisperse_fast_direct"
        ),
        runtime_total=runtime_total,
        runtime_continuation=runtime_continuation,
        runtime_cleanup=runtime_cleanup,
        runtime_validation=runtime_validation,
        iterations_total=continuation_iterations + int(cleanup_stats["iterations"]),
        iterations_continuation=continuation_iterations,
        iterations_cleanup=int(cleanup_stats["iterations"]),
        gradient_evaluations=(
            continuation_gradients + int(cleanup_stats["gradient_evaluations"])
        ),
        tree_builds_total=continuation_builds + cleanup_builds + 1,
        tree_builds_continuation=continuation_builds,
        tree_builds_cleanup=cleanup_builds,
        neighbor_list_reuses=state.reuses,
        reuse_ratio=(state.reuses / list_calls if list_calls else 0.0),
        pair_evaluations=(
            continuation_pairs + int(cleanup_stats["pair_evaluations"])
            + validation.candidate_pair_count
        ),
        particle_number=int(points.shape[0]),
        v1_actual_physical=float(v1_actual),
        collision_vf_actual=float(collision_vf),
        diameter_mean=diameter_mean,
        diameter_cv=diameter_cv,
        diameter_min=float(np.min(physical_diameters)),
        diameter_max=float(np.max(physical_diameters)),
        final_potential=validation.potential,
        final_max_overlap=validation.max_overlap,
        final_relative_max_overlap=validation.max_relative_overlap,
        final_overlap_pair_count=validation.overlap_pair_count,
        final_violating_overlap_pair_count=validation.violating_overlap_pair_count,
        minimum_center_distance=validation.minimum_center_distance,
        minimum_surface_gap=validation.minimum_surface_gap,
        minimum_normalized_separation=validation.minimum_normalized_separation,
        nan_count=validation.nan_count,
        inf_count=validation.inf_count,
        continuation_attempted_stages=int(attempted_stages),
        continuation_accepted_stages=int(accepted_stages),
        continuation_rescue_calls=int(rescue_calls),
        continuation_successful_rescues=int(successful_rescues),
        continuation_rollbacks=int(continuation_rollbacks),
        continuation_stages=stages,
        rebuild_diagnostics=state.rebuild_diagnostics,
    )
