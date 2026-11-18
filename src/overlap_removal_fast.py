"""Fast periodic overlap removal for monodisperse circles and spheres.

The solver improves the original fixed-diameter algorithm through:

1. safe Verlet-type neighbor-list reuse;
2. adaptive radius continuation at high collision volume fractions;
3. exact final cleanup with a fresh periodic KDTree at every iteration.

The module is directly executable and can also be imported by benchmark scripts.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numba import njit
from scipy.spatial import cKDTree


# ============================================================================
# Configuration and result containers
# ============================================================================


@dataclass
class FastSolverConfig:
    """Configuration of the fast overlap-removal solver.

    Parameters
    ----------
    alpha : float
        Gradient-descent step size.
    norm_eps : float
        Distance threshold used to handle coincident particles.
    skin_ratio : float
        Neighbor-list skin divided by the target collision diameter.
    neighbor_safety_factor : float
        Fraction of the skin budget at which the list is rebuilt.
    phi_easy_2d, phi_easy_3d : float
        Easy collision fractions used to determine the initial continuation
        diameter in two and three dimensions.
    lambda_min : float
        Lower bound on the initial continuation diameter ratio.
    delta_lambda_initial, delta_lambda_min, delta_lambda_max : float
        Initial and bounded continuation increments.
    stage_overlap_ratio : float
        Intermediate-stage overlap tolerance divided by target diameter.
    stage_growth_fraction : float
        Fraction of the current diameter increment used in the stage tolerance.
    stage_fast_iterations, stage_slow_iterations : int
        Thresholds used to adapt the continuation increment.
    max_iterations_stage : int
        Maximum neighbor-list iterations in one continuation stage.
    max_iterations_stage_rescue : int
        Maximum exact-KDTree rescue iterations for a failed stage.
    max_iterations_cleanup : int
        Maximum exact cleanup iterations at the target diameter.
    strict_overlap_ratio : float
        Final maximum-overlap tolerance divided by target diameter.
    max_move_ratio : float
        Maximum continuation displacement divided by current diameter.
    max_move_skin_ratio : float
        Maximum continuation displacement divided by the skin.
    cleanup_max_move_ratio : float
        Maximum cleanup displacement divided by target diameter.
    stagnation_window : int
        Iteration window used to detect stage stagnation.
    stagnation_relative_drop : float
        Minimum relative reduction over the stagnation window.
    print_interval : int
        Iteration interval for verbose progress output.
    verbose : bool
        Print continuation and cleanup progress when True.
    """

    alpha: float = 0.4
    norm_eps: float = 1.0e-12

    skin_ratio: float = 0.05
    neighbor_safety_factor: float = 0.80

    phi_easy_2d: float = 0.55
    phi_easy_3d: float = 0.42
    phi_continuation_2d: float = 0.70
    phi_continuation_3d: float = 0.62
    lambda_min: float = 0.75
    delta_lambda_initial: float = 0.02
    delta_lambda_min: float = 0.0025
    delta_lambda_max: float = 0.08

    stage_overlap_ratio: float = 1.0e-3
    stage_growth_fraction: float = 0.10
    stage_fast_iterations: int = 100
    stage_slow_iterations: int = 500
    max_iterations_stage: int = 10_000
    max_iterations_stage_rescue: int = 3_000
    max_iterations_precleanup: int = 50_000
    precleanup_overlap_ratio: float = 3.0e-8

    max_iterations_cleanup: int = 100_000
    strict_overlap_ratio: float = 1.0e-8

    max_move_ratio: float = 0.05
    max_move_skin_ratio: float = 0.20
    cleanup_max_move_ratio: float = 0.02

    stagnation_window: int = 100
    stagnation_relative_drop: float = 1.0e-4

    print_interval: int = 200
    verbose: bool = True


@dataclass
class ValidationResult:
    """Independent geometric validation result."""

    candidate_pair_count: int
    potential: float
    max_overlap: float
    relative_max_overlap: float
    overlap_pair_count: int
    violating_overlap_pair_count: int
    minimum_center_distance: float
    nan_count: int
    inf_count: int
    coordinates_inside_box: bool


@dataclass
class SolverResult:
    """Result and diagnostics returned by an overlap-removal solver."""

    points: np.ndarray
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

    diameter_physical: float
    diameter_target: float
    particle_number: int
    v1_actual_physical: float
    collision_vf_actual: float

    final_potential: float
    final_max_overlap: float
    final_relative_max_overlap: float
    final_overlap_pair_count: int
    final_violating_overlap_pair_count: int
    minimum_center_distance: float
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
        """Return a JSON-serializable summary without the coordinate array."""

        data = dict(vars(self))
        data.pop("points", None)
        return data


@dataclass
class NeighborListState:
    """State of a safely reusable periodic neighbor list."""

    pairs: np.ndarray | None = None
    reference_points: np.ndarray | None = None
    diameter_at_build: float = 0.0
    tree_builds: int = 0
    reuses: int = 0
    reuse_lengths: list[int] = field(default_factory=list)
    current_reuse_length: int = 0
    rebuild_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    pending_rebuild_reason: str = "initialization"

    def invalidate(self, reason: str = "invalidated") -> None:
        """Invalidate the current list while retaining accumulated statistics."""

        if self.current_reuse_length > 0:
            self.reuse_lengths.append(self.current_reuse_length)
        self.pairs = None
        self.reference_points = None
        self.diameter_at_build = 0.0
        self.current_reuse_length = 0
        self.pending_rebuild_reason = reason

    def finalize(self) -> None:
        """Close the current reuse segment for reporting."""

        if self.current_reuse_length > 0:
            self.reuse_lengths.append(self.current_reuse_length)
            self.current_reuse_length = 0


# ============================================================================
# Geometry and low-level kernels
# ============================================================================


def query_pairs(tree: cKDTree, radius: float) -> np.ndarray:
    """Return KDTree pairs as an ``(M, 2)`` int64 array.

    Parameters
    ----------
    tree : scipy.spatial.cKDTree
        Periodic KDTree.
    radius : float
        Pair-search radius.

    Returns
    -------
    numpy.ndarray
        Pair-index array with shape ``(M, 2)``.
    """

    try:
        pairs = tree.query_pairs(r=radius, output_type="ndarray")
        return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    except TypeError:
        raw_pairs = list(tree.query_pairs(r=radius))
        if raw_pairs:
            return np.asarray(raw_pairs, dtype=np.int64).reshape(-1, 2)
        return np.empty((0, 2), dtype=np.int64)


@njit(cache=True)
def _deterministic_unit_direction(
    particle_i: int,
    particle_j: int,
    seed: int,
    dim: int,
) -> np.ndarray:
    """Construct a reproducible nonzero direction for coincident particles."""

    direction = np.empty(dim, dtype=np.float64)
    norm_sq = 0.0
    index_low = particle_i if particle_i < particle_j else particle_j
    index_high = particle_j if particle_i < particle_j else particle_i
    orientation_sign = 1.0 if particle_i <= particle_j else -1.0
    for axis in range(dim):
        phase = (
            (index_low + 1) * 12.9898
            + (index_high + 1) * 78.233
            + (seed + 1) * 37.719
            + (axis + 1) * 19.913
        )
        value = math.sin(phase) + 0.5 * math.cos(0.731 * phase)
        direction[axis] = orientation_sign * value
        norm_sq += value * value

    if norm_sq <= 1.0e-30:
        for axis in range(dim):
            direction[axis] = 0.0
        direction[0] = 1.0
        return direction

    inv_norm = 1.0 / math.sqrt(norm_sq)
    for axis in range(dim):
        direction[axis] *= inv_norm
    return direction


@njit(cache=True)
def compute_gradients_and_metrics(
    points: np.ndarray,
    pairs: np.ndarray,
    box_size: np.ndarray,
    collision_diameter: float,
    gradients: np.ndarray,
    norm_eps: float,
    seed: int,
) -> tuple[float, float, int]:
    """Compute overlap potential, maximum overlap, and analytical gradient.

    Parameters
    ----------
    points : numpy.ndarray
        Primary particle centers with shape ``(N, dim)``.
    pairs : numpy.ndarray
        Candidate pair indices.
    box_size : numpy.ndarray
        Periodic box lengths.
    collision_diameter : float
        Current collision diameter.
    gradients : numpy.ndarray
        Preallocated gradient array, overwritten in place.
    norm_eps : float
        Coincidence threshold.
    seed : int
        Seed used for deterministic coincidence separation.

    Returns
    -------
    potential : float
        Total quadratic overlap potential.
    max_overlap : float
        Maximum pair overlap.
    overlap_pair_count : int
        Number of truly overlapping pairs.
    """

    gradients[:, :] = 0.0
    potential = 0.0
    max_overlap = 0.0
    overlap_pair_count = 0
    dim = points.shape[1]

    for pair_index in range(pairs.shape[0]):
        particle_i = pairs[pair_index, 0]
        particle_j = pairs[pair_index, 1]

        delta = np.empty(dim, dtype=np.float64)
        norm_sq = 0.0
        for axis in range(dim):
            value = points[particle_i, axis] - points[particle_j, axis]
            value -= box_size[axis] * round(value / box_size[axis])
            delta[axis] = value
            norm_sq += value * value

        distance = math.sqrt(norm_sq)
        overlap = collision_diameter - distance

        if overlap > 0.0:
            overlap_pair_count += 1
            potential += 0.5 * overlap * overlap
            if overlap > max_overlap:
                max_overlap = overlap

            if distance > norm_eps:
                inv_distance = 1.0 / distance
                for axis in range(dim):
                    grad_component = -overlap * delta[axis] * inv_distance
                    gradients[particle_i, axis] += grad_component
                    gradients[particle_j, axis] -= grad_component
            else:
                direction = _deterministic_unit_direction(
                    particle_i,
                    particle_j,
                    seed,
                    dim,
                )
                for axis in range(dim):
                    grad_component = -overlap * direction[axis]
                    gradients[particle_i, axis] += grad_component
                    gradients[particle_j, axis] -= grad_component

    return potential, max_overlap, overlap_pair_count


@njit(cache=True)
def maximum_periodic_displacement(
    points: np.ndarray,
    reference_points: np.ndarray,
    box_size: np.ndarray,
) -> float:
    """Return the maximum minimum-image displacement from reference positions."""

    max_displacement_sq = 0.0
    for particle in range(points.shape[0]):
        displacement_sq = 0.0
        for axis in range(points.shape[1]):
            value = points[particle, axis] - reference_points[particle, axis]
            value -= box_size[axis] * round(value / box_size[axis])
            displacement_sq += value * value
        if displacement_sq > max_displacement_sq:
            max_displacement_sq = displacement_sq
    return math.sqrt(max_displacement_sq)


def warm_up_numba(dim: int) -> None:
    """Compile Numba kernels before formal timing.

    Parameters
    ----------
    dim : int
        Spatial dimension, either 2 or 3.
    """

    dummy_points = np.zeros((2, dim), dtype=np.float64)
    dummy_pairs = np.array([[0, 1]], dtype=np.int64)
    dummy_box = np.ones(dim, dtype=np.float64)
    dummy_gradients = np.zeros_like(dummy_points)
    compute_gradients_and_metrics(
        dummy_points,
        dummy_pairs,
        dummy_box,
        0.5,
        dummy_gradients,
        1.0e-12,
        0,
    )
    maximum_periodic_displacement(dummy_points, dummy_points, dummy_box)


def particle_measure(diameter: float, dim: int) -> float:
    """Return circle area or sphere volume for a given diameter.

    Parameters
    ----------
    diameter : float
        Particle diameter.
    dim : int
        Spatial dimension, 2 or 3.

    Returns
    -------
    float
        Circle area for ``dim=2`` or sphere volume for ``dim=3``.
    """

    if dim == 2:
        return math.pi * diameter**2 / 4.0
    if dim == 3:
        return math.pi * diameter**3 / 6.0
    raise ValueError("Only dim=2 and dim=3 are supported.")


def compute_particle_number(
    box_size: np.ndarray,
    diameter_physical: float,
    v1_nominal: float,
) -> int:
    """Compute particle count using the original ceiling convention."""

    dim = int(box_size.size)
    return int(
        math.ceil(
            float(np.prod(box_size))
            * v1_nominal
            / particle_measure(diameter_physical, dim)
        )
    )


def compute_actual_fractions(
    particle_number: int,
    box_size: np.ndarray,
    diameter_physical: float,
    diameter_target: float,
) -> tuple[float, float]:
    """Return actual physical and target collision volume fractions."""

    dim = int(box_size.size)
    box_measure = float(np.prod(box_size))
    v1_actual = (
        particle_number * particle_measure(diameter_physical, dim) / box_measure
    )
    collision_vf = (
        particle_number * particle_measure(diameter_target, dim) / box_measure
    )
    return v1_actual, collision_vf


def generate_initial_points(
    particle_number: int,
    box_size: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Generate benchmark points with NumPy's modern random generator."""

    rng = np.random.default_rng(seed)
    return rng.random((particle_number, box_size.size)) * box_size


def generate_initial_points_legacy(
    particle_number: int,
    box_size: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Reproduce ``np.random.seed`` plus ``np.random.rand`` initialization.

    Notes
    -----
    Use this function only when reproducing the supplied historical script.
    Formal original-versus-fast benchmarks should generate one initial array
    with :func:`generate_initial_points` and pass copies to both solvers.
    """

    rng = np.random.RandomState(seed)
    return rng.rand(particle_number, box_size.size) * box_size


def _limited_update(
    points: np.ndarray,
    gradients: np.ndarray,
    box_size: np.ndarray,
    alpha: float,
    max_move: float,
) -> np.ndarray:
    """Apply a capped gradient update and periodic wrapping."""

    displacement = -alpha * gradients
    displacement_norms = np.linalg.norm(displacement, axis=1)
    largest_move = float(np.max(displacement_norms)) if displacement_norms.size else 0.0
    if largest_move > max_move > 0.0:
        displacement *= max_move / largest_move
    return (points + displacement) % box_size


# ============================================================================
# Neighbor-list management
# ============================================================================


def periodic_displacement_statistics(
    points: np.ndarray,
    reference_points: np.ndarray,
    box_size: np.ndarray,
) -> dict[str, float | None]:
    """Return rebuild diagnostics for per-particle periodic displacements."""

    delta = points - reference_points
    delta -= box_size * np.rint(delta / box_size)
    norms = np.linalg.norm(delta, axis=1)
    u_max = float(np.max(norms)) if norms.size else 0.0
    u_95 = float(np.percentile(norms, 95.0)) if norms.size else 0.0
    u_99 = float(np.percentile(norms, 99.0)) if norms.size else 0.0
    return {
        "u_max": u_max,
        "u_95": u_95,
        "u_99": u_99,
        "u_max_over_u_95": u_max / u_95 if u_95 > 0.0 else None,
        "u_max_over_u_99": u_max / u_99 if u_99 > 0.0 else None,
    }


def ensure_safe_neighbor_list(
    points: np.ndarray,
    box_size: np.ndarray,
    current_diameter: float,
    skin: float,
    safety_factor: float,
    state: NeighborListState,
) -> tuple[np.ndarray, bool, float]:
    """Return a complete candidate list, rebuilding it when required.

    The list built at diameter ``D_b`` with cutoff ``D_b + s`` is reused only
    while

    ``2*u_max + (D - D_b) < safety_factor*s``.

    Parameters
    ----------
    points : numpy.ndarray
        Current primary particle centers.
    box_size : numpy.ndarray
        Periodic box dimensions.
    current_diameter : float
        Current continuation collision diameter.
    skin : float
        Neighbor-list skin distance.
    safety_factor : float
        Conservative fraction of the skin budget.
    state : NeighborListState
        Mutable list state.

    Returns
    -------
    pairs : numpy.ndarray
        Complete candidate pair list.
    rebuilt : bool
        True if a new KDTree was constructed.
    safety_measure : float
        Value of ``2*u_max + diameter_growth`` before rebuilding.
    """

    must_rebuild = state.pairs is None or state.reference_points is None
    safety_measure = math.inf
    displacement_stats: dict[str, float | None] | None = None
    displacement_contribution: float | None = None
    diameter_growth: float | None = None
    rebuild_reason = state.pending_rebuild_reason
    previous_build_diameter = (
        float(state.diameter_at_build) if state.reference_points is not None else None
    )

    if not must_rebuild:
        displacement_stats = periodic_displacement_statistics(
            points, state.reference_points, box_size
        )
        max_displacement = float(displacement_stats["u_max"])
        diameter_growth = max(0.0, current_diameter - state.diameter_at_build)
        displacement_contribution = 2.0 * max_displacement
        safety_measure = displacement_contribution + diameter_growth
        threshold = safety_factor * skin
        must_rebuild = safety_measure >= threshold
        if must_rebuild:
            displacement_trigger = displacement_contribution >= threshold
            growth_trigger = diameter_growth >= threshold
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
        tree = cKDTree(points, boxsize=box_size)
        pairs = query_pairs(tree, current_diameter + skin)
        state.pairs = pairs
        state.reference_points = points.copy()
        state.diameter_at_build = current_diameter
        state.tree_builds += 1
        state.current_reuse_length = 0
        diagnostic: dict[str, Any] = {
            "tree_build_index": state.tree_builds,
            "reason": rebuild_reason,
            "diameter_current": float(current_diameter),
            "diameter_at_previous_build": previous_build_diameter,
            "skin": float(skin),
            "safety_threshold": float(safety_factor * skin),
            "displacement_contribution": displacement_contribution,
            "diameter_growth_contribution": diameter_growth,
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


# ============================================================================
# Exact validation and cleanup
# ============================================================================


def validate_configuration(
    points: np.ndarray,
    box_size: np.ndarray,
    collision_diameter: float,
    norm_eps: float,
    seed: int,
    overlap_tolerance: float = 0.0,
) -> ValidationResult:
    """Independently validate a periodic particle configuration.

    Parameters
    ----------
    points : numpy.ndarray
        Primary particle centers.
    box_size : numpy.ndarray
        Periodic box lengths.
    collision_diameter : float
        Collision diameter used for validation.
    norm_eps : float
        Coincidence threshold.
    seed : int
        Coincidence-direction seed.
    overlap_tolerance : float, optional
        Overlaps not exceeding this value are accepted as numerical contact.

    Returns
    -------
    ValidationResult
        Exact overlap and coordinate diagnostics.
    """

    nan_count = int(np.isnan(points).sum())
    inf_count = int(np.isinf(points).sum())
    inside_box = bool(
        nan_count == 0
        and inf_count == 0
        and np.all(points >= 0.0)
        and np.all(points < box_size)
    )
    if nan_count > 0 or inf_count > 0:
        return ValidationResult(
            candidate_pair_count=0,
            potential=math.inf,
            max_overlap=math.inf,
            relative_max_overlap=math.inf,
            overlap_pair_count=-1,
            violating_overlap_pair_count=-1,
            minimum_center_distance=math.nan,
            nan_count=nan_count,
            inf_count=inf_count,
            coordinates_inside_box=False,
        )

    gradients = np.zeros_like(points)
    tree = cKDTree(points, boxsize=box_size)
    pairs = query_pairs(tree, collision_diameter)
    potential, max_overlap, overlap_count = compute_gradients_and_metrics(
        points,
        pairs,
        box_size,
        collision_diameter,
        gradients,
        norm_eps,
        seed,
    )

    if points.shape[0] < 2:
        minimum_distance = math.inf
    else:
        nearest_distances, _ = tree.query(points, k=2)
        minimum_distance = float(np.min(nearest_distances[:, 1]))

    if pairs.shape[0] == 0:
        violating_overlap_count = 0
    else:
        pair_delta = points[pairs[:, 0]] - points[pairs[:, 1]]
        pair_delta -= box_size * np.rint(pair_delta / box_size)
        pair_distances = np.linalg.norm(pair_delta, axis=1)
        pair_overlaps = collision_diameter - pair_distances
        violating_overlap_count = int(
            np.count_nonzero(pair_overlaps > overlap_tolerance)
        )

    return ValidationResult(
        candidate_pair_count=int(pairs.shape[0]),
        potential=float(potential),
        max_overlap=float(max_overlap),
        relative_max_overlap=float(max_overlap / collision_diameter),
        overlap_pair_count=int(overlap_count),
        violating_overlap_pair_count=violating_overlap_count,
        minimum_center_distance=minimum_distance,
        nan_count=nan_count,
        inf_count=inf_count,
        coordinates_inside_box=inside_box,
    )


def strict_cleanup(
    points: np.ndarray,
    box_size: np.ndarray,
    diameter_target: float,
    seed: int,
    config: FastSolverConfig,
) -> tuple[np.ndarray, bool, str, dict[str, Any]]:
    """Remove residual overlaps using an exact KDTree at every iteration."""

    points = points.copy()
    gradients = np.zeros_like(points)
    strict_tolerance = config.strict_overlap_ratio * diameter_target
    pair_evaluations = 0
    tree_builds = 0
    last_potential = math.inf
    last_max_overlap = math.inf
    last_overlap_count = -1

    for iteration in range(config.max_iterations_cleanup + 1):
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
        last_potential = float(potential)
        last_max_overlap = float(max_overlap)
        last_overlap_count = int(overlap_count)

        if config.verbose and (
            iteration % config.print_interval == 0
            or max_overlap <= strict_tolerance
        ):
            print(
                "Cleanup "
                f"iteration {iteration}: potential={potential:.6e}, "
                f"max_overlap={max_overlap:.6e}, pairs={overlap_count}"
            )

        if max_overlap <= strict_tolerance:
            return points, True, "converged", {
                "iterations": iteration,
                "gradient_evaluations": iteration + 1,
                "tree_builds": tree_builds,
                "pair_evaluations": pair_evaluations,
                "potential": last_potential,
                "max_overlap": last_max_overlap,
                "overlap_count": last_overlap_count,
            }

        if iteration == config.max_iterations_cleanup:
            break

        max_move = config.cleanup_max_move_ratio * diameter_target
        points = _limited_update(
            points,
            gradients,
            box_size,
            config.alpha,
            max_move,
        )

        if not np.all(np.isfinite(points)):
            return points, False, "invalid_coordinates", {
                "iterations": iteration + 1,
                "gradient_evaluations": iteration + 1,
                "tree_builds": tree_builds,
                "pair_evaluations": pair_evaluations,
                "potential": last_potential,
                "max_overlap": last_max_overlap,
                "overlap_count": last_overlap_count,
            }

    return points, False, "cleanup_failed", {
        "iterations": config.max_iterations_cleanup,
        "gradient_evaluations": config.max_iterations_cleanup + 1,
        "tree_builds": tree_builds,
        "pair_evaluations": pair_evaluations,
        "potential": last_potential,
        "max_overlap": last_max_overlap,
        "overlap_count": last_overlap_count,
    }


# ============================================================================
# Continuation stages
# ============================================================================


def _stage_has_stagnated(
    potential_history: list[float],
    overlap_history: list[float],
    window: int,
    relative_drop_required: float,
) -> bool:
    """Return True when both potential and maximum overlap stagnate."""

    if window <= 0 or len(potential_history) <= window:
        return False

    old_potential = potential_history[-window - 1]
    new_potential = potential_history[-1]
    old_overlap = overlap_history[-window - 1]
    new_overlap = overlap_history[-1]

    potential_drop = (old_potential - new_potential) / max(
        abs(old_potential), 1.0e-30
    )
    overlap_drop = (old_overlap - new_overlap) / max(abs(old_overlap), 1.0e-30)

    return (
        potential_drop < relative_drop_required
        and overlap_drop < relative_drop_required
    )


def relax_continuation_stage(
    points: np.ndarray,
    box_size: np.ndarray,
    current_diameter: float,
    diameter_target: float,
    stage_tolerance: float,
    seed: int,
    config: FastSolverConfig,
    neighbor_state: NeighborListState,
    max_iterations: int | None = None,
) -> tuple[np.ndarray, bool, str, dict[str, Any]]:
    """Relax one continuation stage using safely reused neighbor lists."""

    points = points.copy()
    gradients = np.zeros_like(points)
    skin = config.skin_ratio * diameter_target
    potential_history: list[float] = []
    overlap_history: list[float] = []
    pair_evaluations = 0
    starting_tree_builds = neighbor_state.tree_builds
    starting_reuses = neighbor_state.reuses
    last_potential = math.inf
    last_max_overlap = math.inf
    last_overlap_count = -1
    iteration_limit = (
        config.max_iterations_stage if max_iterations is None else max_iterations
    )

    for iteration in range(iteration_limit + 1):
        pairs, rebuilt, safety_measure = ensure_safe_neighbor_list(
            points,
            box_size,
            current_diameter,
            skin,
            config.neighbor_safety_factor,
            neighbor_state,
        )
        pair_evaluations += int(pairs.shape[0])

        potential, max_overlap, overlap_count = compute_gradients_and_metrics(
            points,
            pairs,
            box_size,
            current_diameter,
            gradients,
            config.norm_eps,
            seed,
        )
        last_potential = float(potential)
        last_max_overlap = float(max_overlap)
        last_overlap_count = int(overlap_count)
        potential_history.append(last_potential)
        overlap_history.append(last_max_overlap)

        if config.verbose and (
            iteration % config.print_interval == 0
            or max_overlap <= stage_tolerance
        ):
            print(
                "  Stage "
                f"iteration {iteration}: D={current_diameter:.8g}, "
                f"potential={potential:.6e}, max_overlap={max_overlap:.6e}, "
                f"pairs={overlap_count}, rebuilt={rebuilt}"
            )

        if max_overlap <= stage_tolerance:
            return points, True, "converged", {
                "iterations": iteration,
                "gradient_evaluations": iteration + 1,
                "tree_builds": neighbor_state.tree_builds - starting_tree_builds,
                "neighbor_reuses": neighbor_state.reuses - starting_reuses,
                "pair_evaluations": pair_evaluations,
                "potential": last_potential,
                "max_overlap": last_max_overlap,
                "overlap_count": last_overlap_count,
                "safety_measure": safety_measure,
            }

        if iteration == iteration_limit:
            break

        if _stage_has_stagnated(
            potential_history,
            overlap_history,
            config.stagnation_window,
            config.stagnation_relative_drop,
        ):
            return points, False, "stagnated", {
                "iterations": iteration,
                "gradient_evaluations": iteration + 1,
                "tree_builds": neighbor_state.tree_builds - starting_tree_builds,
                "neighbor_reuses": neighbor_state.reuses - starting_reuses,
                "pair_evaluations": pair_evaluations,
                "potential": last_potential,
                "max_overlap": last_max_overlap,
                "overlap_count": last_overlap_count,
                "safety_measure": safety_measure,
            }

        max_move = min(
            config.max_move_ratio * current_diameter,
            config.max_move_skin_ratio * skin,
        )
        points = _limited_update(
            points,
            gradients,
            box_size,
            config.alpha,
            max_move,
        )

        if not np.all(np.isfinite(points)):
            return points, False, "invalid_coordinates", {
                "iterations": iteration + 1,
                "gradient_evaluations": iteration + 1,
                "tree_builds": neighbor_state.tree_builds - starting_tree_builds,
                "neighbor_reuses": neighbor_state.reuses - starting_reuses,
                "pair_evaluations": pair_evaluations,
                "potential": last_potential,
                "max_overlap": last_max_overlap,
                "overlap_count": last_overlap_count,
                "safety_measure": safety_measure,
            }

    return points, False, "max_iterations", {
        "iterations": iteration_limit,
        "gradient_evaluations": iteration_limit + 1,
        "tree_builds": neighbor_state.tree_builds - starting_tree_builds,
        "neighbor_reuses": neighbor_state.reuses - starting_reuses,
        "pair_evaluations": pair_evaluations,
        "potential": last_potential,
        "max_overlap": last_max_overlap,
        "overlap_count": last_overlap_count,
        "safety_measure": math.nan,
    }


def rescue_stage_exact(
    points: np.ndarray,
    box_size: np.ndarray,
    current_diameter: float,
    stage_tolerance: float,
    seed: int,
    config: FastSolverConfig,
) -> tuple[np.ndarray, bool, str, dict[str, Any]]:
    """Rescue a failed stage using an exact KDTree and a reduced step size."""

    points = points.copy()
    gradients = np.zeros_like(points)
    pair_evaluations = 0
    last_potential = math.inf
    last_max_overlap = math.inf
    last_overlap_count = -1
    rescue_alpha = 0.5 * config.alpha

    for iteration in range(config.max_iterations_stage_rescue + 1):
        tree = cKDTree(points, boxsize=box_size)
        pairs = query_pairs(tree, current_diameter)
        pair_evaluations += int(pairs.shape[0])
        potential, max_overlap, overlap_count = compute_gradients_and_metrics(
            points,
            pairs,
            box_size,
            current_diameter,
            gradients,
            config.norm_eps,
            seed,
        )
        last_potential = float(potential)
        last_max_overlap = float(max_overlap)
        last_overlap_count = int(overlap_count)

        if max_overlap <= stage_tolerance:
            return points, True, "rescue_converged", {
                "iterations": iteration,
                "gradient_evaluations": iteration + 1,
                "tree_builds": iteration + 1,
                "neighbor_reuses": 0,
                "pair_evaluations": pair_evaluations,
                "potential": last_potential,
                "max_overlap": last_max_overlap,
                "overlap_count": last_overlap_count,
            }

        if iteration == config.max_iterations_stage_rescue:
            break

        max_move = config.max_move_ratio * current_diameter
        points = _limited_update(
            points,
            gradients,
            box_size,
            rescue_alpha,
            max_move,
        )

        if not np.all(np.isfinite(points)):
            return points, False, "invalid_coordinates", {
                "iterations": iteration + 1,
                "gradient_evaluations": iteration + 1,
                "tree_builds": iteration + 1,
                "neighbor_reuses": 0,
                "pair_evaluations": pair_evaluations,
                "potential": last_potential,
                "max_overlap": last_max_overlap,
                "overlap_count": last_overlap_count,
            }

    return points, False, "rescue_failed", {
        "iterations": config.max_iterations_stage_rescue,
        "gradient_evaluations": config.max_iterations_stage_rescue + 1,
        "tree_builds": config.max_iterations_stage_rescue + 1,
        "neighbor_reuses": 0,
        "pair_evaluations": pair_evaluations,
        "potential": last_potential,
        "max_overlap": last_max_overlap,
        "overlap_count": last_overlap_count,
    }


# ============================================================================
# Input validation
# ============================================================================


def theoretical_packing_limit(dim: int) -> float:
    """Return the densest-packing upper bound for circles or spheres."""

    if dim == 2:
        return math.pi / (2.0 * math.sqrt(3.0))
    if dim == 3:
        return math.pi / (3.0 * math.sqrt(2.0))
    raise ValueError("Only dim=2 and dim=3 are supported.")


def validate_solver_inputs(
    initial_points: np.ndarray,
    box_size: np.ndarray,
    diameter_physical: float,
    enlargement_factor: float,
    skin_ratio: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Validate and normalize common solver inputs.

    Returns
    -------
    points, box_size, diameter_target
        Validated float64 arrays and target collision diameter.
    """

    points = np.asarray(initial_points, dtype=np.float64).copy()
    box = np.asarray(box_size, dtype=np.float64).copy()
    if points.ndim != 2 or points.shape[1] not in (2, 3):
        raise ValueError("initial_points must have shape (N, 2) or (N, 3).")
    if points.shape[0] < 1:
        raise ValueError("At least one particle center is required.")
    if box.shape != (points.shape[1],):
        raise ValueError("box_size must contain one length per spatial dimension.")
    if not np.all(np.isfinite(points)):
        raise ValueError("initial_points contains NaN or Inf values.")
    if not np.all(np.isfinite(box)) or np.any(box <= 0.0):
        raise ValueError("All box lengths must be finite and positive.")
    if not math.isfinite(diameter_physical) or diameter_physical <= 0.0:
        raise ValueError("diameter_physical must be finite and positive.")
    if not math.isfinite(enlargement_factor) or enlargement_factor <= 0.0:
        raise ValueError("enlargement_factor must be finite and positive.")

    diameter_target = diameter_physical * enlargement_factor
    cutoff = diameter_target
    if skin_ratio is not None:
        if not math.isfinite(skin_ratio) or skin_ratio <= 0.0:
            raise ValueError("skin_ratio must be finite and positive.")
        cutoff += skin_ratio * diameter_target

    if cutoff >= 0.5 * float(np.min(box)):
        raise ValueError(
            "The target cutoff must be smaller than half the minimum box length "
            "for a unique minimum-image neighbor representation."
        )

    points %= box
    return points, box, diameter_target


# ============================================================================
# Main fast solver
# ============================================================================


def solve_fast(
    initial_points: np.ndarray,
    box_size: np.ndarray,
    diameter_physical: float,
    enlargement_factor: float,
    seed: int = 0,
    config: FastSolverConfig | None = None,
) -> SolverResult:
    """Run the fast periodic overlap-removal algorithm.

    Parameters
    ----------
    initial_points : numpy.ndarray
        Initial primary particle centers with shape ``(N, 2)`` or ``(N, 3)``.
    box_size : numpy.ndarray
        Periodic box lengths.
    diameter_physical : float
        Physical particle diameter used to define the physical volume fraction.
    enlargement_factor : float
        Ratio of target collision diameter to physical diameter.
    seed : int, optional
        Reproducibility seed used for coincidence handling.
    config : FastSolverConfig, optional
        Solver settings. Defaults are used when omitted.

    Returns
    -------
    SolverResult
        Final coordinates and complete solver diagnostics.

    Raises
    ------
    ValueError
        If dimensions, coordinates, or input parameters are invalid.
    """

    if config is None:
        config = FastSolverConfig()

    points, box_size, diameter_target = validate_solver_inputs(
        initial_points,
        box_size,
        diameter_physical,
        enlargement_factor,
        skin_ratio=config.skin_ratio,
    )
    if not (0.0 < config.neighbor_safety_factor <= 1.0):
        raise ValueError("neighbor_safety_factor must lie in (0, 1].")
    if config.alpha <= 0.0 or not math.isfinite(config.alpha):
        raise ValueError("alpha must be finite and positive.")
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

    phi_easy = config.phi_easy_2d if dim == 2 else config.phi_easy_3d
    phi_continuation = (
        config.phi_continuation_2d if dim == 2 else config.phi_continuation_3d
    )
    use_continuation = collision_vf >= phi_continuation

    if use_continuation and collision_vf > 0.0:
        lambda_initial = min(1.0, (phi_easy / collision_vf) ** (1.0 / dim))
        lambda_initial = max(config.lambda_min, lambda_initial)
    else:
        # Moderate-density fast mode: reuse a safe list directly at D_target.
        lambda_initial = 1.0

    warm_up_numba(dim)
    start_total = time.perf_counter()

    continuation_start = time.perf_counter()
    neighbor_state = NeighborListState()
    continuation_stages: list[dict[str, Any]] = []
    continuation_iterations = 0
    continuation_gradient_evaluations = 0
    continuation_pair_evaluations = 0
    rescue_tree_builds = 0
    continuation_rollbacks = 0
    delta_lambda = config.delta_lambda_initial
    continuation_success = True
    continuation_status = "converged"

    accepted_points = points.copy()
    accepted_lambda: float | None = None
    target_lambda = lambda_initial

    # Intermediate continuation stages are used only in the high-density regime.
    while use_continuation and target_lambda < 1.0 - 1.0e-15:
        diameter_current = target_lambda * diameter_target
        if accepted_lambda is None:
            growth_for_tolerance = delta_lambda
        else:
            growth_for_tolerance = target_lambda - accepted_lambda
        stage_tolerance = max(
            config.stage_overlap_ratio * diameter_target,
            config.stage_growth_fraction
            * growth_for_tolerance
            * diameter_target,
        )

        if config.verbose:
            print(
                f"Continuation stage: lambda={target_lambda:.6f}, "
                f"D={diameter_current:.8g}, tolerance={stage_tolerance:.6e}"
            )

        stage_start_points = accepted_points.copy()
        points_stage, stage_success, stage_status, stage_stats = (
            relax_continuation_stage(
                stage_start_points,
                box_size,
                diameter_current,
                diameter_target,
                stage_tolerance,
                seed,
                config,
                neighbor_state,
            )
        )

        continuation_iterations += int(stage_stats["iterations"])
        continuation_gradient_evaluations += int(stage_stats["gradient_evaluations"])
        continuation_pair_evaluations += int(stage_stats["pair_evaluations"])

        neighbor_stage_stats = dict(stage_stats)
        rescue_stats: dict[str, Any] | None = None
        used_rescue = False
        if not stage_success:
            if config.verbose:
                print(
                    f"  Neighbor-list stage failed ({stage_status}); "
                    "trying exact rescue."
                )
            rescue_points, rescue_success, rescue_status, rescue_stats = (
                rescue_stage_exact(
                    points_stage,
                    box_size,
                    diameter_current,
                    stage_tolerance,
                    seed,
                    config,
                )
            )
            used_rescue = True
            continuation_iterations += int(rescue_stats["iterations"])
            continuation_gradient_evaluations += int(
                rescue_stats["gradient_evaluations"]
            )
            continuation_pair_evaluations += int(rescue_stats["pair_evaluations"])
            rescue_tree_builds += int(rescue_stats["tree_builds"])
            neighbor_state.invalidate("post_rescue")

            if rescue_success:
                points_stage = rescue_points
                stage_success = True
                stage_status = rescue_status
            else:
                stage_status = rescue_status

        total_stage_iterations = int(neighbor_stage_stats["iterations"]) + (
            int(rescue_stats["iterations"]) if rescue_stats is not None else 0
        )
        continuation_stages.append(
            {
                "stage_type": "continuation",
                "lambda": float(target_lambda),
                "diameter": float(diameter_current),
                "tolerance": float(stage_tolerance),
                "success": bool(stage_success),
                "status": stage_status,
                "used_rescue": used_rescue,
                "iterations": total_stage_iterations,
                "neighbor_stage": neighbor_stage_stats,
                "rescue_stage": rescue_stats,
            }
        )

        if not stage_success:
            if accepted_lambda is None:
                lower_lambda = max(config.lambda_min, target_lambda - delta_lambda)
                if lower_lambda < target_lambda - 1.0e-15:
                    neighbor_state.invalidate("rollback_lower_initial_scale")
                    continuation_rollbacks += 1
                    target_lambda = lower_lambda
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

            # Roll back to the last accepted diameter and retry a smaller growth.
            neighbor_state.invalidate("rollback_smaller_growth")
            continuation_rollbacks += 1
            target_lambda = min(1.0, accepted_lambda + delta_lambda)
            continue

        accepted_points = points_stage
        accepted_lambda = target_lambda

        stage_iterations = total_stage_iterations
        if stage_iterations < config.stage_fast_iterations:
            delta_lambda = min(config.delta_lambda_max, 1.4 * delta_lambda)
        elif stage_iterations > config.stage_slow_iterations:
            delta_lambda = max(config.delta_lambda_min, 0.5 * delta_lambda)

        target_lambda = min(1.0, accepted_lambda + delta_lambda)

    points = accepted_points

    # At D_target, use the safely reused list down to a near-strict tolerance.
    # This leaves only a short exact-KDTree cleanup tail.
    if continuation_success:
        precleanup_tolerance = config.precleanup_overlap_ratio * diameter_target
        if config.verbose:
            mode = "after continuation" if use_continuation else "direct target mode"
            print(
                f"Target-diameter precleanup ({mode}): "
                f"tolerance={precleanup_tolerance:.6e}"
            )

        pre_points, pre_success, pre_status, pre_stats = relax_continuation_stage(
            points,
            box_size,
            diameter_target,
            diameter_target,
            precleanup_tolerance,
            seed,
            config,
            neighbor_state,
            max_iterations=config.max_iterations_precleanup,
        )
        continuation_iterations += int(pre_stats["iterations"])
        continuation_gradient_evaluations += int(pre_stats["gradient_evaluations"])
        continuation_pair_evaluations += int(pre_stats["pair_evaluations"])
        continuation_stages.append(
            {
                "stage_type": "target_precleanup",
                "lambda": 1.0,
                "diameter": float(diameter_target),
                "tolerance": float(precleanup_tolerance),
                "success": bool(pre_success),
                "status": pre_status,
                "used_rescue": False,
                **{key: value for key, value in pre_stats.items()},
            }
        )

        # Even an incomplete precleanup may provide a useful starting point for
        # the exact cleanup. Reject it only when coordinates became invalid.
        if np.all(np.isfinite(pre_points)):
            points = pre_points
        else:
            continuation_success = False
            continuation_status = "invalid_coordinates"

    neighbor_state.finalize()
    runtime_continuation = time.perf_counter() - continuation_start

    # Strict exact cleanup at the target diameter.
    cleanup_start = time.perf_counter()
    if continuation_success:
        points, cleanup_success, cleanup_status, cleanup_stats = strict_cleanup(
            points,
            box_size,
            diameter_target,
            seed,
            config,
        )
    else:
        cleanup_success = False
        cleanup_status = continuation_status
        cleanup_stats = {
            "iterations": 0,
            "gradient_evaluations": 0,
            "tree_builds": 0,
            "pair_evaluations": 0,
            "potential": math.inf,
            "max_overlap": math.inf,
            "overlap_count": -1,
        }
    runtime_cleanup = time.perf_counter() - cleanup_start

    # Independent final validation.
    validation_start = time.perf_counter()
    validation = validate_configuration(
        points,
        box_size,
        diameter_target,
        config.norm_eps,
        seed,
        overlap_tolerance=config.strict_overlap_ratio * diameter_target,
    )
    runtime_validation = time.perf_counter() - validation_start

    strict_tolerance = config.strict_overlap_ratio * diameter_target
    validation_success = (
        validation.max_overlap <= strict_tolerance
        and validation.nan_count == 0
        and validation.inf_count == 0
        and validation.coordinates_inside_box
    )
    success = bool(continuation_success and cleanup_success and validation_success)

    if success:
        status = "converged"
    elif not continuation_success:
        status = continuation_status
    elif not cleanup_success:
        status = cleanup_status
    else:
        status = "validation_failed"

    continuation_tree_builds = neighbor_state.tree_builds + rescue_tree_builds
    cleanup_tree_builds = int(cleanup_stats["tree_builds"])
    total_tree_builds = continuation_tree_builds + cleanup_tree_builds + 1
    total_gradient_evaluations = (
        continuation_gradient_evaluations
        + int(cleanup_stats["gradient_evaluations"])
    )
    total_iterations = continuation_iterations + int(cleanup_stats["iterations"])
    total_pair_evaluations = (
        continuation_pair_evaluations
        + int(cleanup_stats["pair_evaluations"])
        + validation.candidate_pair_count
    )
    neighbor_list_calls = neighbor_state.tree_builds + neighbor_state.reuses
    reuse_ratio = (
        neighbor_state.reuses / neighbor_list_calls
        if neighbor_list_calls > 0
        else 0.0
    )

    runtime_total = time.perf_counter() - start_total

    # Remove coordinate snapshots from stage history before returning/reporting.
    for stage in continuation_stages:
        stage.pop("accepted_points", None)
    continuation_only_stages = [
        stage for stage in continuation_stages
        if stage.get("stage_type") == "continuation"
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

    return SolverResult(
        points=points,
        success=success,
        status=status,
        solver_name="fast_strict",
        solver_path=("fast_continuation" if use_continuation else "fast_direct"),
        runtime_total=runtime_total,
        runtime_continuation=runtime_continuation,
        runtime_cleanup=runtime_cleanup,
        runtime_validation=runtime_validation,
        iterations_total=total_iterations,
        iterations_continuation=continuation_iterations,
        iterations_cleanup=int(cleanup_stats["iterations"]),
        gradient_evaluations=total_gradient_evaluations,
        tree_builds_total=total_tree_builds,
        tree_builds_continuation=continuation_tree_builds,
        tree_builds_cleanup=cleanup_tree_builds,
        neighbor_list_reuses=neighbor_state.reuses,
        reuse_ratio=float(reuse_ratio),
        pair_evaluations=total_pair_evaluations,
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
        continuation_attempted_stages=int(attempted_stages),
        continuation_accepted_stages=int(accepted_stages),
        continuation_rescue_calls=int(rescue_calls),
        continuation_successful_rescues=int(successful_rescues),
        continuation_rollbacks=int(continuation_rollbacks),
        continuation_stages=continuation_stages,
        rebuild_diagnostics=neighbor_state.rebuild_diagnostics,
    )


# ============================================================================
# Output helpers
# ============================================================================


def generate_periodic_images(
    primary_points: np.ndarray,
    box_size: np.ndarray,
    diameter_physical: float,
) -> np.ndarray:
    """Generate only periodic images that intersect the physical RVE."""

    dim = primary_points.shape[1]
    shift_grid = np.array(
        np.meshgrid(*[[-1, 0, 1]] * dim, indexing="ij")
    ).reshape(dim, -1).T
    shifts = shift_grid * box_size
    images = np.vstack([primary_points + shift for shift in shifts])

    lower = -0.5 * diameter_physical
    upper = box_size + 0.5 * diameter_physical
    mask = np.all((images >= lower) & (images <= upper), axis=1)
    return images[mask]


def save_solver_outputs(
    result: SolverResult,
    box_size: np.ndarray,
    output_directory: str | Path,
) -> None:
    """Save primary points, periodic images, and a JSON summary."""

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

    with (output_directory / f"fast_result_{dim}d.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(result.summary_dict(), file, indent=2, ensure_ascii=False)


# ============================================================================
# Direct execution: edit only this parameter block for a single case
# ============================================================================


if __name__ == "__main__":
    # --------------------------- Parameters ---------------------------
    dim = 3
    rve_size = np.ones(dim, dtype=np.float64) * 100.0
    diameter_physical = 20.0
    v1_nominal = 0.55
    enlargement_factor = 1.025
    seed = 0

    config = FastSolverConfig(
        alpha=0.4,
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
    result = solve_fast(
        initial_points=initial_points,
        box_size=rve_size,
        diameter_physical=diameter_physical,
        enlargement_factor=enlargement_factor,
        seed=seed,
        config=config,
    )

    # ---------------------------- Output ------------------------------
    save_solver_outputs(result, rve_size, output_directory="fast_output")

    print("\n================ Fast solver summary ================")
    for key, value in result.summary_dict().items():
        if key != "continuation_stages":
            print(f"{key}: {value}")
