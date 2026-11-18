from __future__ import annotations

from pathlib import Path
from typing import Sequence

# ============================= User Config =============================
# Batch selection. Use [1] for one case, or e.g. [1, 2, 3] for multiple cases.
SELECTED_INDEXES: list[int | str] = [1, 2, 3, 4, 5]
L_D = 8
NUM_ELE = 60
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / f"hybrid_output_ld{L_D}"
OUTPUT_DIR = PROJECT_ROOT / "digimatFE_analysis"

# Material1 (isotropic matrix): adjustable
M1_YOUNG = 0.074
M1_POISSON = 0.33

# Material2 (isotropic particle): adjustable
M2_YOUNG = 0.41
M2_POISSON = 0.19

# Phase2 / mesh / RVE: adjustable
PHASE2_VOLUME_FRACTION = 0.50
PARTICLE_DIAMETER = 20.0
MINIMUM_ELEMENT_SIZE = 0.25
RVE_SIZE = PARTICLE_DIAMETER * L_D
ELEMENT_SIZE = 100 / NUM_ELE

# Other constants (usually unchanged)
MATERIAL1_DENSITY = 1.0
MATERIAL2_DENSITY = 1.0
PHASE1_VOLUME_FRACTION = 1.0 - PHASE2_VOLUME_FRACTION

MATERIAL1_PARAMS: list[tuple[str, float | str]] = [
    ("name", "Material1"),
    ("type", "elastic"),
    ("density", MATERIAL1_DENSITY),
    ("elastic_model", "isotropic"),
    ("Young", M1_YOUNG),
    ("Poisson", M1_POISSON),
]

MATERIAL2_PARAMS: list[tuple[str, float | str]] = [
    ("name", "Material2"),
    ("type", "elastic"),
    ("density", MATERIAL2_DENSITY),
    ("elastic_model", "isotropic"),
    ("Young", M2_YOUNG),
    ("Poisson", M2_POISSON),
]

# ------------------------------- Phases -------------------------------
PHASE1_PARAMS: list[tuple[str, float | str]] = [
    ("name", "Phase1"),
    ("type", "matrix"),
    ("volume_fraction", PHASE1_VOLUME_FRACTION),
    ("material", "Material1"),
]

PHASE2_FIXED_PARAMS: list[tuple[str, float | str]] = [
    ("name", "Phase2"),
    ("type", "inclusion_fe"),
    ("volume_fraction", PHASE2_VOLUME_FRACTION),
    ("material", "Material2"),
    ("inclusion_shape", "ellipsoid"),
    ("aspect_ratio", 1.000000000000000e+00),
    ("phase_definition", "by_size_and_diameter"),
    ("inclusion_diameter", PARTICLE_DIAMETER),
    ("inclusion_size", PARTICLE_DIAMETER),
    ("size_distribution", "fixed"),
    ("orientation", "fixed"),
    ("theta_angle", 9.000000000000000e+01),
    ("phi_angle", 0.000000000000000e+00),
    ("coated", "no"),
    ("interface_behavior", "perfectly_bonded"),
    ("clustering", "no"),
    ("allow_size_reduction", "no"),
    ("track_percolation_onset", "no"),
    ("stop_at_percolation", "no"),
    ("check_final_percolation", "no"),
    ("no_tie_on_fiber_tips","no"),
    ("custom_position_usage", "sequential"),
    ("custom_position_ignore_phase_fraction", "on"),
    ("custom_position_disable_geom_check", "on"),
]

MICROSTRUCTURE_PARAMS: list[tuple[str, float | str]] = [
    ("name", "Microstructure1"),
    ("phase", "Phase1"),
    ("phase", "Phase2"),
]

MECHANICAL_LOADING_PARAMS: list[tuple[str, float | str]] = [
    ("name", "Mechanical"),
    ("type", "strain"),
    ("boundary_conditions", "periodic"),
    ("custom_bc", "no"),
    ("load", "uniaxial_1"),
    ("initial_strain", 0.0),
    ("peak_strain", 3.0e-2),
    ("history", "monotonic"),
    ("quasi_static", "on"),
    ("theta_load", 90.0),
    ("phi_load", 0.0),
    ("required_components", "E1_E2_E3_G12_G23_G13_"),
]

RVE_PARAMS: list[tuple[str, float | str]] = [
    ("type", "classical"),
    ("microstructure", "Microstructure1"),
]

MESH_PARAMS: list[tuple[str, float | str]] = [
    ("mesh_type", "conforming"),
    ("automatic_mesh_sizing", "off"),
    ("element_size", ELEMENT_SIZE),
    ("minimum_element_size", MINIMUM_ELEMENT_SIZE),
    ("use_quadratic_elements", "off"),
    ("use_quadratic_geometric_elements", "off"),
    ("element_shape", "quad_dominated"),
    ("internal_coarsening", "on"),
    ("curvature_control", "on"),
    ("chordal_deviation_ratio", 1.0e-1),
    ("nb_refinement_steps", 5),
    ("model_layer_interfaces", "off"),
    ("seed_size", 5.0),
    ("share_nodes", "off"),
    ("periodic_mesh", "off"),
    ("cohesive_element_size_ratio", 2.0e-1),
]

MECHANICAL_ANALYSISFE_PARAMS: list[tuple[str, float | str]] = [
    ("name", "Analysis1"),
    ("type", "mechanical"),
    ("loading_name", "Mechanical"),
    ("final_time", 1.0),
    ("max_time_inc", 1.0),
    ("min_time_inc", 1.0e-1),
    ("finite_strain", "off"),
    ("initial_time_inc", 1.0),
    ("max_number_increment", 2),
    ("rve_size_definition", "user_defined"),
    ("rve_dimension", "3d"),
    ("size_rve", RVE_SIZE),
    ("periodic", "yes"),
    ("generation_sequence", "proportional"),
    ("generate_matrix", "no"),
    ("track_global_percolation_onset", "no"),
    ("stop_at_global_percolation", "no"),
    ("check_final_global_percolation", "no"),
    ("random_seed_type", "automatic"),
    ("random_seed", -75161927),
    ("fe_solver", "Abaqus/INP"),
    ("unsymmetric_solver", "no"),
    ("default_timestepping", "yes"),
    ("nb_cpus", 8),
    ("fe_solver_type", "iterative"),
    ("fe_field_output_frequency", 1),
    ("use_output_time_points", "yes"),
]

GLOBAL_SETTINGS_PARAMS: list[tuple[str, float | str]] = [
    ("allow_interpenetration", "no"),
    ("allow_coating_interpenetration", "no"),
    ("allow_rim_interpenetration", "no"),
    ("use_median_plane_interpenetration", "no"),
    ("cubic_architecture", "no"),
    ("apply_perturbation", "no"),
    ("minimum_relative_distance_wrt_diameter", 5.0e-2),
    ("minimum_relative_vol", 5.0e-2),
    ("max_number_of_tests", 2000),
    ("OT_norm_tol", 1.0e-1),
    ("max_number_of_geometry_attempts", 10),
    ("minimum_rel_dist_incl_to_face", 0.0),
    ("maximum_interpenetration_amount", 1.0),
    ("default_geometric_options", "yes"),
    ("remove_unconnected_matrix_regions", "no"),
]


# ============================== Internals ==============================
SECTION_MARKER = "##########################################"


def _format_value(value: float | str) -> str:
    """Format one DAF value.

    Parameters
    ----------
    value
        Integer, floating-point, or string parameter value.

    Returns
    -------
    str
        Digimat-compatible textual representation.
    """
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{float(value):.12e}"
    return str(value)


def _read_table(path: Path, min_cols: int) -> list[list[float]]:
    """Read a comma-separated numeric table.

    Parameters
    ----------
    path
        Input text-file path.
    min_cols
        Minimum required number of columns per non-empty row.

    Returns
    -------
    list[list[float]]
        Parsed numeric rows.
    """
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line_no, line in enumerate(file_obj, start=1):
            text = line.strip()
            if not text:
                continue
            parts = text.replace(",", " ").replace(";", " ").split()
            if len(parts) < min_cols:
                raise ValueError(
                    f"{path} line {line_no} has {len(parts)} columns, "
                    f"expected at least {min_cols}."
                )
            try:
                rows.append([float(part) for part in parts])
            except ValueError as exc:
                raise ValueError(
                    f"{path} line {line_no} contains non-numeric data: {text}"
                ) from exc

    if not rows:
        raise ValueError(f"{path} is empty.")
    return rows


def _resolve_input_file(input_dir: Path, filename: str) -> Path:
    """Resolve one required input file by exact filename.

    Parameters
    ----------
    input_dir
        Directory containing the particle-center file.
    filename
        Exact required filename.

    Returns
    -------
    pathlib.Path
        Resolved input path.
    """
    candidate = Path(input_dir) / filename
    if not candidate.is_file():
        raise FileNotFoundError(f"Particle-center file not found: {candidate}")
    return candidate


def _render_section(
    section_name: str,
    params: Sequence[tuple[str, float | str]],
) -> list[str]:
    """Render one DAF section."""
    lines: list[str] = [SECTION_MARKER, section_name]
    for key, value in params:
        lines.append(f"{key} = {_format_value(value)}")
    lines.append("")
    return lines


def _render_phase2_with_custom_positions(
    positions: Sequence[Sequence[float]],
) -> list[str]:
    """Render the spherical-particle phase and prescribed centers.

    Parameters
    ----------
    positions
        Particle-center rows. The first three columns are global ``x``,
        ``y``, and ``z`` coordinates.

    Returns
    -------
    list[str]
        DAF lines for Phase2.
    """
    lines = _render_section("PHASE", PHASE2_FIXED_PARAMS)
    lines.pop()

    for row_no, row in enumerate(positions, start=1):
        if len(row) < 3:
            raise ValueError(
                f"Particle-center row {row_no} must contain at least 3 columns."
            )
        lines.append(
            "custom_position = "
            f"{_format_value(row[0])} ; "
            f"{_format_value(row[1])} ; "
            f"{_format_value(row[2])}"
        )

    lines.append("")
    return lines


def _build_daf_text(positions: Sequence[Sequence[float]]) -> str:
    """Build one complete Digimat FE mechanical DAF input string."""
    lines: list[str] = []
    lines += _render_section("MATERIAL", MATERIAL1_PARAMS)
    lines += _render_section("MATERIAL", MATERIAL2_PARAMS)
    lines += _render_section("PHASE", PHASE1_PARAMS)
    lines += _render_phase2_with_custom_positions(positions)
    lines += _render_section("MICROSTRUCTURE", MICROSTRUCTURE_PARAMS)
    lines += _render_section("LOADING", MECHANICAL_LOADING_PARAMS)
    lines += _render_section("RVE", RVE_PARAMS)
    lines += _render_section("MESH", MESH_PARAMS)
    lines += _render_section("ANALYSISFE", MECHANICAL_ANALYSISFE_PARAMS)
    lines += _render_section("GLOBAL_SETTINGS", GLOBAL_SETTINGS_PARAMS)
    return "\n".join(lines).rstrip() + "\n"


def _normalize_indexes(indexes: Sequence[int | str]) -> list[str]:
    """Normalize and validate selected case identifiers."""
    normalized = [str(index).strip() for index in indexes if str(index).strip()]
    if not normalized:
        raise ValueError("SELECTED_INDEXES is empty.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"SELECTED_INDEXES contains duplicates: {normalized}")
    return normalized


def generate_one_daf(
    index: int | str,
    input_dir: Path = INPUT_DIR,
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    """Generate one mechanical DAF for prescribed spherical particles.

    Parameters
    ----------
    index
        Case identifier used in ``points_primary_3d_<index>.txt`` and the
        output filename.
    input_dir
        Directory containing the particle-center file.
    output_dir
        Directory receiving the generated DAF file.

    Returns
    -------
    pathlib.Path
        Generated DAF path.
    """
    index_tag = str(index).strip()
    if not index_tag:
        raise ValueError("index is empty.")

    point_filename = f"points_primary_3d_{index_tag}.txt"
    point_path = _resolve_input_file(input_dir, point_filename)
    positions = _read_table(point_path, min_cols=3)
    output_text = _build_daf_text(positions)

    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"Analysis_{index_tag}_{L_D}.daf"
    output_path.write_text(output_text, encoding="utf-8")
    return output_path


def generate_batch_dafs(
    indexes: Sequence[int | str] = SELECTED_INDEXES,
    input_dir: Path = INPUT_DIR,
    output_dir: Path = OUTPUT_DIR,
) -> list[Path]:
    """Generate mechanical DAF files for all selected case identifiers."""
    normalized_indexes = _normalize_indexes(indexes)
    return [
        generate_one_daf(
            index=index,
            input_dir=input_dir,
            output_dir=output_dir,
        )
        for index in normalized_indexes
    ]


def main() -> None:
    """Generate all configured mechanical DAF files."""
    generated_paths = generate_batch_dafs()
    print("Generated mechanical DAF files:")
    for path in generated_paths:
        print(path)


if __name__ == "__main__":
    main()
