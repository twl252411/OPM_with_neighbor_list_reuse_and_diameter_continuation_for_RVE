from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence


# ============================= User Config =============================
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR
OUTPUT_DIR: Path | None = None

# Only these DAF files are treated as mechanical cases.
DAF_GLOB = "Analysis_*.daf"

# False: extract completed cases and report cases whose result folders are absent.
# True: stop as soon as one DAF has no same-name result folder.
STRICT_MISSING_RUN_DIRS = False

# Digimat stiffness order:
# [11, 22, 33, 12, 23, 13]
#
# Output order:
# [11, 22, 33, 12, 13, 23]
REORDER_STIFFNESS_TO_STANDARD_VOIGT = True
OUTPUT_FLOAT_FORMAT = ".6e"


_FLOAT_PATTERN = re.compile(
    r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?"
)


def _format_float(value: float) -> str:
    """Format one output value."""
    return format(float(value), OUTPUT_FLOAT_FORMAT)


def _parse_numeric_line(line: str) -> list[float]:
    """Parse numerical values from one ENG line."""
    values = _FLOAT_PATTERN.findall(line)
    return [
        float(value.replace("D", "E").replace("d", "e"))
        for value in values
    ]


def _normalize_heading(line: str) -> str:
    """Normalize one ENG section heading."""
    text = line.strip().lower()
    text = text.lstrip("#").strip()
    text = text.replace(":", " ")
    return " ".join(text.split())


def _find_stiffness_section_start(lines: Sequence[str]) -> int:
    """Find the stiffness matrix section in global axes."""
    accepted_headings = {
        "stiffness matrix in global axes",
        "stiffness tensor in global axes",
    }

    for index, line in enumerate(lines):
        if _normalize_heading(line) in accepted_headings:
            return index

    raise ValueError(
        "Mechanical stiffness section was not found. Expected a heading "
        "such as '# Stiffness Matrix in Global Axes :'."
    )


def _extract_stiffness_block(
    lines: Sequence[str],
) -> list[list[float]]:
    """Extract the first 6x6 stiffness matrix."""
    section_index = _find_stiffness_section_start(lines)
    rows: list[list[float]] = []

    for raw_line in lines[section_index + 1:]:
        stripped = raw_line.strip()

        if not stripped:
            if rows:
                break
            continue

        if stripped.startswith("#"):
            if rows:
                break
            continue

        values = _parse_numeric_line(stripped)
        if not values:
            if rows:
                break
            continue

        if len(values) != 6:
            raise ValueError(
                "Each stiffness row must contain exactly 6 numerical "
                f"values, but got {len(values)} in line: {raw_line!r}"
            )

        rows.append(values)
        if len(rows) == 6:
            break

    if len(rows) != 6:
        raise ValueError(
            f"Stiffness matrix must contain 6 rows, but got {len(rows)}."
        )

    return rows


def _stiffness_reorder_6x6(
    raw: Sequence[Sequence[float]],
) -> list[list[float]]:
    """Reorder the Digimat matrix to standard Voigt order."""
    if len(raw) != 6 or any(len(row) != 6 for row in raw):
        shape = (len(raw), len(raw[0]) if raw else 0)
        raise ValueError(f"Stiffness must be 6x6, got shape {shape}.")

    permutation = [0, 1, 2, 3, 5, 4]
    return [
        [float(raw[i][j]) for j in permutation]
        for i in permutation
    ]


def _write_csv_rows(
    path: Path,
    rows: Sequence[Sequence[float]],
) -> None:
    """Write a matrix as comma-separated values."""
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(
                ",".join(_format_float(value) for value in row)
            )
            file.write("\n")


def _read_text_with_fallback(path: Path) -> str:
    """Read an ENG file using common Windows encodings."""
    encodings = ("utf-8-sig", "utf-8", "cp1252", "mbcs")
    failures: list[str] = []

    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding)
        except LookupError:
            continue
        except UnicodeDecodeError as exc:
            failures.append(f"{encoding}: {exc}")

    raise RuntimeError(
        f"Unable to decode ENG file {path}. Attempts: "
        + "; ".join(failures)
    )


def _resolve_exact_eng_path(
    run_dir: Path,
    job_name: str,
) -> Path:
    """Resolve exactly one ENG file for the DAF-named job."""
    candidates = [
        run_dir / f"{job_name}.eng",
        run_dir / f"{job_name}_Analysis1.eng",
        run_dir / f"{job_name}_Analysis2.eng",
    ]

    existing = [
        candidate
        for candidate in candidates
        if candidate.exists()
    ]

    if len(existing) == 1:
        if existing[0].stat().st_size == 0:
            raise RuntimeError(f"ENG file is empty: {existing[0]}")
        return existing[0]

    if len(existing) > 1:
        names = ", ".join(path.name for path in existing)
        raise RuntimeError(
            f"Ambiguous ENG files for job {job_name}: {names}"
        )

    expected = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"No exact ENG file found for job {job_name}. "
        f"Expected one of: {expected}"
    )


def _list_subdirectories(root: Path) -> str:
    """Return a readable list of existing immediate subdirectories."""
    names = sorted(
        path.name for path in root.iterdir() if path.is_dir()
    )
    return ", ".join(names) if names else "<none>"


def _process_one_daf(
    daf_path: Path,
    out_dir: Path,
) -> Path:
    """Extract one stiffness matrix using the DAF filename as job name."""
    job_name = daf_path.stem
    run_dir = daf_path.parent / job_name

    if not run_dir.is_dir():
        raise FileNotFoundError(
            f"Run directory not found for {daf_path.name}: {run_dir}"
        )

    eng_path = _resolve_exact_eng_path(
        run_dir=run_dir,
        job_name=job_name,
    )
    print(f"Using mechanical ENG file: {eng_path}")

    lines = _read_text_with_fallback(eng_path).splitlines()
    stiffness_raw = _extract_stiffness_block(lines)

    if REORDER_STIFFNESS_TO_STANDARD_VOIGT:
        stiffness = _stiffness_reorder_6x6(stiffness_raw)
    else:
        stiffness = [list(row) for row in stiffness_raw]

    output_path = out_dir / f"{job_name}_Stiffness.txt"
    _write_csv_rows(output_path, stiffness)
    return output_path


def extract_all_and_save(
    input_dir: Path = INPUT_DIR,
    output_dir: Path | None = OUTPUT_DIR,
    daf_glob: str = DAF_GLOB,
    strict_missing_run_dirs: bool = STRICT_MISSING_RUN_DIRS,
) -> list[Path]:
    """Extract stiffness matrices for DAF files with completed runs."""
    root = Path(input_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Input directory not found: {root}")

    out_dir = (
        root if output_dir is None
        else Path(output_dir).resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    daf_files = sorted(
        path for path in root.glob(daf_glob) if path.is_file()
    )

    if not daf_files:
        raise FileNotFoundError(
            f"No DAF files matching {daf_glob!r} were found in {root}."
        )

    saved: list[Path] = []
    missing_run_dirs: list[Path] = []
    failed_cases: list[str] = []

    for daf_path in daf_files:
        run_dir = root / daf_path.stem

        if not run_dir.is_dir():
            missing_run_dirs.append(run_dir)
            if strict_missing_run_dirs:
                raise FileNotFoundError(
                    f"Expected same-name run directory: {run_dir}"
                )
            print(
                f"Skipped {daf_path.name}: run directory does not exist: "
                f"{run_dir}"
            )
            continue

        try:
            saved.append(
                _process_one_daf(
                    daf_path=daf_path,
                    out_dir=out_dir,
                )
            )
        except Exception as exc:
            failed_cases.append(f"{daf_path.name}: {exc}")
            print(f"Failed {daf_path.name}: {exc}")

    if not saved:
        existing_dirs = _list_subdirectories(root)
        expected_dirs = ", ".join(
            str(path.name) for path in missing_run_dirs
        ) or "<none>"

        details = [
            "No mechanical stiffness result was extracted.",
            f"Analysis directory: {root}",
            f"Expected DAF-named run folders: {expected_dirs}",
            f"Existing immediate subdirectories: {existing_dirs}",
            "Run digimatFE_runner_mechanical_overwrite.py first and confirm "
            "that it creates folders such as Analysis_1.",
        ]
        if failed_cases:
            details.append("Case errors: " + " | ".join(failed_cases))

        raise FileNotFoundError("\n".join(details))

    if missing_run_dirs:
        print("Missing run directories:")
        for path in missing_run_dirs:
            print(f"  {path}")

    if failed_cases:
        print("Cases not extracted:")
        for message in failed_cases:
            print(f"  {message}")

    return saved


def main() -> None:
    """Extract all available mechanical stiffness matrices."""
    saved_paths = extract_all_and_save()

    print("Saved mechanical stiffness files:")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
