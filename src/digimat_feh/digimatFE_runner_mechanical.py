from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Sequence


DEFAULT_DIGIMAT_BAT = Path(
    r"C:\MSC.Software\Digimat\2023.1\DigimatFE\exec\DigimatFE.bat"
)
SCRIPT_DIR = Path(__file__).resolve().parent

# ============================= User Config =============================
# One or more case indexes. For one case, use [1], not 1.
SELECTED_INDEXES: list[int | str] = [1, 2, 3, 4, 5]

# Analysis_1.daf, Analysis_2.daf, ... must be stored here.
ANALYSIS_DIR = SCRIPT_DIR
DIGIMAT_BAT = DEFAULT_DIGIMAT_BAT

USE_RUN_FE_WORKFLOW_FLAG = True
FALLBACK_WITHOUT_RUN_FE_WORKFLOW_FLAG = True
TIMEOUT_SECONDS: float | None = None
DRY_RUN = False


def _normalize_indexes(indexes: Sequence[int | str]) -> list[str]:
    """Normalize selected mechanical case identifiers.

    Parameters
    ----------
    indexes
        Identifiers used in ``Analysis_<index>.daf``.

    Returns
    -------
    list[str]
        Valid non-empty identifiers.
    """
    normalized = [
        str(index).strip()
        for index in indexes
        if str(index).strip()
    ]
    if not normalized:
        raise ValueError("SELECTED_INDEXES is empty.")
    return normalized


def _resolve_analysis_dir(analysis_dir: Path) -> Path:
    """Resolve the directory containing the mechanical DAF files."""
    analysis_dir = Path(analysis_dir)

    if analysis_dir.is_absolute():
        return analysis_dir.resolve()

    cwd_candidate = (Path.cwd() / analysis_dir).resolve()
    script_candidate = (SCRIPT_DIR / analysis_dir).resolve()

    if cwd_candidate.exists():
        return cwd_candidate
    if script_candidate.exists():
        return script_candidate
    if analysis_dir.name == SCRIPT_DIR.name:
        return SCRIPT_DIR
    return script_candidate


def _prepare_overwrite_run(
    parent_dir: Path,
    daf_stem: str,
    dry_run: bool = False,
) -> tuple[Path, str]:
    """Create the run folder using the exact DAF stem.

    If the folder already exists, it is deleted completely and recreated.
    The Digimat job name is identical to the DAF filename without ``.daf``.

    Parameters
    ----------
    parent_dir
        Directory containing the DAF file and its run folder.
    daf_stem
        DAF filename without the ``.daf`` suffix.
    dry_run
        When true, determine the path without changing the filesystem.

    Returns
    -------
    tuple[pathlib.Path, str]
        Run-directory path and matching Digimat job name.
    """
    parent_dir = Path(parent_dir).resolve()
    job_name = str(daf_stem).strip()

    if not job_name:
        raise ValueError("DAF stem is empty.")

    working_dir = (parent_dir / job_name).resolve()

    if not dry_run:
        if working_dir.exists():
            if working_dir == parent_dir:
                raise RuntimeError(
                    "Refusing to delete the DAF source directory."
                )
            shutil.rmtree(working_dir)
            print(f"Removed existing run directory: {working_dir}")

        working_dir.mkdir(parents=True, exist_ok=False)

    return working_dir, job_name


def build_digimat_command(
    digimat_bat: Path,
    daf_path: Path,
    working_dir: Path,
    job_name: str,
    use_run_fe_workflow_flag: bool = True,
) -> list[str]:
    """Build the DigimatFE command for one mechanical analysis."""
    command = [str(digimat_bat)]

    if use_run_fe_workflow_flag:
        command.append("-runFEWorkflow")

    command.extend(
        [
            f"input={daf_path}",
            f"workingDir={working_dir}",
            f"jobName={job_name}",
        ]
    )
    return command


def _execute_digimat_command(
    command: Sequence[str],
    timeout: float | None,
    dry_run: bool,
) -> subprocess.CompletedProcess[str] | None:
    """Print and execute one Digimat command."""
    print("Executing:", " ".join(command))

    if dry_run:
        return None

    return subprocess.run(
        list(command),
        check=True,
        text=True,
        timeout=timeout,
    )


def _resolve_exact_eng_path(
    working_dir: Path,
    job_name: str,
) -> Path:
    """Resolve exactly one ENG output belonging to the current job.

    Accepted names are:

    - ``<job_name>.eng``
    - ``<job_name>_Analysis1.eng``
    - ``<job_name>_Analysis2.eng``

    No wildcard or newest-file fallback is allowed.
    """
    working_dir = Path(working_dir).resolve()
    target = working_dir / f"{job_name}.eng"
    generated_candidates = [
        working_dir / f"{job_name}_Analysis1.eng",
        working_dir / f"{job_name}_Analysis2.eng",
    ]

    existing_generated = [
        path for path in generated_candidates if path.exists()
    ]

    if target.exists() and existing_generated:
        names = ", ".join(
            path.name for path in [target, *existing_generated]
        )
        raise RuntimeError(
            f"Ambiguous ENG files for mechanical job {job_name}: {names}"
        )

    if len(existing_generated) > 1:
        names = ", ".join(path.name for path in existing_generated)
        raise RuntimeError(
            f"Ambiguous ENG files for mechanical job {job_name}: {names}"
        )

    if target.exists():
        if target.stat().st_size == 0:
            raise RuntimeError(f"Generated ENG file is empty: {target}")
        return target

    if len(existing_generated) == 1:
        source = existing_generated[0]
        source.replace(target)
        print(f"Renamed ENG: {source.name} -> {target.name}")

        if target.stat().st_size == 0:
            raise RuntimeError(f"Generated ENG file is empty: {target}")
        return target

    expected = [target, *generated_candidates]
    expected_text = ", ".join(str(path) for path in expected)
    raise FileNotFoundError(
        "Digimat finished without an exact ENG result for the current "
        f"mechanical job. Expected one of: {expected_text}"
    )


def run_digimat_daf(
    daf_path: Path,
    digimat_bat: Path = DEFAULT_DIGIMAT_BAT,
    use_run_fe_workflow_flag: bool = True,
    fallback_without_run_fe_workflow_flag: bool = True,
    timeout: float | None = None,
    dry_run: bool = False,
) -> subprocess.CompletedProcess[str] | None:
    """Submit one DAF and overwrite its existing same-name run folder.

    The DAF filename without ``.daf`` is used for both:

    - the run-folder name;
    - the Digimat job name.

    If the first submission fails and fallback is enabled, the same folder is
    deleted and recreated before retrying without ``-runFEWorkflow``.
    """
    daf_path = Path(daf_path).resolve()
    digimat_bat = Path(digimat_bat).resolve()

    if not daf_path.exists():
        raise FileNotFoundError(f"Mechanical DAF file not found: {daf_path}")

    if not dry_run and not digimat_bat.exists():
        raise FileNotFoundError(f"DigimatFE.bat not found: {digimat_bat}")

    working_dir, job_name = _prepare_overwrite_run(
        parent_dir=daf_path.parent,
        daf_stem=daf_path.stem,
        dry_run=dry_run,
    )

    command = build_digimat_command(
        digimat_bat=digimat_bat,
        daf_path=daf_path,
        working_dir=working_dir,
        job_name=job_name,
        use_run_fe_workflow_flag=use_run_fe_workflow_flag,
    )

    try:
        result = _execute_digimat_command(
            command=command,
            timeout=timeout,
            dry_run=dry_run,
        )
    except subprocess.CalledProcessError:
        if not (
            use_run_fe_workflow_flag
            and fallback_without_run_fe_workflow_flag
        ):
            raise

        # Overwrite the failed run folder before fallback retry.
        working_dir, job_name = _prepare_overwrite_run(
            parent_dir=daf_path.parent,
            daf_stem=daf_path.stem,
            dry_run=dry_run,
        )

        fallback_command = build_digimat_command(
            digimat_bat=digimat_bat,
            daf_path=daf_path,
            working_dir=working_dir,
            job_name=job_name,
            use_run_fe_workflow_flag=False,
        )

        print(
            "Retrying without -runFEWorkflow after overwriting "
            "the same-name run directory."
        )
        result = _execute_digimat_command(
            command=fallback_command,
            timeout=timeout,
            dry_run=dry_run,
        )

    if not dry_run:
        eng_path = _resolve_exact_eng_path(
            working_dir=working_dir,
            job_name=job_name,
        )
        print(f"Validated mechanical ENG file: {eng_path}")

    return result


def run_digimat_by_index(
    index: int | str,
    analysis_dir: Path = ANALYSIS_DIR,
    digimat_bat: Path = DEFAULT_DIGIMAT_BAT,
    use_run_fe_workflow_flag: bool = True,
    fallback_without_run_fe_workflow_flag: bool = True,
    timeout: float | None = None,
    dry_run: bool = False,
) -> subprocess.CompletedProcess[str] | None:
    """Run one mechanical DAF selected by index."""
    index_tag = str(index).strip()
    if not index_tag:
        raise ValueError("index is empty.")

    resolved_dir = _resolve_analysis_dir(analysis_dir)
    daf_path = (resolved_dir / f"Analysis_{index_tag}_8.daf").resolve()

    return run_digimat_daf(
        daf_path=daf_path,
        digimat_bat=digimat_bat,
        use_run_fe_workflow_flag=use_run_fe_workflow_flag,
        fallback_without_run_fe_workflow_flag=(
            fallback_without_run_fe_workflow_flag
        ),
        timeout=timeout,
        dry_run=dry_run,
    )


def run_batch(
    indexes: Sequence[int | str] = SELECTED_INDEXES,
    analysis_dir: Path = ANALYSIS_DIR,
    digimat_bat: Path = DIGIMAT_BAT,
    use_run_fe_workflow_flag: bool = USE_RUN_FE_WORKFLOW_FLAG,
    fallback_without_run_fe_workflow_flag: bool = (
        FALLBACK_WITHOUT_RUN_FE_WORKFLOW_FLAG
    ),
    timeout: float | None = TIMEOUT_SECONDS,
    dry_run: bool = DRY_RUN,
) -> list[subprocess.CompletedProcess[str] | None]:
    """Run all selected mechanical cases sequentially."""
    normalized_indexes = _normalize_indexes(indexes)

    results: list[subprocess.CompletedProcess[str] | None] = []
    for index_tag in normalized_indexes:
        print(
            f"=== Running DigimatFE mechanical analysis for "
            f"Analysis_{index_tag}.daf ==="
        )
        results.append(
            run_digimat_by_index(
                index=index_tag,
                analysis_dir=analysis_dir,
                digimat_bat=digimat_bat,
                use_run_fe_workflow_flag=use_run_fe_workflow_flag,
                fallback_without_run_fe_workflow_flag=(
                    fallback_without_run_fe_workflow_flag
                ),
                timeout=timeout,
                dry_run=dry_run,
            )
        )

    return results


def main() -> None:
    """Run the configured mechanical Digimat batch."""
    run_batch()


if __name__ == "__main__":
    main()
