# An accelerated OPM method for generating dense RVEs using neighbor-list reuse and diameter continuation

## 1. Overview

This repository provides the source programs used to generate dense periodic
representative volume elements (RVEs) of particulate composites and to perform
the associated finite-element homogenization (FEH) calculations. The retained
programs cover:

- periodic circular and spherical RVE generation by overlap-potential
  minimization;
- conventional exact-neighbor relaxation and accelerated relaxation using
  protective-skin neighbor-list reuse;
- adaptive collision-diameter continuation and hybrid path dispatch;
- monodisperse and polydisperse particle systems;
- Digimat-FE input generation, batch execution, and stiffness extraction.

This is a source-code release. Precomputed numerical results, saved particle
configurations, manuscript figures, figure-source data, plotting programs,
commercial-solver databases, and temporary solver files are not included.

## 2. Repository Layout

```text
Github_codes/
├─ config/
│  └─ optimal_parameters_frozen.yaml   # Frozen RVE solver parameters
├─ src/
│  ├─ overlap_removal_fast.py          # Exact/reused-neighbor RVE kernels
│  ├─ overlap_removal_hybrid.py        # Continuation and hybrid solvers
│  ├─ overlap_removal_paths.py         # Unified single-case interface
│  ├─ overlap_removal_polydisperse.py  # Polydisperse solver kernels
│  └─ digimat_feh/
│     ├─ daf_generator_fe_mechanical.py
│     ├─ digimatFE_runner_mechanical.py
│     └─ eng_results_fe_mechanical.py
├─ requirements.txt
├─ LICENSE
└─ README.md
```

Generated particle configurations, DAF files, solver outputs, and extracted
stiffness files are user outputs and are not part of this source-only
repository.

## 3. Requirements

### 3.1 Python Dependencies

Python 3.12 is recommended. Install the pinned open-source dependencies from
the repository root:

```bash
python -m pip install -r requirements.txt
```

The implementation uses `numpy`, `scipy`, and `numba`.

### 3.2 Commercial Software

- RVE generation and geometric certification use only the open-source Python
  dependencies listed above.
- Digimat-FE and a compatible Abaqus solver are required to execute the FEH
  analyses.
- The default Digimat-FE executable path is defined by `DEFAULT_DIGIMAT_BAT`
  in `src/digimat_feh/digimatFE_runner_mechanical.py`; update it for the local
  installation.
- Proprietary executables, licenses, solver databases, and runtime files are
  not distributed with this repository.

## 4. Quick Start

Run all available solution paths for one three-dimensional spherical RVE:

```bash
python src/overlap_removal_paths.py \
  --dimension 3 \
  --v1 0.60 \
  --l-over-d0 8 \
  --seed 1000 \
  --paths all
```

The command prints a JSON summary to the terminal. Use `--dimension 2` for
circular RVEs. Run any command-line program with `--help` to inspect its
available options.

## 5. Workflow A: Single-Case RVE Generation

The unified entry point is `src/overlap_removal_paths.py`. It supports the
following path labels:

- `R`: conventional relaxation with exact periodic-neighbor reconstruction;
- `A_Dt`: accelerated direct relaxation using protective-skin neighbor-list
  reuse;
- `A_cont`: accelerated collision-diameter continuation; and
- `hybrid`: parameter-based dispatch between the accelerated paths.

Select one or more paths with a comma-separated argument:

```bash
python src/overlap_removal_paths.py \
  --dimension 3 \
  --v1 0.62 \
  --enlargement 1.0 \
  --diameter 1.0 \
  --l-over-d0 8 \
  --seed 1701 \
  --paths R,A_Dt,A_cont
```

Lengths are expressed in coordinates normalized by the reference diameter
`D0`. If `--l-over-d0` is omitted, the default value is 20 in two dimensions
and 8 in three dimensions.

## 6. Workflow B: Finite-Element Homogenization

The FEH implementation is provided as three standalone modules. Edit the user-
configuration blocks in `src/digimat_feh/` to select the particle-center input
directory, case identifiers, material and mesh settings, Digimat-FE executable,
and output locations. Then run:

```bash
python src/digimat_feh/daf_generator_fe_mechanical.py
python src/digimat_feh/digimatFE_runner_mechanical.py
python src/digimat_feh/eng_results_fe_mechanical.py
```

The first program converts prescribed spherical-particle center files into
Digimat-FE DAF inputs. The second executes the selected DAF cases, and the third
extracts the homogenized stiffness matrices from completed ENG results.

## 7. Configuration and Naming Conventions

- `dimension=2` denotes periodic circular RVEs; `dimension=3` denotes periodic
  spherical RVEs.
- `v1` is the nominal particle fraction.
- `e` or `enlargement` is the particle-diameter enlargement factor.
- `D0` is the reference particle diameter, and `L/D0` is the normalized RVE
  side length.
- `R`, `A_Dt`, and `A_cont` denote the reference, accelerated-direct, and
  accelerated-continuation paths, respectively.
- The frozen YAML file records the protective-skin, continuation, dispatch,
  certification, and timeout parameters associated with the reported
  implementation.

## 8. Troubleshooting

1. **Python import error**: install the pinned dependencies and run the commands
   from the repository root as shown above.
2. **Long first RVE run**: Numba compiles numerical kernels on first use.
3. **Digimat executable not found**: update `DEFAULT_DIGIMAT_BAT` to the local
   Digimat-FE batch executable and confirm that the Abaqus solver is available
   to Digimat-FE.
4. **DAF or ENG file not found**: confirm that the configured input, analysis,
   and output directories use the same case identifiers.

## 9. Citation

If these codes are helpful, please cite:

> Wenlong Tian, Qingliang Shen, Lizheng Su, and Lehua Qi,
> *Efficient generation of dense RVEs of composites with circular and
> spherical inclusions using safe neighbor-list reuse and adaptive
> collision-diameter continuation*, Computational Mechanics, 2026, Revision.

## 10. License

The source code is released under the MIT License in `LICENSE`. Third-party
software remains subject to its own license.
