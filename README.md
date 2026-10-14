# SAD: Birkhoff Interpolation Models for Optimization with Some Available Derivatives

Code accompanying the manuscript:

> J. Larson, M. Menickelly, E. Toler, "Birkhoff Interpolation Models for Optimization with Some
> Available Derivatives," submitted to the *SIAM Journal on Optimization*. Preprint:
> [arXiv:2605.28503](https://arxiv.org/abs/2605.28503).

This repository implements `min_SAD`, a derivative-free/partially-derivative-aware trust-region
optimization method based on Birkhoff interpolation models, along with the benchmarking,
poisedness-illustration, and stellarator-design-example code used to produce the manuscript's
figures.

## Repository layout

```
src/                    core min_SAD solver and its interpolation-model machinery
benchmark/              CUTEst benchmark driver (SAD side) and problem/K-set construction utilities
plot/                   scripts that reproduce the manuscript's figures from benchmark output
stellarator_example/    reduced (n=10) stellarator design example (Section on stellarator example)
tests/                  lightweight regression/unit tests for src/min_SAD.py and its components
patches/                description of the Hermite-baseline modification to Py-BOBYQA, for
                        transparency only (neither Py-BOBYQA nor the driver code that called it
                        is included here; see patches/README.md)
requirements.txt        Python package dependencies
LICENSE                 MIT license (see file; excludes the not-included Py-BOBYQA baseline)
```

The manuscript's "Hermite least-squares" baseline comparison (Py-BOBYQA-based) was run with driver
code written by a collaborator, not the SAD authors, and so is not published in this repository.
`patches/README.md` documents the Py-BOBYQA modification itself for transparency, but the
`benchmark/hermite_baseline/` and `stellarator_example/from_scratch/` driver/comparison scripts
used to produce the manuscript's Hermite-baseline numbers are not included here. **If you need
that driver code, it is not ours to distribute** — contact the authors of 
https://doi.org/10.1007/s11081-023-09795-y for a version.

## Installation

```
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` covers the pip-installable dependencies (numpy, scipy, matplotlib, ipdb,
trustregion, sparse). Several pieces of this code additionally depend on packages that are **not**
pip-installable in the usual way; install only the ones needed for what you want to reproduce:

- **minqsw** (box-constrained QP solver), required by `src/min_SAD.py` and most of `src/` and
  `tests/`:
  ```
  git clone https://github.com/POptUS/MINQ
  export PYTHONPATH="$PWD/MINQ/py/minq5:$PYTHONPATH"
  ```
- **pycutest** (CUTEst interface), required only by `benchmark/` and `plot/plot_cutest_comparison.py`:
  follow the installation instructions at https://github.com/ralna/pycutest exactly (it requires a
  separate CUTEst/SIFDecode/ARCHDefs source install, not just a pip package).
- **mpi4py**, required only by `benchmark/run_test_functions.py`: requires a working MPI
  installation (OpenMPI or MPICH) on your system before `pip install mpi4py`.
- **ibcdfo** (POptUS `pounders` solver, https://github.com/POptUS/IBCDFO), required only by
  `stellarator_example/single_stage_optimization_SAD_reduced.py` (used there for an additional
  derivative-free baseline comparison) and optionally `src/Test_SAD_BenDFO.py`.
- **simsopt** (https://github.com/hiddenSymmetries/simsopt), required only by
  `stellarator_example/`. `simsopt` itself depends on a working VMEC build; see
  https://simsopt.readthedocs.io. See also `build_scripts/simsopt/`, `build_scripts/DESC_jax/`,
  `build_scripts/GX/`, and `build_scripts/PyPlasmaOpt/` in the broader project tree (not part of
  this release) for the exact build recipes used on the cluster where the manuscript's stellarator
  results were generated — they are not reproduced here because they are machine/cluster-specific
  build scripts rather than part of the method itself.
- **Py-BOBYQA ("Hermite" baseline)**: not a dependency of anything in this repository. The
  manuscript's Hermite-baseline comparison used a modified Py-BOBYQA fork (GNU GPLv3) together with
  driver code written by a collaborator; neither the fork nor that driver code is included here, and
  neither is ours to redistribute — contact the manuscript's corresponding author for access.
  `patches/pybobyqa_HermiteLS_v1.2.patch` is a verified unified diff against upstream Py-BOBYQA
  v1.2 documenting what the fork's modification was, for transparency; see `patches/README.md`.

## Reproducing the CUTEst comparison figure (Figure "CUTEst")

This reproduces `absgrad=1.0e-02.png` and `absgrad=1.0e-04.png` (among other gate values), the two
gate values used as the manuscript's CUTEst comparison figure. These reference images are not
bundled in this repository; the steps below regenerate them from scratch.

1. Install `pycutest` and `mpi4py` (see above).
2. Generate SAD results on the CUTEst problem set:
   ```
   cd benchmark
   python run_test_functions.py
   ```
   This populates `benchmark/benchmark_results_model_degree_2_max_deriv_2/`.
3. Produce the figures:
   ```
   cd plot
   python plot_cutest_comparison.py
   ```
   This writes `absgrad=1.0e-XX.png` for each gate value in `np.logspace(-8, -1, 8)`, including
   the two used in the manuscript.

Note: the manuscript's figure also includes a "Hermite least-squares" (Py-BOBYQA-based) baseline,
produced with driver code that is not included in this repository (see "Repository layout" above).
`plot_cutest_comparison.py` tolerates this: it looks for that baseline's results under
`benchmark/hermite_baseline/benchmark_results_mona/`, and if that directory is missing or
incomplete it fills the missing problem/solver combinations with null data, prints a warning, and
still produces a figure — just one with only the SAD and scipy curves, not matching the manuscript.

## Reproducing the poisedness heatmap figures (Figure "heatmap_poisedness")

```
cd src
mkdir -p plots
python heatmap_poisedness.py
```

This writes both panels into `src/plots/` (not bundled in this repository): the **left panel**,
`heatmap_poisedness_diag.png`, and the **right panel**,
`heatmap_poisedness_diag_f_only_plus_f_newpoint.png` (plus two intermediate, didactic variants not
used in the manuscript). The right panel is produced by a separate call to
`heatmap_poisedness()` that passes an explicit `Data` list: it augments the same initial
interpolation set `D` used for the left panel with one additional *fixed, function-value-only*
condition at `(sqrt(2)/2, -sqrt(2)/2)` (drawn with a circle marker), and then — unlike the left
panel, which scans candidate points carrying their full admissible derivative set — evaluates
adding each scanned candidate point as a *function-value-only* condition (no derivative).

## Reproducing the stellarator example (Figure "stellarator_comparison")

This is the most involved reproduction and requires `simsopt` (with a working VMEC build).

```
cd stellarator_example
python single_stage_optimization_SAD_reduced.py
```

This writes `single_stage_SAD_reduced_max_deriv_1_model_degree_2.npy` into `stellarator_example/`.
It uses the included warm-start files `x_opt_FDJ1.npy`, `x_after_stage_2.npy`, and
`inputs/input.nfp4_QH_warm_start`.

The manuscript's figure also overlays a "Hermite least-squares" (Py-BOBYQA-based) baseline on this
same problem, produced with driver code that is not included in this repository (see "Repository
layout" above). We cannot reproduce the full `stellarator_comparison.png` figure from this
repository alone; the SAD-only trajectory above is what this repository can regenerate.

## Running the test suite

```
cd tests
python run_all.py
```

Runs every `test_*.py` as a subprocess and prints a PASS/FAIL/SKIP summary. Most tests require
`minqsw` on `PYTHONPATH` (see Installation) and will print `SKIP` rather than fail if it is
unavailable.

Note: `src/Test_SAD_BenDFO.py` (not part of `tests/`) additionally depends on the `calfun`/`dfoxs`
Python modules from the POptUS BenDFO package (https://github.com/POptUS/BenDFO, BSD 3-Clause
License). They are not included in this repository; clone BenDFO and add its `py/` directory to
`PYTHONPATH` to run this script. (`benchmark/BenDFO/dfo.dat`, used by the CUTEst-free DFO
benchmark problems, is BenDFO's own problem-list data file and is already included here.)
