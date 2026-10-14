# Py-BOBYQA "Hermite least-squares" baseline

The manuscript compares our Birkhoff-interpolation optimizer (`src/min_SAD.py`) against a
derivative-aware baseline built on a modified fork of
[Py-BOBYQA](https://github.com/numericalalgorithmsgroup/pybobyqa) (version 1.2, commit
`2ae99d565f4ed03165e177c71583e81b9d9b93ab`). That fork implements the "Hermite least-squares"
interpolation approach of Fuhrländer & Schöps (see the manuscript's bibliography) and is
distributed under the GNU GPLv3.

**Neither that fork nor the driver code that called it is included in this repository.** The
driver/benchmark scripts that produced the manuscript's Hermite-baseline results were written by a
collaborator, not the SAD authors, so we are not publishing them here alongside our own code;
Py-BOBYQA's own fork is additionally excluded because its GPLv3 license is incompatible with
vendoring it into this MIT-licensed repository. This directory documents, for transparency and
reproducibility, exactly what the fork's modification was — not how it was invoked. **That driver
code and the fork are not ours to distribute; if you need them, contact the 
corresponding author of** https://doi.org/10.1007/s11081-023-09795-y.

`pybobyqa_HermiteLS_v1.2.patch`, included in this directory, is a verified unified diff against
upstream Py-BOBYQA v1.2 (see "How to reproduce the baseline locally" below). Like the fork it
documents, **the patch file itself is derived from and contains excerpts of GPLv3-licensed source
code and is not covered by this repository's MIT license** — it is included here solely as a
derivative-work description under GPLv3's own terms, in the same spirit as a Debian/Gentoo
packaging patch.

## What the modification does

Py-BOBYQA builds a local quadratic model by interpolating/regressing function values at a
trust-region sample set `Y`. The modification adds a `known_derivs` argument (a list of
derivative-component indices, e.g. `[0, 1]` for the first two coordinate partials) and a
`version` flag, threaded from `pybobyqa.solve(...)` down through `solve_main`, `Controller`, and
`Model`:

- The user-supplied objective function is called as `objfun(x, known_derivs, return_all)` and is
  expected to additionally return the requested partial derivatives of `f` at `x` for every index
  in `known_derivs`, on every evaluation. `util.py` renames the original Py-BOBYQA evaluator to
  `eval_objective_Cartis` and adds a new `eval_objective` with this derivative-aware signature
  (plus a small `eval_derivative` helper used elsewhere).
- `model.py`'s `Model` class stores `version`/`known_derivs`/`derivs0`, carries a parallel
  `fderivs` array of sampled derivatives alongside `f_values` (kept in sync across
  `change_point`/`swap_points`/`add_new_point`/`add_new_sample`), and exposes `derivsopt()`. When
  building its interpolation/regression system (`build_interpolation_matrix`, the `version == 1`
  branch), it appends one extra regression row per sample point per known derivative component, in
  addition to the usual function-value rows. Each derivative row is the exact (not
  finite-differenced) partial derivative of the quadratic model basis with respect to that
  component, evaluated at the sample point, optionally weighted by a distance-based weight
  function (`deriv_weights`/`fct_weights`; weighting is disabled by default in the fork as found —
  see `weight_type` in `model.py`). A new `hessian.py` helper, `to_upper_triangular_vector_MF`,
  builds the (generally non-symmetric) quadratic-column contribution of a single derivative row.
- The resulting augmented, generally overdetermined linear system is solved with
  `np.linalg.lstsq` (`solve_system_with_regression_scaled`) instead of the usual square/
  underdetermined solve, giving a least-squares quadratic model that is consistent with both the
  sampled function values and the known partial derivatives at every sample point. This is
  `self.version == 1`; the original Py-BOBYQA interpolation behavior is preserved as a separate
  code path (`version in {2, 3, 4}` branches of `lagrange_polynomial`/`build_interpolation_matrix`
  retain the unmodified least-change update, used only for Lagrange-polynomial/poisedness
  bookkeeping). Supplying `known_derivs = []` does not by itself reproduce stock Py-BOBYQA, since
  the fork always takes the `version == 1` regression path in `interpolate_model`; `version` and
  `known_derivs` are both baked into the benchmark driver's call, not user-facing solver options.
- `solver.py`, `controller.py`, and `hessian.py` thread `version`/`known_derivs`/`derivs`/`derivsmin`
  through the full call chain from `pybobyqa.solve()` to `Model.__init__`, the objective-evaluation
  calls, `change_point`, `soft_restart`, and the final-result tuple (`get_final_results` now also
  returns the derivatives at the returned point). `solver.OptimResults` additionally carries the
  iteration count (`n_iter`).
- `params.py` and `trust_region.py` differ from upstream only in incidental NumPy-deprecation
  fixes unrelated to the above (`np.int`/`np.float` → `int`/`float`, marked inline with "Evan:"
  comments in the patch); they contain no Hermite-least-squares-specific changes.
- `diagnostic_info.py`, `__init__.py`, and `version.py` are unmodified from upstream v1.2.

## How to reproduce the baseline locally

1. Clone upstream Py-BOBYQA and check out the `v1.2` tag:
   ```
   git clone https://github.com/numericalalgorithmsgroup/pybobyqa
   cd pybobyqa && git checkout v1.2
   ```
2. Apply the included patch from the cloned repo's root:
   ```
   patch -p1 < /path/to/this/repo/patches/pybobyqa_HermiteLS_v1.2.patch
   ```
3. Rename the resulting `pybobyqa/` package directory to `pybobyqa_HermiteLS` and place its parent
   directory on `PYTHONPATH` so your own driver code can `import pybobyqa_HermiteLS`.

The patch was generated by diffing the actual modified fork against a fresh checkout of upstream
tag `v1.2` (commit `2ae99d565f4ed03165e177c71583e81b9d9b93ab`), after normalizing the fork's CRLF
line endings to LF, and was verified by applying it to a clean `v1.2` checkout and confirming the
result matches the fork byte-for-byte. `patch -p1` from the cloned repo's root (as in step 2 above)
applies it without any manual edits.
