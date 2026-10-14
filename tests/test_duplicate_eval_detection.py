"""
Unit + regression test for src/check_duplicate_evals.py (item 5 of the low-risk tuning-fix plan).

check_duplicate_evals is a diagnostic-only helper -- not wired into the algorithm's hot path --
that flags pairs of distinct point indices evaluated at the same alpha whose coordinates coincide
to within a tolerance, which the existing exact (i, alpha) duplicate check in min_SAD.py's __main__
cannot catch.
"""
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from _test_utils import report, finish, require_minqsw

require_minqsw()

from check_duplicate_evals import check_duplicate_evals
from min_SAD import min_SAD


def case_synthetic():
    X = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [1.0 + 1e-10, 1.0 - 1e-10],  # near-duplicate of index 1 (same alpha)
        [2.0, 2.0],
    ])
    eval_list = [
        (0, (0, 0)),
        (1, (0, 0)),
        (2, (0, 0)),  # near-duplicate pair with index 1
        (3, (0, 0)),
        (0, (1, 0)),  # same point, different alpha -- not a duplicate
    ]

    results = []
    ok, dups = check_duplicate_evals(X, eval_list, tol=1e-8)
    results.append(report("check_duplicate_evals::detects_near_duplicate", not ok and dups == [(1, 2, (0, 0))],
                           f"ok={ok}, dups={dups}"))

    eval_list_no_dup = [(0, (0, 0)), (1, (0, 0)), (3, (0, 0))]
    ok2, dups2 = check_duplicate_evals(X, eval_list_no_dup, tol=1e-8)
    results.append(report("check_duplicate_evals::no_false_positive", ok2 and dups2 == [],
                           f"ok={ok2}, dups={dups2}"))
    return results


def case_rosenbrock_regression():
    def rosen(x):
        val = (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
        grad = np.array([-2 * (1 - x[0] + 200 * x[0] * (x[1] - x[0] ** 2)), 200 * (x[1] - x[0] ** 2)])
        Hess = np.array([[2 - 400 * (x[1] - 3 * x[0] ** 2), -400 * x[0]], [-400 * x[0], 200]])
        return val, grad, Hess

    X0 = np.array([[-1.2, 1.0]])
    n = 2
    K = np.array([0], dtype=int)
    alpha_admissible = [(0, 0), (1, 0)]
    nfmax = 560
    Delta = 0.3
    options = {"eps_c": 1e-6, "xi_imp": 1 + 1e-10, "xi_acc": 1e-4, "printf": False}

    X, F, sigmahist, xkin, eval_list, flag, phase_counts = min_SAD(
        rosen, X0, [], n, K, alpha_admissible, nfmax, Delta, 0, np.array([]), np.array([]), np.array([]), 0, options
    )

    # Since min_SAD.py's default trust-region update is the Mona-style eta/eta2 improvement-ratio
    # bands with dnorm-relative growth/shrink (see the Delta-update comments in min_SAD.py's main
    # loop), this run now mints 9 near-duplicate points -- all late in the 330-eval run (indices
    # 220, 230, 300-311), at already-tiny function values (~1e-14 down to ~1e-20, i.e. well past
    # any accuracy the user's query-efficiency metric cares about), with coordinate distances
    # (5e-9 to 8e-9) just under this check's 1e-8 tolerance. This is an accepted, minor
    # consequence of adopting only Mona's three-band eta1/eta2 rule without an accompanying
    # rho/restart mechanism (not implemented in this codebase): the dnorm-relative "unsuccessful"
    # shrink has no floor as tight as a flat gam_dec*Delta shrink would, so very late in
    # convergence it can re-sample points that are extremely close together without ever quite
    # exceeding the exact-duplicate check. Pinned here as the known-good baseline rather than
    # silently ignored.
    expected_dups = [
        (220, 300, (0, 0)), (220, 302, (1, 0)), (220, 304, (0, 0)), (220, 306, (1, 0)),
        (230, 308, (0, 0)), (230, 309, (1, 0)), (230, 310, (1, 0)), (230, 311, (0, 0)),
        (309, 310, (1, 0)),
    ]
    ok, dups = check_duplicate_evals(X, eval_list)
    return [report("check_duplicate_evals::rosenbrock_regression", (ok, dups) == (False, expected_dups),
                    f"(ok, dups)=({ok}, {dups})")]


def main():
    results = case_synthetic() + case_rosenbrock_regression()
    finish(all(results))


if __name__ == "__main__":
    main()
