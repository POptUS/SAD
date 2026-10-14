"""
End-to-end regression test of the full min_SAD main loop -- the strongest test in this suite
since it exercises every module in the call chain together (criticality_step, s3a_accept, s3r,
model_improvement, model_validity, min_quadratic, ...), the same way min_SAD.py's own __main__
Rosenbrock smoke test does.

Two cases:
1. The Rosenbrock baseline from min_SAD.py's own __main__ block, pinned to its known-good
   output (len(eval_list)=330, F[xkin]=2.755382252749607e-20, no repeated evals) -- this is the
   exact regression check that caught (and now guards) the s3a_accept.py positional-argument bug.
   Re-pinned when min_SAD.py adopted the Mona-style trust-region update (eta/eta2
   improvement-ratio bands, dnorm-relative growth/shrink) as its default; the old values
   (400 / 9.6469871096959e-24) reflect the pre-Mona-style TR rule and are no longer produced by
   this code.
2. A convex quadratic with the full gradient+Hessian known (K = all indices): since a degree-2
   Birkhoff model of a quadratic function is exact everywhere (see test_quadratic_exactness.py),
   min_SAD should locate the true minimizer to tight tolerance using very few extra evaluations.
"""
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from _test_utils import report, finish, require_minqsw, check_no_repeated_evals

require_minqsw()

from min_SAD import min_SAD


def case_rosenbrock_baseline():
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

    results = []
    results.append(report("min_SAD_e2e::rosenbrock_eval_count", len(eval_list) == 330,
                           f"len(eval_list)={len(eval_list)} (expected 330)"))
    results.append(report("min_SAD_e2e::rosenbrock_objective", abs(F[xkin] - 2.755382252749607e-20) < 1e-30,
                           f"F[xkin]={F[xkin]!r} (expected 2.755382252749607e-20)"))
    ok, dups = check_no_repeated_evals(X, eval_list)
    results.append(report("min_SAD_e2e::rosenbrock_no_repeated_evals", ok, f"{len(dups)} duplicate(s)" if dups else ""))
    return results


def case_convex_quadratic():
    A = np.diag([2.0, 3.0])
    x_star = np.zeros(2)

    def fun(x):
        val = 0.5 * x @ A @ x
        grad = A @ x
        Hess = A
        return val, grad, Hess

    X0 = np.array([[3.0, -2.0]])
    n = 2
    K = np.array([0, 1], dtype=int)
    alpha_admissible = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]
    nfmax = 50
    Delta = 1.0
    options = {"eps_c": 1e-10, "xi_imp": 1 + 1e-10, "xi_acc": 1e-4, "printf": False}

    X, F, sigmahist, xkin, eval_list, flag, phase_counts = min_SAD(
        fun, X0, [], n, K, alpha_admissible, nfmax, Delta, 0, np.array([]), np.array([]), np.array([]), 0, options
    )

    results = []
    results.append(report("min_SAD_e2e::quadratic_converges", F[xkin] < 1e-8,
                           f"F[xkin]={F[xkin]!r} (expected near 0)"))
    results.append(report("min_SAD_e2e::quadratic_within_budget", len(eval_list) <= nfmax,
                           f"len(eval_list)={len(eval_list)} (budget {nfmax})"))
    ok, dups = check_no_repeated_evals(X, eval_list)
    results.append(report("min_SAD_e2e::quadratic_no_repeated_evals", ok, f"{len(dups)} duplicate(s)" if dups else ""))
    return results


def main():
    results = case_rosenbrock_baseline() + case_convex_quadratic()
    finish(all(results))


if __name__ == "__main__":
    main()
