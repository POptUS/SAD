"""
Ground-truth test derived from main.tex's interpolation theory: a degree-2 Hermite/Birkhoff
interpolation model of a function that IS a quadratic must reproduce that quadratic's (c, g, H)
exactly (to machine precision), everywhere -- not merely satisfy the interpolation conditions at
the nodes used to build it (which get_interpolation_model.py's own __main__ block checks, but
that check is tautological: it compares the fitted model against the SAME data used to fit it,
which any linear solve satisfies up to solver error even for a badly-chosen model).

This exercises the real call chain: complete_nonpoised_set (Algorithm 1) picks a poised set of
points/derivatives from scratch, then get_interpolation_model solves the Hermite system for
(c, g, H) -- across several (n, K, model_degree) configurations.
"""
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "benchmark"))

from _test_utils import report, finish, require_minqsw

require_minqsw()

from complete_nonpoised_set import complete_nonpoised_set
from get_interpolation_model import get_interpolation_model
from make_alpha_admissible_from_K import make_alpha_admissible_from_K
from symmetric_matrix2vec import symmetric_matrix2vec


def make_quadratic(n, seed):
    rng = np.random.default_rng(seed)
    c0 = float(rng.standard_normal())
    g0 = rng.standard_normal(n)
    A = rng.standard_normal((n, n))
    H0 = A + A.T  # symmetric

    def fun(x):
        val = c0 + g0 @ x + 0.5 * x @ H0 @ x
        grad = g0 + H0 @ x
        return val, grad, H0

    return fun, c0, g0, H0


def run_case(n, K, seed, Delta=1.0, xi_acc=1e-2):
    model_degree = 2
    qn1 = (n + 1) * (n + 2) // 2
    fun, c0, g0, H0 = make_quadratic(n, seed)
    alpha_admissible = make_alpha_admissible_from_K(K, n, max_deriv=2)

    # Start from just the model center at the origin; complete_nonpoised_set must add
    # whatever points/derivatives it needs to reach a poised size-qn1 set on its own.
    xkin = 0
    X = np.zeros((1 + qn1, n))
    nfs = 1
    nfmax = len(X)
    Data_in = [(0, (0,) * n)]

    X, Data, added_idx, _, model_flag = complete_nonpoised_set(
        X, Data_in, alpha_admissible, xkin, Delta, xi_acc, nfs, nfmax, model_degree=model_degree
    )
    if model_flag != 0:
        return False, f"n={n} K={K}: complete_nonpoised_set returned model_flag={model_flag}"
    if len(Data) != qn1:
        return False, f"n={n} K={K}: len(Data)={len(Data)} != qn1={qn1}"

    nK = len(K)
    npts = 1 + len(added_idx)
    F = np.zeros(len(X))
    DF = np.zeros((len(X), nK))
    D2F = np.zeros((len(X), nK * (nK + 1) // 2))

    for i in range(npts):
        val, grad, Hess = fun(X[i])
        F[i] = val
        if nK:
            DF[i, :] = grad[K]
            D2F[i, :] = symmetric_matrix2vec(Hess[np.ix_(K, K)], nK)

    c, g, H = get_interpolation_model(
        X, K, n, F, DF, D2F, xkin, Data, scalefactor=Delta, model_degree=model_degree, vec_out=False
    )[:3]

    if c is None:
        return False, f"n={n} K={K}: get_interpolation_model reported a singular system"

    tol = 1e-7
    ok = abs(c - c0) < tol and np.allclose(g, g0, atol=tol) and np.allclose(H, H0, atol=tol)
    detail = f"n={n} K={K}: |c-c0|={abs(c - c0):.2e}, ||g-g0||={np.linalg.norm(g - g0):.2e}, ||H-H0||={np.linalg.norm(H - H0):.2e}"
    return ok, detail


def main():
    cases = [
        (2, [], 0),
        (2, [0], 1),
        (2, [0, 1], 2),
        (3, [], 3),
        (3, [1], 4),
        (3, [0, 1, 2], 5),
    ]
    results = []
    for n, K, seed in cases:
        ok, detail = run_case(n, K, seed)
        results.append(report(f"quadratic_exactness n={n} K={K}", ok, detail))

    finish(all(results))


if __name__ == "__main__":
    main()
