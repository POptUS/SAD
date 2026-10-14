"""
Adapts criticality_step.py's own __main__ smoke test into assertions. main.tex's criticality
step must only ever shrink Delta (never grow it), terminate within its internal maxit safety
cap, and its final Delta must equal min(max(Deltanew, beta*sigma), Delta) -- the code's own
completion formula, checked here by recomputing it independently from the returned sigmanew.
"""
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from _test_utils import report, finish, require_minqsw

require_minqsw()

from criticality_step import criticality_step
from get_interpolation_model import get_interpolation_model
from symmetric_matrix2vec import symmetric_matrix2vec
from min_SAD import EvalCounter


def fun(x):
    g = np.array([1.0, 2.0])
    H = np.array([[3.0, 1.0], [1.0, 2.0]])
    val = np.dot(x, g + 0.5 * H @ x) + x[0] ** 3
    grad = H @ x + g + np.array([3 * x[0] ** 2, 0])
    Hess = H.copy()
    Hess[0, 0] += 6 * x[0]
    return val, grad, Hess


def run():
    n = 2
    qn1 = (n + 1) * (n + 2) // 2

    t1 = np.pi / 4
    t2 = np.pi / 4 + 5.0e-1
    rad = 1.0
    X = rad * np.array([[0.0, 0.0], [np.cos(t1), np.sin(t1)], [np.cos(t2), np.sin(t2)]])
    alpha_list = [(0, 0), (1, 0), (0, 0), (1, 0), (0, 0), (1, 0)]

    npad = 10
    X = np.append(X, np.zeros((qn1, n)), axis=0)
    X = np.append(np.zeros((npad, n)), X, axis=0)
    nfmax = len(X)
    nfs = 3 + npad

    Data = [
        (0 + npad, alpha_list[0]), (0 + npad, alpha_list[1]),
        (1 + npad, alpha_list[2]), (1 + npad, alpha_list[3]),
        (2 + npad, alpha_list[4]), (2 + npad, alpha_list[5]),
    ]
    xkin = 0 + npad
    K = np.array([0])
    alpha_admissible = [(0, 0), (1, 0)]
    Delta = 1.0
    eval_counter = EvalCounter()

    nK = len(K)
    F = np.zeros(len(X))
    DF = np.zeros((len(X), nK))
    D2F = np.zeros((len(X), nK * (nK + 1) // 2))
    for i in range(npad, nfs):
        val, grad, Hess = fun(X[i])
        F[i] = val
        DF[i, :] = grad[K]
        D2F[i, :] = symmetric_matrix2vec(Hess[np.ix_(K, K)], nK)

    c, g, H, sigma = get_interpolation_model(X, K, n, F, DF, D2F, xkin, Data, scalefactor=Delta, model_degree=2, vec_out=False)

    omega, mu, flag, r, eps_c, beta, xi_imp, xi_acc, tol_ineq = 0.6, 0.9, 0, 1.0, 0, 0.5, 1.1, 0.05, 0

    (Xnew, Fnew, DFnew, D2Fnew, eval_counter, nfs_new, gnew, Hnew,
     sigmanew, Deltanew, Datanew, flagnew) = criticality_step(
        fun, g, H, sigma, Delta, omega, mu, n, X, F, DF, D2F, eval_counter, flag, nfs, nfmax,
        Data, K, alpha_admissible, xkin, r, eps_c, beta, xi_imp, xi_acc, tol_ineq, model_degree=2
    )
    return Delta, Deltanew, sigmanew, flagnew, Datanew, qn1


def main():
    results = []

    Delta, Deltanew, sigmanew, flagnew, Datanew, qn1 = run()

    results.append(report("criticality_step::shrinks_or_holds", 0 < Deltanew <= Delta,
                           f"Delta={Delta} -> Deltanew={Deltanew}"))

    results.append(report("criticality_step::recognized_flag", flagnew in (0, 1, 4, 5, 7) or flagnew == sigmanew,
                           f"flag={flagnew}"))

    if flagnew == 0:
        beta = 0.5
        mu = 0.9
        # Loop only exits normally (not via the nfmax/invalid-model/singular early returns) when
        # flagnew retains its initial value of 0; in that case the completion clamp must hold.
        expected = min(max(Deltanew, beta * sigmanew), Delta)
        results.append(report("criticality_step::delta_clamp_formula", abs(Deltanew - expected) < 1e-12,
                               f"Deltanew={Deltanew}, expected clamp={expected}"))
        results.append(report("criticality_step::data_size_invariant", len(Datanew) == qn1,
                               f"len(Datanew)={len(Datanew)} (expected {qn1})"))

    finish(all(results))


if __name__ == "__main__":
    main()
