"""
Adapts model_improvement.py's own __main__ example (points deliberately close together, close
to non-poised) into assertions, plus a second case that exercises the "no improvement found"
no-op path -- where main.tex's Algorithm 2 would signal a distinct "failure" outcome but this
implementation silently returns Datanew exactly as complete_nonpoised_set produced it (see
model_improvement.py's own docstring) -- to confirm it returns cleanly (len(Data) == qn1, no
exception/hang) rather than silently corrupting Data.
"""
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from _test_utils import report, finish, require_minqsw

require_minqsw()

from model_improvement import model_improvement
from get_interpolation_matrix import get_interpolation_matrix

COND_THRESHOLD = 1e8


def build_case():
    n = 2
    qn1 = (n + 1) * (n + 2) // 2
    npad = 1

    t1 = np.pi / 4
    t2 = np.pi / 4 + 1.0e-1

    X = np.array([[0.0, 0.0], [np.cos(t1), np.sin(t1)], [np.cos(t2), np.sin(t2)]])
    alpha_list = [(0, 0), (1, 0), (0, 0), (1, 0), (0, 0), (1, 0)]
    X = np.append(np.zeros((npad, n)), X, axis=0)
    Data = [
        (0 + npad, alpha_list[0]), (0 + npad, alpha_list[1]),
        (1 + npad, alpha_list[2]), (1 + npad, alpha_list[3]),
        (2 + npad, alpha_list[4]), (2 + npad, alpha_list[5]),
    ]
    nfs = len(X)
    alpha_admissible = [(0, 0), (1, 0)]
    X = np.append(X, np.zeros((qn1, n)), axis=0)
    xkin = 0 + npad
    Delta = 1.0
    nfmax = len(X)
    return X, Data, alpha_admissible, xkin, Delta, nfs, nfmax, n, qn1


def case_normal_improvement():
    X, Data, alpha_admissible, xkin, Delta, nfs, nfmax, n, qn1 = build_case()
    xi_imp = 1.10
    xi_acc = 1e-2

    Xnew, Datanew, added_idx, flag, phase_map = model_improvement(
        X, Data, alpha_admissible, xkin, Delta, xi_imp, xi_acc, nfs, nfmax, tol_ineq=1e-2, model_degree=2
    )

    ok = flag == 0 and len(Datanew) == qn1 and len(set(Datanew)) == len(Datanew)
    cond = np.linalg.cond(get_interpolation_matrix(Xnew, xkin, Datanew, scalefactor=Delta, model_degree=2))
    ok = ok and np.isfinite(cond) and cond < COND_THRESHOLD
    return ok, f"normal_improvement: flag={flag}, len(Datanew)={len(Datanew)}, cond={cond:.2e}"


def case_no_improvement_found():
    """xi_imp set absurdly high so no replacement pivot can clear the improvement threshold --
    exercises the documented no-flag-on-failure no-op path without crashing."""
    X, Data, alpha_admissible, xkin, Delta, nfs, nfmax, n, qn1 = build_case()
    xi_imp = 1e12
    xi_acc = 1e-2

    Xnew, Datanew, added_idx, flag, phase_map = model_improvement(
        X, Data, alpha_admissible, xkin, Delta, xi_imp, xi_acc, nfs, nfmax, tol_ineq=1e-2, model_degree=2
    )

    ok = flag == 0 and len(Datanew) == qn1
    return ok, f"no_improvement_found: flag={flag}, len(Datanew)={len(Datanew)} (expected {qn1})"


def main():
    results = []
    for fn in (case_normal_improvement, case_no_improvement_found):
        ok, detail = fn()
        results.append(report(f"model_improvement::{fn.__name__}", ok, detail))
    finish(all(results))


if __name__ == "__main__":
    main()
