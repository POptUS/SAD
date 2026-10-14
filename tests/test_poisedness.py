"""
Adapts complete_nonpoised_set.py's own __main__ examples into assertions: for a few starting
configurations (points already scattered around the model center; and starting from the model
center alone), Algorithm 1 (complete_nonpoised_set) must return a poised set of exactly qn1
points/derivatives with a well-conditioned Birkhoff interpolation matrix, flag == 0.
"""
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from _test_utils import report, finish, require_minqsw

require_minqsw()

from complete_nonpoised_set import complete_nonpoised_set
from get_interpolation_matrix import get_interpolation_matrix

COND_THRESHOLD = 1e8


def check_poised(X, Data, added_idx, flag, xkin, Delta, model_degree, n, label):
    qn1 = (n + 1) * (n + 2) // 2
    ok = True
    detail_parts = []

    if flag != 0:
        return False, f"{label}: flag={flag} (expected 0)"

    if len(Data) != qn1:
        ok = False
        detail_parts.append(f"len(Data)={len(Data)} != qn1={qn1}")

    if len(set(Data)) != len(Data):
        ok = False
        detail_parts.append("Data contains duplicate (i, alpha) pairs")

    Mat = get_interpolation_matrix(X, xkin, Data, scalefactor=Delta, model_degree=model_degree)
    cond = np.linalg.cond(Mat)
    if not np.isfinite(cond) or cond > COND_THRESHOLD:
        ok = False
        detail_parts.append(f"cond(Mat)={cond:.2e} exceeds threshold {COND_THRESHOLD:.0e}")
    else:
        detail_parts.append(f"cond(Mat)={cond:.2e}")

    return ok, f"{label}: " + ", ".join(detail_parts)


def case_points_on_circle():
    n = 2
    qn1 = (n + 1) * (n + 2) // 2
    tt = [np.pi / 2, 3 * np.pi / 2]
    X = np.zeros((len(tt) + 1, n))
    rad = 1.0
    for i, t in enumerate(tt):
        X[i + 1] = rad * np.array([np.cos(t), np.sin(t)])
    nfs = len(X)
    X = np.append(X, np.zeros((qn1, n)), axis=0)
    nfmax = len(X)

    xkin = 0
    Delta = 1.0
    xi_acc = 0.05
    alpha_admissible = [(0, 0), (1, 0)]
    Data_in = [(i, alpha_admissible[0]) for i in range(nfs)]

    X, Data, added_idx, Upper, flag = complete_nonpoised_set(
        X, Data_in, alpha_admissible, xkin, Delta, xi_acc, nfs, nfmax, model_degree=2
    )
    return check_poised(X, Data, added_idx, flag, xkin, Delta, 2, n, "points_on_circle")


def case_center_only():
    n = 2
    qn1 = (n + 1) * (n + 2) // 2
    X = np.zeros((1, n))
    nfs = len(X)
    X = np.append(X, np.zeros((qn1, n)), axis=0)
    nfmax = len(X)

    xkin = 0
    Delta = 1.0
    xi_acc = 0.05
    alpha_admissible = [(0, 0), (1, 0)]
    Data_in = [(0, alpha_admissible[0])]

    X, Data, added_idx, Upper, flag = complete_nonpoised_set(
        X, Data_in, alpha_admissible, xkin, Delta, xi_acc, nfs, nfmax, model_degree=2
    )
    return check_poised(X, Data, added_idx, flag, xkin, Delta, 2, n, "center_only")


def case_nfmax_exceeded():
    """
    Regression check on the flag==1 path: if nfmax is too small to admit the points Algorithm 1
    needs to add, it must report flag==1 rather than silently truncating Data.
    """
    n = 2
    X = np.zeros((1, n))
    nfs = len(X)
    nfmax = nfs  # no room to add any new points
    xkin = 0
    Delta = 1.0
    xi_acc = 0.05
    alpha_admissible = [(0, 0), (1, 0)]
    Data_in = [(0, alpha_admissible[0])]

    X, Data, added_idx, Upper, flag = complete_nonpoised_set(
        X, Data_in, alpha_admissible, xkin, Delta, xi_acc, nfs, nfmax, model_degree=2
    )
    ok = flag == 1
    return ok, f"nfmax_exceeded: flag={flag} (expected 1)"


def main():
    results = []
    for fn in (case_points_on_circle, case_center_only, case_nfmax_exceeded):
        ok, detail = fn()
        results.append(report(f"poisedness::{fn.__name__}", ok, detail))
    finish(all(results))


if __name__ == "__main__":
    main()
