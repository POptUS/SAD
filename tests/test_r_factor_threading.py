"""
Regression test for item 4 of the low-risk tuning-fix plan: complete_nonpoised_set.py's own
"remove points outside trust region" filter used plain `Delta`, discarding the `r` factor that
its caller (s3a_accept.py) already applies when building the candidate set. This meant a
candidate point between Delta and r*Delta from xkin was always excluded, regardless of r,
silently making r do nothing anywhere except model_validity.py's accept/reject decision.

Exercises the fixed filter (`r*Delta + tol_ineq`) directly: a candidate point at 1.3*Delta from
xkin must be excluded from consideration with r=1.0 (forcing 3 new evaluations to complete the
model) but eligible for reuse with r=2.0 (only 2 new evaluations needed, since the existing far
point fills one of the required rows).
"""
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from _test_utils import report, finish, require_minqsw

require_minqsw()

from complete_nonpoised_set import complete_nonpoised_set


def _run(r):
    n = 2
    alpha_admissible = [(0, 0), (1, 0)]
    xkin = 0
    Delta = 1.0
    xi_acc = 0.05

    X = np.zeros((10, n))
    X[1] = [0.5, 0.0]
    X[2] = [0.0, 0.5]
    X[3] = [1.3, 0.0]  # 1.3*Delta from xkin: excluded when r=1.0, eligible when r=2.0
    nfs = 4
    nfmax = 10
    Data_in = [(0, (0, 0)), (1, (0, 0)), (2, (0, 0)), (3, (0, 0))]

    X_, Data, added_idx, Upper, flag = complete_nonpoised_set(
        X, list(Data_in), alpha_admissible, xkin, Delta, xi_acc, nfs, nfmax, model_degree=2, r=r
    )
    far_point_reused = 3 in [i for i, _ in Data]
    return flag, far_point_reused, len(added_idx)


def main():
    results = []

    flag1, reused1, n_added1 = _run(r=1.0)
    results.append(report("r_factor::r1_excludes_far_point", flag1 == 0 and not reused1,
                           f"flag={flag1}, far_point_reused={reused1}, added={n_added1}"))

    flag2, reused2, n_added2 = _run(r=2.0)
    results.append(report("r_factor::r2_includes_far_point", flag2 == 0 and reused2,
                           f"flag={flag2}, far_point_reused={reused2}, added={n_added2}"))

    results.append(report("r_factor::r2_needs_fewer_new_evals", n_added2 < n_added1,
                           f"added_idx count: r=1.0 -> {n_added1}, r=2.0 -> {n_added2}"))

    finish(all(results))


if __name__ == "__main__":
    main()
