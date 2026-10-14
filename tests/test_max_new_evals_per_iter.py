"""
Regression test for the max_new_evals_per_iter / trace_iterations ablation added to min_SAD.py
(the "incremental geometry completion" plan). Capping new-point-minting to 1 per pass only
throttles how many outer-loop passes complete_nonpoised_set needs to fully repopulate Data --
it must not change WHICH points/derivatives end up selected, since Delta and xkin are left
untouched (the main loop `continue`s) on every model_flag==3 partial pass. So a capped run
should reach an identical trajectory to the uncapped baseline: same eval_list, same final
objective, same exit flag -- just more (repeated) `it` values recorded in the trace while a
given outer iteration's geometry is being completed incrementally.
"""
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "benchmark"))

from _test_utils import report, finish, require_minqsw, check_no_repeated_evals

require_minqsw()

from min_SAD import min_SAD
from make_alpha_admissible_from_K import make_alpha_admissible_from_K
import s3a_accept
import s3r
import criticality_step


def _count_internal_cap_hits():
    """
    trace_iterations only records the mandatory top-of-loop ("line 2") complete_nonpoised_set
    call. Since the max_new_evals_per_iter fix, cap hits inside s3a_accept.py's/s3r.py's own
    complete_nonpoised_set call and criticality_step.py's/s3r.py's model_improvement call are
    resolved via a LOCAL retry loop and never bubble a model_flag==3 up to the trace -- that's
    the whole point of the fix (see s3a_accept.py's docstring). So to confirm the cap actually
    engages somewhere, monkeypatch those call sites to count model_flag==3 returns directly,
    restoring the originals afterward.
    """
    counts = {"n": 0}

    orig_cnps = s3a_accept.complete_nonpoised_set
    def counting_cnps(*args, **kwargs):
        result = orig_cnps(*args, **kwargs)
        if result[-1] == 3:
            counts["n"] += 1
        return result
    s3a_accept.complete_nonpoised_set = counting_cnps

    orig_mi_r = s3r.model_improvement
    def counting_mi_r(*args, **kwargs):
        result = orig_mi_r(*args, **kwargs)
        if result[3] == 3:
            counts["n"] += 1
        return result
    s3r.model_improvement = counting_mi_r

    orig_mi_c = criticality_step.model_improvement
    def counting_mi_c(*args, **kwargs):
        result = orig_mi_c(*args, **kwargs)
        if result[3] == 3:
            counts["n"] += 1
        return result
    criticality_step.model_improvement = counting_mi_c

    def restore():
        s3a_accept.complete_nonpoised_set = orig_cnps
        s3r.model_improvement = orig_mi_r
        criticality_step.model_improvement = orig_mi_c

    return counts, restore


def case_capped_matches_uncapped_quadratic():
    # n=2 with fully-known derivatives no longer stresses the cap under min_SAD.py's default
    # Mona-style trust-region update (eta/eta2 improvement-ratio bands, dnorm-relative growth/
    # shrink -- see the comments around the Delta update in min_SAD.py's main loop): it converges
    # to Data being poised in one top-of-loop pass with zero brand-new evaluations, since every
    # needed (point, alpha) pivot is already satisfied by known-derivative info at already-
    # evaluated points. n=4 still needs several genuinely new evaluations per geometry-completion
    # pass, so the cap keeps engaging.
    # asymmetric A/X0 (no repeated eigenvalues or matching-magnitude coordinates): a symmetric
    # choice here produces a tied pivot at one point where complete_nonpoised_set's greedy
    # selection legitimately picks a different (but equally valid) alpha under capping, which
    # would make same_eval_set flaky without indicating an actual bug.
    A = np.diag([2.0, 3.3, 1.7, 2.9])

    def fun(x):
        val = 0.5 * x @ A @ x
        grad = A @ x
        Hess = A
        return val, grad, Hess

    X0 = np.array([[3.1, -2.3, 1.7, -1.1]])
    n = 4
    K = np.arange(n, dtype=int)
    alpha_admissible = make_alpha_admissible_from_K(K, n, 2)
    nfmax = 80
    Delta = 1.0
    base_options = {"eps_c": 1e-10, "xi_imp": 1 + 1e-10, "xi_acc": 1e-4, "printf": False}

    X_u, F_u, _, xkin_u, eval_list_u, flag_u, phase_counts_u = min_SAD(
        fun, X0, [], n, K, alpha_admissible, nfmax, Delta, 0, np.array([]), np.array([]), np.array([]), 0,
        {**base_options}
    )

    capped_options = {**base_options, "max_new_evals_per_iter": 1, "trace_iterations": True}
    cap_counts, restore_cap_counting = _count_internal_cap_hits()
    try:
        X_c, F_c, _, xkin_c, eval_list_c, flag_c, phase_counts_c, trace = min_SAD(
            fun, X0, [], n, K, alpha_admissible, nfmax, Delta, 0, np.array([]), np.array([]), np.array([]), 0,
            capped_options
        )
    finally:
        restore_cap_counting()

    results = []

    results.append(report("max_new_evals_per_iter::same_eval_set",
                           set(eval_list_u) == set(eval_list_c),
                           f"len(uncapped)={len(eval_list_u)}, len(capped)={len(eval_list_c)}"))

    results.append(report("max_new_evals_per_iter::same_objective",
                           abs(F_u[xkin_u] - F_c[xkin_c]) < 1e-15,
                           f"F_u={F_u[xkin_u]!r}, F_c={F_c[xkin_c]!r}"))

    results.append(report("max_new_evals_per_iter::same_flag", flag_u == flag_c,
                           f"flag_u={flag_u}, flag_c={flag_c}"))

    results.append(report("max_new_evals_per_iter::same_phase_counts", phase_counts_u == phase_counts_c,
                           f"uncapped={phase_counts_u}, capped={phase_counts_c}"))

    ok, dups = check_no_repeated_evals(X_c, eval_list_c)
    results.append(report("max_new_evals_per_iter::capped_no_repeated_evals", ok,
                           f"{len(dups)} duplicate(s)" if dups else ""))

    # every capped call minted at most 1 new point per pass
    max_added = max((rec["n_added"] for rec in trace), default=0)
    results.append(report("max_new_evals_per_iter::trace_respects_cap", max_added <= 1,
                           f"max n_added in trace = {max_added}"))

    # Delta/xkin must be unchanged across consecutive retries of the SAME outer iteration
    # (model_flag==3 passes never advance `it`, since the main loop `continue`s before the
    # bottom-of-loop `it += 1`).
    stable = True
    detail = ""
    for prev, cur in zip(trace, trace[1:]):
        if prev["model_flag"] == 3 and cur["it"] == prev["it"]:
            if cur["Delta"] != prev["Delta"] or cur["xkin"] != prev["xkin"]:
                stable = False
                detail = f"it={prev['it']}: Delta {prev['Delta']}->{cur['Delta']}, xkin {prev['xkin']}->{cur['xkin']}"
                break
    results.append(report("max_new_evals_per_iter::delta_xkin_stable_during_partial_passes", stable, detail))

    # sanity: the cap actually engaged at least once on this problem (qn1=6 > 1, so the first
    # iteration's geometry completion needs multiple passes) -- otherwise the test above would
    # be vacuously true. Cap hits are resolved internally within s3a_accept/s3r/criticality_step
    # now (see _count_internal_cap_hits' docstring), so check the monkeypatched counter rather
    # than the trace, which only ever sees the mandatory top-of-loop call.
    results.append(report("max_new_evals_per_iter::cap_actually_engaged", cap_counts["n"] > 0,
                           f"{cap_counts['n']} internal partial pass(es) recorded"))

    return results


def main():
    results = case_capped_matches_uncapped_quadratic()
    finish(all(results))


if __name__ == "__main__":
    main()
