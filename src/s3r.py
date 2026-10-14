import numpy as np
from model_update import model_update
from model_improvement import model_improvement

def s3r(fun, sigma, Delta, n, X, F, DF, D2F, eval_counter, flag, nfs, nfmax, Data, K, alpha_admissible, xkin, added_idx, model_flag, xi_imp, xi_acc, tol_ineq, model_degree, printf, weight_exponent=3, r=1.0, reuse_existing_points=False, full_rebuild_on_reject=False, max_new_evals_per_iter=None):
    """
    Reject path (rho_ratio < eta): xkin does NOT move. Improve model geometry instead, by
    appending the just-rejected trial point to the CURRENT Data (not a fresh filter of the full
    evaluation history) and running model_improvement (Algorithm 2) on that.

    This is the key place this codebase deviates from main.tex's literal Algorithm 3: the paper
    reruns Algorithm 1 on the full accumulated history D-bar at the top of every iteration,
    accept or reject. This function instead only ever extends the previous iteration's already-
    poised Data by one point -- points that fell out of Data on an earlier iteration are not
    reconsidered as pivot candidates again until the next s3a_accept.py call rebuilds Data from
    eval_counter.queries. Likely an intentional efficiency trade-off (avoids an O(|D-bar|) scan
    every reject), but a genuine behavioral difference from the paper.

    full_rebuild_on_reject (ablation, default False): when True, mirrors s3a_accept.py's rebuild
    -- Data is rebuilt from eval_counter.queries filtered to r*Delta of xkin -- instead of just
    appending the one rejected point, testing whether that documented paper-vs-code deviation is
    itself a source of min_SAD's extra evaluations.

    reuse_existing_points (ablation, default False): forwarded to model_improvement.py's own
    reuse-vs-mint ablation of the same name.

    Handoff contract: entry and exit both require len(Data) == qn1 (asserted); xkin is passed
    through unchanged in the return signature (contrast with s3a_accept.py, which returns a new
    xkin) since this is exactly why the caller (min_SAD.py) does not reassign xkin on this path.

    max_new_evals_per_iter cap-hit handling: if model_improvement's own complete_nonpoised_set
    call can't fully repopulate Data in one pass (model_flag==3), this function retries
    internally -- see the matching comment in s3a_accept.py for why this must not bubble up to
    min_SAD.py's main loop as a `continue` (it would skip this reject event's own Delta-shrink
    decision). The retry pool passed back into model_improvement is the union of this call's
    ORIGINAL candidate pool (the previous poised Data, or the full history if
    full_rebuild_on_reject) with whatever partial progress the interrupted attempt already made
    -- NOT eval_counter's full history unconditionally, since that would silently make every
    cap-interrupted reject behave like full_rebuild_on_reject=True regardless of that option.
    """

    assert(len(Data)==(n+1)*(n+2)//2)
    if full_rebuild_on_reject:
        Data = [(i, alpha) for (i, alpha) in eval_counter.queries if np.linalg.norm(X[i] - X[xkin]) <= r*Delta + tol_ineq]
    else:
        # potentially add candidate point to model geometry
        Data.append( (nfs - 1, (0,)*n) )

    candidate_pool = list(Data)

    # improve the model
    X, Data, added_idx, model_flag, phase_map = model_improvement(X, Data, alpha_admissible, xkin, Delta, xi_imp, xi_acc, nfs, nfmax, tol_ineq, model_degree, printf, weight_exponent=weight_exponent, r=r, reuse_existing_points=reuse_existing_points, eval_counter=eval_counter, max_new_evals_per_iter=max_new_evals_per_iter)

    # entry_flag: model_update's success path returns `flag` unchanged from whatever was passed
    # in -- it only ever overwrites it on a genuine exit condition (4/5/sigma). On a cap-hit pass
    # it deliberately sets flag=3 as the retry signal for the loop below; feeding that 3 back in
    # as the NEXT call's input would leak into a final successful return (flag stuck at 3 with
    # exiting=False) since the success path never clears it. So every retry call must use the
    # original entry flag, not the previous iteration's returned flag.
    entry_flag = flag

    while True:
        # model_flag==3 (max_new_evals_per_iter cap hit): Data is intentionally partial here, so
        # the usual "Data is poised" invariant does not hold yet -- skip the assert on this pass.
        if model_flag != 3:
            assert(len(Data)==(n+1)*(n+2)//2)
        g, H, sigma, X, F, DF, D2F, eval_counter, flag, nfs, exiting \
                = model_update(fun, sigma, Delta, n, X, F, DF, D2F, eval_counter, entry_flag, nfs, nfmax, Data, K, alpha_admissible, xkin, added_idx, model_flag, model_degree, phase_map=phase_map)

        if not (model_flag == 3 and flag == 3):
            break

        candidate_pool = list(dict.fromkeys(candidate_pool + Data))
        X, Data, added_idx, model_flag, phase_map = model_improvement(X, candidate_pool, alpha_admissible, xkin, Delta, xi_imp, xi_acc, nfs, nfmax, tol_ineq, model_degree, printf, weight_exponent=weight_exponent, r=r, reuse_existing_points=reuse_existing_points, eval_counter=eval_counter, max_new_evals_per_iter=max_new_evals_per_iter)

    if flag != 3:
        assert(len(Data)==(n+1)*(n+2)//2)
    return g, H, sigma, X, F, DF, D2F, eval_counter, flag, nfs, Data, exiting
