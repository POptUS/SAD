import numpy as np
from complete_nonpoised_set import complete_nonpoised_set
from model_update import model_update

def s3a_accept(fun, sigma, Delta, n, X, F, DF, D2F, eval_counter, flag, nfs, nfmax, Data, K, alpha_admissible, xkin, Deltamax, rho_ratio, r, xi_acc, eta, gam_inc, stry, tol_eq, tol_ineq, model_degree, printf, weight_exponent=3, max_new_evals_per_iter=None):
    """
    Accept path (rho_ratio > 0.0): advance xkin to the newly evaluated trial point and rebuild
    Data from scratch around it, filtered from the FULL evaluation history (eval_counter.queries,
    which plays the role of main.tex's D-bar) restricted to points within r*Delta of the new
    xkin. This is where the code actually performs main.tex's "rerun Algorithm 1 on all of
    D-bar" step -- the reject path (s3r.py) does NOT do this full-history rebuild (see its own
    docstring for that paper-vs-code deviation), only this accept path does.

    Handoff contract: on return, Data has been rebuilt (via complete_nonpoised_set, Algorithm 1)
    and the model (g, H, sigma) has been rebuilt around it via model_update -- both to len(Data)
    == qn1 on success. Delta growth (the paper's "if ||s^(k)||=Delta^(k), increase Delta" line)
    is NOT handled in this function despite being conceptually part of "line 18" -- check the
    caller (min_SAD.py) for where Deltamax/gam_inc/stry are actually used for that decision.

    max_new_evals_per_iter cap-hit handling: if complete_nonpoised_set can't fully repopulate
    Data in one pass (model_flag==3), this function retries internally -- evaluating/tagging
    whatever was minted so far (via model_update's own model_flag==3 branch), then re-running
    complete_nonpoised_set with the enlarged candidate pool -- until Data is genuinely poised or
    a real exit condition is hit. This is deliberately NOT bubbled up to min_SAD.py's main loop
    as a `continue`: doing so would force a brand-new TRSP/accept-reject cycle for this same `it`
    using a stale (not-yet-grown) Delta before this accept event's own geometry finished, which
    silently changed the trajectory relative to an uncapped run (found via the
    max_new_evals_per_iter ablation's regression test). xkin/Delta are fixed for the whole loop
    below, so a capped run resolves to exactly the same final Data an uncapped call would.
    """
    # accept the point
    xkin = nfs - 1

    # add to the model set of points
    Data.append( (xkin, (0,)*n) )

    # complete potentially nonpoised set
    Data = [(i, alpha) for (i, alpha) in eval_counter.queries if np.linalg.norm(X[i] - X[xkin]) <= r*Delta + tol_ineq]
    X, Data, added_idx, _, model_flag = complete_nonpoised_set(X, Data, alpha_admissible, xkin, Delta, xi_acc, nfs, nfmax, model_degree, tol_ineq, printf, weight_exponent=weight_exponent, r=r, max_new_evals_per_iter=max_new_evals_per_iter)

    # entry_flag: model_update's success path returns `flag` unchanged from whatever was passed
    # in -- it only ever overwrites it on a genuine exit condition (4/5/sigma). On a cap-hit pass
    # it deliberately sets flag=3 as the retry signal for the loop below; feeding that 3 back in
    # as the NEXT call's input would leak into a final successful return (flag stuck at 3 with
    # exiting=False) since the success path never clears it. So every retry call must use the
    # original entry flag, not the previous iteration's returned flag.
    entry_flag = flag

    while True:
        g, H, sigma, X, F, DF, D2F, eval_counter, flag, nfs, exiting \
                = model_update(fun, sigma, Delta, n, X, F, DF, D2F, eval_counter, entry_flag, nfs, nfmax, Data, K, alpha_admissible, xkin, added_idx, model_flag, model_degree, phase_map="geometry_completion")

        if not (model_flag == 3 and flag == 3):
            break

        Data = [(i, alpha) for (i, alpha) in eval_counter.queries if np.linalg.norm(X[i] - X[xkin]) <= r*Delta + tol_ineq]
        X, Data, added_idx, _, model_flag = complete_nonpoised_set(X, Data, alpha_admissible, xkin, Delta, xi_acc, nfs, nfmax, model_degree, tol_ineq, printf, weight_exponent=weight_exponent, r=r, max_new_evals_per_iter=max_new_evals_per_iter)

    return g, H, sigma, Delta, X, F, DF, D2F, xkin, Data, eval_counter, flag, nfs, exiting
