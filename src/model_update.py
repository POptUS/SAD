import numpy as np
from get_interpolation_model import get_interpolation_model
from call_fun_script import call_fun_script


def model_update(fun, sigma, Delta, n, X, F, DF, D2F, eval_counter, flag, nfs, nfmax, Data, K, alpha_admissible, xkin, added_idx, model_flag, model_degree, phase_map="geometry_completion"):
    """
    Shared tail end of both the accept path (s3a_accept.py) and reject path (s3r.py): evaluate
    fun at every newly-added point (added_idx, from whichever complete_nonpoised_set/
    model_improvement call preceded this), record eval usage in eval_counter, then rebuild the
    interpolation model (get_interpolation_model) and stationarity measure sigma for the current
    Data. This is the "build m^(k) using D" / "sigma^(k) = max(||g||, -lambda_min(H))" step of
    main.tex's Algorithm 3.

    phase_map: either a single phase-name string applied uniformly to every (i, alpha) pair in
    Data (the common case -- geometry-completion-only callers), or a dict mapping a point index i
    to a phase name (the reject-path case, where model_improvement.py's added_idx is a mix of
    "geometry_completion" points from complete_nonpoised_set and at most one "model_improvement"
    point from its own improvement step). Indices already evaluated in an earlier iteration and
    simply reappearing in Data are never re-tagged/re-counted by EvalCounter's own dedup, so a
    missing index in a phase_map dict is harmless.

    Handoff contract: expects model_flag as produced by complete_nonpoised_set (0/1/2/3) and
    propagates it into flag (model_flag==1 -> flag holds sigma's current value, the nfmax-
    exceeded overload used throughout this codebase; model_flag==2 -> flag=7). Also sets flag=5
    if get_interpolation_model reports sigma is None (singular interpolation matrix even after
    its own fallback re-solve). On any of these exit paths, exiting=True and g/H are left as
    zeros -- callers must check exiting before trusting g/H/sigma.

    model_flag==3 (max_new_evals_per_iter cap hit inside complete_nonpoised_set, geometry not
    yet fully poised): evaluates the points in added_idx and tags them in eval_counter exactly
    like the normal path, but does NOT build an interpolation model (Data is not yet poised, so
    get_interpolation_model would be meaningless) and does NOT touch sigma -- it is returned
    unchanged from whatever was passed in. exiting=True here does not mean "terminate the
    algorithm": callers (min_SAD.py's main loop) must check for flag==3 specifically and treat it
    as "retry this step on another pass" rather than a real exit condition.
    """

    exiting = False
    g = np.zeros(n)
    H = np.zeros((n,n))

    if model_flag == 1:
        print("Maximum number of function evaluations exceeded.")
        flag = sigma
        exiting = True
        return g, H, sigma, X, F, DF, D2F, eval_counter, flag, nfs, exiting
    elif model_flag == 2:
        print("Encountered an invalid model.")
        flag = 7
        exiting = True
        return g, H, sigma, X, F, DF, D2F, eval_counter, flag, nfs, exiting
    elif model_flag == 3:
        # partial geometry completion -- evaluate/tag the new point(s) minted so far, but do not
        # build a model on a not-yet-poised Data; signal the caller to retry via flag==3.
        for i in added_idx:
            F, DF, D2F, flag, nfs = call_fun_script(fun, X[i], alpha_admissible, n, F, DF, D2F, flag, nfs)
            if nfs >= nfmax:
                print("maximum number of function evaluations exceeded")
                flag = sigma
                exiting = True
                return g, H, sigma, X, F, DF, D2F, eval_counter, flag, nfs, exiting
        for (i, alpha) in Data:
            phase = phase_map[i] if isinstance(phase_map, dict) and i in phase_map else (
                phase_map if not isinstance(phase_map, dict) else "geometry_completion"
            )
            eval_counter.count(i, alpha, phase=phase)
        if flag == 4:
            exiting = True
            return g, H, sigma, X, F, DF, D2F, eval_counter, flag, nfs, exiting
        flag = 3
        exiting = True
        return g, H, sigma, X, F, DF, D2F, eval_counter, flag, nfs, exiting
    # evaluate the function at the new points
    for i in added_idx:
        F, DF, D2F, flag, nfs = call_fun_script(fun, X[i], alpha_admissible, n, F, DF, D2F, flag, nfs)

        if nfs >= nfmax:
            print("maximum number of function evaluations exceeded")
            flag = sigma
            exiting = True
            return g, H, sigma, X, F, DF, D2F, eval_counter, flag, nfs, exiting
    # update usage history for counting f evals
    for (i, alpha) in Data:
        phase = phase_map[i] if isinstance(phase_map, dict) and i in phase_map else (
            phase_map if not isinstance(phase_map, dict) else "geometry_completion"
        )
        eval_counter.count(i, alpha, phase=phase)
    if flag == 4:
        exiting = True
        return g, H, sigma, X, F, DF, D2F, eval_counter, flag, nfs, exiting

    # make the interpolation model
    # points are normalized by their trust region radius
    if model_degree == 2:
        _, g, H, sigma = get_interpolation_model(X, K, n, F, DF, D2F, xkin, Data, Delta, model_degree, vec_out=False)
    elif model_degree == 1:
        _, g, H, sigma = get_interpolation_model(X, K, n, F, DF, D2F, xkin, Data, Delta, model_degree, vec_out=False)
    if sigma is None:
        flag = 5
        exiting = True
        return g, H, sigma, X, F, DF, D2F, eval_counter, flag, nfs, exiting

    return g, H, sigma, X, F, DF, D2F, eval_counter, flag, nfs, exiting
