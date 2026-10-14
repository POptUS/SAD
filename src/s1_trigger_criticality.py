import numpy as np
from criticality_step import criticality_step
from prepare_outputs_before_return import prepare_outputs_before_return


####################################
def s1_trigger_criticality(fun, g, H, sigma, sigmahist, Delta, crit_omega, Deltamin, mu, n, X, F, DF, D2F, eval_counter, flag, nfs, nfmax, Data, K, alpha_admissible, xkin, r, eps_c, beta, xi_imp, xi_acc, tol_ineq, model_degree, printf, weight_exponent=3, max_new_evals_per_iter=None):
    """
    Wraps criticality_step (the mechanics of main.tex's criticality step) and dispatches its
    outcome into a termination decision -- the "exit-condition" half of Algorithm 3's
    criticality-step logic. Always runs criticality_step once; whether that step actually did
    anything depends on the mu*sigma <= Delta condition checked inside criticality_step itself.

    Handoff contract -- how each criticality_step outcome is interpreted here:
    - flag == sigma (the flag-holds-sigma overload: on nfmax-exceeded inside criticality_step,
      flag is literally set equal to sigma's current value rather than an integer code) -->
      exiting=True, sigmahist appended, printed message about exceeding nfmax.
    - flag in [4, 5, 7] (Inf/NaN eval, singular interpolation matrix, unrecoverable poisedness
      failure respectively) --> exiting=True immediately, no further checks.
    - sigma <= eps_c --> convergence declared: flag reset to 0, exiting=True.
    - flag == 1 (Delta shrank below threshold inside criticality_step without sigma becoming
      small enough) --> exiting=True, flag passed through unchanged.
    - otherwise, if the (possibly-updated) Delta <= Deltamin --> flag=2, exiting=True.
    - else --> exiting=False, caller (min_SAD.py) proceeds to the trust-region-subproblem step.
    """

    exiting = False # will the optimization algorithm terminate after this subroutine?

    X, F, DF, D2F, eval_counter, nfs, g, H, sigma, Delta, Data, flag = criticality_step(
        fun, g, H, sigma, Delta, crit_omega, mu, n, X, F, DF, D2F, eval_counter, flag, nfs, nfmax,
        Data, K, alpha_admissible, xkin, r, eps_c, beta, xi_imp, xi_acc, tol_ineq, model_degree, printf,
        weight_exponent=weight_exponent, max_new_evals_per_iter=max_new_evals_per_iter
    )

    # max_new_evals_per_iter cap hit inside criticality_step's own model_improvement call: Data is
    # only partially poised, and Delta was intentionally left untouched. Signal the caller
    # (min_SAD.py) to retry via flag==3 rather than treating this as a termination condition.
    if flag == 3:
        exiting = True
        return X, F, DF, D2F, eval_counter, nfs, g, H, sigma, sigmahist, Delta, Data, flag, exiting

    # parse exiting conditions from criticality
    if flag == sigma:

        sigmahist = np.append(sigmahist, sigma)
        exiting = True
        if printf:
            print("Maximum number of function evaluations exceeded.")
            print("Terminating.")
        
        return X, F, DF, D2F, eval_counter, nfs, g, H, sigma, sigmahist, Delta, Data, flag, exiting

    if flag in [4, 5, 7]:
        exiting = True
        return X, F, DF, D2F, eval_counter, nfs, g, H, sigma, sigmahist, Delta, Data, flag, exiting

    # declare convergence if model optimality "sigma" is still small after model improvement during criticality
    if sigma <= eps_c:

        sigmahist = np.append(sigmahist, sigma)
        flag = 0
        exiting = True
        if printf:
            print("Terminating.")

        return X, F, DF, D2F, eval_counter, nfs, g, H, sigma, sigmahist, Delta, Data, flag, exiting

    elif flag == 1:  # Delta was small but sigma is still not sufficiently small

        sigmahist = np.append(sigmahist, sigma)
        exiting = True
        return X, F, DF, D2F, eval_counter, nfs, g, H, sigma, sigmahist, Delta, Data, flag, exiting

    # after criticality, check if TR radius is too small
    if Delta <= Deltamin:

        sigmahist = np.append(sigmahist, sigma)
        flag = 2
        exiting = True
        if printf:
            print("Delta has become too small. Terminating.")
        return X, F, DF, D2F, eval_counter, nfs, g, H, sigma, sigmahist, Delta, Data, flag, exiting

    return X, F, DF, D2F, eval_counter, nfs, g, H, sigma, sigmahist, Delta, Data, flag, exiting

# end of file
