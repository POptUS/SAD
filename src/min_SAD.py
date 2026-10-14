import os
import sys
from collections import defaultdict
import numpy as np

try:
    from minqsw import minqsw
except ModuleNotFoundError as e:
    print(e)
    sys.exit("Ensure a python implementation of MINQ is available. For example, clone https://github.com/POptUS/minq and add minq/py/minq5 to the PYTHONPATH environment variable")

from call_fun_script import call_fun_script
from complete_nonpoised_set import complete_nonpoised_set
from prepare_outputs_before_return import prepare_outputs_before_return
from model_validity import model_validity
from model_update import model_update

# main loop subroutines
from s1_trigger_criticality import s1_trigger_criticality
from s2_TRSP_calculation import s2_TRSP_calculation
from s3a_accept import s3a_accept
from s3r import s3r

###
class EvalCounter:
    def __init__(self, Pre_Evaluated_Pairs=[]):
        self.queries = Pre_Evaluated_Pairs
        self.phase_counts = defaultdict(int)

    def count(self, i, alpha, phase="unspecified"):
        if not( (i, alpha) in self.queries ):
            self.queries.append( (i, alpha) )
            self.phase_counts[phase] += 1

###

def exit_protocol(X, F, it, nfs, xkin, Delta, rho_ratio, sigma, g, H, stry, flag, printf):
    if (printf and (sigma is not None)):
        print(
            "Iter: %4i  num_fun_evals: %4i  fun value: %1.6e  Delta: %1.1e  rho: %+1.3e  sigma: %1.3e  ||g||: %1.3e  cond(H): %1.3e  xkin: %4i  ||s||: %1.1e"
            % (it, nfs, F[xkin], Delta, rho_ratio, sigma, np.linalg.norm(g), np.linalg.cond(H), xkin, np.linalg.norm(stry))
        )

    X, F, flag = prepare_outputs_before_return(X, F, nfs, flag)
    return X, F, flag



def min_SAD(fun, X0, Pre_Evaluated_Pairs, n, K, alpha_admissible, nfmax, Delta, nfs, F0, DF0, D2F0, xkin=0, options=None):
    """
    Calling sequence:
        [X, F, sigmahist, xkin, flag] = min_SAD(fun, X0, n, K, alpha_admissible, nfmax, Delta, nfs, F0, DF0, D2F0, xkin, options)

    This code minimizes output from a blackbox function, solving
    min { f(X) }
    where the user-provided blackbox f is specified in the handle fun. Evaluation
    of this f must result in the return of a scalar, and gradient and Hessian information if applicable.
    In each iteration, the algorithm forms an interpolating quadratic model
    of the function from function evaluations and first- and second-derivative
    and minimizes it in a 2-norm trust region.

    -INPUTS:--------------------------------------------------------
    fun     [f h] objective function with calling sequence
                [val, gradK, HessKK] = fun(x)
    X0      [dbl] [max(nfs,1)-by-n] Set of initial points  (zeros(1,n))
    n       [int] Dimension (number of continuous variables)
    K       [int] [nK] indices of grad(fun) which are available
    alpha_admissible [list of tuples] derivative multi-indices which are available for fun
    nfmax   [int] Maximum number of function evaluations (>n+1) (100)
    Delta   [dbl] Positive trust region radius
    nfs     [int] Number of function values (at points in X0) known in advance (0)
    F0      [dbl] [nfs] Set of known function values  ([])
    DF0     [dbl] [ndfs-by-nK] set of known first derivative values ([])
    D2F0    [dbl] [ndfs-by-nK*(nK+1)/2] ''' second derivative ''''' ([])
    xkin    [int] Index of point in X0 serving as the initial model center (0)
    printf  [bool] False: No printing to screen (default)
                   True: Debugging level of output to screen after each iter

    -OUTPUTS:--------------------------------------------------------
    X       [dbl] [(nfmax+nfs)-by-n] Locations of evaluated points
    F       [dbl] [nfmax+nfs] Function values of evaluated points
    sigmahist [dbl] [nfmax+nfs] stationarity/optimality metrics of the model at each iter
    flag    [dbl] Termination criteria flag:
                  = 0 normal termination because of small sigma,
                  > 0 exceeded nfmax evals,   flag = sigma at final X
                  = 1 trust radius smaller than Deltamin 
                  = 2 trust radius small during criticality step
                  = 3 inappropriate user input
                  = 4 encountered an Inf or NaN from function evaluation
                  = 5 error during model construction (nonpoised data)
                  = 8 a geometry/QP-subproblem invariant broke down from floating-point
                      degeneracy (Delta shrank below the machine-precision floor set by the
                      magnitude of the current iterate); best point found so far is returned
    xkin    [int] Index of point in X representing approximate minimizer

    This is Algorithm 3 in the paper.
    """

    ###########################################################################################
    ###########################################################################################
    # set algorithm parameters
    ###########################################################################################
    ###########################################################################################

    default_options = {
        "Deltamax"  : 1e3 * Delta,   # maximum trust region radius
        "Deltamin"  : 1e-14 * Delta, # minimum '''''''''''''''''''
        "eta"       : 0.1,           # Mona-style eta1: below this improvement ratio a step is "unsuccessful" (resizes Delta down); no longer gates whether xkin moves
        "eta2"      : 0.7,           # Mona-style eta2: above this improvement ratio a step is "very successful" and Delta grows more aggressively
        "gam_dec"   : 0.5,           # factor by which to reduce/hold the trust region radius, 0<gam_dec<1 (Mona noiseless default)
        "gam_inc"   : 2.0,           # '''''''''''''''''' expand ''''''''''''''''''''''', 1<gam_inc
        "gam_inc_overline": 4.0,     # dnorm-relative growth factor applied on very successful (ratio>eta2) steps
        "crit_omega": 0.6,           # factor by which criticality_step shrinks Delta each of its iterations, 0<crit_omega<1
        "eps_c"     : 1e-6,          # enter criticality step if norm(grad) at TR center is <= eps_c
        "mu"        : 0.9,           # factor to determine whether to run the criticality step
        "beta"      : 0.5,           # factor to determine the new trust radius in criticality step
        "xi_acc"    : 0.05,          # pivot threshold
        "xi_imp"    : 1.1,           # model-improvement threshold
        "r"         : 1.0,           # trust-region radius factor
        "tol_ineq"  : 1e-8,          # tolerance for model_improvement to detect when points are within Delta-ball
        "tol_eq"    : 1e-14,         # tolerance for satisfying equality that ||step||=Delta to enlarge TR radius
        "tol_step_ratio" : 0.1,      # tolerance on the ratio between step size and Delta
        "weight_exponent": 3,        # exponent in the pivot distance-weighting penalty max(Delta/scalefactor, dist)**weight_exponent
        "model_degree"   : 2,
        "reuse_existing_points": False,  # ablation: model_improvement prefers an already-evaluated point over minting a new one when comparably good
        "full_rebuild_on_reject": False, # ablation: s3r rebuilds Data from full eval history (like s3a_accept) instead of extending by one point
        "max_new_evals_per_iter": None,  # ablation: cap new-point-minting in complete_nonpoised_set to this many per pass (None = unlimited); when hit, model_flag=3 is returned and the main loop retries on another pass instead of shrinking Delta/moving xkin
        "trace_iterations": False,      # diagnostic: record a per-pass trace (it, Delta, xkin, len(Data), phase_counts, ...), returned as an 8th tuple element
        "printf"         : False,
        "plotf"          : False,
        }
    
    # ensure a dictionary
    options = options or {}

    # Merge user options with defaults
    opts = {**default_options, **options}

    # read inputs
    Deltamax    = opts["Deltamax"]
    Deltamin    = opts["Deltamin"]
    eta         = opts["eta"     ]
    eta2        = opts["eta2"    ]
    gam_dec     = opts["gam_dec" ]
    gam_inc     = opts["gam_inc" ]
    gam_inc_overline = opts["gam_inc_overline"]
    crit_omega  = opts["crit_omega"]
    eps_c       = opts["eps_c"   ]
    mu          = opts["mu"      ]
    beta        = opts["beta"    ]
    xi_acc      = opts["xi_acc"  ]
    xi_imp      = opts["xi_imp"  ]
    r           = opts["r"       ]
    tol_ineq    = opts["tol_ineq"]
    tol_eq      = opts["tol_eq"  ]
    tol_step_ratio = opts["tol_step_ratio"]
    weight_exponent = opts["weight_exponent"]
    model_degree   = opts["model_degree"]
    reuse_existing_points = opts["reuse_existing_points"]
    full_rebuild_on_reject = opts["full_rebuild_on_reject"]
    max_new_evals_per_iter = opts["max_new_evals_per_iter"]
    trace_iterations = opts["trace_iterations"]
    printf      = opts["printf"]
    plotf       = opts["plotf"]

    assert model_degree == 2


    ###########################################################################################
    ###########################################################################################
    # Initialization and setup
    ###########################################################################################
    ###########################################################################################
 
    eval_counter = EvalCounter(Pre_Evaluated_Pairs)

    # important numbers
    qn1 = (n + 1) * (n + 2) // 2
    nK = len(K)
    n_HessKK = nK * (nK + 1) // 2  # number of entries in KK block of Hessian
    if model_degree == 2:
        n_interp = qn1
    elif model_degree == 1:
        n_interp = n + 1

    # initialize exit flag
    flag = 0

    sigmahist = np.array([], dtype=float)

    if nfmax < nfs + n_interp:  # nfmax inappropriate from user input
        print("nfmax is too small to create an initial model. Number of function evaluations required = {n_interp}.")
        flag = 3
        if printf:
            print("Terminating.")
        X = X0
        F = F0
        xkin = None
        if trace_iterations:
            return X, F, sigmahist, xkin, eval_counter.queries, flag, dict(eval_counter.phase_counts), []
        return X, F, sigmahist, xkin, eval_counter.queries, flag, dict(eval_counter.phase_counts)

    if printf:
        print(f"r = {r}")

    ###########################################################################################
    ###########################################################################################
    # process initial points and function values
    ###########################################################################################
    ###########################################################################################

    if nfs == 0:

        X = np.vstack((X0, np.zeros((nfmax - 1, n))))
        F = np.zeros(nfmax)
        DF = np.zeros((nfmax, nK))
        D2F = np.zeros((nfmax, n_HessKK))
        eval_counter.count(nfs, (0,)*n, phase="geometry_completion")
        F, DF, D2F, flag, nfs = call_fun_script(fun, X[nfs], alpha_admissible, n, F, DF, D2F, flag, nfs)
        if printf:
            print("%4i    Initial point  %11.5e\n" % (0, F[0]))  # nfs-1 since it was updated in the call script

        if flag == 4:
            X = X[0:nfs]
            F = F[0:nfs]
            xkin = None
            if trace_iterations:
                return X, F, sigmahist, xkin, eval_counter.queries, flag, dict(eval_counter.phase_counts), []
            return X, F, sigmahist, xkin, eval_counter.queries, flag, dict(eval_counter.phase_counts)

    else:

        X = np.vstack((X0[0 : max(1, nfs), :], np.zeros((nfmax, n))))
        F = np.concatenate((F0[0:nfs], np.zeros(nfmax)))
        DF = np.vstack( (DF0[0:ndfs], np.zeros((nfmax, nK))) )
        D2F = np.vstack((D2F0[0:ndfs], np.zeros((nfmax, n_HessKK))))
        nfmax += nfs
        if printf:
            print("%4i    Initial point  %11.5e\n" % (0, F[xkin]))

    Data = [(i, alpha) for (i, alpha) in eval_counter.queries if np.linalg.norm(X[i] - X[xkin]) <= r*Delta + tol_ineq]

    ###########################################################################################
    ###########################################################################################
    # main loop
    ###########################################################################################
    ###########################################################################################

    it = 1
    rho_ratio = np.nan
    sigma = None
    g = None
    H = None
    stry = np.zeros(n)
    trace = [] if trace_iterations else None
    prev_xkin = xkin

    while nfs <= nfmax:  # (it <= maxit) and (nfs <= nfmax):

        # line 2
        data_len_before_line2 = len(Data) if trace_iterations else None
        try:
            X, Data, added_idx, _, model_flag = complete_nonpoised_set(X, Data, alpha_admissible, xkin, Delta, xi_acc, nfs, nfmax, model_degree, tol_ineq, printf, weight_exponent=weight_exponent, r=r, max_new_evals_per_iter=max_new_evals_per_iter)
        except (AssertionError, ValueError, SystemExit) as e:
            if printf:
                print(f"Numerical degeneracy near the floating-point precision floor ({type(e).__name__}: {e}).")
            flag = 8
            X, F, flag = exit_protocol(X, F, it, nfs, xkin, Delta, rho_ratio, sigma, g, H, stry, flag, printf)
            break

        if trace_iterations:
            trace.append({
                "it": it,
                "Delta": Delta,
                "xkin": xkin,
                "xkin_changed": xkin != prev_xkin,
                "data_len_before": data_len_before_line2,
                "data_len_after": len(Data),
                "n_added": len(added_idx),
                "model_flag": model_flag,
                "phase_counts": dict(eval_counter.phase_counts),
            })
            prev_xkin = xkin

        if model_flag in (1, 2):

            if printf:
                print(
                    "Iter: %4i  num_fun_evals: %4i  fun value: %1.6e  Delta: %1.1e  rho: %+1.3e  sigma: %1.3e  ||g||: %1.3e  cond(H): %1.3e  xkin: %4i  ||s||: %1.1e"
                    % (it, nfs, F[xkin], Delta, rho_ratio, sigma, np.linalg.norm(g), np.linalg.cond(H), xkin, np.linalg.norm(stry))
                )

            if model_flag == 1:
                flag = sigma
            elif model_flag == 2:
                flag = 7

            X, F, flag = prepare_outputs_before_return(X, F, nfs, flag)
            break

        g, H, sigma, X, F, DF, D2F, eval_counter, flag, nfs, exiting \
            = model_update(fun, sigma, Delta, n, X, F, DF, D2F, eval_counter, flag, nfs, nfmax, Data, K, alpha_admissible, xkin, added_idx, model_flag, model_degree, phase_map="geometry_completion")

        if exiting and flag == 3:
            # max_new_evals_per_iter cap hit (model_flag==3): geometry is not yet fully poised.
            # The new point(s) minted this pass have been evaluated/tagged above; retry
            # complete_nonpoised_set on another pass rather than treating this as an exit --
            # Delta/xkin are untouched, so the trust region cannot shrink/move until poisedness
            # is eventually certified (model_flag==0).
            # Reset flag to 0 before retrying: unlike every other "special" flag value in this
            # loop (4, 5, 7, sigma-as-flag), flag==3 is NOT immediately followed by a break, so
            # it must not be left stale in `flag` -- otherwise a later, unrelated exiting=True
            # would be misread as another cap-hit retry by the flag==3 checks below/downstream.
            # Rebuild Data from the full eval history (same filter as the pre-loop init above and
            # s3a_accept.py's own rebuild): complete_nonpoised_set only returns the partial pivot
            # list it managed to select before the cap hit, NOT the rest of Data_in it was given
            # (the broader pool of already-evaluated (i, alpha) candidates near xkin). Without this
            # rebuild, that pool is silently lost on every retry pass, forcing needless new-point
            # minting instead of the "just spread the same work over more passes" the ablation
            # assumes.
            Data = [(i, alpha) for (i, alpha) in eval_counter.queries if np.linalg.norm(X[i] - X[xkin]) <= r*Delta + tol_ineq]
            flag = 0
            continue

        if exiting:
            X, F, flag = exit_protocol(X, F, it, nfs, xkin, Delta, rho_ratio, sigma, g, H, stry, flag, printf)
            break

        sigmahist = np.append(sigmahist, sigma)

        #########################################################################
        # 1. Criticality step
        #########################################################################

        # enter criticality step if stationarity condition is satisfied
        if (sigma <= eps_c) and (r * Delta > mu * sigma):
            # lines 5--9
            try:
                X, F, DF, D2F, eval_counter, nfs, g, H, sigma, sigmahist, Delta, Data, flag, exiting \
                    = s1_trigger_criticality(fun, g, H, sigma, sigmahist, Delta, crit_omega, Deltamin, mu, n, X, F, DF, D2F,
                            eval_counter, flag, nfs, nfmax, Data, K, alpha_admissible, xkin, r, eps_c, beta, xi_imp, xi_acc, tol_ineq, model_degree, printf,
                            weight_exponent=weight_exponent, max_new_evals_per_iter=max_new_evals_per_iter)
            except (AssertionError, ValueError, SystemExit) as e:
                if printf:
                    print(f"Numerical degeneracy near the floating-point precision floor ({type(e).__name__}: {e}).")
                flag = 8
                X, F, flag = exit_protocol(X, F, it, nfs, xkin, Delta, rho_ratio, sigma, g, H, stry, flag, printf)
                break

        # NOTE: criticality_step.py now resolves any max_new_evals_per_iter cap hit inside its own
        # model_improvement call internally (a retry loop keeping Deltanew frozen until poised),
        # so flag==3 can no longer reach s1_trigger_criticality's return here.

        if exiting:
            X, F, flag = exit_protocol(X, F, it, nfs, xkin, Delta, rho_ratio, sigma, g, H, stry, flag, printf)
            break


        #########################################################################
        # 2. TRSP Step Calculation
        #########################################################################

        # lines 10--12
        try:
            X, F, DF, D2F, eval_counter, nfs, stry, ftry, rho_ratio, exiting, flag \
                    = s2_TRSP_calculation(fun, g, H, Delta, alpha_admissible, n, X, F, DF, D2F, eval_counter, flag, nfs, nfmax, Data, xkin, sigma, it, printf, plotf)
        except (AssertionError, ValueError, SystemExit) as e:
            if printf:
                print(f"Numerical degeneracy near the floating-point precision floor ({type(e).__name__}: {e}).")
            flag = 8
            X, F, flag = exit_protocol(X, F, it, nfs, xkin, Delta, rho_ratio, sigma, g, H, stry, flag, printf)
            break

        if exiting:
            X, F, flag = exit_protocol(X, F, it, nfs, xkin, Delta, rho_ratio, sigma, g, H, stry, flag, printf)
            break

        #########################################################################
        # 3. Model Update
        #########################################################################

        # print out status of current iteration
        if printf:
            print(
                "Iter: %4i  num_fun_evals: %4i  fun value: %1.6e  Delta: %1.1e  rho: %+1.3e  sigma: %1.3e  ||g||: %1.3e  cond(H): %1.3e  xkin: %4i  ||s||: %1.1e"
                % (it, nfs, F[xkin], Delta, rho_ratio, sigma, np.linalg.norm(g), np.linalg.cond(H), xkin, np.linalg.norm(stry))
            )

        # Did the TRSP point lead to decrease?
        # NOTE: I have not changed the calling sequences during refactoring, so there may be redundancies.

        dnorm = float(np.linalg.norm(stry))
        move_dec = rho_ratio > 0.0  # Mona-style: move to the new point on any real decrease
        valid, dists = model_validity(Delta, X, Data, xkin, r, tol_ineq)  # kept for the trace only

        if trace_iterations:
            trace[-1].update({
                "stry_norm": dnorm,
                "rho_ratio": float(rho_ratio) if rho_ratio is not None and not np.isnan(rho_ratio) else None,
                "move_dec": bool(move_dec),
                "valid": bool(valid),
            })

        # line 13
        try:
            if move_dec: # function value went down

                # line 14
                assert len(Data) == qn1
                g, H, sigma, Delta, X, F, DF, D2F, xkin, Data, eval_counter, flag, nfs, exiting \
                        = s3a_accept(fun, sigma, Delta, n, X, F, DF, D2F, eval_counter, flag, nfs, nfmax, Data, K, alpha_admissible, xkin,
                                Deltamax, rho_ratio, r, xi_acc, eta, gam_inc, stry, tol_eq, tol_ineq, model_degree, printf,
                                weight_exponent=weight_exponent, max_new_evals_per_iter=max_new_evals_per_iter)

            else: # reject the new point and improve the model

                g, H, sigma, X, F, DF, D2F, eval_counter, flag, nfs, Data, exiting \
                     = s3r(fun, sigma, Delta, n, X, F, DF, D2F, eval_counter, flag, nfs, nfmax, Data, K, alpha_admissible, xkin,
                             added_idx, model_flag, xi_imp, xi_acc, tol_ineq, model_degree, printf,
                             weight_exponent=weight_exponent, r=r, reuse_existing_points=reuse_existing_points,
                             full_rebuild_on_reject=full_rebuild_on_reject, max_new_evals_per_iter=max_new_evals_per_iter)
        except (AssertionError, ValueError, SystemExit) as e:
            if printf:
                print(f"Numerical degeneracy near the floating-point precision floor ({type(e).__name__}: {e}).")
            flag = 8
            X, F, flag = exit_protocol(X, F, it, nfs, xkin, Delta, rho_ratio, sigma, g, H, stry, flag, printf)
            break

        # NOTE: s3a_accept.py/s3r.py now resolve any max_new_evals_per_iter cap hit inside their
        # own complete_nonpoised_set/model_improvement calls internally (a retry loop keeping
        # Delta/xkin frozen until poised), so flag==3 can no longer reach here -- the growth/
        # shrink decision below always applies to a genuinely finished accept/reject event.

        if exiting:
            X, F, flag = exit_protocol(X, F, it, nfs, xkin, Delta, rho_ratio, sigma, g, H, stry, flag, printf)
            break

        # Mona-style trust-region radius update (eta=eta1 / eta2 bands, dnorm-relative), decoupled
        # from the move decision above.
        if rho_ratio > eta2:  # very successful

            Delta = min(max(gam_inc * Delta, gam_inc_overline * dnorm), Deltamax)

        elif rho_ratio >= eta:  # acceptable

            Delta = max(gam_dec * Delta, dnorm)

        else:  # unsuccessful (rho_ratio < eta, including outright rejections)

            Delta = max(min(gam_dec * Delta, dnorm), Deltamin)
            if Delta == Deltamin:
                exiting = True

        if exiting:
            X, F, flag = exit_protocol(X, F, it, nfs, xkin, Delta, rho_ratio, sigma, g, H, stry, flag, printf)
            break

        #########################################################################
        # Prepare for next iteration
        #########################################################################

        sigmahist = np.append(sigmahist, sigma)
        it += 1

    # end main loop

    if trace_iterations:
        return X, F, sigmahist, xkin, eval_counter.queries, flag, dict(eval_counter.phase_counts), trace
    return X, F, sigmahist, xkin, eval_counter.queries, flag, dict(eval_counter.phase_counts)



if __name__ == "__main__":

    def rosen(x):
        val = (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
        grad = np.array([-2 * (1 - x[0] + 200 * x[0] * (x[1] - x[0] ** 2)), 200 * (x[1] - x[0] ** 2)])
        Hess = np.array([[2 - 400 * (x[1] - 3 * x[0] ** 2), -400 * x[0]], [-400 * x[0], 200]])
        return val, grad, Hess


    X0 = np.array([[-1.2, 1.0]])
    Pre_Evaluated_Pairs = []
    n = 2
    K = np.array([0], dtype=int)
    alpha_admissible = [ (0,0), (1,0) ]
    nfmax = 560
    Delta = 0.3
    nfs = 0
    F0 = np.array([])
    DF0 = np.array([])
    D2F0 = np.array([])
    xkin = 0
    options = {
        "eps_c"     : 1e-6,             # enter criticality step if norm(grad) at TR center is <= eps_c
        "xi_imp"    : 1+1e-10,
        "xi_acc"    : 1e-4,
        "printf"    : True,
        }

    X, F, sigmahist, xkin, eval_list, flag, phase_counts = min_SAD(rosen, X0, Pre_Evaluated_Pairs, n, K, alpha_admissible, nfmax, Delta, nfs, F0, DF0, D2F0, xkin, options)
    print("Done")

    print("eval history =")
    print(eval_list)
    print(f"len(eval_list) = {len(eval_list)}")
    print(f"Final objective value = {F[xkin]}")
    print(f"phase_counts = {phase_counts}")

    # Take a look at the eval history and check for any repeats
    X_alpha_all = [(X[i].tolist(), alpha) for i, alpha in eval_list]
    seen = []
    np.set_printoptions(precision=15) # threshold=np.inf)
    for i,tup in enumerate(X_alpha_all):
        if tup in seen:
            j = X_alpha_all.index(tup)
            ix = eval_list[i][0]
            jx = eval_list[j][0]
            print(f"Evals {i} and {j} are the same: {eval_list[i]} vs. {eval_list[j]}.")
            print(f"X[{ix}] = {X[ix]}")
            print(f"X[{jx}] = {X[jx]}")
        else:
            seen.append(tup)
    repeat = not( len(seen)==len(X_alpha_all) )
    print(f"Are there evals repeated in the count? {repeat}")

    # Diagnostic-only: also check for near-duplicate evals (same alpha, coordinates
    # within tol but not an exact repeat of an already-known (i, alpha) pair).
    from check_duplicate_evals import check_duplicate_evals
    near_ok, near_dups = check_duplicate_evals(X, eval_list)
    print(f"Are there near-duplicate evals (within tol)? {not near_ok}")
    if near_dups:
        print(f"Near-duplicate (i, j, alpha) tuples: {near_dups}")
