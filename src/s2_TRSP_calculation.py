import numpy as np
import trustregion

try:
    from minqsw import minqsw
except ModuleNotFoundError as e:
    print(e)
    sys.exit("Ensure a python implementation of MINQ is available. For example, clone https://github.com/POptUS/minq and add minq/py/minq5 to the PYTHONPATH environment variable")

from call_fun_script import call_fun_script

##########################################
def s2_TRSP_calculation(fun, g, H, Delta, alpha_admissible, n, X, F, DF, D2F, eval_counter, flag, nfs, nfmax, Data, xkin, sigma, it, printf, plotf):
    """
    Solve the trust-region subproblem, evaluate fun at the resulting trial point, and compute
    the improvement ratio rho_ratio. main.tex's Algorithm 3 states this step abstractly as
    "compute a step minimizing m^(k) over Ball(x^(k), Delta^(k))"; this implementation is more
    specific than that description:

    - It solves the subproblem TWO ways -- `trustregion.solve` (an exact/near-exact 2-norm-ball
      TRSP solver) and `minqsw` (a box-constrained ["-Delta,Delta]^n] QP solver), with the box
      solution then projected onto the 2-norm sphere (stry2 = Delta*Xsp/||Xsp||) -- and keeps
      whichever candidate step gives the lower model value. This dual-solver-and-pick-best
      strategy is a code-level enhancement not described anywhere in main.tex.
    - modelxk is taken to be 0 since the model's constant term cancels in the rho_ratio
      difference, matching main.tex's rho^(k) = [f(x)-f(x+s)] / [m(x)-m(x+s)] formula exactly.
    - flag=6 is a sanity check not in the paper's pseudocode: it fires if the "minimizing" step
      nonetheless has a higher model value than the model center, which should not happen for a
      correct solve and signals a numerical/solver problem upstream.

    Handoff contract: on return, if exiting=False, rho_ratio is a finite float and X/F/DF/D2F/
    eval_counter/nfs reflect exactly one new function evaluation at the trial point (Data is
    NOT modified here -- the accept/reject step decides how Data changes in response to
    rho_ratio). If exiting=True, flag explains why (sigma-valued flag = nfmax exceeded before
    evaluating the trial point; flag=4 = Inf/NaN from fun; flag=6 = the sanity check above).
    """

    exiting = False # will the optimization algorithm terminate after this subroutine?
    rho_ratio = np.nan
    ftry = np.nan

    stry = trustregion.solve(g, H, Delta) # the normalized step, looking at B(0, Delta) after centering
    [Xsp, _, minq_err, _] = minqsw(0, g, H, -Delta*np.ones(n), Delta*np.ones(n), 0, np.zeros((n, 1)))
    Xsp = Xsp.squeeze()
    stry2 = Delta*Xsp/np.linalg.norm(Xsp) # Box-norm solution projected to two-norm sphere

    modeltry = stry.dot(0.5 * H.dot(stry) + g)  # model at candidate point
    modeltry2 = stry2.dot(0.5 * H.dot(stry2) + g)  # model at candidate point

    if modeltry2 < modeltry:
        stry = stry2
        modeltry = modeltry2

    xtry = X[xkin] + stry  # the corresponding point, candidate for a new iterate

    # does the new candidate iter exhaust the max iteration count?
    if nfs+1 >= nfmax:
        
        if printf:
            print("maximum number of function evaluations exceeded") 
        flag = sigma
        exiting = True
        return X, F, DF, D2F, eval_counter, nfs, stry, ftry, rho_ratio, exiting, flag

    # count this function value eval
    eval_counter.count(nfs, (0,)*n, phase="trial_step") # before nfs is incremented in the function caller

    # evaluate function and store new point in history
    X[nfs] = xtry
    F, DF, D2F, flag, nfs = call_fun_script(fun, xtry, alpha_admissible, n, F, DF, D2F, flag, nfs)  # evaluate the derivatives of a potential iterate here
    ftry = F[nfs - 1]  # nfs is incremented in call_fun_script

    if flag == 4:
        exiting = True
        return X, F, DF, D2F, eval_counter, nfs, stry, ftry, rho_ratio, exiting, flag

    modelxk = 0  # model value at x_k; constant c not necessary due to differencing

    if modeltry > modelxk:

        if printf:
            print("Alert! The TRSP returned a point with model value greater than the model center")
        exiting = True
        flag = 6
        return X, F, DF, D2F, eval_counter, nfs, stry, ftry, rho_ratio, exiting, flag

    rho_ratio = (F[xkin] - ftry) / (modelxk - modeltry)

    return X, F, DF, D2F, eval_counter, nfs, stry, ftry, rho_ratio, exiting, flag
