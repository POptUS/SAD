import numpy as np

from call_fun_script import call_fun_script
from get_interpolation_model import get_interpolation_model
from matrix2polynomial import matrix2polynomial
from model_improvement import model_improvement
from vec2symmetric_matrix import vec2symmetric_matrix

def criticality_step(fun, g, H, sigma, Delta, omega, mu, n, X, F, DF, D2F, eval_counter, flag, nfs, nfmax, Data, K, alpha_admissible, xkin, r, eps_c, beta, xi_imp, xi_acc, tol_ineq, model_degree=2, printf=True, weight_exponent=3, max_new_evals_per_iter=None):
    """
    Perform the criticality step: repeatedly tighten the model geometry (via model_improvement,
    Algorithm 2) and shrink Delta by omega until either Delta is small enough relative to the
    updated sigma (Deltanew < mu*sigma is no longer true) or a hard exit condition is hit. This
    is the mechanics underlying main.tex's Algorithm 3 criticality step; s1_trigger_criticality.py
    is the wrapper that decides whether to call this at all and how to interpret its outcome.

    Handoff contract:
    - On normal completion, returns Deltanew = min(max(Deltanew, beta*sigma), Delta) exactly as
      in main.tex's criticality-step radius update, plus the (possibly rebuilt) Data/model
      (gnew, Hnew, sigma) consistent with that Delta.
    - flag is left unchanged (whatever the caller passed in) unless one of these fires:
      flag <- sigma (nfmax exceeded inside model_improvement -- see min_SAD.py's flag-holds-sigma
      convention: on this path flag is NOT an integer code but literally equals the current sigma
      value, which is how s1_trigger_criticality.py's `if flag == sigma:` check detects it);
      flag=7 (model_improvement hit an unrecoverable poisedness failure, model_flag==2);
      flag=4 (fun returned Inf/NaN at a newly added point); flag=5 (interpolation matrix
      singular, sigma is None); flag=1 (Deltanew shrank below sqrt-threshold before mu*sigma
      condition was satisfied -- loop exits via `break`, not `return`, so the final
      Deltanew-clamp line above still runs).
    - NOTE: the model_improvement(...) call below hardcodes model_degree=2 regardless of this
      function's own model_degree argument. This is currently unreachable dead weight since
      min_SAD.py asserts model_degree==2 at the top level, but would be a bug if model_degree=1
      support is ever completed -- fix this call site first if that happens.

    INPUTS
    fun         function handle of objective
    coefvec     vector of current quadratic model coefficients
    Delta       current trust region radius
    omega       trust radius reduction factor
    mu          threshold for criticality (enter while Delta >= mu*sigma, where sigma is the stationarity measure)
    n           problem dimension
    X           history of points where fun is evaluated
    F           history of fun values
    DF          history of first derivative values
    D2F         '''''''''' second ''''''''''''''''
    flag        exit flag
    nfs         number of function evaluations so far in optimization algorithm
    nfmax       maximum number of function evals allowed
    K           known partial derivative indices
    xkin        current best iterate index within X (or F)
    dxkin       ''''''''''''''''''''''''''''''''' DF, D2F
    model_idx   indices of X contributing to current model of fun
    """

    # initialize the new model as the current model
    gnew = np.copy(g)
    Hnew = np.copy(H)

    Deltanew = Delta

    it = 0
    enter_once = True
    maxit = 1000 # adding to prevent infinite loop when sigma is very small

    while ( ( Deltanew >= mu * sigma or enter_once ) and (it < maxit) ):

        enter_once = False

        # Call the model improvement algorithm to improve poisedness on Ball(x_k,omega^(i-1)*r*Delta_tilde).
        # NOTE: model_degree is hardcoded to 2 here, not passed through from this function's own
        # model_degree argument -- see the handoff-contract note in this function's docstring.
        candidate_pool = list(Data)
        X, Data, added_idx, model_flag, _phase_map = model_improvement(X, candidate_pool, alpha_admissible, xkin, Deltanew, xi_imp, xi_acc, nfs, nfmax, tol_ineq, model_degree=2, printf=printf, weight_exponent=weight_exponent, r=r, max_new_evals_per_iter=max_new_evals_per_iter)

        # max_new_evals_per_iter cap-hit handling: retry internally (evaluate/tag whatever was
        # minted so far, then re-run model_improvement with the enlarged pool) until this pass's
        # geometry is genuinely poised or a real exit condition is hit, keeping Deltanew frozen
        # (not yet shrunk for this pass) the whole time -- see the matching comment in
        # s3a_accept.py for why this must not bubble a `continue` up to min_SAD.py's main loop.
        while True:
            if model_flag == 1: # ran out of fun evals in model_improvement
                nfs = max([dat[0] for dat in eval_counter.queries]) + 1
                flag = sigma
                return X, F, DF, D2F, eval_counter, nfs, gnew, Hnew, sigma, Deltanew, Data, flag
            elif model_flag == 2:
                nfs = max([dat[0] for dat in eval_counter.queries]) + 1
                flag = 7
                return X, F, DF, D2F, eval_counter, nfs, gnew, Hnew, sigma, Deltanew, Data, flag

            # evaluate the function at new points
            for i in added_idx:
                F, DF, D2F, flag, nfs = call_fun_script(fun, X[i], alpha_admissible, n, F, DF, D2F, flag, nfs)

            # update usage history for counting f evals
            for (i, alpha) in Data:
                eval_counter.count(i, alpha, phase="criticality")

            if flag == 4:
                return X, F, DF, D2F, eval_counter, nfs, gnew, Hnew, sigma, Deltanew, Data, flag

            if model_flag != 3:
                break

            candidate_pool = list(dict.fromkeys(candidate_pool + Data))
            X, Data, added_idx, model_flag, _phase_map = model_improvement(X, candidate_pool, alpha_admissible, xkin, Deltanew, xi_imp, xi_acc, nfs, nfmax, tol_ineq, model_degree=2, printf=printf, weight_exponent=weight_exponent, r=r, max_new_evals_per_iter=max_new_evals_per_iter)

        # compute new model
        cnew, gnew, Hnew, sigma = get_interpolation_model(X, K, n, F, DF, D2F, xkin, Data, Deltanew, model_degree, vec_out=False)
        
        if sigma is None:
            flag = 5
            return X, F, DF, D2F, eval_counter, nfs, gnew, Hnew, sigma, Deltanew, Data, flag

        # decrease trust region radius
        Deltanew *= omega

        if np.sqrt(Deltanew) < max(eps_c, np.finfo(float).eps):
            print("sqrt(Deltanew) has decreased below the gradient threshold.")
            flag = 1
            break

        it += 1

    # end while

    Deltanew = min(max(Deltanew, beta * sigma), Delta)

    return X, F, DF, D2F, eval_counter, nfs, gnew, Hnew, sigma, Deltanew, Data, flag



#########################################################################################
if __name__ == "__main__":

    n = 2
    qn1 = (n+1) * (n+2) // 2

    t1 = np.pi/4
    t2 = np.pi/4 + 5.0e-1

    rad = 1.0
    X = rad * np.array([[0.,0.], [np.cos(t1),np.sin(t1)], [np.cos(t2), np.sin(t2)]])
    alpha_list = [(0,0),(1,0),(0,0),(1,0),(0,0),(1,0)]
    Data = [(0, alpha_list[0]), (0, alpha_list[1]), (1, alpha_list[2]), (1, alpha_list[3]), (2, alpha_list[4]), (2, alpha_list[5])]

    nfs = len(X)

    # pad X with data before and after current model
    X = np.append(X, np.zeros((qn1,n)), axis=0)
    npad = 10
    X = np.append(np.zeros((npad,n)), X, axis=0)
    nfmax = len(X)
    nfs += npad

    Data = [(0+npad, alpha_list[0]), (0+npad, alpha_list[1]), (1+npad, alpha_list[2]), (1+npad, alpha_list[3]), (2+npad, alpha_list[4]), (2+npad, alpha_list[5])]
    xkin = 0+npad                    # X[xkin] == model center
    K = np.array([0])
    alpha_admissible = [ (0,0), (1,0) ]
    Delta = 1.0
    from min_SAD import EvalCounter
    eval_counter = EvalCounter()

    def fun(x):
        g = np.array([1.0, 2.0])
        H = np.array([[3.0, 1.0],
                      [1.0, 2.0]])
        val = np.dot(x, g + 0.5 * H @ x) + x[0]**3
        grad = H @ x + g + np.array([3*x[0]**2, 0])
        Hess = H
        Hess[0,0] += 6*x[0]
        return val, grad, Hess

    # evaluate function for interpolation data
    nK = len(K)
    F = np.zeros(len(X))
    DF = np.zeros((len(X), nK))
    D2F = np.zeros((len(X), nK * (nK + 1) // 2))

    from symmetric_matrix2vec import symmetric_matrix2vec
    for i in range(npad, nfs):
        val, grad, Hess = fun(X[i])
        F[i] = val
        DF[i,:] = grad[K]
        D2F[i,:] = symmetric_matrix2vec(Hess[np.ix_(K,K)], nK)

    # compute model
    c, g, H, sigma = get_interpolation_model(X, K, n, F, DF, D2F, xkin, Data, scalefactor=Delta, model_degree=2, vec_out=False)
    coefvec = np.concatenate(([c], g, symmetric_matrix2vec(H, n)))

    print("previous sigma =")
    print(sigma)

    omega = 0.6
    mu = 0.9
    flag = 0
    r = 1.0
    eps_c = 0
    beta = 0.5
    xi_imp = 1.1
    xi_acc = 0.05
    tol_ineq = 0

    Xnew, Fnew, DFnew, D2Fnew, eval_counter, nfs, gnew, Hnew, sigmanew, Deltanew, Datanew, flag = criticality_step(fun, g, H, sigma, Delta, omega, mu, n, X, F, DF, D2F, eval_counter, flag, nfs, nfmax, Data, K, alpha_admissible, xkin, r, eps_c, beta, xi_imp, xi_acc, tol_ineq, model_degree=2)

    print("Xnew =")
    print(Xnew)
    print("Fnew =")
    print(Fnew)
    print("Datanew =")
    print(Datanew)
    print("Eval queries =")
    print(eval_counter.queries)
    print("nfs =")
    print(nfs)
    print("gnew =")
    print(gnew)
    print("Hnew =")
    print(Hnew)
    print("sigmanew =")
    print(sigmanew)
