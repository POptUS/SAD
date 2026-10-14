import numpy as np

from max_abs_quadratic import max_abs_quadratic
from polynomial2matrix import polynomial2matrix
from shift_and_scale import shift_and_scale
from phi2eval import Dalpha_phieval, Dpoly2
from complete_nonpoised_set import complete_nonpoised_set, get_Upper1

def model_improvement(X, Data, alpha_admissible, xkin, Delta, xi_imp, xi_acc, nfs, nfmax, tol_ineq=1e-2, model_degree=2, printf=True, weight_exponent=3, r=1.0, reuse_existing_points=False, eval_counter=None, max_new_evals_per_iter=None):
    """
    Implements Algorithm 2 ("model_improvement") of main.tex: first runs Algorithm 1
    (complete_nonpoised_set, i.e. model_completion) to obtain a poised set, then tries to
    replace exactly one interpolation condition -- the last pivot -- with a better one
    found by searching the whole trust region, if doing so improves that pivot's value by
    at least a factor of xi_imp.

    Handoff contract:
    - Data may be ANY size on entry (this function is called with a poised set of size qn1
      from criticality_step.py, and with a poised set plus one extra rejected point, size
      qn1+1, from s3r.py) -- complete_nonpoised_set always normalizes it to size qn1 first.
    - On success (model_flag==0), returns Datanew with len(Datanew) == qn1 == (n+1)(n+2)/2,
      always -- whether or not the improvement replacement itself succeeded.
    - If no candidate replacement beats xi_imp * previous_pivot_value, this function does
      NOT set an error flag (main.tex's Algorithm 2 treats "no improvement found" as a
      distinct "failure" outcome; here it is a silent no-op -- Datanew is returned exactly
      as complete_nonpoised_set produced it, with only a printed message). Callers do not
      currently branch on this case, so treat it as "geometry left as poised-by-Algorithm-1"
      rather than a real error.
    - flag==0 normal/no-op; flag==1 nfmax exceeded; flag==2 the trust-region search over
      alpha_admissible produced no usable (nonzero) pivot value; otherwise flag is
      model_flag propagated from complete_nonpoised_set (1 = nfmax exceeded there, 2 =
      unrecoverable poisedness failure, 3 = max_new_evals_per_iter cap hit -- Datanew is
      partial, len < qn1, and the improvement step below is skipped entirely).
    - Also returns phase_map, a dict {point index -> phase name} for eval-count instrumentation:
      every index in added_idx (from complete_nonpoised_set) maps to "geometry_completion";
      if the improvement step below mints a brand-new point, that index is overridden to
      "model_improvement". Callers (s3r.py, criticality_step.py) forward this into model_update.
    - reuse_existing_points (ablation, default False): when True and eval_counter is given, before
      minting a brand-new point for the improved pivot, check already-evaluated points (from
      eval_counter.queries) not already part of Datanew for one lying in the trust region whose
      pivot value for the winning alpha is within a factor of xi_imp of the newly-found optimum;
      if found, reuse that existing point's index instead of minting a new one (so it keeps its
      "geometry_completion"/whatever phase it already had, rather than being counted as a fresh
      model-improvement evaluation).

    INPUTS:
    X       [dbl] [?-by-n] history of points queried for interpolation models
    Data    [lst] [q+1] current model interpolation equations
    alpha...[lst] [?] all available derivative multiindices
    Delta   [dbl] trust region radius
    K       [int] [nK] indices of known partial derivatives of the objective function
    y0      [dbl] [n] model center point within Y
    Data    [tup] [(int, n-int)] interpolation point index in X and corresponding derivative multiindex
    f       [dbl] [p_ini1]
    xi_acc  [>0 ] acceptance threshold for determining sufficient poisedness

    OUTPUTS:
    Y       [dbl] [(qn-qnK+1)-by-n] a poised set of points
    f       [dbl] [qn-qnK+1] function evaluations associated with each point in Y
                    Evaluations at new points have the placeholder value np.inf
    Y_added [dbl] [?-by-n] points which were added to Y to make it a poised set
    Y_removed [dbl] [?-by-n] points '''''''' removed from Y '''''''''''''''''''
    perm_idx [int] indicates the permuted ordering of the points in Y upon output,
                where -1 indicates a new point
    """
    n = np.shape(X)[1]

    # call complete_nonpoised_set to perform pivoting, then change last added data pair
    X, Datanew, added_idx, Upper, model_flag = complete_nonpoised_set(X, Data, alpha_admissible, xkin, Delta, xi_acc, nfs, nfmax, model_degree, tol_ineq, printf, weight_exponent=weight_exponent, r=r, max_new_evals_per_iter=max_new_evals_per_iter)

    phase_map = {i: "geometry_completion" for i in added_idx}

    if model_flag!=0:
        if printf:
            print("Model completion terminated early or unsuccessfully.")
        flag = model_flag
        return X, Datanew, added_idx, flag, phase_map

    assert(len(Datanew)==(n+1)*(n+2)//2)

    from model_validity import model_validity
    valid,_ = model_validity(Delta, X, Datanew, xkin, r=r, tol_ineq=tol_ineq)
    # ipdb.set_trace(cond=not valid, context=21)  # breadcrumb: complete_nonpoised_set should
    # always return a set within r*Delta of xkin; re-enable if that invariant is ever violated.
    assert valid, "complete_nonpoised_set returned a Data set violating the r*Delta trust-region bound"

    idx_in = list(dict.fromkeys([j for j, _ in Datanew])) # unique point indices in Data, maintaining order of first instance in Data
    y0 = X[xkin]
    Y = np.array([X[j] for j in idx_in]) # unique points
    Y, scalefactor = shift_and_scale(Y, y0, backupfactor=Delta, printf=printf)  # rescale by Delta (Y)

    assert(len(Datanew)==(n+1)*(n+2)//2)

    idx_XY = dict(zip( idx_in, range(len(idx_in)) )) # correspondence between global (X) indices and local (Y) indices

    _, n = Y.shape
    n_alpha = len(alpha_admissible)

    #################################################
    # improvement step: optimize the last pivot poly
    #################################################

    idx_X, alpha = Datanew[-1] # last Data pair will now be replaced
    idx_Y = idx_XY[idx_X]
    Upper1 = get_Upper1(Upper, alpha)
    vander = Dalpha_phieval(Y[idx_Y], alpha=(0,)*n) # Vandermonde vector for last point

    weight = max(Delta / scalefactor, np.linalg.norm(Y[idx_Y]))**weight_exponent
    previous_pivot_value = np.abs(vander @ Upper1[-1,:]) / weight

    assert(len(Datanew)==(n+1)*(n+2)//2)
    #################################################
    # Optimize over normalized y in the trust region
    #################################################

    flag = 0

    improved_pivot_value = 0.0
    j_i = None      #alpha^{j_i} solves max_alpha max_{y in Ball(0,1)} |D^{alpha} phi_{i}(y)|

    for j in range(n_alpha): # loop alpha

        alpha = alpha_admissible[j]
        deg = sum(alpha)

        poly = Dpoly2(Upper[-1,:], alpha, n)
        c, g, H = polynomial2matrix(poly, n)
        y, ab_val = max_abs_quadratic(c, g, H, Delta / scalefactor)

        # weight = max(Delta / scalefactor, np.linalg.norm(ynew - Y[ykin]))**3 = 1 by construction of TRSP solution

        if ab_val > improved_pivot_value:
            improved_pivot_value = ab_val
            j_i = j
            alpha_i = alpha
            ynew = y

    if j_i is None:
        if printf:
            print("Error: TRSP failed to produce a nonzero pivot value.")
        flag = 2
        return X, Datanew, added_idx, flag, phase_map

    if nfs + 1 >= nfmax:
        if printf:
            print("maximum number of function evaluations exceeded")
        flag = 1 # ran out of function evals
        return X, Datanew, added_idx, flag, phase_map

    deg = sum(alpha_i)

    assert(len(Datanew)==(n+1)*(n+2)//2)
    # determine if the new (y,alpha) pair has sufficient geometry improvement
    if improved_pivot_value > xi_imp * previous_pivot_value: # add to data

        reused_idx = None
        if reuse_existing_points and eval_counter is not None:
            Upper1_new = get_Upper1(Upper, alpha_i)
            candidate_indices = sorted(set(i for (i, _) in eval_counter.queries) - set(idx_in))
            best_val = 0.0
            for i in candidate_indices:
                y_cand = (X[i] - y0) / scalefactor
                if np.linalg.norm(y_cand) > Delta / scalefactor + tol_ineq:
                    continue
                vander_c = Dalpha_phieval(y_cand, alpha=(0,)*n)
                pivot_val = np.abs(vander_c @ Upper1_new[-1, :])
                if pivot_val > best_val:
                    best_val = pivot_val
                    reused_idx = i
            if reused_idx is not None and best_val * xi_imp < improved_pivot_value:
                reused_idx = None  # existing point not comparably good; mint a new one instead

        if reused_idx is not None:
            Datanew[-1] = (reused_idx, alpha_i)
        else:
            idx_new = max([nfs] + [i+1 for i in added_idx]) # == max(nfs, max(added_idx)+1); nfs if no points replaced in complete_nonpoised_set
            Datanew[-1] = (idx_new, alpha_i)
            X[idx_new] = ynew * scalefactor + y0
            added_idx.append(idx_new)
            phase_map[idx_new] = "model_improvement"

    else:

        if printf:
            print("Model-improvement step failed.")

    assert(len(Datanew)==(n+1)*(n+2)//2)
    return X, Datanew, added_idx, flag, phase_map


####################################################################################
if __name__ == "__main__":

    n = 2
    qn1 = (n+1) * (n+2) // 2
    npad = 1

    # choose points which are close together, but not so close that they trigger non-poisedness with pivot values < xi_acc
    t1 = np.pi/4
    t2 = np.pi/4 + 1.0e-1

    X = np.array([[0.,0.], [np.cos(t1),np.sin(t1)], [np.cos(t2), np.sin(t2)]])
    alpha_list = [(0,0),(1,0),(0,0),(1,0),(0,0),(1,0)]
    X = np.append(np.zeros((npad, n)), X, axis=0) # simulate some earlier iterations
    Data = [(0+npad, alpha_list[0]), (0+npad, alpha_list[1]), (1+npad, alpha_list[2]), (1+npad, alpha_list[3]), (2+npad, alpha_list[4]), (2+npad, alpha_list[5])]
    nfs = len(X)

    alpha_admissible = [ (0,0), (1,0) ]

    # pad X with extra space for additional points added by model improvement/complete nonpoised data
    X = np.append(X, np.zeros((qn1,n)), axis=0)

    # parameters
    xkin = 0+npad                    # X[xkin] == model center
    Delta = 1.0
    xi_imp = 1.10
    xi_acc = 1e-2
    nfmax = len(X)

    Xnew, Datanew, added_idx, flag, phase_map = model_improvement(X, Data, alpha_admissible, xkin, Delta, xi_imp, xi_acc, nfs, nfmax, tol_ineq=1e-2, model_degree=2)
    print("Xnew =")
    print(Xnew)
    print("Datanew =")
    print(Datanew)
    print("added_idx =")
    print(added_idx)
    print("flag =")
    print(flag)
    from get_interpolation_matrix import get_interpolation_matrix
    Mhat = get_interpolation_matrix(Xnew, xkin, Datanew, scalefactor=Delta, model_degree=2)
    print("Matrix condition number =")
    print(np.linalg.cond(Mhat))
