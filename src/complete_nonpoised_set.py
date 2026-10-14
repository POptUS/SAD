import numpy as np

from max_abs_quadratic import max_abs_quadratic
from multiindex2derivatives import multiindex2derivatives
from phi2eval import Dalpha_phieval, Dpoly2
from polyval2 import Dpolyval2_combined
from polynomial2matrix import polynomial2matrix
from shift_and_scale import shift_and_scale, undo_shift_and_scale

# cache dictionary shared across calls
_TRIU_CACHE = {}

def get_triu_indices(n):
    """Return cached upper-triangular indices for dimension n."""
    if n not in _TRIU_CACHE:
        iu = np.triu_indices(n)
        diag_mask = (iu[0] == iu[1])
        _TRIU_CACHE[n] = (iu, diag_mask)
    return _TRIU_CACHE[n]


### Auxiliary functions

def eliminate(Upper, Vander, i, j, deg, Upper1=None):
    if deg == 0:
        normalize_factor = np.dot(Upper[i, :], Vander[j, :])
        Upper[i + 1:, :] -= np.outer(np.dot(Upper[i + 1:, :], Vander[j, :]) / normalize_factor, Upper[i, :])
    elif deg == 1:
        normalize_factor = np.dot(Upper1[i, :], Vander[j, :])
        Upper[i + 1:, :] -= np.outer(np.dot(Upper1[i + 1:, :], Vander[j, :]) / normalize_factor, Upper[i, :])
    return Upper

###
def get_used_points_idx(Data, alpha):
    """
    return indices of points already used in Data_idx for given multiindex alpha
    """
    return [dat[0] for dat in Data if dat[1] == alpha]

###
def orthogonal_component(row, Qbasis):
    """
    Decompose row (a raw interpolation-matrix row, in the natural monomial basis) into its
    component orthogonal to the row space of Qbasis (rows of Qbasis assumed orthonormal).
    Returns (residual, residual_norm).
    """
    if Qbasis.shape[0] == 0:
        return row, np.linalg.norm(row)
    resid = row - Qbasis.T @ (Qbasis @ row)
    return resid, np.linalg.norm(resid)

###
def get_Vander(Y, n):
    qn1 = (n + 1) * (n + 2) // 2
    Vander = np.zeros((len(Y), qn1))
    for j,y in enumerate(Y):
        Vander[j,:] = Dalpha_phieval(np.atleast_2d(y), alpha=(0,)*n)
    return Vander

###
def get_Upper1(Upper, alpha):
    n = len(alpha)
    _, qn1 = np.shape(Upper)
    Upper1 = np.copy(Upper)
    for k in range(qn1):
        Upper1[k,:] = Dpoly2(Upper[k,:], alpha, n)
    return Upper1


def get_Upper1_combined(Upper, alpha):
    """
    Apply d^alpha to all rows of Upper (monomial coefficient vectors)
    in one vectorized operation.
    """
    n = len(alpha)
    deg = sum(alpha)
    ilist = multiindex2derivatives(alpha)
    qn1 = Upper.shape[0]
    ncoeff = int(0.5 * (n + 1) * (n + 2))

    if deg == 0:
        return Upper.copy()

    Upper1 = np.zeros((qn1, ncoeff))
    lin_start, quad_start = 1, 1 + n

    # cache upper-triangular index mapping once
    iu, _ = get_triu_indices(n)

    if deg == 1:
        i = ilist[0]
        # constant term derivative = linear coefficient
        Upper1[:, 0] = Upper[:, lin_start + i]
        # linear term derivatives: derivative of x_i*x_j
        # Each column j gets contribution from upper-triangular coefficient (i,j)
        # map min(i,j),max(i,j) to vector index
        for j in range(n):
            idx = int(n * min(i,j) - 0.5 * min(i,j) * (min(i,j) + 1) + max(i,j))
            Upper1[:, 1 + j] = Upper[:, quad_start + idx]
    elif deg == 2:
        i, j = ilist
        idx = int(n * min(i,j) - 0.5 * min(i,j) * (min(i,j) + 1) + max(i,j))
        Upper1[:, 0] = Upper[:, 1 + n + idx]

    return Upper1

###
def break_pivot_ties(pivots, xkin, Data_in):
    """
    pivots = local indices of data pairs with a maximal pivot value
    xkin   = global index of current model center
    idx_XY = dictionary of global points to local points
    Return index in pivots which breaks ties, if present
    """

    if len(pivots) == 1: # no ties
        return pivots[0]

    candidates = [Data_in[p] for p in pivots] # candidate tuples
    idx_X = [i for (i, alpha) in candidates] # global indices of points for maximal pivots

    if xkin in idx_X:

        with_xkin = [(p, Data_in[p][1]) for p in pivots if Data_in[p][0] == xkin] # (pivot index wrt Data_in, alpha)
        xkin_largest_degree = max(with_xkin, key=lambda pair: sum(pair[1]))
        return xkin_largest_degree[0]  # return index into Data_in

    else:

        # model center not present; take most recent (highest index) point
        most_recent = max(pivots, key=lambda p: Data_in[p][0])
        return most_recent


#################################################################################################
# main function: let alphas used in data set change / don't pair (y, alpha) a priori for pivoting
#################################################################################################
def complete_nonpoised_set(X, Data_in, alpha_admissible, xkin, Delta, xi_acc, nfs, nfmax, model_degree=2, tol_ineq=1e-2, printf=True, weight_exponent=3, r=1.0, max_new_evals_per_iter=None):
    """
    Given a set Y with potentially != qn+1 points,
    generate a poised set by adding and/or deleting points

    Implements Algorithm 1 ("model_completion") of main.tex: the greedy pivot-polynomial
    procedure that builds a poised Birkhoff interpolation set for the natural monomial basis,
    one basis function at a time. For each basis function i, it first searches Data_in for the
    best-scoring (point, derivative-multiindex) candidate; if none clears the xi_acc threshold,
    it falls back to searching the whole trust region (via max_abs_quadratic, over every
    multiindex in alpha_admissible) for a point that does, adding a brand-new evaluation point
    if necessary. Candidate pivot values are weighted by max(Delta/scalefactor, dist)**3 to
    penalize points more than Delta from xkin -- this implements main.tex's optional weighted
    variant of Algorithm 1, not a deviation from it. Ties among maximal pivots are broken by
    break_pivot_ties (preferring the model center, else the most recently added point) -- a
    tie-breaking rule the paper's Algorithm 1 leaves unspecified.

    Rank-revealing guard (deviation from the paper, added to fix a singular-matrix failure
    mode): every candidate (point, alpha) pair, from either Data_in or the trust-region
    fallback search, is checked against Qbasis -- an incrementally-maintained orthonormal
    basis for the ACTUAL rows (Dalpha_phieval(y, alpha)) already accepted into Data -- before
    being finally accepted. A candidate whose row is nearly in that span is skipped (not
    removed from Data_in, so a later pivot may still use it) even if its abstract Upper-space
    pivot value cleared xi_acc; the next-best candidate is tried instead. Without this, the
    greedy algorithm's per-pivot scalar check (computed via the Upper elimination bookkeeping)
    can accept a sequence of individually-plausible pivots that are nevertheless collectively
    near-linearly-dependent once assembled into the real interpolation matrix -- observed in
    practice both as two points differing only along a direction some other pivot had already
    eliminated (e.g. via an exactly-known K x K second-derivative condition), and, more
    generally, as many different alphas concentrating on the same point once the trust region
    has shrunk, until some derivative condition there becomes expressible as a combination of
    other accepted rows.

    Handoff contract:
    - Data_in may be ANY size/state on entry (not necessarily poised, may be empty aside from
      the required model-center pair) -- this function is the one place a possibly-nonpoised
      candidate set gets turned into a poised one.
    - On success (flag==0), returns Data with len(Data) == qn1 == (n+1)(n+2)/2 exactly, always.
    - flag==0 success; flag==1 nfmax would be exceeded by the points this call would need to add
      (Data/added_idx are not meaningful in this case -- callers must check flag before using
      them); flag==2 no candidate pivot value cleared xi_acc anywhere in the trust region
      (unrecoverable poisedness failure for this model_degree/xi_acc/Delta combination); flag==3
      max_new_evals_per_iter was reached before all qn1 pivots were assigned -- Data holds
      whatever pivots were completed so far (length < qn1) plus added_idx for the new point(s)
      minted this call; the caller is expected to evaluate those points and call this function
      again on a later pass to keep making progress (see max_new_evals_per_iter below).
    - Callers: s3a_accept.py (rebuilding Data from the full evaluation history after an accept)
      and model_improvement.py/Algorithm 2 (as its first step, on whatever Data it is given).
    - max_new_evals_per_iter (ablation, default None = unlimited): caps how many brand-new points
      this call may mint via the trust-region-search fallback (the `else` branch below that
      appends to Y_added) in a single call. Does NOT cap reuse of already-evaluated candidates
      from Data_in, nor the use_existing_point case (reusing xkin's coordinates with a new alpha,
      which costs no new function evaluation since call_fun_script evaluates every admissible
      derivative at a point in one call) -- neither of those adds to n_added. When the cap is hit,
      stops early with flag=3 instead of continuing to flag=0/qn1 pivots.

    INPUTS:
    X       [dbl] [?-by-n] history of points queried for interpolation models
    idx_in  [dbl] [list] tuples (i, alpha) for which X[i] is in the trust region and "partial^alpha f(X[i])" has has already been evaluated
    Y       [dbl] [p_ini1-by-n] points from which to choose Hermite interpolation conditions
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

    # remove points outside trust region
    Data_in = [(i, alpha) for (i, alpha) in Data_in if np.linalg.norm(X[i] - X[xkin]) <= r*Delta + tol_ineq]

    y0 = X[xkin]
    idx_in = np.unique([i for (i, _) in Data_in])
    idx_in = idx_in.tolist()
    Y = X[idx_in] # Y has unique vectors if idx_in has no repeats
    Y, scalefactor = shift_and_scale(Y, y0, backupfactor=Delta, printf=printf)  # rescale by radius of Data_in

    npts, n = Y.shape
    n_alpha = len(alpha_admissible)
    qn1 = (n + 1) * (n + 2) // 2  # Dimension of the polynomial space

    if model_degree == 2:
        n_interp = qn1
    elif model_degree == 1:
        n_interp = n + 1

    assert (0,)*n in alpha_admissible
    assert (xkin, (0,)*n) in Data_in

    #################################################################################
    # preallocate and initialize pivot polynomials
    #################################################################################

    # make original Vandermonde-like matrix ...
    Vander = get_Vander(Y, n)

    # initial guess for HIPs WRT the natural basis
    Upper = np.eye(qn1)

    # number of points added to Y
    n_added = 0

    # set weights for preferring to replace points more than Delta from xk (when r > 1)
    dists = np.linalg.norm(Y, axis=1)  # Y has been shifted and scaled
    weights = np.maximum(Delta / scalefactor, dists)  # penalize points which are more than Delta from xk in the un-normalized space
    weights = weights**weight_exponent

    # initialize
    Data = [ (xkin, (0,)*n) ] # force interpolation of function value at the model center
    Data_in.remove( (xkin, (0,)*n) ) # this pair is used for initialization; we need not consider it for future updates to Data

    # dictionary mapping corresponding point indices in global history (X) and local consideration (Y)
    idx_XY = dict(zip( idx_in, range(len(idx_in)) ))

    # Incremental orthonormal basis (built via modified Gram-Schmidt) spanning the ACTUAL rows
    # of the interpolation matrix (Dalpha_phieval(y, alpha) in the natural monomial basis)
    # accepted into Data so far. Used as a rank-revealing check: a candidate is rejected if its
    # raw row is nearly in this span, even when its abstract Upper-space pivot value clears
    # xi_acc -- the two checks are not equivalent once several derivative conditions
    # concentrate on nearby points (e.g. many alphas assigned to the same point once the trust
    # region has shrunk), which can leave the ACTUAL assembled matrix numerically singular even
    # though every individual greedy pivot looked acceptable in isolation (see main.tex
    # Algorithm 1 discussion).
    Qbasis = np.zeros((0, qn1))
    rank_tol = 1e-6  # reject a candidate whose row component orthogonal to Qbasis is this small, relative to its own norm

    Y_added = np.zeros((0,n))

    flag = 0

    #################################################################################
    # begin main loop
    for i in range(1, qn1): # loop basis functions

        maxval = -np.inf
        k_i = None

        if Data_in: # non-empty list

            # search data first
            mod_ab_values = np.zeros(len(Data_in))
            for p, (ix, alpha) in enumerate(Data_in):
                iy = idx_XY[ix]
                ab_val = np.abs( Dpolyval2_combined(Upper[i,:], Y[iy], alpha) )

                if (ix, alpha) in Data:
                    mod_ab_values[p] = 0 # zero out data that is already used
                else:
                    mod_ab_values[p] =  ab_val / weights[iy]

            # Walk candidates best-first; skip (without removing from Data_in) any candidate
            # whose actual row is nearly redundant with the rows already accepted into Data.
            excluded = set()
            while True:
                remaining = [p for p in range(len(Data_in)) if p not in excluded]
                if not remaining:
                    break

                best_val = np.amax(mod_ab_values[remaining])
                if best_val <= max(xi_acc, maxval):
                    break

                pivots = [p for p in remaining if mod_ab_values[p] == best_val]
                k = break_pivot_ties(pivots, xkin, Data_in) # use helper function here

                # Now we have that [Data_in[p] for p in pivots] are maximal (x, alpha) pairs for the given pivot poly
                # And Data_in[k] breaks a tie, if one is present.

                ix_c, alpha_c = Data_in[k]
                iy_c = idx_XY[ix_c]
                row_c = Dalpha_phieval(np.atleast_2d(Y[iy_c]), alpha_c)[0]
                resid_c, resid_c_norm = orthogonal_component(row_c, Qbasis)
                if resid_c_norm < rank_tol * max(np.linalg.norm(row_c), 1e-300):
                    excluded.add(k)
                    continue

                maxval = mod_ab_values[k]
                ix, alpha_i = Data_in.pop(k) # remove from Data_in so we don't check it in future iters
                deg = sum(alpha_i)
                k_i = idx_XY[ix]

                Data.append( (ix, alpha_i) )
                Qbasis = np.vstack([Qbasis, resid_c / resid_c_norm])
                break

        if k_i is None: # search the trust region for a valid (x, alpha) pair

            j_i = None      #|D^{alpha^{j_i}} phi_i(y)| is maximal from the unit ball, if it surpasses the xi_acc threshold

            # for each alpha^j, find y which maximizes |D^{alpha^j} phi_i(y)| by two TRSP's
            # for the grand maximal y/alpha pair, add to the data and pivot

            # loop over alpha, computing the best (point, value) pair each admits via two TRSPs
            candidates = []  # (ab_val, prefer_center, j, alpha, y)
            for j in range(n_alpha): # loop alpha

                alpha = alpha_admissible[j]

                poly = Dpoly2(Upper[i,:], alpha, n)
                c, g, H = polynomial2matrix(poly, n)

                # detect if the polynomial is constant; if so, choose every point is maximal; take the model center
                if np.linalg.norm(g) == 0 and np.linalg.norm(H) == 0:
                    y, ab_val = np.zeros(n), np.abs(c)
                else:
                    y, ab_val = max_abs_quadratic(c, g, H, Delta / scalefactor)

                # weight = max(Delta / scalefactor, np.linalg.norm(ynew - Y[ykin]))**3 = 1 by construction of TRSP solution

                candidates.append((ab_val, np.linalg.norm(y) == 0, j, alpha, y))

            # best value first; among ties, prefer the model center (matches the original
            # single-pass tie-break, which preferred y=0 whenever it tied the running best)
            candidates.sort(key=lambda cand: (-cand[0], not cand[1]))

            # walk candidates best-first, skipping any whose row is nearly redundant with the
            # rows already accepted into Data (see rank-revealing guard docstring above)
            j_i = None
            for ab_val, prefer_center, j, alpha, y in candidates:
                if ab_val <= max(xi_acc, maxval):
                    break  # candidates are sorted descending; nothing further can pass either
                row_y = Dalpha_phieval(np.atleast_2d(y), alpha)[0]
                resid_y, resid_y_norm = orthogonal_component(row_y, Qbasis)
                if resid_y_norm < rank_tol * max(np.linalg.norm(row_y), 1e-300):
                    continue
                maxval = ab_val
                j_i = j
                alpha_i = alpha
                ynew = y
                use_existing_point = prefer_center
                winning_resid, winning_resid_norm = resid_y, resid_y_norm
                break

            if j_i is None:
                print("Error: Model completion failed to produce a nonzero pivot value.")
                flag = 2
                break

            deg = sum(alpha_i)

            if use_existing_point: # hard-coded for the model center

                # update Data
                Data.append( (xkin, alpha_i) )
                k_i = idx_XY[xkin]
                Qbasis = np.vstack([Qbasis, winning_resid / winning_resid_norm])

            else:

                if max_new_evals_per_iter is not None and n_added >= max_new_evals_per_iter:
                    flag = 3
                    break

                n_added += 1

                # update Data
                Y_added = np.append(Y_added, [ynew], axis=0)
                Y = np.append(Y, [ynew], axis=0)
                weights = np.append(weights, max(Delta / scalefactor, np.linalg.norm(ynew))**weight_exponent)
                k_i = len(Y) - 1

                idx_added = nfs-1 + n_added
                Data.append( (idx_added, alpha_i) )
                idx_in.append(idx_added)
                idx_XY.update({idx_added: len(idx_in)-1})
                Qbasis = np.vstack([Qbasis, winning_resid / winning_resid_norm])

                # generate new row of Vandermonde matrix
                new_row = Dalpha_phieval(np.atleast_2d(ynew), alpha=(0,)*n)
                Vander = np.append(Vander, new_row, axis=0)

        Upper1 = get_Upper1_combined(Upper, alpha_i)

        # Gaussian elimination
        # i = pivot poly index, j = Vandermonde matrix row index
        Upper = eliminate(Upper, Vander, i, k_i, deg, Upper1)


    if flag == 0 and nfs + n_added >= nfmax:
        # Only overrides a would-be SUCCESS (flag==0) into "ran out of evals" -- flag==2 (no valid
        # pivot found) and flag==3 (max_new_evals_per_iter cap hit) already reflect why the loop
        # stopped early and must not be clobbered by this check, or a genuine partial-progress/
        # unrecoverable-poisedness outcome would be silently misreported as an nfmax exit.
        print("maximum number of function evaluations exceeded")
        added_idx = []
        flag = 1 # ran out of function evals
        return X, Data, added_idx, Upper, flag

    added_idx = list(range(nfs, nfs + n_added))
    Y_added = undo_shift_and_scale(Y_added, y0, scalefactor)
    X[added_idx] = Y_added

    return X, Data, added_idx, Upper, flag


if __name__ == "__main__":

    n = 2
    qn1 = (n+1) * (n+2) // 2

    tt = [np.pi/2, 3*np.pi/2]
    X = np.zeros((len(tt)+1, n)) # model center at origin
    rad = 1.0
    for i, t in enumerate(tt):
        X[i+1] = rad * np.array([np.cos(t), np.sin(t)])
    nfs = len(X)

    # pad X with enough space
    X = np.append(X, np.zeros((qn1,n)), axis=0)
    nfmax = len(X)

    idx_in = [i for i in range(nfs)]    # indices of points in X to use for model-building
    xkin = 0                    # X[xkin] == model center
    K = np.array([0])
    Delta = 1.0
    xi_acc = 0.05               # acceptance threshold for pivot polynomials

    alpha_admissible = [(0,0), (1,0)]
    Data_in = [ (i, alpha_admissible[0]) for i in range(nfs) ]

    Xnew, Data, added_idx, Upper, flag = complete_nonpoised_set(X, Data_in, alpha_admissible, xkin, Delta, xi_acc, nfs, nfmax, model_degree=2)
    print("Xnew = ")
    print(Xnew)
    print("Data = ")
    print(Data)
    print("added_idx =")
    print(added_idx)
    print("Upper =")
    print(Upper)
    print("flag = ")
    print(flag)
    from get_interpolation_matrix import get_interpolation_matrix
    Mat = get_interpolation_matrix(Xnew, xkin, Data, scalefactor=Delta, model_degree=2)
    print("Birkhoff Vandermonde matrix condition number =")
    print(np.linalg.cond(Mat))

    # start with model center only
    X = np.zeros((1, n)) # model center at origin
    nfs = len(X)
    # pad X with enough space
    X = np.append(X, np.zeros((qn1,n)), axis=0)
    nfmax = len(X)
    idx_in = [i for i in range(nfs)]    # indices of points in X to use for model-building
    Data_in = [ (i, alpha_admissible[0]) for i in range(nfs) ]
    Xnew, Data, added_idx, Upper, flag = complete_nonpoised_set(X, Data_in, alpha_admissible, xkin, Delta, xi_acc, nfs, nfmax, model_degree=2)
    print("Xnew = ")
    print(Xnew)
    print("Data = ")
    print(Data)
    print("added_idx =")
    print(added_idx)
    print("flag = ")
    print(flag)
    from get_interpolation_matrix import get_interpolation_matrix
    Mat = get_interpolation_matrix(Xnew, xkin, Data, scalefactor=Delta, model_degree=2)
    print("Birkhoff Vandermonde matrix condition number =")
    print(np.linalg.cond(Mat))
