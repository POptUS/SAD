import numpy as np
from model_update import model_update
from complete_nonpoised_set import complete_nonpoised_set

def s3r_valid(fun, sigma, Delta, n, X, F, DF, D2F, eval_counter, flag, nfs, nfmax, Data, K, alpha_admissible, xkin, r, gam_dec, Deltamin, xi_acc, added_idx, model_flag, model_degree):

    exiting = False

    # decrease trust radius
    Delta = max(gam_dec * Delta, Deltamin)
    
    if Delta < np.finfo(float).eps * np.linalg.norm(X[xkin]):  # Delta too small for floating point
        flag = 2
        if printf:
            print("Delta has become too small. Terminating.")
        exiting = True
        return

    # potentially add candidate point to model geometry
    Data.append( (nfs - 1, (0,)*n) )

    # rebuild the model with smaller Delta
    Data = [(i, alpha) for (i, alpha) in eval_counter.queries if np.linalg.norm(X[i] - X[xkin]) <= r*Delta] # get rid of all points outside new Delta-ball
    X, Data, added_idx, _, model_flag = complete_nonpoised_set(X, Data, alpha_admissible, xkin, Delta, xi_acc, nfs, nfmax, model_degree)

    g, H, sigma, X, F, DF, D2F, eval_counter, flag, nfs, exiting \
            = model_update(fun, sigma, Delta, n, X, F, DF, D2F, eval_counter, flag, nfs, nfmax, Data, K, alpha_admissible, xkin, added_idx, model_flag, model_degree)

    return g, H, sigma, Delta, X, F, DF, D2F, Data, eval_counter, flag, nfs, exiting
