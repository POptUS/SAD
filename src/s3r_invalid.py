from model_update import model_update
from model_improvement import model_improvement

def s3r_invalid(fun, sigma, Delta, n, X, F, DF, D2F, eval_counter, flag, nfs, nfmax, Data, K, alpha_admissible, xkin, added_idx, model_flag, xi_imp, xi_acc, tol_ineq, model_degree, printf):

    # potentially add candidate point to model geometry
    Data.append( (nfs - 1, (0,)*n) )

    # improve the model
    X, Data, added_idx, model_flag = model_improvement(X, Data, alpha_admissible, xkin, Delta, xi_imp, xi_acc, nfs, nfmax, tol_ineq, model_degree, printf)

    g, H, sigma, X, F, DF, D2F, eval_counter, flag, nfs, exiting \
            = model_update(fun, sigma, Delta, n, X, F, DF, D2F, eval_counter, flag, nfs, nfmax, Data, K, alpha_admissible, xkin, added_idx, model_flag, model_degree)

    return g, H, sigma, X, F, DF, D2F, eval_counter, flag, nfs, Data, exiting
