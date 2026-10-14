import numpy as np

from idx_uppertri2vec import idx_uppertri2vec


def get_nonK_idx(K, n, model_degree=2):
    """
    given linear or quadratic natural basis functions, return the
    indices of the functions (in standard upper triangular order)
    which do not depend solely on the variables indexed by K,
    the known partial derivatives.
    """

    # important numbers
    nK = len(K)
    nU = n - nK
    qn1 = (n + 1) * (n + 2) // 2
    qnK1 = (nK + 1) * (nK + 2) // 2

    if model_degree == 2:
        n_interp = qn1 - qnK1 + 1
    elif model_degree == 1:
        n_interp = nU + 1

    # the unavailable derivatives
    U = np.setdiff1d(np.arange(n), K)

    # main loop
    idx = np.zeros(n_interp, dtype=int)  # preallocate and set idx[0]=0; keep the constant function
    idx[1 : nU + 1] = U + 1  # linear functions

    if model_degree == 2:

        count = nU + 1

        for i in range(n):  # keep only basis functions with at least one component in U

            if i in U:

                # process the entire (upper triangular) row of matrix
                for iprime in range(i, n):
                    idx[count] = (n + 1) + idx_uppertri2vec(i, iprime, n)
                    count += 1

            else:

                # i in K; process only the (upper tri) columns of matrix corresp. to U
                for iprime in U[U > i]:
                    idx[count] = (n + 1) + idx_uppertri2vec(i, iprime, n)
                    count += 1

    return idx
