import numpy as np

from get_interpolation_matrix import get_interpolation_matrix
from polynomial2matrix import polynomial2matrix
from idx_uppertri2vec import idx_uppertri2vec
from shift_and_scale import shift_and_scale
from multiindex2derivatives import multiindex2derivatives
from symmetric_matrix2vec import symmetric_matrix2vec
from get_sigma import get_sigma


def get_interpolation_model(X, K, n, F, DF, D2F, xkin, Data, scalefactor=1.0, model_degree=2, vec_out=True):
    """
    compute the parameters c, g, and H for the Hermite interpolation model
        m(x) = c + dot(g, x) + dot(x, H @ x)
    NOTE: Input objective function "fun" to cheat for debugging and see all derivative information
    """
    qn1 = (n + 1) * (n + 2) // 2
    assert len(Data) == qn1

    local_K_map = {k: iloc for iloc, k in enumerate(K)} # iloc = 0, ..., nK-1
    nK = len(K)

    Mat = get_interpolation_matrix(X, xkin, Data, scalefactor, model_degree)

    # build right-hand side from function and derivative values
    if model_degree == 2:
        
        rhs = np.zeros(qn1)
        for i, dat in enumerate(Data):
            j, alpha = dat
            deg = sum(alpha)
            if deg == 0:
                rhs[i] = F[j]
            elif deg == 1:
                k = multiindex2derivatives(alpha)[0]
                ik = local_K_map[k]
                rhs[i] = scalefactor * DF[j,ik] # k -> ik
            elif deg == 2:
                k1, k2 = multiindex2derivatives(alpha)
                ik1 = local_K_map[k1]
                ik2 = local_K_map[k2]
                ik = idx_uppertri2vec(ik1, ik2, nK)
                rhs[i] = scalefactor**2 * D2F[j,ik]

    try:  # try to prevent instability from small scalings
    
        coefvec = np.linalg.solve(Mat, rhs)

    except np.linalg.LinAlgError:

        # re-build the system without normalizing by scalefactor
        scalefactor = 1.0

        Mat = get_interpolation_matrix(X, xkin, Data, scalefactor, model_degree)

        if model_degree == 2:
            
            rhs = np.zeros(qn1)
            for i, dat in enumerate(Data):
                j, alpha = dat
                deg = sum(alpha)
                if deg == 0:
                    rhs[i] = F[j]
                elif deg == 1:
                    k = multiindex2derivatives(alpha)[0]
                    ik = local_K_map[k]
                    rhs[i] = scalefactor * DF[j,ik] # k -> ik
                elif deg == 2:
                    k1, k2 = multiindex2derivatives(alpha)
                    ik1 = local_K_map[k1]
                    ik2 = local_K_map[k2]
                    ik = idx_uppertri2vec(ik1, ik2, nK)
                    rhs[i] = scalefactor**2 * D2F[j,ik]

        try:

            coefvec = np.linalg.solve(Mat, rhs)

        except np.linalg.LinAlgError:

            print("Encountered Singular Interpolation Matrix.")
            c = None
            g = None
            H = None
            sigma = None
            return c, g, H, sigma

    # end try/except


    # undo the scaling from the linear solve
    coefvec[1 : n + 1] /= scalefactor
    if model_degree == 2:
        coefvec[n + 1 :] /= scalefactor**2


    # compute the measure of stationarity
    if model_degree == 2:
        c, g, H = polynomial2matrix(coefvec, n)

    sigma = get_sigma(g, H, model_degree)

    # return in long vector form? ...
    if vec_out:
        if model_degree == 1:
            coefvec = np.concatenate(([c], g, symmetric_matrix2vec(H, n)))
        return coefvec, sigma

    # ... or matrix form?
    else:
        return c, g, H, sigma


##########################################################################

if __name__ == "__main__":

    n = 2
    qn1 = (n+1) * (n+2) // 2

    x0 = np.array([1, 2])

    tt = [0, np.pi/4, np.pi/2, 3*np.pi/2, -np.pi/3]
    X = np.zeros((len(tt)+1, n)) + x0 # model center at origin
    Delta = 2.0
    rad = Delta
    for i, t in enumerate(tt):
        X[i+1] = x0 + rad * np.array([np.cos(t), np.sin(t)])
    nfs = len(X)

    # pad X with data before and after current model
    X = np.append(X, np.zeros((qn1,n)), axis=0)
    npad = 10
    X = np.append(np.zeros((npad,n)), X, axis=0)
    nfmax = len(X)
    nfs += npad

    Data = [(0+npad, (0,0)), (0+npad, (1,0)), (2+npad, (0,0)), (5+npad, (0,0)), (2+npad, (1,0)), (5+npad, (1,0))]
    xkin = 0+npad                    # X[xkin] == model center
    K = np.array([0])
    
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

    for i in range(npad, nfs):
        val, grad, Hess = fun(X[i])
        F[i] = val
        DF[i,:] = grad[K]
        D2F[i,:] = symmetric_matrix2vec(Hess[np.ix_(K,K)], nK)

    # compute model
    c, g, H, sigma = get_interpolation_model(X, K, n, F, DF, D2F, xkin, Data, scalefactor=Delta, model_degree=2, vec_out=False)
    coefvec = np.concatenate(([c], g, symmetric_matrix2vec(H, n)))

    # check error by comparing polynomial model to function info
    from polyval2 import polyval2, Dpolyval2
    err = np.zeros(len(Data))
    local_K_map = {k: iloc for iloc, k in enumerate(K)} # iloc = 0, ..., nK-1
    nK = len(K)
    for i,dat in enumerate(Data):
        j, alpha = dat
        deg = sum(alpha)
        err[i] = Dpolyval2(coefvec, X[j] - x0, alpha)
        if deg == 0:
            err[i] -= F[j]
        elif deg == 1:
            k = multiindex2derivatives(alpha)[0]
            ik = local_K_map[k]
            err[i] -= DF[j,ik]
        elif deg == 2:
            k1, k2 = multiindex2derivatives(alpha)
            ik1 = local_K_map[k1]
            ik2 = local_K_map[k2]
            ik = idx_uppertri2vec(ik1, ik2, nK)
            err[i] -= D2F[j,ik]

    print("X =")
    print(X)
    print("Data =")
    print(Data)
    print("Model (c, g, H) =")
    print((c, g, H))
    print("Interpolation Error =")
    print(np.linalg.norm(err))


###################################################################
