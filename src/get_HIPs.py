import numpy as np

from get_interpolation_matrix import get_interpolation_matrix
from polyval2 import polyval2
from shift_and_scale import shift_and_scale

def get_HIPs(X, xkin, Data, scalefactor, model_degree=2, Xeval=None):
    """
    evaluate the Hermite interpolating polynomials (HIPs), defined by Y and K, at Xeval.
    Xeval is an m-by-n array, where m is the number of points to evaluate
    Returns an m-by-(q(n)+1) array
    """

    Mhat = get_interpolation_matrix(X, xkin, Data, scalefactor, model_degree)  # rescale to unit ball
    n = len(X[0, :])

    degrees = [sum(tup[1]) for tup in Data]

    # dimension of the polynomial space
    if model_degree == 2:
        npoly = (n + 1) * (n + 2) // 2

    # solve for the HIP coefficients
    try:
        Coefs = np.linalg.solve(Mhat, np.eye(npoly))
    except:
        Coefs = np.empty_like(Mhat)
        Coefs.fill(np.nan)
        if Xeval is None:
            return Coefs
        else:
            m = len(Xeval)
            lambdab = np.empty((m, npoly))
            lambdab.fill(nan)
            return Coefs, lambdab

    # evaluate
    if Xeval is None:
        return Coefs
    else:
        m = len(Xeval)
        lambdab = np.zeros((m, npoly))  # "lambdab" = "lambda bold"

        # normalize Xeval to the same space as the HIPs
        Xeval, _ = shift_and_scale(Xeval, X[xkin], scalefactor)

        if model_degree == 2:
            for j in range(npoly):
                lambdab[:, j] = polyval2(Coefs[:, j], Xeval)

        return Coefs, lambdab


#########################################################################################################
if __name__ == "__main__":

    from polyval2 import Dpolyval2

    n = 2
    qn1 = (n+1) * (n+2) // 2

    x0 = np.array([1,2.])

    tt = [0, np.pi/4, np.pi/2, 3*np.pi/2, -np.pi/4]
    X = np.zeros((len(tt)+1, n)) + x0
    rad = 1.0
    for i, t in enumerate(tt):
        X[i+1] = x0 + rad * np.array([np.cos(t), np.sin(t)])

    idx_in = [0, 3, 4, 0, 1, 2]    # indices of points in X to use for model=building
    alpha_list = [(0,0), (0,0), (0,0), (1,0), (1,0), (1,0)]
    Data = [(idx_in[i], alpha_list[i]) for i in range(qn1)]
    xkin = 0                    # X[xkin] == model center
    Delta = 1.0

    COEF = get_HIPs(X, xkin, Data, Delta, model_degree=2)

    print("COEF =")
    print(COEF)

    CheckMat = np.zeros((qn1,qn1))
    Xhat, _ = shift_and_scale(X, X[xkin], Delta)
    for j in range(qn1):
        kj, alpha = Data[j]
        for i in range(qn1):
            CheckMat[i,j] = Dpolyval2(COEF[:,i], Xhat[kj], alpha)
    print("Error =")
    print(np.linalg.norm(CheckMat - np.eye(qn1)))


    ###################################
    print("Double Delta:")
    Delta = 2*Delta
    X = 2*(X - x0) + x0
    print("X=")
    print(X)
    Data = [(idx_in[i], alpha_list[i]) for i in range(qn1)]
    COEF = get_HIPs(X, xkin, Data, Delta, model_degree=2)
    
    print("COEF =")
    print(COEF)

    CheckMat = np.zeros((qn1,qn1))
    Xhat, _ = shift_and_scale(X, X[xkin], Delta)
    for j, (kj, alpha) in enumerate(Data):
        for i in range(qn1):
            CheckMat[i,j] = Dpolyval2(COEF[:,i], Xhat[kj], alpha)
    print("Error =")
    print(np.linalg.norm(CheckMat - np.eye(qn1)))

