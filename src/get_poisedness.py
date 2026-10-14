import numpy as np

from get_HIPs import get_HIPs
from max_abs_quadratic import max_abs_quadratic
from polynomial2matrix import polynomial2matrix
from phi2eval import Dpoly2

def get_poisedness(X, alpha_admissible, xkin, Data, Delta, model_degree=2):
    """
    Determine the poisedness of the points Y with respect to known derivative indices K
    on the ball of radius Delta centered at y0.
    Assumes that y0 is the first point in Y.
    Y is an npts-by-n matrix
    ykin is index of Y corresponding to y0
    """

    Lambda = 0.0
    n = len(X[0])

    # generate HIPs
    CoefMat = get_HIPs(X, xkin, Data, Delta, model_degree)
    npoly, _ = np.shape(CoefMat)

    if np.isnan(CoefMat).any():  # interpolation matrix was singular, and Y is not poised
        return np.inf

    if model_degree == 2:

        # for each HIP
        for j in range(npoly):

            for k, alpha in enumerate(alpha_admissible):

                # take appropriate derivative and translate to matrix form
                coefvec = Dpoly2(CoefMat[:, j], alpha, n)
                c, g, H = polynomial2matrix(coefvec, n)

                _, val = max_abs_quadratic(c, g, H, 1.0)  # use unit ball here since points are normalized

                # candidate for Lambda constant
                if val > Lambda:
                    Lambda = val

    return Lambda



if __name__ == "__main__":

    n = 2
    qn1 = (n+1) * (n+2) // 2

    tt = [0, np.pi/4, np.pi/2, 3*np.pi/2, -np.pi/4]
    X = np.zeros((len(tt)+1, n)) # model center at origin
    rad = 1.0
    for i, t in enumerate(tt):
        X[i+1] = rad * np.array([np.cos(t), np.sin(t)])

    # pad X with enough space
    X = np.append(X, np.zeros((qn1,n)), axis=0)

    idx_in = [0, 3, 4, 0, 1, 2]    # indices of points in X to use for model=building
    alpha_list = [(0,0), (0,0), (0,0), (1,0), (1,0), (1,0)]
    Data = [(idx_in[i], alpha_list[i]) for i in range(qn1)]
    alpha_admissible = [ (0,0), (1,0) ]
    xkin = 0                    # X[xkin] == model center
    Delta = 1.0

    COEF = get_HIPs(X, xkin, Data, Delta, model_degree=2)
    Lambda = get_poisedness(X, alpha_admissible, xkin, Data, Delta, model_degree=2)
    print("Lambda =")
    print(Lambda)
    print("COEF =")
    print(COEF)


    print("Double Delta:")
    Delta = 2*Delta
    X = 2*X
    print("X=")
    print(X)
    Data = [(idx_in[i], alpha_list[i]) for i in range(qn1)]
    COEF = get_HIPs(X, xkin, Data, Delta, model_degree=2)
    Lambda = get_poisedness(X, alpha_admissible, xkin, Data, Delta, model_degree=2)
    
    print("Lambda =")
    print(Lambda)
    print("COEF =")
    print(COEF)

    #############################################################
    # 1-poised set
    #############################################################
    t1 = np.pi/4
    X = np.array([[0,0], [0,1], [0,-1], [np.cos(t1),np.sin(t1)], [np.cos(-t1),np.sin(-t1)], [-1,0]])
    alpha_list = [(0,0),(0,0),(0,0),(1,0),(1,0),(1,0)]
    Data = [(i, alpha_list[i]) for i in range(qn1)]
    xkin = 0                    # X[xkin] == model center
    Delta = 1.0
    
    Lambda = get_poisedness(X, alpha_admissible, xkin, Data, Delta, model_degree=2)

    print("Lambda =")
    print(Lambda)


