import numpy as np

from phi2eval import Dalpha_phieval
from shift_and_scale import shift_and_scale


def get_interpolation_matrix(X, xkin, Data, scalefactor=1.0, model_degree=2):
    """
    generate the system matrix for Birkhoff interpolation
    scalefactor [dbl] rescaling factor in order to put matrix entries to order 1
    """

    _ , n = np.shape(X)

    # normalize points
    Xhat, _ = shift_and_scale(X, X[xkin], scalefactor)

    # preallocate
    if model_degree == 2:
        qn1 = (n + 1) * (n + 2) // 2
        Mat = np.zeros((qn1, qn1))  # when fully determined, npts = qn1-qnK1+1

    if model_degree == 2:

        for i, dat in enumerate(Data):
            j, alpha = dat
            Mat[i, :] = Dalpha_phieval(np.atleast_2d(Xhat[j]), alpha)

    return Mat

###################################################################################

if __name__ == "__main__":

    n = 2
    qn1 = (n+1) * (n+2) // 2

    tt = [0, np.pi/4, np.pi/2, 3*np.pi/2, -np.pi/4]
    X = np.zeros((len(tt)+1, n)) # model center at origin
    rad = 1.0
    for i, t in enumerate(tt):
        X[i+1] = rad * np.array([np.cos(t), np.sin(t)])
    nfs = len(X)

    # pad X with data before and after current model
    X = np.append(X, np.zeros((qn1,n)), axis=0)
    npad = 10
    X = np.append(np.zeros((npad,n)), X, axis=0)
    nfmax = len(X)
    nfs += npad

    Data = [(0+npad, (0,0)), (0+npad, (1,0)), (5+npad, (0,0)), (2+npad, (0,0)), (2+npad, (1,0)), (3+npad, (1,0))]
    xkin = 0+npad                    # X[xkin] == model center
    Delta = 1.0

    Mat = get_interpolation_matrix(X, xkin, Data, scalefactor=Delta, model_degree=2)

    print("X =")
    print(X)
    print("Data =")
    print(Data)
    print("Interpolation Matrix =")
    print(Mat)
    print("Condition number =")
    print(np.linalg.cond(Mat))

