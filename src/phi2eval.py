"""
quadratic basis function evaluation
taken from IBCDFO/pounders
"""

import numpy as np
from idx_uppertri2vec import idx_uppertri2vec
from multiindex2derivatives import multiindex2derivatives

def phi2eval(X):
    """
    Input:
    X is a [m-by-n] numpy array
    Output:
    Phi is a [m-by-(n*(n+1)/2)] numpy array
    """
    nd = X.ndim
    X = np.atleast_2d(X)

    [m, n] = np.shape(X)
    
    Phi = np.zeros((m, int(0.5 * n * (n + 1))))
    X2 = 0.5 * np.atleast_2d(X**2)

    j = 0
    for k in range(0, n):
        Phi[:, j] = X2[:, k]
        j += 1
        # Hadamard product of column vectors
        Phi[:, j : j + n - k - 1] = np.multiply(X[:, [k]], X[:, k + 1 :])  # / np.sqrt(2) # Evan removes this factor
        j += n - k - 1

    if nd == 1:
        Phi = Phi[0, :]

    return Phi


################################################################


def Dphi2eval(X, i):
    """
    Evaluate the first derivatives of monomial basis functions with respect to the i-th variable.

    Input:
    X is a [m-by-n] numpy array (points where derivatives are evaluated)
    i is the index of the variable with respect to which the derivative is computed (0-based index)

    Output:
    dPhi is a [m-by-(n*(n+1)/2)] numpy array containing the derivatives.
    """
    nd = X.ndim
    X = np.atleast_2d(X)
    [m, n] = np.shape(X)
    
    dPhi = np.zeros((m, int(0.5 * n * (n + 1))))

    j = 0
    for k in range(0, n):
        if k == i:
            # Derivative of X[k]^2/2 with respect to X[i] is X[k]
            dPhi[:, j] = X[:, k]
        j += 1

        # Derivative of X[k] * X[l] (k < l) with respect to X[i]
        for l in range(k + 1, n):
            if k == i:
                dPhi[:, j] = X[:, l]
            elif l == i:
                dPhi[:, j] = X[:, k]
            j += 1

    if nd == 1:
        dPhi = dPhi[0, :]

    return dPhi

def D2phi2eval(X, i, j):
    """
    Evaluate the second derivatives of monomial basis functions with respect to the i-th & j-th variables.

    Input:
    X is a [m-by-n] numpy array (points where derivatives are evaluated)
    i,j are indices of the variables with respect to which the derivative is computed

    Output:
    d2Phi is a [m-by-(n*(n+1)/2)] numpy array containing the derivatives.
    """
    nd = X.ndim
    if nd == 1: 
        n = len(X)
        d2Phi = np.zeros(int(0.5 * n * (n + 1)))
        d2Phi[idx_uppertri2vec(i,j,n)] = 1.0
    elif nd == 2:
        [m, n] = np.shape(X)
        d2Phi = np.zeros((m, int(0.5 * n * (n + 1))))
        d2Phi[:, idx_uppertri2vec(i,j,n)] = 1.0

    return d2Phi

def Dalpha_phieval(X, alpha):
    """
    Evaluate the second derivatives of monomial basis functions with respect to the multiindex alpha.

    Input:
    X is a [m-by-n] numpy array (points where derivatives are evaluated)
    i,j are indices of the variables with respect to which the derivative is computed

    Output:
    d2Phi is a [m-by-(n*(n+1)/2)] numpy array containing the derivatives.
    """
    nd = X.ndim
    if nd == 1:
        m = 1
        n = len(X)
    elif nd == 2:
        [m, n] = np.shape(X)
    
    assert len(alpha) == n
    
    Dphi = np.zeros((m, int(0.5 * (n + 1) * (n + 2))))
    deg = sum(alpha) # order of the derivative
    ilist = multiindex2derivatives(alpha)

    assert deg <= 2

    if deg == 0:

        Dphi[:, 0] = 1.0 
        Dphi[:, 1 : n + 1] = X 
        Dphi[:, n + 1 :] = phi2eval(X)

    elif deg == 1:

        i = ilist[0]
        Dphi[:, 0] = 0.0
        Dphi[:, 1 + i] = 1.0 # [1:n+1] will be e^i unit vec
        Dphi[:, n + 1 :] = Dphi2eval(X, i) 


    elif deg == 2:

        i,j = ilist
        Dphi[:, 0] = 0.0
        Dphi[:, 1 : n + 1] = 0.0
        Dphi[:, n + 1 :] = D2phi2eval(X, i, j)

    if nd == 1:
        Dphi = Dphi[0,:]

    return Dphi

def Dpoly2(poly, alpha, n):
    """
    Apply d^alpha to monomial basis representation of poly
    """
    assert len(alpha) == n

    deg = sum(alpha)
    ilist = multiindex2derivatives(alpha)
    dpoly = np.zeros(int(0.5 * (n + 1) * (n + 2)))
    
    if deg == 0:
        return poly # dpoly = poly
    elif deg == 1:
        i = ilist[0]
        dpoly[0] = poly[1 + i]
        for j in range(n):
            dpoly[1 + j] = poly[1 + n + idx_uppertri2vec(min(i,j),max(i,j),n)]
    elif deg == 2:
        i,j = ilist
        dpoly[0] = poly[1 + n + idx_uppertri2vec(i,j,n)]
    
    return dpoly

