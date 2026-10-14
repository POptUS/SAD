import numpy as np

from symmetric_matrix2vec import symmetric_matrix2vec


def matrix2polynomial(c, g, H):
    """
    This function takes a polynomial of the form:
        .5*x.T@H@x + g.T@x + c
    and turns it into the long vector a_0, ..., a_{(n+1)*(n+2)/2 - 1}
    so that the polynomial is equivalent to
        a_0
        + a_1 x_0 + ... + a_n x_{n-1}
        + a_{n+1} .5*x_0^2 + a_{n+2} x_0 x_1 + ... + a_{end} .5*x_{n-1}^2

    Ported from CSV_DFO
    """

    n = len(g)

    coefvec = np.zeros((n + 1) * (n + 2) // 2)

    coefvec[0] = c
    coefvec[1 : n + 1] = g
    coefvec[n + 1 :] = symmetric_matrix2vec(H, n)

    return coefvec
