import numpy as np

from vec2symmetric_matrix import vec2symmetric_matrix


def polynomial2matrix(coefvec, n):
    """
    This function takes a polynomial of the form:
        a_0
        + a_1 x_0 + ... + a_n x_{n-1}
        + a_{n+1} .5*x_0^2 + a_{n+2} x_0 x_1 + ... + a_{end} .5*x_{n-1}^2
    and turns it into the scalar (c), vector (g), and  matrix (H).
    The polynomial is equivalent to .5*x.T@H@x + g.T@x + c

    Ported from CSV_DFO
    """

    c = coefvec[0]
    g = coefvec[1 : n + 1]
    coefvec = coefvec[n + 1 :]  # Remove the constant and linear terms
    H = vec2symmetric_matrix(coefvec, n)

    return c, g, H
