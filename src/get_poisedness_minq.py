import numpy as np
import trustregion

from get_HIPs import get_HIPs

try:
    from minqsw import minqsw
except ModuleNotFoundError as e:
    print(e)
    sys.exit("Ensure a python implementation of MINQ is available. For example, clone https://github.com/POptUS/minq and add minq/py/minq5 to the PYTHONPATH environment variable")


def get_poisedness(Y, K):
    """
    Determine the poisedness of the points Y with respect to known derivative indices K
    on the ball of radius 1 centered at y0.
    Assumes that y0 is the first point in Y
    Y is an npts-by-n matrix
    """

    Delta = 1.0
    Lambda = 0.0
    npts, n = np.shape(Y)

    # generate HIPs
    CoefMat = get_HIPs(Y, K)
    qn1 = (n + 1) * (n + 1) // 2  # size of CoefMat

    # for each HIP
    for j in range(qn1):

        # translate to matrix form
        _, g, H = polynomial_to_matrix(CoefMat[:, j], n)

        # maximize absolute value by minimizing (+1)* and (-1)* HIP
        sl = -Delta * np.ones(n)
        su = Delta * np.ones(n)
        [s1, val1, minq_err, _] = minqsw(0, g, H, sl, su, 0, np.zeros((n, 1)))  # constant term 0, printlevel 0
        val1 = val1.item()  # minqsw returns a matrix
        if minq_err != 0:
            print("WARNING: minq did not find a local minimum")
        # val1 = g.dot(s1) + 0.5*s1.dot(H.dot(s1))

        [s2, val2, minq_err, _] = minqsw(0, -g, -H, sl, su, 0, np.zeros((n, 1)))  # constant term 0, printlevel 0
        val2 = val2.item()
        if minq_err != 0:
            print("WARNING: minq did not find a local minimum")
        # val2 = -g.dot(s2) - 0.5*s2.dot(H.dot(s2))

        # candidate for Lambda
        val = max(-val1, -val2)
        if val > Lambda:
            Lambda = val

    return Lambda


#####################################################################


def polynomial_to_matrix(coefvec, n):
    """
    This function takes a polynomial of the form:
        a_0
        + a_1 x_0 + ... + a_n x_{n-1}
        + a_{n+1} .5*x_0^2 + a_{n+2} x_0 x_1 + ... + a_{end} .5*x_{n-1}^2
    and turns it into the matrix (H), the vector (g), and the scalar (c).
    The polynomial is equivalent to .5*x.T@H@x + g.T@x + c

    Ported from CSV_DFO
    """

    c = coefvec[0]
    g = coefvec[1 : n + 1]
    coefvec = coefvec[n + 1 :]  # Remove the constant and linear terms
    H = np.zeros((n, n))  # Build the quadratic term
    for k in range(n):
        H[k, k:] = coefvec[0 : n - k]
        coefvec = coefvec[n - k :]

    # Make H symmetric
    d = np.diag(H)
    H = H - np.diag(d)
    H = H + H.T + np.diag(d)

    return c, g, H
