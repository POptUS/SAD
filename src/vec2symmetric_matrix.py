import numpy as np


def vec2symmetric_matrix(vec, n):
    """
    Given a vector vec of length n*(n+1)/2, return the symmetric matrix
    with upper triangular entries given in vec, assuming row major ordering
    """

    Mat = np.zeros((n, n))  # Build the quadratic term
    for k in range(n):
        Mat[k, k:] = vec[0 : n - k]
        vec = vec[n - k :]

    # Make Mat symmetric
    d = np.diag(Mat)
    Mat = Mat - np.diag(d)
    Mat = Mat + Mat.T + np.diag(d)

    return Mat
