import numpy as np

from idx_uppertri2vec import idx_uppertri2vec


def symmetric_matrix2vec(Mat, n):
    """
    Given a symmetric n-by-n matrix Mat, return the defining upper triangular entries
    in a single vector according to row major ordering
    """

    assert Mat.shape == (n, n)

    vec = np.zeros(n * (n + 1) // 2)

    idx = 0
    for i in range(n):
        vec[idx : idx + n - i] = Mat[i, i:n]
        idx += n - i
        # for j in range(i,n):
        #    vec[idx_uppertri2vec(i,j,n)] = Mat[i, j]

    return vec
