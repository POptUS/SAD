import numpy as np


def get_points_in_trust_region(X, xkin, Delta, npts):
    """
    Determine the points in X which are contained in a Delta-radius ball centered
    at the current iterate indexed by xkin

    INPUTS
        X       [dbl] [m-by-n]
        xkin    [int] index of the center of the Euclidean ball within X
        Delta   [dbl] radius within which to find points
        npts    [int] only search the first npts entries in X
    """

    # center of the trust region
    y0 = X[xkin, :]

    # get distances from all other points to the center point
    dists = np.linalg.norm(X[0:npts] - y0, axis=1)  # Euclidean norm

    # indices of points within Delta of center
    idx_in_trust_region = np.nonzero(dists <= Delta)[0]  # use [0] to coerce to numpy 1d array
    idx_in_trust_region = idx_in_trust_region.tolist()
    # the points
    Y = X[idx_in_trust_region, :]

    return Y, idx_in_trust_region
