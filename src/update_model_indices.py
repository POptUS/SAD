import numpy as np


def update_model_indices(X, model_idx, perm_idx, npts, n_interp):
    """
    Update indices of the points used in determining a Hermite interpolation model

    INPUTS:
    X           [dbl] [npts-by-n] iterate history
                    recommended to input history only up to current optimization time
    model_idx   [int] model indices before updating
    perm_idx    [int] [>=n_interp] permutation of indices from complete_nonpoised_set
    npts        [int] number of points in X prior to the additions and removals here
    """

    # indices of new points and existing (old) points
    idx_new = np.nonzero(perm_idx == -1)[0]
    idx_old = np.nonzero(perm_idx >= 0)[0]

    # shuffle the existing points based on the pivoting order
    model_idx[idx_old] = model_idx[perm_idx[idx_old]]

    # add the new points
    for i in idx_new:
        model_idx[i] = npts
        npts += 1

    # if model_idx came with too many points for interpolation, we have removed points from a nonpoised set
    # from complete_nonpoised_set, points beyond the first n_interp should be removed from the model
    if len(model_idx) > n_interp:
        model_idx = model_idx[0:n_interp]

    # make sure the model has the right number of degrees of freedom
    assert len(model_idx) == n_interp

    return model_idx
