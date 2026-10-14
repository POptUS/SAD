"""
This function shifts and scales the set of points (Y). 
The first entry of Y is assumed to be the trust region center.

Y is npts-by-n
"""

import numpy as np


def shift_and_scale(Y, yc, scalefactor=None, tol=1e-12, backupfactor=1.0, printf=True):
    """
    do an affine transformation of the points Y so that it lies in the unit ball,
    with at least one point on the sphere.
    INPUTS:
        Y [npts-by-n] is a set of points
        yc is model center "y0"
        scalefactor [dbl] factor by which to normalize the centered points, e.g., the trust region radius
            if None, calculates the smallest radius from y0 containing Y
    OUTPUTS:
        Yhat [npts-by-n] is the normalized transformation of Y
        scalefactor is the least positive number R so Y is contained in Ball(y0, R)
    """

    npts, n = np.shape(Y)

    # preallocate
    Yhat = np.copy(Y)

    # center
    Yhat = Yhat - yc

    # scale
    if npts > 1:

        if scalefactor is None:
            ynorms = np.linalg.norm(Yhat, axis=1)  # row norms
            scalefactor = max(ynorms)

            if scalefactor < tol: # detect when Y contains (copies of) a single point
                scalefactor = backupfactor
                if printf:
                    print(f"Detected Delta(Y)=0. Rescaling using factor {backupfactor} instead.")
            # end if
        # end if

        Yhat = Yhat / scalefactor

    elif scalefactor is None:
        scalefactor = backupfactor  # default value if Y is only one point

    return Yhat, scalefactor


def undo_shift_and_scale(Y, yc, scalefactor):
    return scalefactor * Y + yc
