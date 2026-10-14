import numpy as np

from polyval2 import polyval2


def hermite_interp_check(fun, model_coefvec, Y, ykin, K):
    """
    Check if the inputted model satisfies the Hermite interpolation conditions
    """

    npts, n = Y.shape

    Y_shifted = Y - Y[ykin]

    fun_values = np.zeros(npts)
    for i in range(npts):
        fun_values = fun(Y[i], nargout=1)

    # deriv values

    model_values = polyval2(model_coefvec, Y_shifted)
