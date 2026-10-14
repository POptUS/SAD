import numpy as np


def make_K_set(deriv_prop, n, rng):
    """
    Create a set K of known derivative directions
    which is a given proportion of the total number of variables.
    The outputted set K will contain between 1 and n-1 variables.
    To request |K| = 0 or n is straightforward and should be performed
    by the user outside of this function.

    INPUTS
        deriv_prop  [real] proportion of variables which should be assigned to K
        n           [int] total number of decision variables
        rng         [seed] rng seed to create reproducible results
    """

    nK = deriv_prop * n

    nK = round(nK)

    # K = np.arange(n)[ rng.random(n) < 0.5 ] # each coordinate has Pr=0.5 chance to have its derivative known

    if nK == n:  # make sure not all derivs are known
        nK = n - 1
    elif nK == 0:  # make sure at least one deriv is known
        nK = 1

    K = rng.choice(n, size=nK, replace=False)
    K = np.sort(K)

    return K
