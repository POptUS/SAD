import numpy as np
from make_K_set import make_K_set

def make_grand_K_list(problems, deriv_props, rng):
    # deriv_props = list of known derivative proportions

    ns = len(deriv_props)

    # history for plotting
    nprob = len(problems)

    # make K sets first for reproducibility before encountering the solver
    K_list = [[np.array([], dtype=int) for p in range(nprob)] for s in range(ns)]
    for p in range(nprob):

        n = problems[p].n

        for s in range(ns):

            deriv_prop = deriv_props[s]

            if deriv_prop > 0:
                if deriv_prop == 1:
                    K = np.arange(n)
                else:
                    K = make_K_set(deriv_prop, n, rng)

                K_list[s][p] = K

    return K_list
