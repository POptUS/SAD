import numpy as np

def model_validity(Delta, X, Data, xkin, r, tol_ineq):
    """
    Check whether every point currently used in Data lies within r*Delta of the model center
    X[xkin]. Called from min_SAD.py's main loop right before the accept/reject branch to decide
    (on the reject path only) whether Delta should shrink -- main.tex's Algorithm 3 uses exactly
    this "does D still lie in the current trust region" test to gate the Delta-decrease line.
    Pure read-only check: does not modify Data, X, or Delta itself.
    """

    # check if model is valid
    # determined by checking if all model points are within r*Delta of the model center
    Xmodel = np.array([X[i] for (i, _) in Data])
    dists = np.linalg.norm(Xmodel - X[xkin], axis=1)
    valid = all(dists <= r*Delta + tol_ineq)

    return valid, dists
