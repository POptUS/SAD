import numpy as np


def prepare_outputs_before_return(X, F, nfs, flag):
    """
    parse the outputs of min_SAD before terminating the optimization procedure
    """
    X = X[:nfs]
    F = F[:nfs]

    if flag == 8:
        print("A geometry/QP-subproblem invariant broke down from floating-point degeneracy "
              "at a vanishingly small trust-region radius. Terminating with best point found so far.")
    elif flag == 7:
        print("Model-building has failed.")
    elif flag == 6:
        print("TRSP failed to find model decrease.")
    elif flag == 5:
        print("Unable to determine an interpolation model. Terminating.")
    elif flag == 4:
        print("A NaN or Inf was encountered during function evaluation. Terminating.")
    elif flag == 3:
        print("Evaluation budget is too low. Terminating.")
    elif flag == 2:
        print("Trust region radius has become smaller than Deltamin. Terminating.")
    elif flag == 1:
        print("Trust region radius has become small in the criticality step. Terminating.")
    elif flag == 0:
        print("Stationarity is sufficiently small.")

    return X, F, flag
