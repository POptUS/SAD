import os
import sys

import numpy as np

sys.path.append("../src")
import pycutest
from make_K_set import make_K_set

from mpi4py import MPI

from make_pyCUTEst_problem_list import make_pyCUTEst_problem_list
from min_SAD import min_SAD
from make_alpha_admissible_from_K import make_alpha_admissible_from_K
from make_grand_K_list import make_grand_K_list

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

########################################
########################################
# loop over problems
########################################
########################################

problems = make_pyCUTEst_problem_list()

flags = -1 * np.ones(len(problems))

# set reproducibility
rng = np.random.default_rng(seed=122333)

# solver parameters -- how many known derivative directions?
deriv_props = [0, 0.25, 0.5, 0.75, 1]  # proportions of known derivatives for synthesizing K sets
ns = len(deriv_props)

# history for plotting
nprob = len(problems)
print(f"Num problems = {nprob}")
nfmaxs = np.zeros(nprob, dtype=int)
for p in range(nprob):
    nfmaxs[p] = 1000 * (problems[p].n + 1)

model_degree = 2    # polynomial degree for model (1->linear, 2->quadratic)
max_deriv = 2       # maximum partial derivative order to use in interpolation

outdir_str = f"./benchmark_results_model_degree_{model_degree}_max_deriv_{max_deriv}/"
if not os.path.exists(outdir_str):
    os.makedirs(outdir_str)

# make K sets first for reproducibility before encountering the solver
K_list = make_grand_K_list(problems, deriv_props, rng)

for s in range(ns):

    deriv_prop = deriv_props[s]

    for p in range(nprob):

        problem = problems[p]

        print("--------------------------------")
        print(f"Problem: %s" % (problem.name))
        print("--------------------------------")

        # objective function with gradient and Hessian evals
        def objfun(x):
            val = problem.obj(x)
            grad = problem.grad(x)
            Hess = problem.hess(x)
            return val, grad, Hess

        x0 = problem.x0

        X0 = np.array([x0])
        n = problem.n

        assert n <= 15

        K = K_list[s][p]
        Pre_Evaluated_Pairs = []
        max_deriv = 2 # max |alpha| for interpolation
        alpha_admissible = make_alpha_admissible_from_K(K, n, max_deriv)
        nfmax = nfmaxs[p] # min(nfmaxs[p], 3000)
        Delta = 1.0
        if problem.name in ["SENSORS", "MODBEALE"]:
            Delta = 0.9 # prevent initial model from selecting points which are local max or saddle points, for specific problems
        nfs = 0
        F0 = np.array([])
        DF0 = np.array([])
        D2F0 = np.array([])
        xkin = 0
        options = {
            "Deltamax"  : 1e3 * Delta,   # maximum trust region radius
            "Deltamin"  : 1e-14 * Delta, # minimum '''''''''''''''''''
            "eta"       : 0.1,           # step acceptance threshold for the improvement ratio
            "gam_dec"   : 0.5,           # factor by which to reduce the trust region radius, 0<gam_dec<1
            "gam_inc"   : 2.0,           # '''''''''''''''''' expand ''''''''''''''''''''''', 1<gam_inc
            "eps_c"     : 0,             # enter criticality step if norm(grad) at TR center is <= eps_c NOTE: run forever when eps_c=0
            "mu"        : 0.9,           # factor to determine whether to run the criticality step
            "beta"      : 0.5,           # factor to determine the new trust radius in criticality step
            "xi_acc"    : 1e-1,          # pivot threshold
            "xi_imp"    : 1.01,          # model-improvement threshold
            "r"         : 1.0,           # trust-region radius factor
            "tol_ineq"  : 1e-12,         # tolerance for model_improvement to detect when points are within Delta-ball
            "tol_eq"    : 1e-14,         # tolerance for satisfying equality that ||step||=Delta to enlarge TR radius
            "tol_step_ratio" : 0.1,      # tolerance on the ratio between step size and Delta
            "model_degree"   : 2,
            "printf"         : True,
            "plotf"          : False,
            }

        if len(K) == 0:
            K_string = "None"
        else:
            K_string = "_".join(map(str, K))
        outfilename = f"probname={problem.name}_nfmax={nfmax}_K={K_string}.npy"

        task_id = s * nprob + p
        print("task", task_id)

        # Only process if the task_id modulo size is equal to rank
        if task_id % size != rank:
            continue

        if os.path.exists(outdir_str + outfilename):
            print("Already solved: ", outfilename)
            continue

        print(f"n = {n}")
        print(f"K = {K}")

        X, F, sigmahist, xkin, eval_list, flag, phase_counts = min_SAD(objfun, X0, Pre_Evaluated_Pairs, n, K, alpha_admissible, nfmax, Delta, nfs, F0, DF0, D2F0, xkin, options)

        import json
        import socket

        hostname = socket.gethostname()
        outfile_txt = f"min_SAD_output_{hostname}.json"

        def to_serializable(obj):
            import numpy as np
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return str(obj)

        with open(outfile_txt, "w") as f:
            json.dump({
                "X": X,
                "F": F,
                "sigmahist": sigmahist,
                "xkin": xkin,
                "eval_list": eval_list,
                "flag": flag
            }, f, default=to_serializable, indent=2)

        
        print(f"Finished Problem: %s" % (problem.name))
        print(f"n = {n}")
        print(f"K = {K}")
        print("Exit flag is:")
        print(flag)
        print("min_SAD returns best decision variables:")
        print(X[xkin])
        print("Best function value found is:")
        print(F[xkin])

        # get FULL gradient values for optimization history as a postprocessing step
        DF_norm = np.zeros(len(X))
        for i in range(len(X)):
            gradf = problem.grad(X[i])
            DF_norm[i] = np.linalg.norm(gradf)

        print("Full gradient's norm at X[xkin]:")
        print(DF_norm[xkin])

        Results = {"X": X, "F": F, "DF_norm": DF_norm, "xkin": xkin, "eval_list": eval_list, "flag": flag}

        np.save(outdir_str + outfilename, Results)

    # end for
# end for

print("Ending run_test_functions.py")
