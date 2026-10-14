import os
import sys
import re

import matplotlib.pyplot as plt
plt.rcParams["text.usetex"] = True
plt.rcParams.update({
    "font.size": 16,         # default text size
    "axes.titlesize": 18,    # title
    "axes.labelsize": 18,    # x and y labels
    "xtick.labelsize": 14,   # x tick labels
    "ytick.labelsize": 14,   # y tick labels
    "legend.fontsize": 14,   # legend
    "figure.titlesize": 20,  # figure title
})
import numpy as np
import sparse

sys.path.append("../src")
import pycutest

sys.path.append("../benchmark")
from make_pyCUTEst_problem_list import make_pyCUTEst_problem_list
from make_K_set import make_K_set
from make_grand_K_list import make_grand_K_list
from plot_data_profile import plot_data_profile

##############################################

usefile = False # use saved file if available; False to rerun data amalgamation

problems = make_pyCUTEst_problem_list()

# SAD solver parameters -- how many known derivative directions?
deriv_props = [0.25, 0.5, 0.75]  # proportions of known derivatives for synthesizing K sets
ns_SAD = len(deriv_props)

# Adding scipy solvers
# scipy_solvers = ["BFGS", "trust-ncg", "L-BFGS-B", "NM"]
scipy_solvers = []
ns_scipy = len(scipy_solvers)

# adding Mona solvers
deriv_props_mona = [0.25, 0.5, 0.75]
# deriv_props_mona = [] # currently ignore Mona's method
ns_mona = len(deriv_props_mona)

# total number of solvers
ns = ns_SAD + ns_scipy + ns_mona 


# history for plotting
nprob = len(problems)
N = np.inf * np.ones(nprob) # normalization factor for counting function evals
nfmaxs = np.zeros(nprob, dtype=int)
for p in range(nprob):
    N[p] = problems[p].n + 1
    nfmaxs[p] = 1000 * N[p]
bundle_cutoff = 500 # stop plotting after 500*N evals

# expanded these sizes to accomodate evaluating the full gradient at every point
NMAX = 15
FHIST = np.nan * np.ones((max(nfmaxs)*(NMAX+1)*(NMAX+2)//2, nprob, ns))  # history of function values
DFHIST = np.nan * np.ones((max(nfmaxs)*(NMAX+1)*(NMAX+2)//2, nprob, ns))  # history of gradient norm

# model degree and data directory
model_degree = 2
max_deriv = 2
outdir_str = f"../benchmark/benchmark_results_model_degree_{model_degree}_max_deriv_{max_deriv}/"
if not os.path.exists(outdir_str):
    os.makedirs(outdir_str)

# set reproducibility
rng = np.random.default_rng(seed=122333)

K_list = make_grand_K_list(problems, deriv_props, rng)

# read in data if available
filepath = "DFHIST_sp.npz"

if usefile and os.path.exists(filepath):
    
    DFHIST = sparse.load_npz(filepath)
    DFHIST = DFHIST.todense()

else:
    
    ###########################################################
    # Loop for SAD results

    _,_,files = next(os.walk(outdir_str), (None, None, []))

    MAXEVALS = 1 # find most number of evals used in a run
    for s in range(ns_SAD):
        deriv_prop = deriv_props[s]
        for p in range(nprob):
            
            problem = problems[p]
            n = problem.n

            K = K_list[s][p]

            if len(K) == 0:
                K_string = "None"
            else:
                K_string = "_".join(map(str, K))

            outfileregex = f"probname={problem.name}" + r".*" + re.escape(f"K={K_string}.npy")
            regex = re.compile(outfileregex)

            # Load results
            file = [fname for fname in files if regex.search(fname)]
            file_found = (file != [])
            
            if file_found:
                file = file[0]
                Results = np.load(outdir_str + file, allow_pickle=True).item()
            else:
                print(f"Did not find results for problem {problem.name} and K={K_string}. Loading null data.")
                Results = {"X": np.zeros((0,n)), "F": np.zeros(0), "DF_norm": np.zeros(0), "eval_list": [], "xkin": np.nan, "flag": np.nan}

            F = Results["F"]
            DF_norm = Results["DF_norm"]
            eval_list = Results["eval_list"]

            MAXEVALS = max(MAXEVALS, len(eval_list))

            for num, (i, alpha) in enumerate(eval_list):
                FHIST[num, p, s] = F[i]

            # get FULL gradient values for optimization history as a postprocessing step
            for num, (i, alpha) in enumerate(eval_list):
                DFHIST[num, p, s] = DF_norm[i]

    ###########################################################
    # Add results for Mona (Hermite least-squares baseline)
    for solver_idx, deriv_prop_mona in enumerate(deriv_props_mona):
        s = deriv_props.index(deriv_prop_mona)
        for p in range(nprob):
            problem = problems[p]
            n = problem.n

            nfmax = nfmaxs[p]

            K = K_list[s][p]
            nK = len(K)
            if nK == 0:
                K_string = "None"
            else:
                K_string = "_".join(map(str, K))

            outdir_str = "../benchmark/hermite_baseline/benchmark_results_mona/"
            outfilename = f"probname={problem.name}_nfmax={nfmax}_K={K_string}.npy"

            # Load results if available
            if os.path.exists(outdir_str + outfilename):
                Results = np.load(outdir_str + outfilename, allow_pickle=True).item()
                F = Results["F"]
                DF_norm = Results["DF_norm"]

                # Add to FHIST and DFHIST for this solver
                # Mona's method uses function value and first derivative information to build a regression system
                # eval at each x includes f and df/dx_i for nK values of i
                FHIST[0 : len(F)*(1 + nK), p, ns_SAD + ns_scipy + solver_idx] = np.repeat(F, 1 + nK) 
                DFHIST[0 : len(F)*(1 + nK), p, ns_SAD + ns_scipy + solver_idx] = np.repeat(DF_norm, 1 + nK)

            else:
                print("Results not found for Mona on " + outfilename)

    # truncate
    FHIST = FHIST[0:MAXEVALS, :, :]
    DFHIST = DFHIST[0:MAXEVALS, :, :]

    # # compress and save
    # DFHIST_sp = sparse.COO.from_numpy(DFHIST, fill_value=np.nan)
    # sparse.save_npz(filepath, DFHIST_sp) # save

# end if

# # only look at n<=8 objectives
# inds = (N<=9)
# DFHIST = DFHIST[:,inds,:]

# Plotting
ngate = 8
opt_type = "absgrad"
# legendstr = ["n_K/n="+str(pr) for pr in deriv_props] + scipy_solvers + ["Mona " + str(prm) for prm in deriv_props_mona]
legendstr = (
        [rf"$n_K/n = {pr:.2f}$ " + "Birkhoff" for pr in deriv_props] +
        [rf"$n_K/n = {prm:.2f}$ " + "Hermite" for prm in deriv_props_mona]
)
# legendstr[7] = "BFGS (DFO)" # hard-coded
print(f"Shape of DFHIST = {np.shape(DFHIST)}")
for gate in np.logspace(-ngate, -1, ngate):
    plot_data_profile(DFHIST, N, gate, optimality_type=opt_type, legendstr=legendstr)
    # plt.title(f"Objective gradient tolerance = {gate:1.1e}")
    # plt.title("CUTEst Problems Performance by Proportion of Known Derivatives")
    plt.xlabel(r"Number of Function/Derivative Evaluations / ($n+1$)")  # overwrite xlabel
    plt.xlim((0, bundle_cutoff))
    # plt.show()
    plt.tight_layout()
    plt.savefig(opt_type + f"={gate:1.1e}.png", dpi=400, bbox_inches="tight", pad_inches=0.01)
    plt.close()
