#!/usr/bin/env python3
r"""
In this example we both a stage 1 and stage 2 optimization problems using the
single stage approach of R. Jorge et al in https://arxiv.org/abs/2302.10622
The objective function in this case is J = J_stage1 + coils_objective_weight*J_stage2
To accelerate convergence, a stage 2 optimization is done before the single stage one.
Rogerio Jorge, April 2023
"""

deriv_method = "SAD" # options are "DFO", "DFOJ1", "FDJ1", "SAD", "SADFDJ1opt"
# "FDJ1"  -> use finite differences for approximate gradients of J_stage_1
# "DFO"   -> set x_coil to approximate optimal values and use pounders for J_stage_1
# "DFOJ1" -> build quadratic model for J_stage_1 to get gradients; use available gradients for J_stage_2
# "SAD"   -> Some Available Derivatives, but use nK=2 coil params and nU=8 surface params. Fix remaining 135 coil params.
# SADFDJ1opt" -> not implemented
MAXNF_eval = 150 # default budget for the "DFO"/"DFOJ1" paths; the "SAD" case below overrides this to match the paper's 500(n+1) budget

# import SAD library (absolute paths: this script later os.chdir()s into a
# VMEC working directory, so plain relative sys.path entries would break)
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# ibcdfo.pounders needs MINQ (https://github.com/POptUS/MINQ) as its subproblem
# solver; clone it and point MINQ5_PATH at its py/minq5 directory.
sys.path.insert(1, os.environ.get("MINQ5_PATH", os.path.join(SCRIPT_DIR, "..", "..", "minq", "py", "minq5")))
sys.path.insert(1, os.path.join(SCRIPT_DIR, "..", "src"))
from min_SAD import min_SAD
sys.path.append(os.path.join(SCRIPT_DIR, "..", "benchmark"))
from make_alpha_admissible_from_K import make_alpha_admissible_from_K

import numpy as np
from pathlib import Path
from scipy.optimize import minimize
from simsopt.util import MpiPartition, proc0_print, comm_world
from simsopt._core.derivative import Derivative
from simsopt.mhd import Vmec, QuasisymmetryRatioResidual
from simsopt._core.finite_difference import MPIFiniteDifference
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.objectives import SquaredFlux, QuadraticPenalty, LeastSquaresProblem
from simsopt.geo import (CurveLength, CurveCurveDistance, MeanSquaredCurvature,
                         LpCurveCurvature, ArclengthVariation, curves_to_vtk, create_equally_spaced_curves)
from ibcdfo.pounders import formquad, bmpts, pounders
#from ibcdfo.manifold_sampling import build_p_models

mpi = MpiPartition()
parent_path = str(Path(__file__).parent.resolve())
os.chdir(parent_path)
##########################################################################################
############## Input parameters
##########################################################################################
MAXITER_stage_2 = 10
MAXITER_single_stage = 10 # default 10
max_mode = 1
vmec_input_filename = os.path.join(parent_path, 'inputs', 'input.nfp4_QH_warm_start')
ncoils = 3
aspect_ratio_target = 7.0
CC_THRESHOLD = 0.08
LENGTH_THRESHOLD = 3.3
CURVATURE_THRESHOLD = 7
MSC_THRESHOLD = 10
nphi_VMEC = 34
ntheta_VMEC = 34
nmodes_coils = 7
coils_objective_weight = 1e+3
aspect_ratio_weight = 1
diff_method = "forward"
R0 = 1.0 # major axis of initial guess circular coils
R1 = 0.6 # minor axis """""""""""""""""""""""""""""""
quasisymmetry_target_surfaces = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
finite_difference_abs_step = 1e-7
finite_difference_rel_step = 0
J_THRESHOLD = 100.
LENGTH_CON_WEIGHT = 0.1     # Weight on the quadratic penalty for the curve length
LENGTH_WEIGHT = 1e-8        # Weight on the curve lengths in the objective function
CC_WEIGHT = 1e+0            # Weight for the coil-to-coil distance penalty in the objective function
CURVATURE_WEIGHT = 1e-3     # Weight for the curvature penalty in the objective function
MSC_WEIGHT = 1e-3           # Weight for the mean squared curvature penalty in the objective function
ARCLENGTH_WEIGHT = 1e-9     # Weight for the arclength variation penalty in the objective function

##########################################################################################
############### SAD parameters
##########################################################################################
if deriv_method == "SAD" or deriv_method == "SADFDJ1opt":
   
    Delta = 1.0e-5
    nfs = 0
    Pre_Evaluated_Pairs = []
    xkin = 0
    F0 = np.array([])
    DF0 = np.array([])
    D2F0 = np.array([])
    options = {
        "eps_c"     :  0.0,             # exhaust function count
        "printf"    : True,
        "model_degree" : 2,
        "r"         : 1.0,              # containment-radius generalization (Algorithm 2); r=1 recovers the paper's stated algorithm
        }
    max_deriv = 1                      # maximum derivative degree available WRT known variables


##########################################################################################
##########################################################################################
directory = 'optimization_QH'
vmec_verbose = False
# Create output directories
this_path = os.path.join(parent_path, directory)
os.makedirs(this_path, exist_ok=True)
os.chdir(this_path)
vmec_results_path = os.path.join(this_path, "vmec")
coils_results_path = os.path.join(this_path, "coils")
if comm_world.rank == 0:
    os.makedirs(vmec_results_path, exist_ok=True)
    os.makedirs(coils_results_path, exist_ok=True)
##########################################################################################
##########################################################################################
# Stage 1
proc0_print(f' Using vmec input file {vmec_input_filename}')
vmec = Vmec(vmec_input_filename, mpi=mpi, verbose=vmec_verbose, nphi=nphi_VMEC, ntheta=ntheta_VMEC, range_surface='half period')
surf = vmec.boundary
##########################################################################################
##########################################################################################
#Stage 2
base_curves = create_equally_spaced_curves(ncoils, surf.nfp, stellsym=True, R0=R0, R1=R1, order=nmodes_coils, numquadpoints=128)
base_currents = [Current(1) * 1e5 for _ in range(ncoils)]
base_currents[0].fix_all()
##########################################################################################
##########################################################################################
# Save initial surface and coil data
coils = coils_via_symmetries(base_curves, base_currents, surf.nfp, True)
curves = [c.curve for c in coils]
bs = BiotSavart(coils)
bs.set_points(surf.gamma().reshape((-1, 3)))
Bbs = bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3))
BdotN_surf = np.sum(Bbs * surf.unitnormal(), axis=2)
if comm_world.rank == 0:
    curves_to_vtk(curves, os.path.join(coils_results_path, "curves_init"))
    pointData = {"B_N": BdotN_surf[:, :, None]}
    surf.to_vtk(os.path.join(coils_results_path, "surf_init"), extra_data=pointData)
##########################################################################################
##########################################################################################
Jf = SquaredFlux(surf, bs, definition="local")
Jls = [CurveLength(c) for c in base_curves]
Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=len(curves))
Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for i, c in enumerate(base_curves)]
Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
Jals = [ArclengthVariation(c) for c in base_curves]
J_LENGTH = LENGTH_WEIGHT * sum(Jls)
J_CC = CC_WEIGHT * Jccdist
J_CURVATURE = CURVATURE_WEIGHT * sum(Jcs)
J_MSC = MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD) for i, J in enumerate(Jmscs))
J_ALS = ARCLENGTH_WEIGHT * sum(Jals)
J_LENGTH_PENALTY = LENGTH_CON_WEIGHT * sum([QuadraticPenalty(Jls[i], LENGTH_THRESHOLD) for i in range(len(base_curves))])
JF = Jf + J_CC + J_LENGTH + J_LENGTH_PENALTY + J_CURVATURE + J_MSC
##########################################################################################
proc0_print('  Starting optimization')
##########################################################################################
# Initial stage 2 optimization
##########################################################################################
## The function fun_coils defined below is used to optimize the coils at the beginning for a warm start,
## and then later optimize the coils and the surface together. This makes the overall optimization
## more efficient as the number of iterations needed to achieve a good solution is reduced.


def fun_coils(dofss, info):
    info['Nfeval'] += 1
    # info['X'].append(dofss)
    JF.x = dofss
    J = JF.J()
    # info['F'].append(J)
    grad = JF.dJ()
    if mpi.proc0_world:
        jf = Jf.J()
        Bbs = bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3))
        BdotN_surf = np.sum(Bbs * surf.unitnormal(), axis=2)
        BdotN = np.mean(np.abs(BdotN_surf))
        # BdotNmax = np.max(np.abs(BdotN_surf))
        outstr = f"fun_coils#{info['Nfeval']} - J={J:.1e}, Jf={jf:.1e}, ⟨B·n⟩={BdotN:.1e}"  # , B·n max={BdotNmax:.1e}"
        outstr += f", ║∇J coils║={np.linalg.norm(JF.dJ()):.1e}, C-C-Sep={Jccdist.shortest_distance():.2f}"
        cl_string = ", ".join([f"{j.J():.1f}" for j in Jls])
        kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
        msc_string = ", ".join(f"{j.J():.1f}" for j in Jmscs)
        outstr += f" lengths=sum([{cl_string}])={sum(j.J() for j in Jls):.1f}, curv=[{kap_string}],msc=[{msc_string}]"
        print(outstr)
    return J, grad

##########################################################################################
##########################################################################################
# Define database evaluation parameters (function-eval caches are written
# alongside this script so repeated runs can reuse expensive VMEC evaluations)
OUTDIR = SCRIPT_DIR
database_J_stage_1 = os.path.join(OUTDIR, "database_J_stage_1.npy")
database_J_stage_2 = os.path.join(OUTDIR, "database_J_stage_2.npy")
database_grad_J_stage_1_FD = os.path.join(OUTDIR, "database_grad_J_stage_1_FD.npy")
use_DB = True
rtol = 1e-12
atol = 1e-12

##########################################################################################
##########################################################################################

def J_stage_1_database_eval(
    database,
    dofs,
    info={'N_J_stage_1_eval': 0, 'N_J_stage_2_eval': 0},
    use_DB=True,
    rtol=1e-12,
    atol=1e-12,
 ):
    """
    database    (path) path to function evaluation history over multiple runs
    dofs        (array) independent variables, including x_surf and x_coil
    info        (dict) information about the current optimization process
    use_DB      (bool) true -> draw values from database when available; false -> evaluate functions from scratch each time
    rtol, atol  (real>0) tolerances for determining if inputs are already in the database
    """
    info['N_J_stage_1_eval'] += 1
    # info['X'].append(dofs) # asdf
    DB = []
    match = 0
    x_surf = dofs[-number_vmec_dofs:] # last components are x_surf

    # search database for existing entry
    if use_DB and os.path.exists(database):
        DB = np.load(database, allow_pickle=True)
        for db_entry in DB:
            if np.allclose(db_entry["x_surf"], x_surf, rtol=rtol, atol=atol):
                J_stage_1 = db_entry["J_stage_1"]
                match = 1
                break
    
    # evaluate the function if no match
    if match == 0:
        # print("Evaluating J_stage_1")
        prob.x = x_surf
        os.chdir(vmec_results_path)
        J_stage_1 = prob.objective()
        if use_DB:
            to_save = {
                    "x_surf": x_surf,
                    "J_stage_1": J_stage_1,
                    }
            DB = np.append(DB, to_save)
            np.save(database, DB)

    return J_stage_1 

##########################################################################################
##########################################################################################

def J_stage_2_database_eval(
    database,
    dofs,
    info={'N_J_stage_1_eval': 0, 'N_J_stage_2_eval': 0},
    use_DB=True,
    rtol=1e-12,
    atol=1e-12,
):
    info['N_J_stage_2_eval'] += 1
    DB = []
    match = 0
    x_coil = dofs[:-number_vmec_dofs]
    x_surf = dofs[-number_vmec_dofs:]

    if use_DB and os.path.exists(database):
        DB = np.load(database, allow_pickle=True)
        for db_entry in DB:
            if np.allclose(db_entry["x_surf"], x_surf, rtol=rtol, atol=atol) and np.allclose(db_entry["x_coil"], x_coil, rtol=rtol, atol=atol):
                J_stage_2 = db_entry["J_stage_2"]
                coils_dJ  = db_entry["coils_dJ"]
                mixed_dJ  = db_entry["mixed_dJ"]
                match = 1
                break

    if match == 0: # evaluate
        # print("Evaluating J_stage_2")
        JF.x = dofs[:-number_vmec_dofs] # first components are x_coils
        bs.set_points(surf.gamma().reshape((-1, 3)))
        #os.chdir(vmec_results_path)
        J_stage_2 = JF.J().item() # Evan adds .item() to coerce to scalar

        ## derivative of J_stage_2 (function of x_surf and x_coils)
        coils_dJ = JF.dJ() # dJ2/dx1
        ## Mixed term - derivative of squared flux with respect to the surface shape
        n = surf.normal()
        absn = np.linalg.norm(n, axis=2)
        B = bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3))
        dB_by_dX = bs.dB_by_dX().reshape((nphi_VMEC, ntheta_VMEC, 3, 3))
        Bcoil = bs.B().reshape(n.shape)
        unitn = n * (1./absn)[:, :, None]
        Bcoil_n = np.sum(Bcoil*unitn, axis=2)
        mod_Bcoil = np.linalg.norm(Bcoil, axis=2)
        B_n = Bcoil_n
        B_diff = Bcoil
        B_N = np.sum(Bcoil * n, axis=2)
        assert Jf.definition == "local"
        dJdx = (B_n/mod_Bcoil**2)[:, :, None] * (np.sum(dB_by_dX*(n-B*(B_N/mod_Bcoil**2)[:, :, None])[:, :, None, :], axis=3))
        dJdN = (B_n/mod_Bcoil**2)[:, :, None] * B_diff - 0.5 * (B_N**2/absn**3/mod_Bcoil**2)[:, :, None] * n
        deriv = surf.dnormal_by_dcoeff_vjp(dJdN/(nphi_VMEC*ntheta_VMEC)) + surf.dgamma_by_dcoeff_vjp(dJdx/(nphi_VMEC*ntheta_VMEC))
        mixed_dJ = Derivative({surf: deriv})(surf) # dJ2/dx2

        if use_DB:
            to_save = {
                    "x_surf": x_surf,
                    "x_coil": x_coil,
                    "J_stage_2": J_stage_2,
                    "coils_dJ": coils_dJ,
                    "mixed_dJ": mixed_dJ
                    }
            DB = np.append(DB, to_save)
            np.save(database, DB)

    return J_stage_2, coils_dJ, mixed_dJ

##########################################################################################
##########################################################################################

def fun_for_SAD(x_SAD, return_full_coil_grad=False):

    x_coils_free = x_SAD[:nk_coil]
    x_coils_fixed = dofs_FDJ1_opt[nk_coil:-number_vmec_dofs]
    x_surf = x_SAD[nk_coil:]
    dofs = np.concatenate((x_coils_free, x_coils_fixed, x_surf)) # reconstruct the canonical ordering of variables for objective J

    J_stage_1 = J_stage_1_database_eval(database_J_stage_1, dofs, use_DB=use_DB, rtol=rtol, atol=atol)
    J_stage_2, coils_dJ, mixed_dJ = J_stage_2_database_eval(database_J_stage_2, dofs, use_DB=use_DB, rtol=rtol, atol=atol) 
    J = J_stage_1 + coils_objective_weight * J_stage_2
    
    # flag if objective function is too big
    if J > J_THRESHOLD or np.isnan(J):
        proc0_print(f"Exception caught during function evaluation with J={J}. Returning J={J_THRESHOLD}")
        J = J_THRESHOLD
        grad_with_respect_to_surface = [0.0] * number_vmec_dofs
        grad_with_respect_to_coils = [0.0] * len(JF.x)
    else:
        grad_with_respect_to_coils = coils_objective_weight * coils_dJ

    # give option to return full coil grad for post-optimization analysis
    if return_full_coil_grad:
        gradK = np.array(grad_with_respect_to_coils)
    else:
        gradK = np.array(grad_with_respect_to_coils[:nk_coil]) # only the current grads are returned
    
    HessKK = np.zeros((len(gradK), len(gradK))) # unavailable Hessian
    
    return J, gradK, HessKK

##########################################################################################
##########################################################################################
### set differentiable vars to approximate optimal values (from FDJ1 execution)
### no real "FDJ1" run has been performed here; as a stand-in for a small trial run,
### this is set to the stage-2-warm-started dofs below, right after that warm start completes.
dofs_FDJ1_opt = None

##########################################################################################
#############################################################
## Perform optimization
#############################################################
##########################################################################################

# initial stage 2
surf.fix_all()
surf.fixed_range(mmin=0, mmax=max_mode, nmin=-max_mode, nmax=max_mode, fixed=False)
surf.fix("rc(0,0)")
number_vmec_dofs = int(len(surf.x))
qs = QuasisymmetryRatioResidual(vmec, quasisymmetry_target_surfaces, helicity_m=1, helicity_n=-1)
objective_tuple = [(vmec.aspect, aspect_ratio_target, aspect_ratio_weight), (qs.residuals, 0, 1)]
prob = LeastSquaresProblem.from_tuples(objective_tuple)
dofs = np.concatenate((JF.x, vmec.x))
bs.set_points(surf.gamma().reshape((-1, 3)))
Jf = SquaredFlux(surf, bs, definition="local")
proc0_print(f"Aspect ratio before optimization: {vmec.aspect()}")
proc0_print(f"Mean iota before optimization: {vmec.mean_iota()}")
proc0_print(f"Quasisymmetry objective before optimization: {qs.total()}")
proc0_print(f"Magnetic well before optimization: {vmec.vacuum_well()}")
proc0_print(f"Squared flux before optimization: {Jf.J()}")
proc0_print(f'  Performing stage 2 optimization with ~{MAXITER_stage_2} iterations')
res = minimize(fun_coils, dofs[:-number_vmec_dofs], jac=True, args=({'Nfeval': 0}), method='L-BFGS-B', options={'maxiter': MAXITER_stage_2, 'maxcor': 300}, tol=1e-12)

# process stage 2 results
bs.set_points(surf.gamma().reshape((-1, 3)))
Bbs = bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3))
BdotN_surf = np.sum(Bbs * surf.unitnormal(), axis=2)
if comm_world.rank == 0:
    curves_to_vtk(curves, os.path.join(coils_results_path, "curves_after_stage2"))
    pointData = {"B_N": BdotN_surf[:, :, None]}
    surf.to_vtk(os.path.join(coils_results_path, "surf_after_stage2"), extra_data=pointData)

##########################################################################################
##########################################################################################
# set starting guess as output from stage 2
proc0_print(f'  Performing single stage optimization.')
dofs = np.concatenate((JF.x, vmec.x))
dofs_FDJ1_opt = dofs.copy()  # stand-in for a real "FDJ1" result; see note above
print("Number of VMEC dofs", number_vmec_dofs)

if mpi.proc0_world:
    
    match deriv_method:
        
        case "DFO":
            
            # scipy Nelder-Mead example (old)
            #info = {'N_J_stage_1_eval': 0, 'N_J_stage_2_eval': 0}
            #res = minimize(fun_with_modeled_grad, dofs, args=(None, info), jac=False, method='Nelder-Mead', options={'maxiter': MAXITER_single_stage, 'maxfev': MAXNF_eval}, tol=1e-15)
            
            X0 = x0[-number_vmec_dofs:].reshape((1,-1)) # initial x_surface as row
            ninterpmax = (number_vmec_dofs+1)*(number_vmec_dofs+2)//2 # max number of interpolation points
            gtol = 1e-4 # gradient tolerance (of model)
            delta = 0.01 # initial trust region radius
            nfs = 0 # num function values known in advance
            F0 = []  # ^ the known function values
            xkin = 0 # index of point to start from
            L = -np.inf * np.ones(number_vmec_dofs)
            U = np.inf * np.ones(number_vmec_dofs)

            [X,F,flag,xkin] = pounders(
                    fun_for_pounders_only_DFOdofs,
                    X0,
                    number_vmec_dofs,
                    ninterpmax,
                    MAXNF_eval,
                    gtol,
                    delta,
                    nfs,
                    1, # residual components; only a scalar
                    F0,
                    xkin,
                    L,
                    U,
                    printf=1,
                    spsolver=2,
                    hfun=None,
                    combinemodels=None
                    )
        
        case "FDJ1":

            outdir_str = OUTDIR + "/"
            filename = "single_stage_FDJ1.npy"
            if os.path.exists(outdir_str + filename):
                Results = np.load(outdir_str + filename, allow_pickle=True).item()
                F = Results["F"]
            else:
                info = {'X': [], 'F': [], 'N_J_stage_1_vec': [], 'N_J_stage_1_eval': 0, 'N_J_stage_2_eval': 0}
                with MPIFiniteDifference(prob.objective, mpi, diff_method=diff_method, abs_step=finite_difference_abs_step, rel_step=finite_difference_rel_step) as prob_jacobian:
                    res = minimize(fun_with_modeled_grad, dofs, args=(prob_jacobian, info), jac=True, method='BFGS', options={'maxiter': MAXITER_single_stage, "return_all": True}, tol=1e-15)
                # print(dir(res))
                X = np.array(info['X'])
                F = np.array(info['F'])
                N_J_stage_1_vec = np.array(info['N_J_stage_1_vec'])
                Results = {"X": X, "F": F, "N_J_stage_1_vec": N_J_stage_1_vec}
                np.save(outdir_str + filename, Results)
        
        case "DFOJ1":
            
            info = {
                    'N_J_stage_1_eval': 0, 'N_J_stage_2_eval': 0,
                    "X_surf_hist": np.zeros((MAXNF_eval,number_vmec_dofs)), # x_surf inputs
                    "J_stage_1_hist": np.zeros((MAXNF_eval,1)), # empty array initialization
                    "J_hist": np.zeros((MAXNF_eval,1)), # same for total objective
                    "xkin": 0 # best index so far: intialize as initial point
                   }
            res = minimize(fun_with_modeled_grad, dofs, args=(None, info), jac=True, method='BFGS', options={'maxiter': MAXITER_single_stage}, tol=1e-15)
        
        case "SAD":
            
            # # simulate no coil derivatives available
            # nk_coil = 0
            # K = np.array([], dtype=int) 
            
            # set coil parameters to their previously found optimal values
            # dofs[:-number_vmec_dofs] = dofs_FDJ1_opt[:-number_vmec_dofs]
            # x0 = dofs[-number_vmec_dofs:]
            # X0 = np.atleast_2d(x0) 

            # simulate two coil current derivatives available (idx=0,1) (third is fixed for uniqueness)
            nk_coil = 2
            K = np.arange(nk_coil) # first DOFs are coil DOFs; last DOFs are surface DOFs
            # set fixed coil parameters to their previously found optimal values
            dofs[nk_coil:-number_vmec_dofs] = dofs_FDJ1_opt[nk_coil:-number_vmec_dofs]
            x0 = np.concatenate(( dofs[:nk_coil] , dofs[-number_vmec_dofs:] ))
            X0 = np.atleast_2d( x0 )

            n = number_vmec_dofs + nk_coil  # number unknown derivs + known coil derivs
            MAXNF_eval = 500 * (n + 1)  # paper's shared oracle budget for this instance
            alpha_admissible = make_alpha_admissible_from_K(K, n, max_deriv)

            outdir_str = OUTDIR + "/"
            filename = f"single_stage_SAD_reduced_max_deriv_{max_deriv}_model_degree_{options['model_degree']}.npy"

            if os.path.exists(outdir_str + filename):
                
                Results = np.load(outdir_str + filename, allow_pickle=True).item()
                X = Results["X"]
                xkin = Results["xkin"]
                F = Results["F"]
                sigmahist = Results["sigmahist"]

            else:

                X, F, sigmahist, xkin, eval_list, flag, phase_counts = min_SAD(fun_for_SAD, X0, Pre_Evaluated_Pairs, n, K, alpha_admissible, MAXNF_eval, Delta, nfs, F0, DF0, D2F0, xkin, options)
                Results = {"X": X, "F": F, "sigmahist": sigmahist, "xkin": xkin, "eval_list": eval_list, "flag": flag, "phase_counts": phase_counts}
                np.save(outdir_str + filename, Results)

            import matplotlib.pyplot as plt
            F_lowest = np.copy(F)
            for i in range(1, len(F)):
                F_lowest[i] = min(F_lowest[i-1], F[i])
            n_to_plot = min(2000, len(F)) # up to len(F)
            plt.semilogy(range(n_to_plot), F_lowest[:n_to_plot], "b.-")
            plt.xlabel("Number of function evaluations")
            plt.ylabel("Objective function value")
            plt.plot()
            # plt.show()
            figname = os.path.join(OUTDIR, "Fhist_reduced.png")
            plt.savefig(figname, dpi=300)

            x_opt_SAD = np.concatenate((dofs_FDJ1_opt[:nk_coil], dofs_FDJ1_opt[-number_vmec_dofs:]))
            _, grad_FDJ1, _ = fun_for_SAD(x_opt_SAD, return_full_coil_grad=True)
            _, grad_SAD,  _ = fun_for_SAD(X[xkin], return_full_coil_grad=True)
            grad_diff_norm = np.linalg.norm(grad_FDJ1 - grad_SAD)

            print(f"Coil grad difference between methods = {grad_diff_norm:1.5e}")
            
        
        case _:

            raise ValueError('deriv_method must be one of "DFO", "FDJ1", "DFOJ1", or "SAD".')

mpi.comm_world.Bcast(dofs, root=0)
Bbs = bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3))
BdotN_surf = np.sum(Bbs * surf.unitnormal(), axis=2)
if comm_world.rank == 0:
    curves_to_vtk(curves, os.path.join(coils_results_path, "curves_opt"))
    pointData = {"B_N": BdotN_surf[:, :, None]}
    surf.to_vtk(os.path.join(coils_results_path, "surf_opt"), extra_data=pointData)
bs.save(os.path.join(coils_results_path, "biot_savart_opt.json"))
vmec.write_input(os.path.join(this_path, 'input.final'))
proc0_print(f"Aspect ratio after optimization: {vmec.aspect()}")
proc0_print(f"Mean iota after optimization: {vmec.mean_iota()}")
proc0_print(f"Quasisymmetry objective after optimization: {qs.total()}")
proc0_print(f"Magnetic well after optimization: {vmec.vacuum_well()}")
proc0_print(f"Squared flux after optimization: {Jf.J()}")
BdotN_surf = np.sum(Bbs * surf.unitnormal(), axis=2)
BdotN = np.mean(np.abs(BdotN_surf))
BdotNmax = np.max(np.abs(BdotN_surf))
outstr = f"Coil parameters: ⟨B·n⟩={BdotN:.1e}, B·n max={BdotNmax:.1e}"
outstr += f", ║∇J coils║={np.linalg.norm(JF.dJ()):.1e}, C-C-Sep={Jccdist.shortest_distance():.2f}"
cl_string = ", ".join([f"{j.J():.1f}" for j in Jls])
kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
msc_string = ", ".join(f"{j.J():.1f}" for j in Jmscs)
outstr += f" lengths=sum([{cl_string}])={sum(j.J() for j in Jls):.1f}, curv=[{kap_string}], msc=[{msc_string}]"
proc0_print(outstr)
