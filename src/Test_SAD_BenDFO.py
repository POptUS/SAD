"""
Unit test of compute function
"""

import os
import unittest

import ibcdfo.pounders as pdrs
import numpy as np
import scipy as sp
from calfun import calfun
from dfoxs import dfoxs
from min_SAD import min_SAD


if not os.path.exists("benchmark_results"):
    os.makedirs("benchmark_results")

dfo = np.loadtxt("dfo.dat")

spsolver = 2
nf_max = 50
g_tol = 1e-13
factor = 10

for row, (nprob, n, m, factor_power) in enumerate(dfo):
    n = int(n)
    m = int(m)

    def Ffun(y):
        out = calfun(y, m, int(nprob), "smooth", 0, num_outs=1)
        grad = np.zeros(len(y))
        Hess = np.zeros((len(y), len(y)))
        return np.squeeze(out), grad, Hess

    X0 = dfoxs(n, nprob, int(factor**factor_power))

    K = np.array([], dtype=int)
    alpha_admissible = [(0,) * n]
    nfmax = 1000
    Delta = 2.0
    nfs = 0
    F0 = np.array([])
    DF0 = np.array([])
    D2F0 = np.array([])
    xkin = 0
    options = {
        "eps_c": 1e-6,  # enter criticality step if norm(grad) at TR center is <= eps_c
        "xi_imp": 1 + 1e-10,
        "xi_acc": 1e-4,
        "printf": True,
    }

    X, F, sigmahist, xkin, flag = min_SAD(Ffun, X0, n, K, alpha_admissible, nfmax, Delta, nfs, F0, DF0, D2F0, xkin, options)

    Results["SAD_4py_" + str(row) + "_" + str(hfun_cases)] = {}
    Results["SAD_4py_" + str(row) + "_" + str(hfun_cases)]["alg"] = "SAD_4py"
    Results["SAD_4py_" + str(row) + "_" + str(hfun_cases)]["problem"] = "problem " + str(row) + " from More/Wild"
    Results["SAD_4py_" + str(row) + "_" + str(hfun_cases)]["Fvec"] = F
    Results["SAD_4py_" + str(row) + "_" + str(hfun_cases)]["H"] = hF
    Results["SAD_4py_" + str(row) + "_" + str(hfun_cases)]["X"] = X

    sp.io.savemat(filename, Results)
