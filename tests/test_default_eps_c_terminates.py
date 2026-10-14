"""
Regression test for item 1 of the low-risk tuning-fix plan: default_options["eps_c"] was 0,
which is the gate on the *only* clean-convergence exit (s1_trigger_criticality.py: `if sigma <=
eps_c`). With eps_c=0 that exit can never fire (sigma == 0 exactly is not achievable in floating
point for Rosenbrock), so a run that omits eps_c from options burns its entire nfmax budget even
after converging -- flagged by a past comment in test/BenDFO/test_pounder_and_SAD.py ("NOTE: run
forever when eps_c=0"). This runs the Rosenbrock case with eps_c omitted, in a subprocess with a
hard wall-clock timeout, and checks it returns promptly with len(eval_list) <= nfmax.
"""
import os
import subprocess
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from _test_utils import report, finish, require_minqsw

require_minqsw()

TIMEOUT_SECONDS = 60

CHILD_SCRIPT = """
import sys, os
sys.path.append({src_dir!r})
import numpy as np
from min_SAD import min_SAD

def rosen(x):
    val = (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
    grad = np.array([-2 * (1 - x[0] + 200 * x[0] * (x[1] - x[0] ** 2)), 200 * (x[1] - x[0] ** 2)])
    Hess = np.array([[2 - 400 * (x[1] - 3 * x[0] ** 2), -400 * x[0]], [-400 * x[0], 200]])
    return val, grad, Hess

X0 = np.array([[-1.2, 1.0]])
n = 2
K = np.array([0], dtype=int)
alpha_admissible = [(0, 0), (1, 0)]
nfmax = 560
Delta = 0.3
options = {{"xi_imp": 1 + 1e-10, "xi_acc": 1e-4, "printf": False}}  # eps_c intentionally omitted -> default

X, F, sigmahist, xkin, eval_list, flag, phase_counts = min_SAD(
    rosen, X0, [], n, K, alpha_admissible, nfmax, Delta, 0, np.array([]), np.array([]), np.array([]), 0, options
)
print(f"len(eval_list)={{len(eval_list)}}")
print(f"nfmax={{nfmax}}")
assert len(eval_list) <= nfmax
"""


def main():
    src_dir = os.path.join(os.path.dirname(__file__), "..", "src")
    script = CHILD_SCRIPT.format(src_dir=src_dir)

    results = []
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=TIMEOUT_SECONDS,
        )
        finished_in_time = True
        detail = proc.stdout.strip().splitlines()[-2:] if proc.stdout.strip() else [proc.stderr[-500:]]
        exit_ok = proc.returncode == 0
    except subprocess.TimeoutExpired:
        finished_in_time = False
        exit_ok = False
        detail = [f"did not return within {TIMEOUT_SECONDS}s"]

    results.append(report("default_eps_c::terminates_within_timeout", finished_in_time,
                           "; ".join(detail)))
    results.append(report("default_eps_c::exits_cleanly_within_budget", exit_ok,
                           "; ".join(detail)))

    finish(all(results))


if __name__ == "__main__":
    main()
