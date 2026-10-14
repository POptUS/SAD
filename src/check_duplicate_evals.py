import numpy as np


def check_duplicate_evals(X, eval_list, tol=1e-8):
    """Diagnostic-only near-duplicate check over an eval history.

    eval_counter/min_SAD.py's existing self-check only catches *exact* (i, alpha) or
    (X[i], alpha) duplicates. This additionally flags pairs of distinct indices i != j
    evaluated at the same alpha whose coordinates coincide to within `tol`, which the
    exact check misses. Not wired into the algorithm's hot path -- purely for post-hoc
    inspection of a completed run's eval_list.

    Returns (ok, dups) where ok is True iff no near-duplicate pair was found, and dups
    is a list of (i, j, alpha) tuples for each near-duplicate pair found (i < j).
    """
    by_alpha = {}
    for idx, alpha in eval_list:
        by_alpha.setdefault(alpha, []).append(idx)

    dups = []
    for alpha, indices in by_alpha.items():
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                i, j = indices[a], indices[b]
                if i == j:
                    continue
                if np.linalg.norm(X[i] - X[j]) < tol:
                    dups.append((min(i, j), max(i, j), alpha))

    dups = sorted(set(dups))
    return len(dups) == 0, dups
