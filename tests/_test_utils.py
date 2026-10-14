import numpy as np


def check_no_repeated_evals(X, eval_list):
    """
    Given the point history X and eval_list (an eval_counter.queries-style list of (i, alpha)
    pairs actually evaluated), verify that no (point, derivative) pair was queried more than
    once. Factored out of min_SAD.py's __main__ repeat-eval smoke check so it can be reused by
    test_min_SAD_end_to_end.py as a standalone, non-interactive check.

    Returns (ok, duplicates) where duplicates is a list of ((i,alpha),(j,alpha)) collision pairs.
    """
    tagged = [(tuple(np.round(X[i], 12)), alpha) for (i, alpha) in eval_list]
    seen = {}
    duplicates = []
    for k, tup in enumerate(tagged):
        if tup in seen:
            duplicates.append((eval_list[seen[tup]], eval_list[k]))
        else:
            seen[tup] = k
    return (len(duplicates) == 0), duplicates


def report(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    suffix = f" -- {detail}" if detail else ""
    print(f"[{status}] {name}{suffix}")
    return passed


def finish(all_passed):
    print("RESULT: " + ("PASS" if all_passed else "FAIL"))
    raise SystemExit(0 if all_passed else 1)


def skip(reason):
    print(f"[SKIP] {reason}")
    print("RESULT: SKIP")
    raise SystemExit(0)


def require_minqsw():
    """
    Nearly every subroutine below complete_nonpoised_set.py (max_abs_quadratic -> min_quadratic)
    imports minqsw at module load time, so most of this test suite needs it on PYTHONPATH too.
    Skip cleanly rather than crash when it isn't available, matching min_quadratic.py's own
    instructive-error convention for the same missing-dependency case.
    """
    try:
        import minqsw  # noqa: F401
    except ModuleNotFoundError:
        skip(
            "minqsw not importable -- clone https://github.com/POptUS/minq and add "
            "minq/py/minq5 to PYTHONPATH before running this test"
        )
