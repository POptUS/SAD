"""
Regression test for the class of bug just fixed in s3a_accept.py: a call site that supplies the
right NUMBER of positional arguments, but with one omitted partway through the list, so every
argument after the gap silently shifts into the wrong parameter slot. A plain arg-count check
cannot catch this (the count is often still "valid" -- it just fills the wrong names), so instead
this walks src/*.py with `ast`, and for every call site of a target function, checks each simple
Name-valued positional argument against the callee's parameter name AT THAT POSITION. This
codebase's convention is that call-site variable names match callee parameter names almost
everywhere, so a positional argument whose variable name matches a DIFFERENT parameter of the
callee (not the one at its own position) is a strong, low-false-positive signal of exactly this
shift bug.
"""
import ast
import inspect
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from _test_utils import report, finish, require_minqsw

require_minqsw()

SRC_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "src"))

TARGET_FUNCS = [
    "complete_nonpoised_set",
    "model_improvement",
    "model_update",
    "criticality_step",
    "s1_trigger_criticality",
    "s2_TRSP_calculation",
    "s3a_accept",
    "s3r",
]


def import_target(name):
    module = __import__(name)
    return getattr(module, name)


def positional_arg_name(node):
    """Return the variable name of a positional call argument if it's a bare Name, else None."""
    if isinstance(node, ast.Name):
        return node.id
    return None


def find_call_sites(func_name):
    """Return (path, lineno, arg_names, has_star, has_dstar) for each call to func_name in SRC_DIR.

    arg_names: list with one entry per positional argument (None for non-Name expressions).
    has_star/has_dstar: True if the call uses *args/**kwargs, which we can't statically verify.
    """
    sites = []
    for fname in sorted(os.listdir(SRC_DIR)):
        if not fname.endswith(".py"):
            continue
        path = os.path.join(SRC_DIR, fname)
        with open(path) as f:
            src = f.read()
        try:
            tree = ast.parse(src, filename=path)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == func_name:
                has_star = any(isinstance(a, ast.Starred) for a in node.args)
                has_dstar = any(kw.arg is None for kw in node.keywords)
                arg_names = [positional_arg_name(a) for a in node.args if not isinstance(a, ast.Starred)]
                sites.append((path, node.lineno, arg_names, has_star, has_dstar))
    return sites


def check_function(func_name):
    func = import_target(func_name)
    param_names = [p.name for p in inspect.signature(func).parameters.values()]
    param_name_set = set(param_names)

    problems = []
    sites = find_call_sites(func_name)
    for path, lineno, arg_names, has_star, has_dstar in sites:
        if has_star or has_dstar:
            continue  # can't statically verify through *args/**kwargs unpacking

        if len(arg_names) > len(param_names):
            problems.append(
                f"{os.path.relpath(path, SRC_DIR)}:{lineno}: {len(arg_names)} positional args "
                f"exceeds {func_name}'s {len(param_names)} parameters"
            )
            continue

        for pos, name in enumerate(arg_names):
            if name is None:
                continue
            expected = param_names[pos]
            if name != expected and name in param_name_set:
                problems.append(
                    f"{os.path.relpath(path, SRC_DIR)}:{lineno}: positional arg #{pos} is variable "
                    f"'{name}', which is {func_name}'s parameter name at a DIFFERENT position "
                    f"(position {pos} expects '{expected}') -- likely an argument-order/shift bug"
                )

    return len(sites), problems


def main():
    results = []
    total_sites = 0
    for func_name in TARGET_FUNCS:
        n_sites, problems = check_function(func_name)
        total_sites += n_sites
        ok = len(problems) == 0
        detail = f"{n_sites} call site(s) checked" if ok else "; ".join(problems)
        results.append(report(f"call_signature::{func_name}", ok, detail))

    print(f"({total_sites} total call sites scanned across TARGET_FUNCS)")
    finish(all(results))


if __name__ == "__main__":
    main()
