"""
Runs every test_*.py in this directory as a subprocess (judging each by its printed output
rather than collecting it with pytest) and prints a final PASS/FAIL/SKIP summary table. Each
test_*.py prints a trailing "RESULT: PASS|FAIL|SKIP" line that this script parses; a test
crashing without printing that line (e.g. an uncaught exception) is reported as FAIL.
"""
import glob
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run_one(path):
    proc = subprocess.run([sys.executable, path], cwd=HERE, capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    result = "FAIL"
    for line in out.splitlines():
        if line.startswith("RESULT: "):
            result = line.split("RESULT: ", 1)[1].strip()
    return result, out


def main():
    test_files = sorted(
        f for f in glob.glob(os.path.join(HERE, "test_*.py"))
    )

    summary = []
    for path in test_files:
        name = os.path.basename(path)
        result, out = run_one(path)
        summary.append((name, result))
        print(f"===== {name} =====")
        print(out.rstrip())
        print()

    print("========== SUMMARY ==========")
    width = max(len(n) for n, _ in summary) if summary else 0
    for name, result in summary:
        print(f"{name.ljust(width)}  {result}")

    n_fail = sum(1 for _, r in summary if r == "FAIL")
    n_pass = sum(1 for _, r in summary if r == "PASS")
    n_skip = sum(1 for _, r in summary if r == "SKIP")
    print(f"\n{n_pass} passed, {n_fail} failed, {n_skip} skipped ({len(summary)} total)")

    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
