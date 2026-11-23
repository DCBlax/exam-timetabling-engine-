"""Single-command reproduction:  python reproduce.py [--quick]

Runs the unit tests, the benchmark and the analysis.  Everything lands in results/.
--quick is a ~30 s smoke run; without it the full study (n = 10..500, 5 s limit) is run.
"""
import subprocess
import sys

quick = ["--quick"] if "--quick" in sys.argv else []
py = sys.executable
steps = [
    [py, "-m", "pytest", "-q"],
    [py, "experiments/run_experiments.py", *quick],
    [py, "experiments/analyze.py"],
]
for s in steps:
    print("\n$", " ".join(s), flush=True)
    if subprocess.call(s) != 0:
        sys.exit("step failed: " + " ".join(s))
print("\nDone. See results/summary.md and results/fig_*.png")
