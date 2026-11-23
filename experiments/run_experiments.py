"""Run the benchmark.   python experiments/run_experiments.py [--quick]

For every (family, n, seed) instance every algorithm is run under the same
wall-clock limit and one CSV row is written.  Seeds are fixed, so the instance
set is identical on every run (timings naturally vary with the machine).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ete.backtracking import solve_fc_mrv, solve_naive
from ete.bnb import solve_bnb
from ete.common import is_valid, violations
from ete.generators import GENERATORS
from ete.greedy import solve_dsatur, solve_greedy
from ete.hybrid import solve_hybrid
from ete.model import lower_bound

SIZES = [10, 20, 30, 50, 75, 100, 200, 300, 500]
FIELDS = ["family", "n", "seed", "edges", "density", "lb", "algorithm", "slots", "status",
          "nodes", "seconds", "conflicts", "overloads", "valid", "timed_out", "pruned", "ls_iters"]


def algorithms(limit: float):
    return [
        ("Greedy", lambda i: solve_greedy(i)),
        ("DSATUR", lambda i: solve_dsatur(i)),
        ("Naive BT", lambda i: solve_naive(i, limit)),
        ("BT+FC+MRV", lambda i: solve_fc_mrv(i, limit)),
        ("B&B", lambda i: solve_bnb(i, limit)),
        ("Hybrid B&B+LS", lambda i: solve_hybrid(i, limit, seed=0)),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small smoke run (~1 min)")
    ap.add_argument("--time-limit", type=float, default=5.0, help="seconds per algorithm run")
    ap.add_argument("--reps", type=int, default=5, help="instances per (family, n) for n<=100")
    ap.add_argument("--reps-large", type=int, default=3, help="instances per (family, n) for n>100")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results",
                                                   "raw_results.csv"))
    args = ap.parse_args()
    sizes, reps, reps_large, limit = SIZES, args.reps, args.reps_large, args.time_limit
    if args.quick:
        sizes, reps, reps_large, limit = [10, 30, 50, 100], 2, 2, 2.0

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    t_start = time.time()
    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, FIELDS)
        writer.writeheader()
        for family, gen in GENERATORS.items():
            for n in sizes:
                for seed in range(reps if n <= 100 else reps_large):
                    inst = gen(n, seed)
                    lb, _ = lower_bound(inst)
                    for name, run in algorithms(limit):
                        res = run(inst)
                        viol = violations(inst, res.assignment)
                        writer.writerow({
                            "family": family, "n": n, "seed": seed, "edges": inst.num_edges,
                            "density": round(inst.density, 4), "lb": lb, "algorithm": name,
                            "slots": res.slots, "status": res.status, "nodes": res.nodes,
                            "seconds": round(res.seconds, 5),
                            "conflicts": len(viol["conflicts"]) + len(viol["unassigned"]),
                            "overloads": len(viol["overloaded"]),
                            "valid": int(is_valid(inst, res.assignment)),
                            "timed_out": int(res.status != "optimal" and res.seconds >= 0.95 * limit
                                             and name in ("Naive BT", "BT+FC+MRV", "B&B",
                                                          "Hybrid B&B+LS")),
                            "pruned": res.extra.get("pruned", ""),
                            "ls_iters": res.extra.get("ls_iters", ""),
                        })
                        fh.flush()
                    print(f"[{time.time() - t_start:6.0f}s] {family} n={n} seed={seed} done",
                          flush=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
