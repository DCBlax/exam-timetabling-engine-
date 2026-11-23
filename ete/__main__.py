"""Command line:  python -m ete (--instance FILE | --family random|uz --n N --seed S)
                               [--solver hybrid] [--time-limit 10] [--out timetable.json]
"""
from __future__ import annotations

import argparse
import json

from . import SOLVERS, allocate_venues, is_valid, load_instance, lower_bound
from .generators import GENERATORS


def main() -> None:
    ap = argparse.ArgumentParser(prog="ete", description="Exam Timetabling Engine")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--instance", help="JSON instance file (see ete.generators.load_instance)")
    src.add_argument("--family", choices=sorted(GENERATORS), help="generate a synthetic instance")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--solver", choices=sorted(SOLVERS), default="hybrid")
    ap.add_argument("--time-limit", type=float, default=10.0)
    ap.add_argument("--out", help="write the timetable (slots + venue seating) as JSON")
    a = ap.parse_args()

    inst = load_instance(a.instance) if a.instance else GENERATORS[a.family](a.n, a.seed)
    solver = SOLVERS[a.solver]
    res = solver(inst) if a.solver in ("greedy", "dsatur") else solver(inst, a.time_limit)
    lb, _ = lower_bound(inst)
    print(f"{inst.name}: {inst.n} exams, {inst.num_edges} conflicts, capacity {inst.capacity}/slot")
    print(f"{res.algorithm}: {res.slots} slots (lower bound {lb}), status={res.status}, "
          f"nodes={res.nodes}, {res.seconds:.3f}s, valid={is_valid(inst, res.assignment)}")
    if a.out:
        seating = allocate_venues(inst, res.assignment)
        data = {"instance": inst.name, "slots": res.slots, "status": res.status,
                "timetable": {str(s): [{"exam": inst.labels[e], "venue": f"V{v + 1}",
                                        "seats": seats} for e, v, seats in rows]
                              for s, rows in seating.items()}}
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=1)
        print("timetable written to", a.out)


if __name__ == "__main__":
    main()
