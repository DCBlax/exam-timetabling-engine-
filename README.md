# Exam Timetabling Engine (ETE)

A hybrid CSP / graph-colouring solver that builds conflict-free exam timetables with the fewest
time slots, and a benchmark suite comparing six algorithms.

Exams are variables, slots are values. Two exams conflict when they share a student cohort or a
lecturer, and every slot has a seat capacity (the sum of the venue capacities). Without the
capacity constraint this is exactly graph colouring. See `docs/theory.md` for the formal model,
the NP-completeness proof (3-SAT ≤ 3-COLOURING ≤ ETE), correctness proofs, the clique and
capacity lower bounds, the greedy approximation analysis and complexity.

## Layout

```
ete/
  model.py         Instance (CSP), conflict graph, clique + capacity lower bounds
  greedy.py        Welsh-Powell (Greedy) and DSATUR approximations
  backtracking.py  Naive backtracking; backtracking + forward checking + MRV + degree
  bnb.py           Branch-and-bound (DSATUR branching, clique/capacity bound)
  hybrid.py        TabuCol local search to tighten the incumbent, then B&B
  generators.py    Synthetic 'random' and 'uz' instances, JSON load/save
  common.py        Budget (time/node limits), Result, validation, venue allocation
  __main__.py      Command line
experiments/       run_experiments.py (benchmark) and analyze.py (tables, tests, plots)
tests/             pytest unit and edge-case tests
docs/theory.md     proofs and complexity
results/           generated CSVs, summary.md and figures
```

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q                         # tests
python reproduce.py --quick                 # tests + ~30 s benchmark + analysis
python reproduce.py                         # full study (n = 10..500, 5 s per run)

python -m ete --family uz --n 200 --solver hybrid --time-limit 10 --out timetable.json
python -m ete --instance data/my_instance.json --solver bnb
```

Solvers for `--solver`: `greedy`, `dsatur`, `naive`, `fcmrv`, `bnb`, `hybrid`.
Timetables are always verified; `--out` writes the slot-by-slot venue seating.

## Using real data

Provide JSON and pass it with `--instance`:

```json
{"name": "2026 Sem 1",
 "venues":  [{"name": "Great Hall", "capacity": 400}],
 "exams":   [{"code": "CS101", "lecturer": "Dr Moyo"}],
 "cohorts": [{"size": 120, "exams": ["CS101", "MA101"]}]}
```

The bundled generators are synthetic. `uz` mimics the faculty / year / shared-service-course
structure of a Zimbabwean university but contains no real UZ data.

## How the algorithms relate

All exact solvers share the same preprocessing (lower bound `lb`, DSATUR upper bound `ub`).
If `ub == lb` the instance is solved with 0 search nodes. Otherwise they differ only in the
search, so node counts isolate the effect of each pruning technique:

| Algorithm | Search |
|---|---|
| Naive BT | static order, constraints checked on assignment, k = lb, lb+1, ... |
| BT+FC+MRV | forward checking, MRV + degree, capacity filtering, symmetry breaking, same k loop |
| B&B | DSATUR branching, prune when `max(used, lb) ≥ best`, anytime |
| Hybrid B&B+LS | TabuCol removes slots first, then B&B with the better incumbent |

On a timeout every solver returns its best valid timetable with status `feasible`. Node counts
for Hybrid exclude local-search iterations (recorded separately as `ls_iters`).

## Reproducibility

Instance seeds are fixed, so the instance set is identical on every run. Timings vary with the
machine. The 5 s limit means large instances are right-censored: `timed_out_pct` and
`proved_optimal_pct` in `results/summary_table.csv` report how many.
