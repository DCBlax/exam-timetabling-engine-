"""Branch-and-Bound for the minimum-slot timetable (DSATUR branching).

Bounding function
-----------------
At any node let u = number of slots opened so far and
    LB = max(clique_size, ceil(total_enrolment / capacity))
be the root lower bound.  Every completion of the node uses at least
max(u, LB) slots, so the node is pruned when  max(u, LB) >= best  (it cannot
beat the incumbent).  Slots are opened one at a time, so a branch can only
use slot indices 0 .. best-2.

Branching
---------
The uncoloured exam with the largest saturation degree (number of distinct
slots among its already-placed neighbours) is chosen, ties broken by the
number of unplaced neighbours.  The vertices of a heuristic maximum clique are
pre-placed in distinct slots (they must be distinct anyway), which both fixes
the symmetry of slot labels and makes the clique bound tight at the root.

The incumbent is initialised with DSATUR (or any supplied timetable), so the
method is an *anytime* algorithm.  If the search ends without hitting the
budget the incumbent is optimal.
"""
from __future__ import annotations

from .common import Budget, BudgetExceeded, Result, compact, num_colors
from .greedy import dsatur
from .model import Instance, lower_bound


class _Optimal(Exception):
    """Incumbent reached the lower bound: stop, it is optimal."""


def branch_and_bound(
    inst: Instance,
    budget: Budget,
    incumbent: list[int] | None = None,
    name: str = "B&B",
) -> Result:
    n, adj, sizes, cap = inst.n, inst.adj, inst.sizes, inst.capacity
    lb, clique = lower_bound(inst)
    best = compact(incumbent) if incumbent is not None else dsatur(inst)
    state = {"best": best, "k": num_colors(best), "pruned": 0}
    if n == 0 or state["k"] <= lb:
        return Result(name, best, state["k"], 0, budget.elapsed(), "optimal", lb,
                      {"pruned": 0})

    K = state["k"]                                   # array width for counters
    col = [-1] * n
    cnt = [[0] * K for _ in range(n)]                # neighbours of v in slot c
    sat = [0] * n                                    # distinct slots around v
    load = [0] * K

    def place(v: int, c: int) -> None:
        col[v] = c
        load[c] += sizes[v]
        for u in adj[v]:
            if cnt[u][c] == 0:
                sat[u] += 1
            cnt[u][c] += 1

    def unplace(v: int, c: int) -> None:
        col[v] = -1
        load[c] -= sizes[v]
        for u in adj[v]:
            cnt[u][c] -= 1
            if cnt[u][c] == 0:
                sat[u] -= 1

    # pre-place the clique (distinct slots 0..w-1)
    for i, v in enumerate(clique):
        place(v, i)
    start_used = len(clique)
    placed0 = len(clique)

    def rec(placed: int, used: int) -> None:
        budget.tick()
        best_k = state["k"]
        if max(used, lb) >= best_k:                  # bound
            state["pruned"] += 1
            return
        if placed == n:                              # complete, and used < best
            state["best"] = col[:]
            state["k"] = used
            if used <= lb:
                raise _Optimal
            return
        # branching variable: max saturation, tie -> most unplaced neighbours
        v, bs, bd = -1, -1, -1
        for u in range(n):
            if col[u] >= 0:
                continue
            s = sat[u]
            if s > bs:
                v, bs, bd = u, s, -1
            elif s == bs:
                if bd < 0:
                    bd = sum(1 for w in adj[v] if col[w] < 0)
                d = sum(1 for w in adj[u] if col[w] < 0)
                if d > bd:
                    v, bd = u, d
        c = 0
        while c <= used and c < state["k"] - 1:      # slot `used` = open a new one
            if cnt[v][c] == 0 and load[c] + sizes[v] <= cap:
                place(v, c)
                rec(placed + 1, max(used, c + 1))
                unplace(v, c)
            c += 1
        if c == 0:
            state["pruned"] += 1

    optimal = True
    try:
        rec(placed0, start_used)
    except _Optimal:
        pass
    except BudgetExceeded:
        optimal = False
    k = state["k"]
    status = "optimal" if (optimal or k <= lb) else "feasible"
    return Result(name, state["best"], k, budget.nodes, budget.elapsed(), status, lb,
                  {"pruned": state["pruned"]})


def solve_bnb(inst: Instance, time_limit: float | None = None,
              node_limit: int | None = None) -> Result:
    return branch_and_bound(inst, Budget(time_limit, node_limit))
