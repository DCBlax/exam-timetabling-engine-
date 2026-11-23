"""Exact backtracking solvers for the minimum-slot timetable.

Both solvers share the same outer loop (iterative deepening on the number of
slots k) and differ ONLY in the inner decision search, so differences in node
counts are attributable to the pruning techniques alone.

Preprocessing shared by all exact solvers (as in :mod:`ete.bnb`):
    lb  = max(clique bound, capacity bound)
    ub  = DSATUR timetable (also the fallback returned on timeout)
If ub == lb the instance is solved without any search (0 nodes).  Otherwise
k = lb, lb+1, ..., ub-1 is tried in turn; the first k that admits a timetable
is optimal because every smaller k was refuted exhaustively.  If every k < ub
is refuted, the DSATUR timetable is optimal.

* naive_decision : chronological backtracking, static exam order, slots tried
                   0..k-1, constraints checked only when a value is assigned,
                   no symmetry breaking.  Worst case O(k^n).
* fc_mrv_decision: forward checking (domains of unplaced exams are filtered
                   after every placement, wipe-out => backtrack), MRV variable
                   ordering with the degree heuristic as tie-break, capacity
                   filtering, and symmetry breaking (at most one fresh slot is
                   tried per node).  Still exponential in the worst case.

On a timeout the best known valid timetable (DSATUR) is returned with status
'feasible', so the returned assignment is always valid.
"""
from __future__ import annotations

from .common import Budget, BudgetExceeded, Result
from .greedy import dsatur
from .model import Instance, lower_bound
from .common import num_colors


# ------------------------------------------------------------------ decisions
def naive_decision(inst: Instance, k: int, budget: Budget) -> list[int] | None:
    """Is there a valid timetable with slots 0..k-1?  Plain backtracking."""
    n, adj, sizes, cap = inst.n, inst.adj, inst.sizes, inst.capacity
    col = [-1] * n
    load = [0] * k

    def rec(v: int) -> bool:
        budget.tick()
        if v == n:
            return True
        sv = sizes[v]
        for c in range(k):
            if load[c] + sv > cap:
                continue
            if any(col[u] == c for u in adj[v]):      # u < v are the only placed ones
                continue
            col[v] = c
            load[c] += sv
            if rec(v + 1):
                return True
            load[c] -= sv
            col[v] = -1
        return False

    return col[:] if rec(0) else None


def fc_mrv_decision(inst: Instance, k: int, budget: Budget) -> list[int] | None:
    """Backtracking + forward checking + MRV (+degree) + symmetry breaking."""
    n, adj, sizes, cap = inst.n, inst.adj, inst.sizes, inst.capacity
    full = (1 << k) - 1
    dom = [full] * n                                  # bitmask of allowed slots
    col = [-1] * n
    load = [0] * k
    deg = [len(a) for a in adj]
    popcount = int.bit_count

    def rec(placed: int, used: int) -> bool:
        budget.tick()
        if placed == n:
            return True
        # MRV over the symmetry-reduced domain (used slots + one fresh slot)
        sym = (1 << min(used + 1, k)) - 1
        v, best_sz, best_deg = -1, k + 2, -1
        for u in range(n):
            if col[u] >= 0:
                continue
            sz = popcount(dom[u] & sym)
            if sz < best_sz or (sz == best_sz and deg[u] > best_deg):
                v, best_sz, best_deg = u, sz, deg[u]
        if best_sz == 0:
            return False
        mask = dom[v] & sym
        sv = sizes[v]
        while mask:
            low = mask & -mask
            mask ^= low
            c = low.bit_length() - 1
            col[v] = c
            load[c] += sv
            trail: list[tuple[int, int]] = []
            ok = True
            for u in adj[v]:                          # conflict propagation
                if col[u] < 0 and dom[u] & low:
                    trail.append((u, dom[u]))
                    dom[u] &= ~low
                    if dom[u] == 0:
                        ok = False
                        break
            if ok:                                    # capacity propagation
                room = cap - load[c]
                for u in range(n):
                    if col[u] < 0 and dom[u] & low and sizes[u] > room:
                        trail.append((u, dom[u]))
                        dom[u] &= ~low
                        if dom[u] == 0:
                            ok = False
                            break
            if ok and rec(placed + 1, max(used, c + 1)):
                return True
            for u, d in reversed(trail):
                dom[u] = d
            load[c] -= sv
            col[v] = -1
        return False

    return col[:] if rec(0, 0) else None


# ---------------------------------------------------------------- outer loop
def _solve(inst: Instance, decision, name: str, time_limit: float | None,
           node_limit: int | None) -> Result:
    budget = Budget(time_limit, node_limit)
    lb, _ = lower_bound(inst)
    ub_assign = dsatur(inst)
    ub = num_colors(ub_assign)
    if inst.n == 0 or ub <= lb:
        return Result(name, ub_assign, ub, 0, budget.elapsed(), "optimal", lb)
    best, best_k, optimal = ub_assign, ub, True
    try:
        for k in range(max(lb, 1), ub):
            sol = decision(inst, k, budget)
            if sol is not None:
                best, best_k = sol, k
                break
    except BudgetExceeded:
        optimal = False
    return Result(name, best, best_k, budget.nodes, budget.elapsed(),
                  "optimal" if optimal else "feasible", lb)


def solve_naive(inst: Instance, time_limit: float | None = None,
                node_limit: int | None = None) -> Result:
    return _solve(inst, naive_decision, "Naive BT", time_limit, node_limit)


def solve_fc_mrv(inst: Instance, time_limit: float | None = None,
                 node_limit: int | None = None) -> Result:
    return _solve(inst, fc_mrv_decision, "BT+FC+MRV", time_limit, node_limit)
