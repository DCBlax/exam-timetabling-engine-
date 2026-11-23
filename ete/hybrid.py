"""Local search (TabuCol with a capacity penalty) and the hybrid solver.

Hybrid B&B + local search
-------------------------
1. DSATUR gives an initial timetable (upper bound).
2. Local search phase: repeatedly try to *remove one slot*.  The smallest slot
   is dissolved and TabuCol repairs the resulting conflicts / overloads with
   k-1 slots.  Success gives a better incumbent; failure ends the phase.
3. Branch-and-bound phase: the exact B&B is started with the improved
   incumbent, so the bound prunes far more of the tree.  If it finishes, the
   result is proved optimal; otherwise the best timetable is returned.
"""
from __future__ import annotations

import random
import time

from .bnb import branch_and_bound
from .common import Budget, Result, compact, num_colors
from .greedy import dsatur
from .model import Instance, lower_bound


def tabucol(
    inst: Instance,
    k: int,
    init: list[int],
    rng: random.Random,
    max_iters: int,
    deadline: float | None,
) -> tuple[list[int] | None, int]:
    """Try to repair `init` (slots in 0..k-1, possibly invalid) into a valid
    k-slot timetable.  Cost = #conflicting pairs + (seats over capacity)/mean
    exam size.  Returns (timetable or None, iterations used)."""
    n, adj, sizes, cap = inst.n, inst.adj, inst.sizes, inst.capacity
    col = init[:]
    gamma = [[0] * k for _ in range(n)]              # neighbours of v in slot c
    for v in range(n):
        for u in adj[v]:
            gamma[v][col[u]] += 1
    conf = sum(gamma[v][col[v]] for v in range(n)) // 2
    load = [0] * k
    for v in range(n):
        load[col[v]] += sizes[v]
    over = sum(max(0, l - cap) for l in load)
    w = 1.0 / (sum(sizes) / n)
    tabu = [[0] * k for _ in range(n)]
    cost = conf + w * over
    best_cost = cost

    for it in range(1, max_iters + 1):
        if conf == 0 and over == 0:
            return col, it
        if deadline is not None and (it & 31) == 0 and time.perf_counter() > deadline:
            return None, it
        overloaded = {c for c in range(k) if load[c] > cap}
        bestd = float("inf")
        moves: list[tuple[int, int]] = []
        n_conf = 0
        for v in range(n):
            cv = col[v]
            gv = gamma[v][cv]
            if gv == 0 and cv not in overloaded:
                continue
            n_conf += 1
            sv = sizes[v]
            lo_old = max(0, load[cv] - cap)
            lo_new = max(0, load[cv] - sv - cap)
            gv_row = gamma[v]
            for c in range(k):
                if c == cv:
                    continue
                d = gv_row[c] - gv + w * ((lo_new - lo_old)
                                          + (max(0, load[c] + sv - cap) - max(0, load[c] - cap)))
                if tabu[v][c] > it and not (cost + d < best_cost - 1e-9):
                    continue
                if d < bestd - 1e-12:
                    bestd, moves = d, [(v, c)]
                elif d <= bestd + 1e-12:
                    moves.append((v, c))
        if not moves:
            v = rng.randrange(n)
            c = rng.randrange(k)
            if c == col[v]:
                continue
            moves = [(v, c)]
            bestd = 0.0
        v, c = rng.choice(moves)
        cv = col[v]
        sv = sizes[v]
        conf += gamma[v][c] - gamma[v][cv]
        over -= max(0, load[cv] - cap) + max(0, load[c] - cap)
        load[cv] -= sv
        load[c] += sv
        over += max(0, load[cv] - cap) + max(0, load[c] - cap)
        for u in adj[v]:
            gamma[u][cv] -= 1
            gamma[u][c] += 1
        col[v] = c
        tabu[v][cv] = it + int(0.6 * n_conf) + rng.randrange(10)
        cost = conf + w * over
        if cost < best_cost:
            best_cost = cost
    return None, max_iters


def _drop_smallest_slot(inst: Instance, assign: list[int], k: int,
                        rng: random.Random) -> list[int]:
    """Dissolve the lightest slot; re-insert its exams where they hurt least."""
    members: dict[int, list[int]] = {}
    for v, c in enumerate(assign):
        members.setdefault(c, []).append(v)
    victim = min(members, key=lambda c: sum(inst.sizes[v] for v in members[c]))
    keep = [c for c in sorted(members) if c != victim]
    remap = {c: i for i, c in enumerate(keep)}
    new = [remap.get(c, -1) for c in assign]
    load = [0] * (k - 1)
    for v, c in enumerate(new):
        if c >= 0:
            load[c] += inst.sizes[v]
    for v in sorted(members[victim], key=lambda x: -inst.degree(x)):
        scores = []
        for c in range(k - 1):
            clash = sum(1 for u in inst.adj[v] if new[u] == c)
            ov = max(0, load[c] + inst.sizes[v] - inst.capacity)
            scores.append((clash + ov / max(1, inst.sizes[v]), rng.random(), c))
        c = min(scores)[2]
        new[v] = c
        load[c] += inst.sizes[v]
    return new


def solve_hybrid(
    inst: Instance,
    time_limit: float | None = None,
    node_limit: int | None = None,
    seed: int = 0,
    ls_share: float = 0.4,
    ls_iters_per_k: int = 20000,
) -> Result:
    t0 = time.perf_counter()
    rng = random.Random(seed)
    lb, _ = lower_bound(inst)
    best = dsatur(inst)
    k = num_colors(best)
    ls_deadline = t0 + ls_share * time_limit if time_limit else None
    total_iters = 0
    while k > lb:
        start = _drop_smallest_slot(inst, best, k, rng)
        sol, its = tabucol(inst, k - 1, start, rng, ls_iters_per_k, ls_deadline)
        total_iters += its
        if sol is None:
            break
        best = compact(sol)
        k = num_colors(best)
    remaining = None if time_limit is None else max(0.0, time_limit - (time.perf_counter() - t0))
    res = branch_and_bound(inst, Budget(remaining, node_limit), incumbent=best,
                           name="Hybrid B&B+LS")
    res.seconds = time.perf_counter() - t0
    res.extra["ls_iters"] = total_iters
    return res
