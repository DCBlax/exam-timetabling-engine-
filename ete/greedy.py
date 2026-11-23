"""Greedy approximation algorithms (capacity-aware).

* first_fit      : process exams in a given order, give each the smallest slot
                   that has no conflicting exam and enough free seats.  O(V+E)
                   (plus the cost of the ordering).  Uses at most Delta+1 slots
                   when capacity does not bind.
* welsh_powell   : first-fit in order of non-increasing degree (largest first).
* dsatur         : Brelaz' dynamic ordering by saturation degree.
"""
from __future__ import annotations

import heapq
import time

from .common import Result, num_colors
from .model import Instance, lower_bound


def first_fit(inst: Instance, order: list[int]) -> list[int]:
    col = [-1] * inst.n
    load: list[int] = []
    cap = inst.capacity
    for v in order:
        forbidden = {col[u] for u in inst.adj[v] if col[u] >= 0}
        c = 0
        while True:
            if c == len(load):
                load.append(0)
            if c not in forbidden and load[c] + inst.sizes[v] <= cap:
                break
            c += 1
        col[v] = c
        load[c] += inst.sizes[v]
    return col


def welsh_powell_order(inst: Instance) -> list[int]:
    return sorted(range(inst.n), key=lambda v: (-inst.degree(v), v))


def welsh_powell(inst: Instance) -> list[int]:
    return first_fit(inst, welsh_powell_order(inst))


def dsatur(inst: Instance) -> list[int]:
    """DSATUR heuristic with capacity check (lazy-deletion heap)."""
    n = inst.n
    col = [-1] * n
    nbr_cols: list[set[int]] = [set() for _ in range(n)]
    load: list[int] = []
    cap = inst.capacity
    heap = [(0, -inst.degree(v), v) for v in range(n)]      # (-sat, -deg, v)
    heapq.heapify(heap)
    done = 0
    while done < n:
        negsat, negdeg, v = heapq.heappop(heap)
        if col[v] >= 0 or -negsat != len(nbr_cols[v]):
            continue                                        # stale entry
        c = 0
        while True:
            if c == len(load):
                load.append(0)
            if c not in nbr_cols[v] and load[c] + inst.sizes[v] <= cap:
                break
            c += 1
        col[v] = c
        load[c] += inst.sizes[v]
        done += 1
        for u in inst.adj[v]:
            if col[u] < 0 and c not in nbr_cols[u]:
                nbr_cols[u].add(c)
                heapq.heappush(heap, (-len(nbr_cols[u]), -inst.degree(u), u))
    return col


def _wrap(name: str, fn, inst: Instance) -> Result:
    t0 = time.perf_counter()
    assign = fn(inst)
    secs = time.perf_counter() - t0
    lb, _ = lower_bound(inst)
    k = num_colors(assign)
    return Result(name, assign, k, inst.n, secs, "optimal" if k == lb else "feasible", lb)


def solve_greedy(inst: Instance) -> Result:
    """The 'Greedy colouring' algorithm of the project (Welsh-Powell)."""
    return _wrap("Greedy", welsh_powell, inst)


def solve_dsatur(inst: Instance) -> Result:
    return _wrap("DSATUR", dsatur, inst)
