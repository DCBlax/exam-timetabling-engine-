"""Shared machinery: search budget, result record, verification."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .model import Instance


class BudgetExceeded(Exception):
    """Raised inside a search when the time or node limit is hit."""


class Budget:
    """Counts search nodes and enforces wall-clock / node limits."""

    __slots__ = ("time_limit", "node_limit", "t0", "nodes", "deadline")

    def __init__(self, time_limit: float | None = None, node_limit: int | None = None):
        self.t0 = time.perf_counter()
        self.time_limit = time_limit
        self.node_limit = node_limit
        self.nodes = 0
        self.deadline = self.t0 + time_limit if time_limit else None

    def tick(self) -> None:
        self.nodes += 1
        if self.node_limit is not None and self.nodes > self.node_limit:
            raise BudgetExceeded
        if self.deadline is not None and (self.nodes & 63) == 0:
            if time.perf_counter() > self.deadline:
                raise BudgetExceeded

    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    def remaining(self) -> float | None:
        if self.deadline is None:
            return None
        return max(0.0, self.deadline - time.perf_counter())


@dataclass
class Result:
    algorithm: str
    assignment: list[int] | None     # slot of every exam (always valid if not None)
    slots: int | None                # number of slots used
    nodes: int                       # search nodes explored
    seconds: float
    status: str                      # 'optimal' (proved) | 'feasible' (not proved)
    lower_bound: int
    extra: dict = field(default_factory=dict)


def num_colors(assign: list[int]) -> int:
    return (max(assign) + 1) if assign else 0


def compact(assign: list[int]) -> list[int]:
    """Relabel slots to 0..k-1 with no gaps (does not change validity)."""
    remap: dict[int, int] = {}
    out = []
    for c in assign:
        if c not in remap:
            remap[c] = len(remap)
        out.append(remap[c])
    return out


def violations(inst: Instance, assign: list[int]) -> dict:
    """Return every hard-constraint violation of an assignment."""
    unassigned = [v for v, c in enumerate(assign) if c is None or c < 0]
    conflicts = [
        (u, v) for u, v in inst.edges()
        if assign[u] is not None and assign[u] == assign[v] and assign[u] >= 0
    ]
    load: dict[int, int] = {}
    for v, c in enumerate(assign):
        if c is not None and c >= 0:
            load[c] = load.get(c, 0) + inst.sizes[v]
    overloaded = [(c, l) for c, l in sorted(load.items()) if l > inst.capacity]
    return {"unassigned": unassigned, "conflicts": conflicts, "overloaded": overloaded}


def is_valid(inst: Instance, assign: list[int] | None) -> bool:
    if assign is None or len(assign) != inst.n:
        return False
    v = violations(inst, assign)
    return not (v["unassigned"] or v["conflicts"] or v["overloaded"])


def allocate_venues(inst: Instance, assign: list[int]) -> dict[int, list[tuple[int, int, int]]]:
    """Seat the exams of every slot in the venues.

    Largest exam first; an exam may be split over several venues.  Because the
    capacity constraint guarantees total enrolment of a slot <= sum of venue
    capacities, this always succeeds for a valid assignment.
    Returns slot -> list of (exam, venue_index, seats).
    """
    out: dict[int, list[tuple[int, int, int]]] = {}
    slots = sorted(set(assign))
    for s in slots:
        free = list(inst.venues)
        exams = sorted((v for v in range(inst.n) if assign[v] == s), key=lambda v: -inst.sizes[v])
        rows: list[tuple[int, int, int]] = []
        for e in exams:
            need = inst.sizes[e]
            # prefer a single venue that fits the whole exam (best fit)
            fits = [i for i, f in enumerate(free) if f >= need]
            if fits:
                i = min(fits, key=lambda j: free[j])
                free[i] -= need
                rows.append((e, i, need))
                continue
            for i in sorted(range(len(free)), key=lambda j: -free[j]):
                if need == 0:
                    break
                take = min(free[i], need)
                if take:
                    free[i] -= take
                    need -= take
                    rows.append((e, i, take))
            if need:
                raise RuntimeError("venue allocation failed: capacity constraint violated")
        out[s] = rows
    return out
