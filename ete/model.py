"""Problem model for the Exam Timetabling Engine (ETE).

The timetabling problem is modelled as a Constraint Satisfaction Problem (CSP)
that generalises graph colouring:

* Variables  : one per exam e, with domain {0, ..., k-1} (time slots).
* Constraint 1 (conflict): exams that share a student cohort or a lecturer
  must be placed in different slots  ->  edge in the conflict graph.
* Constraint 2 (capacity): the total enrolment of the exams placed in one slot
  may not exceed the total seat capacity of the venues (an exam may be split
  across several venues, see :func:`ete.verify.allocate_venues`).
* Objective  : minimise the number of slots k (the length of the exam session).

Without constraint 2 this is exactly the chromatic number problem.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence


@dataclass
class Instance:
    n: int                              # number of exams
    sizes: list[int]                    # enrolment of every exam
    adj: list[set[int]]                 # conflict graph (adjacency sets)
    capacity: int                       # seats available per slot (sum of venues)
    venues: list[int] = field(default_factory=list)   # individual venue capacities
    labels: list[str] = field(default_factory=list)   # exam codes
    lecturers: list[int] = field(default_factory=list)
    cohorts: list[tuple[int, tuple[int, ...]]] = field(default_factory=list)
    name: str = "instance"

    def __post_init__(self) -> None:
        if not self.labels:
            self.labels = [f"E{i:03d}" for i in range(self.n)]
        if not self.venues:
            self.venues = [self.capacity]
        if self.n and max(self.sizes) > self.capacity:
            raise ValueError(
                "an exam is larger than the total seating capacity of one slot; "
                "the instance is infeasible"
            )

    # ------------------------------------------------------------------ views
    @property
    def total_size(self) -> int:
        return sum(self.sizes)

    @property
    def num_edges(self) -> int:
        return sum(len(a) for a in self.adj) // 2

    @property
    def density(self) -> float:
        if self.n < 2:
            return 0.0
        return 2.0 * self.num_edges / (self.n * (self.n - 1))

    def degree(self, v: int) -> int:
        return len(self.adj[v])

    def edges(self) -> list[tuple[int, int]]:
        return [(u, v) for u in range(self.n) for v in self.adj[u] if u < v]

    # ------------------------------------------------------------ constructors
    @classmethod
    def from_graph(
        cls,
        n: int,
        edges: Iterable[tuple[int, int]],
        sizes: Sequence[int] | None = None,
        capacity: int | None = None,
        name: str = "graph",
    ) -> "Instance":
        """Build an instance directly from a conflict graph (used by tests)."""
        adj = [set() for _ in range(n)]
        for u, v in edges:
            if u != v:
                adj[u].add(v)
                adj[v].add(u)
        sizes = list(sizes) if sizes is not None else [1] * n
        if capacity is None:
            capacity = max(sum(sizes), 1)          # capacity never binds
        return cls(n=n, sizes=sizes, adj=adj, capacity=capacity, name=name)

    @classmethod
    def from_cohorts(
        cls,
        labels: Sequence[str],
        lecturers: Sequence[int],
        cohorts: Sequence[tuple[int, Sequence[int]]],
        venues: Sequence[int],
        name: str = "instance",
    ) -> "Instance":
        """Build an instance from the raw university data.

        cohorts   : list of (number_of_students, exams_they_all_sit)
        lecturers : lecturer id for every exam (a lecturer cannot be in two
                    venues at once, so same-lecturer exams conflict)
        venues    : capacity of every venue available in each slot
        """
        n = len(labels)
        sizes = [0] * n
        adj = [set() for _ in range(n)]
        clean: list[tuple[int, tuple[int, ...]]] = []
        for size, exams in cohorts:
            ex = tuple(sorted(set(exams)))
            clean.append((size, ex))
            for e in ex:
                sizes[e] += size
            for i, u in enumerate(ex):
                for v in ex[i + 1:]:
                    adj[u].add(v)
                    adj[v].add(u)
        by_lecturer: dict[int, list[int]] = {}
        for e, lec in enumerate(lecturers):
            by_lecturer.setdefault(lec, []).append(e)
        for group in by_lecturer.values():
            for i, u in enumerate(group):
                for v in group[i + 1:]:
                    adj[u].add(v)
                    adj[v].add(u)
        return cls(
            n=n, sizes=sizes, adj=adj, capacity=sum(venues), venues=list(venues),
            labels=list(labels), lecturers=list(lecturers), cohorts=clean, name=name,
        )


# ---------------------------------------------------------------- lower bounds
def greedy_clique(inst: Instance, starts: int = 40) -> list[int]:
    """Heuristic maximum clique (lower bound on the chromatic number).

    From each of the `starts` highest-degree vertices a clique is grown by
    repeatedly adding the candidate with most neighbours inside the candidate
    set.  Any clique of size w forces at least w different slots.
    """
    if inst.n == 0:
        return []
    order = sorted(range(inst.n), key=lambda v: -inst.degree(v))[:starts]
    best: list[int] = []
    for s in order:
        clique = [s]
        cand = set(inst.adj[s])
        while cand:
            v = max(cand, key=lambda x: (len(inst.adj[x] & cand), inst.degree(x)))
            clique.append(v)
            cand &= inst.adj[v]
        if len(clique) > len(best):
            best = clique
    return best


def capacity_bound(inst: Instance) -> int:
    """ceil(total enrolment / seats per slot) slots are needed."""
    return max(1, math.ceil(inst.total_size / inst.capacity)) if inst.n else 0


def lower_bound(inst: Instance) -> tuple[int, list[int]]:
    clique = greedy_clique(inst)
    return max(len(clique), capacity_bound(inst)), clique
