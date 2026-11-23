"""Unit tests and edge cases for the ETE solvers."""
import itertools
import json
import random

import pytest

from ete import (Instance, allocate_venues, is_valid, load_instance, lower_bound, save_instance,
                 solve_bnb, solve_dsatur, solve_fc_mrv, solve_greedy, solve_hybrid, solve_naive,
                 violations)
from ete.generators import GENERATORS, random_instance, uz_inspired_instance

EXACT = [solve_naive, solve_fc_mrv, solve_bnb, solve_hybrid]
ALL = [solve_greedy, solve_dsatur] + EXACT


def run(solver, inst):
    return solver(inst) if solver in (solve_greedy, solve_dsatur) else solver(inst, 10.0)


def cycle(n):
    return Instance.from_graph(n, [(i, (i + 1) % n) for i in range(n)])


def complete(n):
    return Instance.from_graph(n, itertools.combinations(range(n), 2))


def petersen():
    outer = [(i, (i + 1) % 5) for i in range(5)]
    spokes = [(i, i + 5) for i in range(5)]
    inner = [(5 + i, 5 + (i + 2) % 5) for i in range(5)]
    return Instance.from_graph(10, outer + spokes + inner)


def gnp(n, p, seed):
    rng = random.Random(seed)
    return Instance.from_graph(n, [(u, v) for u in range(n) for v in range(u + 1, n)
                                   if rng.random() < p])


# ------------------------------------------------------------- known optima
@pytest.mark.parametrize("solver", EXACT)
@pytest.mark.parametrize("make,chi", [
    (lambda: cycle(5), 3), (lambda: cycle(8), 2), (lambda: complete(6), 6),
    (petersen, 3), (lambda: Instance.from_graph(7, []), 1),
    (lambda: Instance.from_graph(1, []), 1),
])
def test_known_chromatic_numbers(solver, make, chi):
    inst = make()
    res = run(solver, inst)
    assert is_valid(inst, res.assignment)
    assert res.slots == chi and res.status == "optimal"


def test_empty_instance():
    inst = Instance.from_graph(0, [])
    for s in ALL:
        res = run(s, inst)
        assert res.slots == 0 and is_valid(inst, res.assignment)


# --------------------------------------------------- heuristics: validity
@pytest.mark.parametrize("solver", ALL)
@pytest.mark.parametrize("fam", ["random", "uz"])
def test_all_solvers_return_valid_timetables(solver, fam):
    for seed in range(3):
        inst = GENERATORS[fam](20, seed)
        res = run(solver, inst)
        assert is_valid(inst, res.assignment), (solver.__name__, fam, seed)
        assert res.slots >= res.lower_bound


def test_greedy_delta_plus_one_when_capacity_free():
    for seed in range(10):
        inst = gnp(30, 0.2, seed)
        delta = max(inst.degree(v) for v in range(inst.n))
        assert solve_greedy(inst).slots <= delta + 1
        assert solve_dsatur(inst).slots <= delta + 1


def test_dsatur_optimal_on_bipartite():
    inst = Instance.from_graph(10, [(i, j) for i in range(5) for j in range(5, 10)])
    assert solve_dsatur(inst).slots == 2


# ------------------------------------------------ exact solvers must agree
def test_exact_solvers_agree_on_random_graphs():
    for seed in range(12):
        inst = gnp(11, 0.35, seed)
        ks = {s.__name__: run(s, inst).slots for s in EXACT}
        assert len(set(ks.values())) == 1, (seed, ks)
        assert all(run(s, inst).status == "optimal" for s in EXACT)


def test_exact_solvers_agree_with_capacity_binding():
    rng = random.Random(1)
    for seed in range(8):
        base = gnp(9, 0.25, seed)
        sizes = [rng.randint(1, 9) for _ in range(9)]
        inst = Instance.from_graph(9, base.edges(), sizes=sizes, capacity=max(sizes) + 3)
        res = [run(s, inst) for s in EXACT]
        assert len({r.slots for r in res}) == 1
        assert all(is_valid(inst, r.assignment) for r in res)


def test_capacity_bound_dominates():
    # no conflicts, but 4 exams of size 5 and capacity 10 -> need 2 slots
    inst = Instance.from_graph(4, [], sizes=[5] * 4, capacity=10)
    assert lower_bound(inst)[0] == 2
    for s in EXACT:
        assert run(s, inst).slots == 2


def test_oversized_exam_rejected():
    with pytest.raises(ValueError):
        Instance.from_graph(2, [], sizes=[50, 1], capacity=10)


# ---------------------------------------------------- budgets / anytime
def test_timeout_still_returns_valid_incumbent():
    inst = random_instance(60, 0)
    for s in (solve_naive, solve_fc_mrv, solve_bnb, solve_hybrid):
        res = s(inst, 0.05)
        assert is_valid(inst, res.assignment)
        assert res.slots >= res.lower_bound


def test_node_limit_respected():
    inst = random_instance(40, 1)
    res = solve_bnb(inst, node_limit=50)
    assert res.nodes <= 51 and is_valid(inst, res.assignment)


def test_hybrid_never_worse_than_dsatur():
    for seed in range(4):
        inst = random_instance(40, seed)
        assert solve_hybrid(inst, 3.0, seed=0).slots <= solve_dsatur(inst).slots


def test_fc_mrv_explores_fewer_nodes_than_naive():
    inst = gnp(12, 0.45, 3)
    a, b = solve_naive(inst, 20.0), solve_fc_mrv(inst, 20.0)
    assert a.slots == b.slots
    assert b.nodes <= a.nodes


# --------------------------------------------------- model / verification
def test_violation_detection():
    inst = Instance.from_graph(3, [(0, 1)], sizes=[5, 5, 5], capacity=10)
    v = violations(inst, [0, 0, 1])
    assert v["conflicts"] == [(0, 1)]
    assert is_valid(inst, [0, 1, 0])             # load 10 == capacity is allowed
    assert violations(inst, [0, 1, 1])["overloaded"] == []
    tight = Instance.from_graph(3, [], sizes=[5, 5, 5], capacity=10)
    assert violations(tight, [0, 0, 0])["overloaded"] == [(0, 15)]
    assert not is_valid(inst, [0, 0, 0])
    assert not is_valid(inst, [0, 1])            # wrong length
    assert not is_valid(inst, None)
    assert violations(inst, [0, -1, 1])["unassigned"] == [1]


def test_from_cohorts_builds_student_and_lecturer_conflicts():
    inst = Instance.from_cohorts(["A", "B", "C", "D"], [0, 1, 2, 2],
                                 [(30, [0, 1]), (20, [1, 2])], [100])
    assert inst.adj[0] == {1} and inst.adj[1] == {0, 2}
    assert 3 in inst.adj[2] and 2 in inst.adj[3]        # shared lecturer
    assert inst.sizes == [30, 50, 20, 0]


def test_venue_allocation_respects_capacity():
    inst = uz_inspired_instance(60, 0)
    res = solve_dsatur(inst)
    seating = allocate_venues(inst, res.assignment)
    for rows in seating.values():
        used = {}
        for e, v, seats in rows:
            used[v] = used.get(v, 0) + seats
        assert all(used[v] <= inst.venues[v] for v in used)
    for e in range(inst.n):
        assert sum(s for ss in seating.values() for x, _, s in ss if x == e) == inst.sizes[e]


# ------------------------------------------------------------ generators
@pytest.mark.parametrize("fam", ["random", "uz"])
def test_generators_deterministic_and_sized(fam):
    for n in (2, 10, 57):
        a, b = GENERATORS[fam](n, 7), GENERATORS[fam](n, 7)
        assert a.n >= n - 1 and a.adj == b.adj and a.sizes == b.sizes and a.venues == b.venues
        assert max(a.sizes) <= a.capacity


def test_json_roundtrip(tmp_path):
    inst = uz_inspired_instance(30, 2)
    p = tmp_path / "i.json"
    save_instance(inst, str(p))
    back = load_instance(str(p))
    assert back.n == inst.n and back.sizes == inst.sizes and back.venues == inst.venues
    assert back.adj == inst.adj
    json.loads(p.read_text())


def test_load_rejects_duplicate_codes(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"venues": [{"capacity": 10}], "exams": [{"code": "A"}, {"code": "A"}],
                             "cohorts": []}))
    with pytest.raises(ValueError):
        load_instance(str(p))
