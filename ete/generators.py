"""Instance generators and JSON input/output.

The generators are SYNTHETIC.  'uz_inspired' mimics the structure of a
Zimbabwean university timetable (faculties, year groups, shared service
courses, a few cross-faculty electives) but contains no real UZ data; replace
it with registry data through :func:`load_instance` when available.
"""
from __future__ import annotations

import json
import math
import random

from .model import Instance

FACULTIES = [  # (name, relative number of exams)
    ("Science", 4), ("Engineering", 3), ("Medicine", 3), ("Social Studies", 3),
    ("Arts", 3), ("Commerce", 3), ("Law", 2), ("Agriculture", 2),
    ("Education", 2), ("Veterinary Science", 1),
]


def _make_venues(total_size: int, n: int, max_exam: int, slots_by_capacity: float) -> list[int]:
    cap = max(max_exam, math.ceil(total_size / max(2.0, slots_by_capacity)))
    shares = [0.40, 0.25, 0.20, 0.15]
    venues = [int(cap * s) for s in shares]
    venues[0] += cap - sum(venues)
    return venues


def random_instance(n: int, seed: int, density: float = 0.20) -> Instance:
    """Random small cohorts (2 or 3 exams each).  The number of cohorts is
    tuned so the conflict graph is close to G(n, density), which has a small
    clique number but a much larger chromatic number -> hard instances."""
    rng = random.Random(seed)
    pairs_per_cohort = 2.0                        # mean of 1 (size 2) and 3 (size 3)
    q = pairs_per_cohort / (n * (n - 1) / 2) if n > 1 else 1.0
    target = max(1, math.ceil(math.log(1 - density) / math.log(1 - q))) if q < 1 else 1
    cohorts: list[tuple[int, list[int]]] = []
    order = list(range(n))
    rng.shuffle(order)
    for i in range(0, n, 2):                      # cover every exam at least once
        chunk = order[i:i + 2]
        if len(chunk) == 1:
            chunk.append(rng.choice([e for e in range(n) if e != chunk[0]] or chunk))
        cohorts.append((rng.randint(20, 150), chunk))
    while len(cohorts) < target:
        cohorts.append((rng.randint(20, 150), rng.sample(range(n), min(n, rng.choice((2, 3))))))
    n_lect = max(2, n // 2)
    lecturers = [rng.randrange(n_lect) for _ in range(n)]
    sizes = [0] * n
    for s, ex in cohorts:
        for e in set(ex):
            sizes[e] += s
    venues = _make_venues(sum(sizes), n, max(sizes), max(2, round(0.04 * n)))
    return Instance.from_cohorts([f"R{i:03d}" for i in range(n)], lecturers, cohorts,
                                 venues, name=f"random_n{n}_s{seed}")


def uz_inspired_instance(n: int, seed: int) -> Instance:
    """Faculty / year structured instance with shared service courses."""
    rng = random.Random(seed)
    n_service = max(1, round(0.06 * n))
    rest = n - n_service
    weights = [w for _, w in FACULTIES]
    tot_w = sum(weights)
    per_fac = [max(1, round(rest * w / tot_w)) for w in weights]
    while sum(per_fac) > rest and max(per_fac) > 1:
        per_fac[per_fac.index(max(per_fac))] -= 1
    while sum(per_fac) < rest:
        per_fac[per_fac.index(min(per_fac))] += 1
    if sum(per_fac) > rest:                       # tiny n: fewer faculties
        per_fac = per_fac[:max(1, rest)]
        while sum(per_fac) > rest:
            per_fac[-1] -= 1
        per_fac = [p for p in per_fac if p > 0]

    labels: list[str] = []
    lecturers: list[int] = []
    groups: list[list[int]] = []                  # (faculty, year) exam pools
    group_fac: list[int] = []
    fac_exams: list[list[int]] = []
    lec_id = 0
    for fi, cnt in enumerate(per_fac):
        fname = FACULTIES[fi][0][:3].upper()
        ids = []
        n_lec = max(1, cnt // 3)
        lec_pool = list(range(lec_id, lec_id + n_lec))
        lec_id += n_lec
        for j in range(cnt):
            ids.append(len(labels))
            labels.append(f"{fname}{(j // 8) + 1}{(j % 8) + 1:02d}")
            lecturers.append(rng.choice(lec_pool))
        fac_exams.append(ids)
        years = max(1, min(4, math.ceil(cnt / 6)))
        for y in range(years):
            pool = ids[y::years]
            if pool:
                groups.append(pool)
                group_fac.append(fi)
    service = []
    for j in range(n_service):
        service.append(len(labels))
        labels.append(f"GEN{j + 1:02d}")
        lecturers.append(lec_id + (j % 2))
    all_ids = list(range(len(labels)))

    cohorts: list[tuple[int, list[int]]] = []
    for gi, pool in enumerate(groups):
        for _ in range(rng.randint(1, 3)):        # programmes in this year group
            m = min(len(pool), rng.randint(4, 6))
            ex = rng.sample(pool, m)
            if service and rng.random() < 0.8:
                ex.append(rng.choice(service))
            for _x in range(rng.randint(0, 3)):
                other = [e for e in all_ids if e not in pool and e not in service]
                if other:
                    ex.append(rng.choice(other))
            year_first = pool is groups[0] or rng.random() < 0.3
            size = rng.randint(80, 300) if year_first else rng.randint(30, 150)
            cohorts.append((size, ex))
    covered = {e for _, ex in cohorts for e in ex}
    for e in all_ids:                             # make sure everything is examined
        if e not in covered:
            cohorts.append((rng.randint(20, 80), [e, rng.choice([x for x in all_ids if x != e])]
                            if len(all_ids) > 1 else [e]))
    sizes = [0] * len(labels)
    for s, ex in cohorts:
        for e in set(ex):
            sizes[e] += s
    venues = _make_venues(sum(sizes), len(labels), max(sizes), max(2, round(0.05 * n)))
    return Instance.from_cohorts(labels, lecturers, cohorts, venues,
                                 name=f"uz_n{len(labels)}_s{seed}")


GENERATORS = {"random": random_instance, "uz": uz_inspired_instance}


# ----------------------------------------------------------------- JSON I/O
def load_instance(path: str) -> Instance:
    """Load an instance from JSON::

        {"name": "...",
         "venues":  [{"name": "Great Hall", "capacity": 400}, ...],
         "exams":   [{"code": "CS101", "lecturer": "Dr Moyo"}, ...],
         "cohorts": [{"size": 120, "exams": ["CS101", "MA101", ...]}, ...]}
    """
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    codes = [e["code"] for e in data["exams"]]
    index = {c: i for i, c in enumerate(codes)}
    if len(index) != len(codes):
        raise ValueError("duplicate exam codes")
    lec_ids: dict[str, int] = {}
    lecturers = [lec_ids.setdefault(e.get("lecturer", f"_anon{i}"), len(lec_ids))
                 for i, e in enumerate(data["exams"])]
    cohorts = [(int(c["size"]), [index[x] for x in c["exams"]]) for c in data["cohorts"]]
    venues = [int(v["capacity"]) for v in data["venues"]]
    return Instance.from_cohorts(codes, lecturers, cohorts, venues,
                                 name=data.get("name", path))


def save_instance(inst: Instance, path: str) -> None:
    lec_name = {l: f"L{l}" for l in set(inst.lecturers)}
    data = {
        "name": inst.name,
        "venues": [{"name": f"V{i + 1}", "capacity": c} for i, c in enumerate(inst.venues)],
        "exams": [{"code": inst.labels[i], "lecturer": lec_name[inst.lecturers[i]]}
                  for i in range(inst.n)],
        "cohorts": [{"size": s, "exams": [inst.labels[e] for e in ex]}
                    for s, ex in inst.cohorts],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)
