# Theory notes: Exam Timetabling Engine (ETE)

## 1. Formal model (CSP)

* **Variables** `x_e` for every exam `e ∈ E` (|E| = n); **domain** `{0,…,k-1}` (time slots).
* **Conflict graph** `G = (E, F)`: `{e,f} ∈ F` iff the exams share a student cohort or a lecturer.
* **C1 (conflict):** `x_e ≠ x_f` for every `{e,f} ∈ F`.
* **C2 (capacity):** for every slot `s`, `Σ_{e : x_e = s} size(e) ≤ C`, where `C = Σ venues` (an exam may be split across venues; `allocate_venues` seats any slot that satisfies C2).
* **Objective:** minimise `k`.  With C2 removed this is exactly the chromatic number `χ(G)`.

Implemented in `ete/model.py` (`Instance.from_cohorts`), verified independently by `ete/common.py::violations`.

## 2. NP-completeness

**Decision problem ETE-D:** given an instance and `k`, is there a timetable using at most `k` slots?

**In NP.** A timetable is a vector of n slot numbers; C1 and C2 are checked in `O(n² + n)` time.

**3-SAT ≤ₚ 3-COLOURING.** Given a 3-CNF formula φ with variables `x_1..x_m` and clauses `c_1..c_r`, build `G_φ`:

1. Palette triangle `T, F, B`.
2. For each variable `x_i`: nodes `x_i`, `¬x_i`, forming a triangle with `B`. So each literal node is coloured `T` or `F`, and `x_i`, `¬x_i` get different colours.
3. For each clause `(l1 ∨ l2 ∨ l3)`: chain two **OR-gadgets**. An OR-gadget on inputs `a, b` has nodes `p, q, o` with edges `a–p, b–q, p–q, p–o, q–o`; `o` is its output. Use `o1 = OR(l1,l2)`, `o2 = OR(o1,l3)` and join `o2` to both `F` and `B`, so `o2` must be coloured `T`.

*Gadget fact (inputs coloured in {T,F}).* (i) If `a = b = F`, then `p,q ∈ {T,B}` and `p ≠ q`, so `{p,q} = {T,B}`, and `o` (adjacent to both) is forced to `F`. (ii) If at least one input is `T`, `p,q,o` can be coloured so that `o = T` (e.g. a=T,b=F: p=F, q=B, o=T; a=F,b=T: p=B, q=F, o=T; a=b=T: p=F, q=B, o=T).

*(⇒)* Colour `x_i` with `T` if true, else `F`; by (ii) the gadgets can be coloured so every `o2 = T`, which is legal because `o2` is adjacent only to `F, B` and its gadget neighbours. *(⇐)* In a proper 3-colouring, name the colours after the palette. Literal nodes are `T/F` and complementary, giving an assignment. If some clause had all literals `F`, then by (i) `o1 = F`, and again by (i) with inputs `(F,F)`, `o2 = F`, contradicting `o2 ≠ F`. Hence φ is satisfied. The graph has `O(m + r)` nodes: polynomial.

**3-COLOURING ≤ₚ ETE-D.** Given `G = (V,E')`, create one exam per vertex, one distinct lecturer per exam, and one cohort of size 1 for every edge `{u,v}` (sitting exams `u` and `v`); give isolated vertices a singleton cohort. Venue capacity `C = Σ sizes`, so C2 never binds. The conflict graph equals `G`, and `k = 3`. A timetable with ≤3 slots exists iff `G` is 3-colourable. Computable in polynomial time.

Therefore **3-SAT ≤ₚ 3-COLOURING ≤ₚ ETE-D**, and ETE-D is NP-complete; minimising slots is NP-hard.
(The clique route works too: ETE with no capacity is `χ`, which is NP-hard; the clique number gives the lower bound used below.)

## 3. Correctness

**Backtracking** (`ete/backtracking.py`). *Soundness:* a leaf is reached only if every placed exam passed C1 (checked against placed neighbours, or neighbours' domains pruned by forward checking) and C2 (load checked / capacity filtering), so the returned timetable is valid. *Completeness:* forward checking removes only values that violate C1/C2 given the current partial assignment, so no solution is removed; MRV and degree only change the *order* in which variables are branched; symmetry breaking (at most one fresh slot tried) discards only relabelings of an already-explored branch. Hence if a k-slot timetable exists the search finds it, otherwise it exhausts the tree. *Optimality:* k is increased from `lb`; every smaller k was refuted exhaustively and `lb` is a valid lower bound, so the first success is optimal. If all `k < ub` are refuted, the DSATUR timetable (`ub` slots) is optimal.

**Branch and bound** (`ete/bnb.py`). Invariant: the incumbent is always a valid timetable with `best` slots. A node with `used` open slots can only be completed with ≥ `max(used, lb)` slots (slots are never closed; `lb` bounds every timetable). Pruning when `max(used, lb) ≥ best` therefore never discards a strictly better timetable. The clique is pre-placed in distinct slots (any optimal timetable can be relabelled this way), and slot indices are tried in increasing order with at most one fresh slot, which is again only relabelling. When the tree is exhausted the incumbent is optimal; if the budget stops the search, the incumbent is returned as `feasible`.

## 4. Lower bounds

* **Clique bound.** A clique of size `w` in `G` needs `w` pairwise different slots, so `k ≥ w` (`greedy_clique` finds a clique heuristically; any clique is a valid bound even if not maximum).
* **Capacity bound.** Each slot seats at most `C`, so `k ≥ ⌈Σ size / C⌉`.
* `lb = max(clique, capacity)`; B&B stops as soon as the incumbent meets it.

## 5. Greedy approximation

**Claim.** First-fit (any order), when capacity does not bind, uses at most `Δ+1` slots. *Proof.* When exam `v` is placed, at most `deg(v) ≤ Δ` slots are forbidden by placed neighbours, so one of slots `0..Δ` is free. ∎  Since `χ(G) ≥ 1`, the ratio is at most `Δ+1`; with Welsh–Powell ordering the bound sharpens to `max_i min(d_i+1, i)`.

**Limits.** No constant ratio holds in general: on the crown graph (bipartite, χ = 2) a bad order makes first-fit use `n/2` slots, and unless P = NP no polynomial algorithm approximates χ within `n^{1-ε}` (Zuckerman 2007). DSATUR is exact on bipartite graphs and cycles. Empirically (`results/approximation.csv`, `fig_ratio.png`) both heuristics are within about one slot of the best known value on the generated families. When capacity binds the Δ+1 bound no longer applies because a slot may be full as well as conflicting; the implementation still always returns a valid timetable.

## 6. Complexity

| Algorithm | Time | Notes |
|---|---|---|
| Naive BT (per k) | `O(k^n)` worst case | static order, no propagation |
| BT + FC + MRV (per k) | `O(k^n)` worst case, far fewer nodes in practice | each node costs `O(n + deg)` for MRV + propagation |
| B&B | `O(k^n)` worst case | pruned by `max(used, lb) ≥ best`; node cost `O(n·deg)` |
| Greedy (Welsh–Powell) | `O(V log V + E)` first-fit pass `O(V+E)` | at most Δ+1 slots |
| DSATUR | `O((V+E) log V)` with heap | |
| Hybrid B&B+LS | LS: `O(iters · n · k)` per attempt (polynomial); then B&B as above | LS only improves the incumbent; the proof of optimality still needs B&B |
