# NAV: Challenges of Monolithic BehaVerify

**Date:** 2026-09-21
**Baseline (before):** `e2cd0b6b2` — *Remove old monolithic NAV draft*
**Scope:** `examples/NAV/core/nav_boundary_finder.py` (new),
`examples/NAV/core/nav_lipschitz_bound.py` (new),
`examples/NAV/nav_domain_config.yaml` (new), `tests/NAV/` (new).

**Headline:** on a continuous state space there is **no grid resolution at
which the monolithic table is both sound and buildable**. Coarse grids make the
only verifier-free soundness argument collapse to "the control could be
anything"; fine grids make the table larger than anything nuXmv has survived.
The two requirements move in opposite directions and the window between them is
empty.

**Reproduce:**

```bash
cd REPRODUCIBILITY/2026_TBA/examples/NAV
python3 -m core.nav_boundary_finder      # state-space bounds
python3 -m core.nav_lipschitz_bound      # Lipschitz constants + table sizes
cd .. && python3 -m pytest tests/NAV -q  # 29 tests
```

---

## 1. The problem this addresses

The monolithic table stores one control value per state:

```
(x = 7 & y = 10 & v = -7 & th = 0) : u = (-0.616, 0.396)
```

On grid world and closed-loop ACAS Xu that was exact, because the declared
state space *was* the integer lattice — no state existed between table entries.
NAV is the first BehaVerify benchmark where that is false. The state is
genuinely continuous, so each table entry stands for a whole *cell* of states,
and the single stored value is only correct if the network is constant across
that cell.

It is not. Sampling 4000 points inside one 0.25-wide cell and stepping one
control period:

| quantity | at the cell centre | across the whole cell |
|---|---|---|
| `u1` | −0.616 | ranges over [−0.843, −0.236] |
| `u2` | +0.396 | ranges over [−0.011, +0.700] |
| successor cells | 1 | **16** |

The 16 successors are exactly `x ∈ {5,6} × y ∈ {9,10} × v ∈ {−8,−7} ×
th ∈ {0,1}` — one contiguous blob straddling a grid boundary in each of the
four dimensions, which is why 2⁴ = 16 is the ceiling. The centre's answer
covers about 21% of the cell.

**Refining the grid does not fix this.** Measured at three resolutions:

| cell side | successor cells per cell (avg) | worst |
|---|---|---|
| 0.25 | 12.1 | 16 |
| 0.10 | 11.9 | 16 |
| 0.05 | 12.9 | 16 |

Halving the cell halves the successor blob too, so the *count* stays put. There
is no resolution at which one stored value becomes correct.

Stating it once, plainly:

> **The table stores a point where the truth is a range.**

## 2. Lipschitz constants: the cheapest sound repair

To make the table sound you must store the *range* `u` can take over each cell.
Getting that range tightly needs a neural-network verifier. This section gives
monolithic its best verifier-free shot instead, so the later comparison is
against a real effort rather than a strawman.

### 2.1 What a Lipschitz constant is

A function `f` is **L-Lipschitz** when

```
||f(a) - f(b)||  <=  L * ||a - b||     for all a, b
```

Read `L` as a speed limit on the function: move the input a little, and the
output cannot move more than `L` times that much. It is useful here because it
turns knowledge at *one* point into knowledge over a *whole region* — exactly
what a table entry needs. If the cell has diameter `d` and you know
`u(centre)`, then everywhere in the cell

```
u  in  u(centre)  ±  L * d / 2
```

For a feed-forward network the constant comes from the weights alone:

- a linear layer stretches by at most its **spectral norm** `||W||₂` (the
  largest singular value — the most any input direction can be amplified);
- ReLU and tanh never stretch anything, so both are 1-Lipschitz;
- composing functions multiplies their constants.

Hence `L <= ||W₁||₂ · ||W₂||₂ · ||W₃||₂`. Three lines of numpy, no inputs, no
sampling, no verifier, and sound.

It is also **loose by construction**: it assumes every layer stretches
maximally in the same direction simultaneously, and ignores that ReLU zeroes
most neurons on any given input. That looseness is precisely what a verifier
buys back.

### 2.2 Measured

`core/nav_lipschitz_bound.py`:

| network | spectral norms | L | informative below |
|---|---|---|---|
| `nn-nav-set.onnx` | 2.629, 3.831, 1.697 | **17.09** | h = 0.0585 |
| `nn-nav-point.onnx` | 2.662, 5.844, 3.292 | **51.21** | h = 0.0195 |

**Side result worth keeping.** The set-trained controller's constant is 3×
smaller than the point-trained one's. That is a one-line computation from the
weights, independent of CORA, CROWN-Reach, and any simulation, and it
corroborates the benchmark's own claim that set-based training improves
verifiability. It is the cheapest evidence available for that claim.

## 3. When the bound stops saying anything

The output layer is tanh, so every component of `u` already lies in `[-1, 1]`
— a spread of 2.0 — for free, without looking at the weights at all. The
Lipschitz bound is worth having only when it beats that.

**There are two thresholds, not one,** because the stored interval is
`u(centre) ± L·d/2` *intersected with* `[-1, 1]`:

| condition | cell side (`nn-nav-set`) | meaning |
|---|---|---|
| `L·d < 2` | `h < 0.0585` | informative for every cell, wherever its centre sits |
| `L·d ≥ 4` | `h ≥ 0.1170` | trivial for every cell — the interval swallows `[-1, 1]` regardless of centre, since \|u(centre)\| ≤ 1 |

Between them the bound is graded: it narrows only cells whose centre sits near
tanh saturation. Measured over the 30 cells on the true trajectory:

| h | L·d | avg stored width | max | cells fully trivial |
|---|---|---|---|---|
| 0.25 | 8.54 | 2.000 | 2.000 | 30/30 |
| 0.1 | 3.42 | 1.931 | 2.000 | 22/30 |
| 0.058 | 1.98 | 1.417 | 1.880 | 0/30 |
| 0.05 | 1.71 | 1.279 | 1.709 | 0/30 |
| 0.025 | 0.85 | 0.733 | 0.855 | 0/30 |

At `h = 0.1` the average width is 1.93 out of 2.0 — technically nonzero
information, practically none.

**Why the coarse model is small rather than enormous.** Once every cell stores
the same interval `[-1, 1]`, the table has *one distinct row*. There is nothing
to look up, so the encoding collapses to a single unconstrained variable and
the cell count stops mattering. The coarse sound model therefore builds in
seconds, runs in seconds, and reports `false`. The huge cell counts in §4 are
the cost only in the regime where the rows actually differ.

**Terminology, stated carefully.** "Vacuous" has a narrower technical meaning
in formal methods — a property passing because some subformula never fires,
e.g. `AG(p -> q)` where `p` is never true. That is not quite what happens here.
Two more precise terms:

- the **bound is trivial** (or uninformative): it is sound, but weaker than
  what tanh already gives you, so it constrains nothing;
- the resulting `INVARSPEC = false` is a **spurious counterexample**: the
  abstract model finds a collision trajectory that the real robot cannot
  follow. It is an artifact of the abstraction being too coarse, not a bug in
  the controller.

That second term is the standard one and is the same phenomenon recorded for
ACAS Xu contract 319 in `2026_07_11_unreachable_states.md`. CEGAR is the
standard response to it.

So a coarse, *sound* monolithic model does not crash. It builds quickly, runs
quickly, and reports `false` — about an unconstrained robot rather than about
this network.

## 4. The feasibility table

State-space widths come from `core/nav_boundary_finder.py`:

- **sound box** — analytic bounds from `|u| <= 1` alone: `|Δv|, |Δθ| <= U·T`
  and `|Δposition| <= |v₀|ₘₐₓ·T + U·T²/2 = 18`. Widths 36.2 / 36.2 / 12.0 /
  12.0. Safe to declare as SMV domains for any discretization of these
  dynamics.
- **sampled box** — measured closed-loop envelope, widths 3.13 / 3.18 / 1.80 /
  1.84. **Not sound** (finitely many initial points cannot cover an interval),
  included only to show that even an optimistic box does not rescue the
  approach.

`nn-nav-set.onnx`. **Cells** counts the grid; **rows needed** is what the SMV
must actually encode, which is 1 once every cell stores the same interval. Each
distinct row costs one ONNX forward pass at generation time and one `case` line
in the SMV.

| h | u spread | informative | cells (sound box) | cells (sampled box) | rows needed |
|---|---|---|---|---|---|
| 0.5 | 2.00 | **no** | 2,985,984 | 576 | 1 (collapses) |
| 0.25 | 2.00 | **no** | 48,441,600 | 8,281 | 1 (collapses) |
| 0.1 | 2.00 | **no** | 1,887,033,600 | 321,408 | 321,408+ |
| 0.05 | 1.71 | yes | 30,192,537,600 | 5,370,624 | 5,370,624+ |
| 0.025 | 0.85 | yes | 483,080,601,600 | 83,439,000 | 83,439,000+ |
| 0.01 | 0.34 | yes | 18,870,336,000,000 | 3,296,566,080 | 3,296,566,080+ |

For scale: grid world's table is **2,401** entries; the largest ACAS Xu table
is **456,775**, on which nuXmv was aborted after 10 minutes.

**The gap has no feasible point.** The two rows that cost nothing to encode
also carry no information. The first resolution that constrains anything at all
(`h = 0.1`) already needs 321,408 rows on the optimistic box to buy an average
stored width of 1.93 out of 2.0 — nearly the full price of a table for nearly
none of the benefit. The first resolution that constrains *every* cell
(`h = 0.05`) needs 5.4M rows on the optimistic box, twelve times past where
nuXmv gave up on ACAS Xu, and 30 billion on the box that can actually be
justified, where the ONNX enumeration alone would not finish.

## 5. Three failure modes

Worth separating, because they fail at different stages and a reviewer will
ask which one is being claimed:

| configuration | what happens | verdict | justified |
|---|---|---|---|
| centre method, h = 0.25 | builds fast, runs fast | `true` | **no** — the stored value is wrong for ~80% of each cell |
| sound Lipschitz, h ≥ 0.117 | table collapses to one free variable; builds fast, runs fast | `false` | yes, but it is a spurious counterexample |
| sound Lipschitz, h < 0.117 | rows differ per cell; table too large to build or to check | — | n/a |

Only the first produces the right answer, and only because CORA and
CROWN-Reach have already established that this benchmark is safe. Monolithic
would be **right without having proved it**.

## 6. What this does not establish

1. **Nothing has been run through BehaVerify yet.** The table sizes are
   arithmetic over declared domains, not measured SMV generation. The claim
   "nuXmv fails at N entries" is extrapolated from the ACAS Xu data point, not
   demonstrated on NAV. Building the h = 0.25 centre-method model end to end is
   the next step and would convert an extrapolation into a measurement.
2. **A tighter sound bound may exist without a full verifier.** Local Lipschitz
   estimates, per-layer interval propagation, or a bound restricted to the
   reachable region would all beat the global product-of-norms figure. The
   claim here is about *this* bound, which is the cheapest one; it is not a
   proof that no verifier-free bound suffices. Narrowing that gap is open work.
3. **The successor-count measurements are sampled**, 300–4000 points per cell.
   They establish that the centre method is wrong, not the exact size of the
   successor set.
4. **Nothing here is about compositional NAV**, which has not been built. The
   comparison this report sets up is still hypothetical on one side.
5. **The regression-output truncation is unfixed.** `dsl_to_nuxmv.py:1106`
   casts network outputs with `int()`, which zeroes `u ∈ [-1, 1]` entirely. Any
   monolithic NAV run needs the ONNX outputs pre-scaled first. This blocks
   item 1 above.

## 7. Relationship to earlier work

**2025 NEUS (`REPRODUCIBILITY/2025_NEUS/`).** Closed-loop ACAS Xu is the same
construction — a continuous problem forced onto a lattice with centre-method
transitions — and that paper disclaims transfer explicitly: *"our closed-loop
model of ACAS Xu cannot be used to argue for the correctness of ACAS Xu."* The
difference on NAV is that CORA and CROWN-Reach have published verified
continuous results on the identical ONNX files, so the same disclaimer is now
visible rather than incidental. Grid world is unaffected: its state genuinely
is discrete, so its lattice is the whole state space and its table is exact.

**`2026_08_31_liveness_ctl.md` (grid world).** The ~50%-UNSAT continuous-mode
result there is this same phenomenon seen from the contract side: 100%-accurate
networks failing between lattice points. That report established the effect;
this one quantifies what it costs the table.

**Sound alternatives (not used here).** Making a discretization transfer to the
continuous system requires the abstract step to *cover* rather than round —
one cell to a set of cells, preserving a simulation relation. The references
are Reissig, Weber & Rungger (IEEE TAC 62(4), 2017) for feedback refinement
relations, and Sun, Khedr & Shoukry (HSCC 2019) for the same construction with
a neural controller, where infeasible region-to-region transitions are deleted
rather than assumed. Monolithic cannot follow that route, because computing the
successor *set* requires bounding the network over a box. Compositional can,
since CROWN is already in the pipeline.
