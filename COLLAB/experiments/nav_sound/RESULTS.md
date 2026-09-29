# NAV with the abstract-control architecture: the window is empty

Sound architecture, better than anything prior, and it still does not close NAV.

## Phase 0 — soundness precondition: PASSED

Contracts admit every true trajectory: 270 states (9 starts x 30 steps), zero
violations, zero outside the box. Checker validated by corrupting contracts three
ways (inverted rounding 54, shifted 200, collapsed 270 violations detected).

## Phase 1 — nonlinearity: SOLVED, with a measured cost

`dx = v*cos(th)` is abstracted per (v,th) cell rather than encoded, keeping
everything in linear real arithmetic. Bounding dx and dy on the *same* cell keeps
the coupling implicit; bounding them independently inflates the step:

| θ bands | inflation |
|---|---|
| 4 | 1.41x (exactly √2, as predicted) |
| 8 | 1.22x |
| 16 | 1.16x |
| 32 | 1.09x |

## Phases 2 and 3 — the squeeze

**Precision required** (CROWN bound width, vacuous = 2.0):

| contracts | mean width |
|---|---|
| 1,024 | 0.95 |
| 16,384 | 0.40 |
| 82,944 | 0.24 |

**Capacity available** (nuXmv, this model shape):

| contracts | verdict | time |
|---|---|---|
| 1,040 | `is false` (spurious) | 1.0 s |
| 2,500 | `is false` (spurious) | 4.6 s |
| 5,184 | `is false` (spurious) | 9.9 s |
| 9,604 | **SEGFAULT** | 1.7 s |
| 16,448 | **SEGFAULT** | — |

The counterexamples are spurious at every size that runs. At 5,184 contracts the
abstract model reaches (2.000, 2.000) at k=4; the true system is at (2.788,
2.898) and never enters the obstacle at any step from any of 9 starts.

**So: informative bounds need ≥16k contracts; nuXmv segfaults between 5,184 and
9,604. The window is empty** — the same squeeze that killed the original LRA
formulation, one architecture higher up.

## What is nevertheless established

1. The architecture is **sound** — over-approximation is valid for NAV's
   universal properties (`AG`), and the contracts provably admit the true system.
2. Continuous state works at all, which the table path forbade.
3. Nonlinear dynamics stay in linear arithmetic via displacement abstraction.
4. The failure is **precision against capacity**, both measured, not a mystery.

## Reproduce

```bash
python3 COLLAB/experiments/nav_sound/admission_check.py 16 8      # Phase 0
python3 COLLAB/experiments/nav_sound/gen_nav_smv.py 12 6 obstacle # Phase 2/3
nuXmv -source <(printf 'go_msat\nmsat_check_invar_bmc -a een-sorensson\nquit\n') \
      COLLAB/experiments/nav_sound/nav_12_6_obstacle.smv
```

---

# Attacks on the above conclusion, and the sharper limit they produced

## Attack 1 — is the segfault nuXmv's or my encoding's? **nuXmv's.**

16,448 *simple* contracts (1 variable, 2-term antecedent — the shape that ran in
0.63 s at 1,400) segfault identically. Bisected on that simple shape:

| simple contracts | result |
|---|---|
| 6,000 | `is true`, 6.9 s |
| 8,000 | **SEGFAULT** |
| 10,000 / 12,000 | **SEGFAULT** |

**nuXmv's `INVAR` ceiling is 6,000–8,000, independent of model complexity.**

## Attack 2 — a real hole in the model, but not the cause

nuXmv could not prove `|v| <= 2.0`, because contracts exist only *inside* the
operating box; once `v` escapes, `u` is unconstrained. `tanh` bounds `|u| <= 1`
**everywhere**, a free structural fact that was not encoded. Added it. Safety
still `is false`. (Also corrected: true max `|v|` is **1.770**, not the 1.19 I
first reported from the opening steps.)

## The finding that moved the needle: decouple the two grids

Drift grows ~0.4/step, which tracks the **v-cell width (0.667)**, not the CROWN
bound width — so displacement, not the network, was binding. Displacement depends
only on (v, θ), so it costs `nvd²` rather than `nx²·nv²`.

| model | contracts | verdict | violated at |
|---|---|---|---|
| net 12×6, disp 6×6 (coupled) | 5,184 | `is false` | k = 4 |
| net 8×4, disp 48×48 | 3,328 | `is false` | k = 9 |
| net 10×5, disp 32×32 | 3,524 | `is false` | k = 8 |
| net 6×3, disp 64×64 | 4,420 | `is false` | k = 8 |
| **net 8×4, disp 64×64** | **5,120** | `is false` | **k = 9** |
| net 6×3, disp 96×96 | 9,540 | **SEGFAULT** | — |

Decoupling roughly **doubled the proved horizon at lower contract count**.
Refining displacement past 32 bands gained nothing, so the binding constraint
moved back to the network contracts — which cost `nx²·nv²` and hit the ceiling
immediately.

## The limit, stated precisely

**~9 of NAV's 30 steps are proved sound before the over-approximation reaches the
obstacle, and the refinement that would extend it exceeds nuXmv's 6–8k `INVAR`
capacity.** Every counterexample is spurious: the true system never enters the
obstacle from any of 9 starts.

This is not "NAV fails". It is a measured horizon with the binding constraint
identified at each stage: displacement bloat first, then network bound width,
then solver capacity.

---

# Attack 3 — non-uniform / anisotropic grids. **Dead, and cleanly so.**

Every prior configuration had `nx = 2*nv`, and `gen_nav_smv.py` additionally
hardcoded `nth = nv`, so θ always got the same cell count as v despite spanning
2π. Both couplings are now broken. Neither mattered.

## 3a — reallocating the same budget changes nothing

| config | net contracts | disp | total | verdict |
|---|---|---|---|---|
| net 8x4 + disp 64 (prior best) | 1,024 | 4,096 | 5,120 | `is false` at k=8 |
| net 4x16 + disp 32 | 4,096 | 1,024 | 5,120 | `is false` at k=8 |
| net 8x8 + disp 32 | 4,096 | 1,024 | 5,120 | `is false` at k=8 |

Identical budget, opposite allocations, **identical horizon**.

## 3b — which axis actually drives the bound?

CROWN `u1` interval width at the cell where the abstraction fails (the true
trajectory's k=8 state), varying one axis at a time from a 8/8/8 baseline:

| refinement | nx | nv | nth |
|---|---|---|---|
| 4 | 2.507 | 2.077 | 2.290 |
| 8 (base) | 2.002 | 2.002 | 2.002 |
| 16 | 1.704 | 1.382 | 1.771 |
| 32 | 1.479 | 1.050 | 1.656 |
| 64 | 1.399 | **0.892** | 1.596 |

**Velocity is the strong axis; θ is the weakest** — the opposite of the
hypothesis. No single axis alone escapes 0.89.

## 3c — why this kills non-uniform grids specifically

A non-uniform grid buys precision by refining *finely near the obstacle* — that
is, in **x and y**. Those are the two weakest axes (2.002 → 1.399 at 64×). The
blowup lives in **velocity** space, which has no spatial locality near the
obstacle to exploit. A non-uniform spatial grid refines the wrong dimension.

## 3d — the real size of the gap

At the failure cell the width is **2.002** — exactly the vacuous bound
(`u1 ∈ [-1,1]`), i.e. zero information. Joint refinement:

| nx | nv | nth | contracts | width |
|---|---|---|---|---|
| 8 | 8 | 8 | 4,096 | 2.002 |
| 8 | 16 | 16 | 16,384 | 1.116 |
| 8 | 32 | 16 | 32,768 | 0.802 |
| 16 | 64 | 32 | 524,288 | 0.423 |
| 32 | 64 | 64 | 4,194,304 | 0.192 |

nuXmv's ceiling is 6,000–8,000. The gap is **4× to 85×** depending on the width
one is willing to call informative — not the ~2× that a cleverer grid could
plausibly close.

## 3e — the benchmark is not degenerate

The true trajectory's minimum distance to the obstacle is **0.2445**, at k=11,
position (0.980, 2.244), from start (2.90, 3.05). NAV is provable in principle;
the obstruction is solver capacity, not a zero-margin specification.

## Reproduce

```bash
python3 COLLAB/experiments/nav_sound/margin_check.py        # 3e
python3 COLLAB/experiments/nav_sound/slack_check.py 8 8     # width along trajectory
python3 COLLAB/experiments/nav_sound/theta_sensitivity.py   # 3b
python3 COLLAB/experiments/nav_sound/gen_nav_smv.py 8 8 obstacle 32    # 3a
```

---

# Attack 4 — is the bound CROWN's fault? **No. CROWN is nearly tight.**

If the intervals were loose because of the *relaxation* rather than the cell
size, a stronger bounding method (full α,β-CROWN with branch and bound, or exact
MILP) would buy precision at no extra contract cost. Measured against the true
range of `u1` over the same cell by dense sampling (12⁴ = 20,736 points):

| cell (nx/nv/nth) | CROWN | sampled true | CROWN w | true w | ratio |
|---|---|---|---|---|---|
| 8/8/8 | [−1.403, 0.599] | [−0.934, 0.577] | 2.002 | 1.511 | **1.32×** |
| 8/16/16 | [−1.131, −0.014] | [−0.934, −0.016] | 1.116 | 0.918 | **1.22×** |
| 16/32/16 | [−0.911, −0.262] | [−0.834, −0.263] | 0.649 | 0.571 | **1.14×** |

Sampling under-approximates the true range, so these ratios are upper bounds on
the available slack. **The network's output genuinely varies by ~1.5 across the
failure cell** — the width is a fact about the function on that box, not an
artefact of the relaxation. At most ~1.3× is recoverable from better bounding,
against the 4–85× needed.

# Summary: every escape route is measured and closed

| route | result | where |
|---|---|---|
| finer uniform grid | needs 32k–524k contracts vs a 6–8k ceiling | 3d |
| different budget allocation | no effect — 3 configs at 5,120, all k=8 | 3a |
| non-uniform (fine near obstacle) | refines x,y, the **weakest** axes | 3b, 3c |
| tighter bounding method | ≤1.32× available | 4 |
| different solver (UCLID5) | 12–82× **slower** than nuXmv on this architecture | claim #55 |

Precision can come only from smaller cells; cells cost `nx²·nv·nth`; the
solver ceiling is 6,000–8,000. The window is empty, and now provably so rather
than just observedly so.

---

# The night of 09/29: the wall was self-inflicted

Everything above measures one quantity — how deep the abstraction stays faithful
before a spurious counterexample — against one budget, nuXmv's 6–8k `INVAR`
ceiling. Both halves of that framing turned out to be wrong.

## 5 — nuXmv is the wrong tool, twice over

Affine contracts (keeping CROWN's linear relaxation instead of concretizing it)
are 2.5–8× tighter at identical contract count. nuXmv **cannot decide them at
all**: timeout at 1,200 s on 514 contracts and 2,700 s on 5,120, under
`msat_check_invar_bmc`, `check_invar_ic3` and `..._cegar_implabs` alike, and at
every BMC depth from 1 to 6. The contracts are numerically clean (coefficients
in [−1.099, 1.155]).

A direct Z3 encoding of **the same contracts** decides it in **14.6 s**.

The encoding is verified faithful rather than assumed: on interval 8×8+d32 it
returns `unsat` at k=7 and `sat` at k=8, exactly matching nuXmv's `is false`
with a 9-state trace — in 9.3 s against 65 s.

So nuXmv's `INVAR` ceiling *and* its inability to handle dense linear contracts
are properties of one tool, not of the approach.

## 6 — NAV needs no inductive invariant

NAV carries `k : 0..30` and freezes at the horizon, so every reachable state
occurs within 30 steps. **BMC unrolled to depth ≥ 30 is a complete proof**, not
a bounded approximation. The invariant-synthesis wall (claims #52, #53) applies
to proving by *induction*; it does not apply to unrolling, and unrolling is
available because the horizon is short and finite.

## 7 — the contract-count wall was self-inflicted

Every model emitted a contract for **every cell** of the 4-D box. The system
visits a thin tube: at grid 16/16/16 the true trajectories touch **74 of
65,536** cells. Interval propagation cannot find that tube soundly — it swells
to cover the whole grid within ten steps — so `z3_cegar.py` lets the *solver*
name the cells, adding only those a counterexample actually occupies.

Cells outside the active set are unconstrained, so the abstraction is **coarser**
than the fully-emitted one, never finer. That direction is exactly why `UNSAT`
at any iteration is a real proof.

| configuration | network cells | network contracts used | result |
|---|---|---|---|
| net 8×8×8, disp 128, k=8 | 4,096 | 4,096 (full) | `unsat`, 57 s |
| net 8×8×8, disp 128, k=8 | 4,096 | **34** | `unsat`, 108 s |
| net 32×32×32, disp 128, k=8 | 1,048,576 | **165** | `unsat`, **137 s** |

**These counts are network contracts only.** Every one of these runs also
carried all `nvd²` displacement contracts (16,384 at disp 128), so the last
row's true total is 16,549, not 165. The reduction is real but applies to the
*network* family — which is the one that matters, since the `nx²·nv·nth`
explosion that bounded this project lives there, and it collapsed from
1,048,576 to 165. The grid is **128× past nuXmv's entire `INVAR` capacity**.

Refining the displacement cells the same way **fails**, instructively: it cut
iteration cost 45× (137 s → 3 s) and stopped the loop converging, because an
uncovered displacement cell leaves `dx`, `dy` free and the solver walks straight
through it. Refinement helps where contracts are redundant; the plant's are not.

The loop also diagnoses its own limit. At disp 32 it halts with *"counterexample
stays inside the covered cells: the grid is too coarse"* — correctly naming the
**displacement** contracts, not the network contracts, as binding. Raising disp
32 → 128 flipped `sat` to `unsat`.

## 8 — and affine loses after all

Head-to-head depth scan, net 4×4 + disp 16, same solver:

| k | interval | affine |
|---|---|---|
| 5 | `unsat` 0.1 s | `unsat` 88.7 s |
| 6 | `unsat` 0.1 s | `unsat` 42.4 s |
| 7 | `unsat` 0.2 s | timeout (>240 s) |
| 8 | `sat` 0.2 s | — |

Affine buys tightness that refinement now supplies far more cheaply. Partial
affine is no help either: dropping from 4 coupled axes to 2 loses 70% of the
tightness, because CROWN's strength is in the joint linear form.

## Reproduce

```bash
python3 COLLAB/experiments/nav_sound/z3_bmc.py 8 8 32 interval 8      # faithfulness
python3 COLLAB/experiments/nav_sound/z3_cegar.py 32 32 128 interval 8 20
python3 COLLAB/experiments/nav_sound/reachable_cells.py
python3 COLLAB/experiments/nav_sound/theta_sensitivity.py
```


---

# 9 — where this actually leaves NAV

**The proved horizon did not move.** It is still k=8 of 30, the same number
nuXmv gave. Everything around it changed; the number did not.

## Measured boundaries

| net grid | disp | proves | fails |
|---|---|---|---|
| 8³ | 128 | k=8 | k=9 |
| 32³ | 128 | k=8 | k=9 |
| 8³ | 256, 512 | — | k=9 |
| 32³ | 256 | — | k=9 |
| 128³ | 256 | could not seed within 1500 s | — |

## Why k=9 resists — two wrong guesses, then the answer

- **Not** the abstract state escaping the contract grid, where no contract
  applies. The stalled counterexample sits at `x ∈ [1.972, 2.900]`,
  `v ∈ [−1.323, 0.000]` — well inside the modelled box.
- **Not** coverage gaps. Expanding by ±1 *cell* covers a shrinking *physical*
  region as the grid refines, so a finer grid should slip out of the covered
  set more easily. Replacing it with a fixed physical margin did not move the
  barrier.
- **It is contract width**, and the budget model accounts for it exactly: k=9
  needs `w ≤ 0.302`; net 32³ delivers `w = 0.317`. Just short.

The counterexample is confirmed **spurious**: simulating the true closed loop
from its own initial state (2.900, 2.938) clears the obstacle by 0.436.

## What did improve

| | before tonight | now |
|---|---|---|
| contracts for the k=8 proof | 16,562 | **1,327** |
| time | 190 s | **~15 s** |
| grid it holds on | 4,096 cells | **1,048,576** (128× past nuXmv) |
| proof obligation | assumed to need an inductive invariant | **BMC to horizon is complete** |
| distance to k=30 | unknown | `w ≤ 0.0272` needed; 128³ gives 0.0729 |

The wall moved from *"we cannot express this"* to *"we need `w ≤ 0.0272` and
the finest grid we can currently seed gives 0.0729"* — a stated, finite gap
rather than a mystery. Closing it needs either a grid we cannot yet seed, or
a contract form that is tighter per cell without being denser per query — and
§8 shows the obvious candidate, affine contracts, is denser per query by more
than it gains.


---

# 10 — the wall, finally characterised

It is not contract count, not solver capacity, and not cell size.

## 10a — the plant is binding, not the network

One-sided gap in y at k=8 (how far the abstraction's lower bound sits below the
true one; the obstacle margin is 0.2445):

| network grid | plant | gap |
|---|---|---|
| 8³ | disp 32 | 0.591 |
| **32³** | disp 32 | **0.532** |
| 8³ | **disp 128** | **0.452** |

Refining the **network** 4× — sixty-four times the cells — changes it by 0.059,
about 10%. Refining the **plant** changes it by 0.139.

*(An earlier version of this table read 0.595 / 0.594 / 0.452, from a bug: 
`add_cells` emitted contracts for `u1` only, leaving `u2` bounded merely by the
global `|u2| ≤ 1`. Fixed; the numbers above are the corrected ones. The bug
made network refinement look completely useless rather than merely weak.)*

Extrapolating honestly: at a factor of 0.90 per 4× refinement, reaching the
0.2445 margin from 0.532 needs about 7.4 further steps — roughly a **40,000×
finer grid**. This project has spent its effort refining the neural side while
the binding constraint was the displacement abstraction — which is not a neural
network at all, but `v·cos θ`, a known analytic function.

## 10b — why refinement cannot win: a feedback loop

Once the abstract state spans more than one cell, the control it admits is the
**union** of the spanned cells' contracts. A finer grid gives proportionally
*more, smaller* cells covering the same physical spread, so the union is just as
wide. Measured from both sides: refining 8³ → 32³ tightens the per-cell `u2`
bound from **2.80 to 0.74** and leaves the drift **unchanged**.

`cell_span.py` showed θ's abstract spread growing at exactly **0.51 cells/step
= 0.40 rad/step = 2·dt·1.0** — the global `|u2| ≤ 1` bound. That measurement was
an artefact of the missing-`u2` bug above, not evidence about the contracts, and
is withdrawn.

## 10c — the culprit is the heading control

Contract width along the true trajectory at 8³:

| k | `u1` width | `u2` width |
|---|---|---|
| 0 | 0.0018 | 0.1326 |
| 1 | 0.0011 | 0.8551 |
| 2 | 0.0011 | 0.8551 |
| 3 | 0.0194 | **2.8041** |
| 4 | 0.0898 | **2.8434** |

The acceleration contract is essentially exact; the **heading contract is
vacuous by step 3** — wider even than the free bound `|u2| ≤ 1` allows. That is
why the abstraction descends into the obstacle: it is free to steer.

## What this means

**Interval contracts on a closed loop are self-defeating once state uncertainty
exceeds one cell, independent of how many contracts one can afford.** The
horizon stalls at k=8 for a structural reason, and every refinement strategy
tried tonight — finer uniform grids, anisotropic reallocation, non-uniform
grids, demand-driven cell addition, CEGAR cell splitting, physical-margin
coverage — fails for the same one.

## Reproduce

```bash
python3 COLLAB/experiments/nav_sound/drift_profile.py 8 8 32 9 1
python3 COLLAB/experiments/nav_sound/drift_profile.py 32 32 32 9 1
python3 COLLAB/experiments/nav_sound/cell_span.py 8 8 128 9
```
