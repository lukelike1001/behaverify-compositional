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
