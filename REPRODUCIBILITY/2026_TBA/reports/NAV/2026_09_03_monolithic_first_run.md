# Monolithic NAV (First Run)

**Date:** 2026-09-03
**Baseline (before):** `316067a` — *Match ACAS Xu safety invariant for discrete*
**Scope:** `examples/NAV/` (new): `core/`, `networks/` wrappers, `tree/`.

**Status.** The pipeline runs end to end and nuXmv verifies all four
specifications for both networks in about a minute. **That is not a
verification of NAV**, and the gap is not a caveat — it is the main content of
this report.

What we have is a *different closed loop*: a three-level quantised controller
on an explicit-Euler plant, checked from one initial lattice point. The lattice
is neither an over- nor an under-approximation of the benchmark plant, so a
TRUE at an interior point carries very little. §6 says exactly what is and is
not licensed; §7 is the plan to fix it.

An earlier draft of this report claimed the opposite — that monolithic could
not build the model at all. That was wrong twice over, and §5 records why.

---

## 1. The benchmark

ARCH-COMP 2025 AINNCS §3.11. A robot navigates to a goal region while avoiding
an obstacle. State `(x1, x2, x3, x4)` = position-x, position-y, speed, heading:

```
dx1 = x3 cos(x4)      dx3 = u1
dx2 = x3 sin(x4)      dx4 = u2
```

Controller: `4 -> 64 ReLU -> 32 ReLU -> 2 tanh`, so `u in (-1, 1)^2`. Control
period 0.2 s, horizon 6 s (30 periods). Obstacle `[1,2]^2`, goal
`[-0.5,0.5]^2`, initial set `x1,x2 in [2.9,3.1]`, `x3 = x4 = 0`. Two
controllers ship: `point` (standard RL) and `set` (set-based robust training).

**Properties.** Avoid the obstacle on `t in [0,6]`; be in the goal **at**
`t = 6`. Both are safety properties in the Alpern–Schneider sense — each is
refuted by a finite prefix. The reach half is a bounded-time state constraint,
not liveness, and it is *stricter* than `F_{<=N}`: arriving early and leaving
again does not satisfy it.

### 1.1 Where the artifacts disagree with the AINNCS report

Two places. **The artifacts win**, and both are easy to "fix" in the wrong
direction later, so they are recorded here with their evidence.

**Architecture.** The report says the controllers have "two hidden layers with
64 neurons each". They do not — both are `64 -> 32`:

| Source | Evidence |
|---|---|
| `nn-nav-{point,set}.onnx` | `fc_1: [4,64]`, `fc_2: [64,32]`, `fc_3: [32,2]` |
| `nn-nav-{point,set}.mat` | `W[0] (64,4)`, `W[1] (32,64)`, `W[2] (2,32)` |

ONNX, MATLAB, and the set-based RL paper (arXiv 2408.09112) agree. Do not
reshape the network to match the report.

**Variable order.** The report's prose lists the state as position, position,
angle `theta`, velocity `nu` — heading third. The artifacts have speed third
and heading fourth. `dynamics.m` is unambiguous:

```matlab
dx = [ x(3)*cos(x(4)); x(3)*sin(x(4)); u(1); u(2) ];
```

`x(3)` multiplies both `cos` and `sin`, so it is the speed; `x(4)` is the angle.
Equation (15) is an unnamed 4-vector `[ν cos θ; ν sin θ; u(1); u(2)]`. Bound to
the report's English order it contradicts MATLAB (`θ' = u(1)`); bound to MATLAB
it is the plant we run (`ν' = u(1)`). Only the surrounding English disagrees, and
the Unicycle benchmark in the same document uses the other order, which is likely
where the slip came from. Our tree follows `dynamics.m`. Full evidence:
`2026_09_03_paper_mismatch.md`.

ARCH-COMP states that the competition's discrete-time versions are obtained by
forward Euler, `x(k+1) = x(k) + f(x) dt`. So Euler is a sanctioned plant. What
is **not** interchangeable with the benchmark instance is treating one Euler
step of 0.2 s as equivalent to what CORA / NNV / JuliaReach run, which is
sample-and-hold plus integration of the flow between samples. Sampled-time
avoidance is weaker than the continuous tube — the same gap NEUS already admits
for ACAS Xu's 6-second checks.

---

## 2. Why a lattice, and what it costs

The state space is continuous; the table method enumerates NN inputs. So
discretisation is a precondition for monolithic to run, not a modelling
preference. State is stored as `round_half_away(value * S)`, and all model
arithmetic is integer.

**Resolution is not a simple floor.** An earlier draft claimed 0.2 was the
coarsest workable resolution because one control period changes speed and
heading by at most `|u| dt = 0.2`. That reasoning is wrong: under the wrapper
encoding, speed changes iff `|u| S dt >= 0.5`, which only forces a freeze when
`S dt < 0.5`, i.e. resolution coarser than 0.4. Measured over the 9 corner and
midpoint starts of the initial set, using the generator's exact arithmetic:

| Resolution | `S·dt` | Moves? | Reaches goal | Collides |
|---|---|---|---|---|
| 0.5 | 0.40 | **no** | 0 / 9 | 0 / 9 |
| 0.33 | 0.60 | yes | 0 / 9 | **9 / 9** |
| **0.25** | 0.80 | yes | **9 / 9** | **0 / 9** |
| 0.2 | 1.00 | yes | 6 / 9 | **2 / 9** |
| 0.1 | 2.00 | yes | 9 / 9 | 0 / 9 |

The relationship is **non-monotone**: 0.25 is strictly better than 0.2, and 0.33
moves but drives every start into the obstacle. Refinement does not improve
faithfulness monotonically, which is the clearest symptom of the problem in §3.

Position is not protected by the speed bound. After one period `|x3| <= 0.2`,
so `|x3| dt <= 0.04` — five times smaller than a 0.2 cell. Position stays put
until the increment clears the rounding threshold. The 6/9 result above is a
position-quantisation failure, not a speed freeze.

---

## 3. The lattice is not a conservative abstraction

This is the framing correction, and everything in §6 follows from it.

The model rounds each successor to a single lattice point. It therefore admits
behaviours the plant does not have, and omits behaviours the plant does have.
It is neither an over-approximation (which would make `INVARSPEC = true`
meaningful) nor an under-approximation (which would make `false` meaningful).
It is a *different system*.

Two consequences, both observed:

* Refinement is non-monotone (§2).
* At resolution 0.2, **2 of 9** initial-set starts collide with the obstacle on
  the lattice, while the continuous plant collides from none. The midpoint —
  the one state we verify — happens to be safe.

Fixing this is item **E** in §7 and has not been done.

---

## 4. Implementation obstacles

**1. `int()` truncation makes regression networks inert.**
`src/dsl_to_nuxmv.py:1105` stores regression outputs as `int(output)`. NAV's
`tanh` outputs lie in `(-1,1)`, so every control truncates to **0**. Worked
around by wrapping the ONNX: `Div` inputs by `S`, original network, `Mul`
outputs by `S dt`, then `Round`. Opset raised 8 -> 13 for `Round`. No weights
changed; 0 mismatches over 2000 random lattice points.

**This wrapper is also a modelling decision, not just a fix.** With `S dt = 1`
at `S = 5`, the plant sees `u in {-1, 0, +1}`: along the midpoint trajectory
**12 of 30** `u1` values and **16 of 30** `u2` values have `|u| < 0.5` and
become zero. We are verifying a three-level controller, not the tanh network.
Decoupling the scales so this is tunable is item **D** in §7.

NAV is the repository's first `regression` network — grid world and ACAS Xu are
both `classification`, so that path had never been exercised.

**2. No trigonometry in the model.** `cos`/`sin` are precomputed into static
`DEFINE` arrays indexed by the heading lattice, scaled by 1000, following ACAS
Xu. The table's own error is `<= 5e-4`, contributing `<~ 0.002` lattice units —
negligible. The real trig error is heading *reconstruction* (+/-0.1 rad at
`S = 5`), which is the cell-width issue of §3, not the table.

**3. Meta-function inputs added a dimension.** `(rdiv, x1, scale)` produced a
rank-3 tensor. Resolved by moving input scaling into the wrapper, so the tree
passes plain variables as grid world does.

---

## 5. Defects found and fixed

### 5.1 The first-run segfault was the shell's stack limit

`ulimit -s` defaults to 8192 KB; nuXmv recurses over the table's nested `case`
and overflows it. The symptom — signal 11 during `go`, before any
specification — is indistinguishable from running out of capacity, which is how
it was misread. With `ulimit -s unlimited` every box builds:

| Box | Entries | SMV | nuXmv |
|---|---|---|---|
| tiny | 4,356 | 2 MB | 4 s |
| small | 23,409 | 9 MB | 16 s |
| med | 51,623 | 20 MB | 38 s |
| tight | 198,375 | 75 MB | 227 s |

The claim that nuXmv could not build the model is **withdrawn**. It never
should have survived contact with NEUS's 6.25 M-entry grid world.

### 5.2 `idiv` truncation biased the dynamics

`idiv` truncates toward zero, so the position update rounded against motion
every step: **0 of 9** starts reached the goal. Now rounds half away from zero
(shift the numerator half a divisor in its own direction, via a `case` on the
sign): 6 of 9. Not a zero-mean error — one-sided.

### 5.3 The network was evaluated on a state that never exists

The generated table conditioned on `x1_stage_1, x2_stage_1, x3_stage_0,
x4_stage_0` — position *after* the environment update, velocity *before*.

Grid world is the exact structural control (a NEURAL variable whose inputs are
`env` variables that `environment_update` writes) and conditions on all
`stage_0`. So this was a defect in our model, not BehaVerify behaviour.

Cause: reading the NEURAL variable directly from `environment_update`, where
`x1`'s update reads `x3` and `x4` — themselves network inputs — so BehaVerify
resolved the cycle through staging. Grid world never hits this because
`drone_x`'s update reads only `drone_x` and a blackboard variable.

Fixed by latching the control into blackboard variables during the tick, as
grid world latches `network` into `current_action`. The table now conditions on
all `stage_0`.

### 5.4 Two rounding conventions in one model

`to_lattice` used Python `round` (half-to-even) while the position update used
half-away. `round(2.9 * 5) = round(14.5) = 14`, so the state 2.9 was stored as
**2.8** — one cell toward the obstacle, and one of the colliding traces. A
single `_round_half_away` is now used by `to_lattice`, the trig table, and the
position update.

### 5.5 Region boundaries rounded the wrong way

Surfaced by 5.4. A real interval rarely lands on lattice points, and the two
regions need **opposite** rounding to stay conservative:

* obstacle **outward** — a larger modelled box, so avoiding it implies avoiding
  the real one;
* goal **inward** — a smaller modelled box, so reaching it implies reaching the
  real one.

Rounding both the same way is unsound in one of them: half-away mapped the goal
`[-0.5,0.5]` to lattice `[-3,3] = [-0.6,0.6]`, *larger* than the real goal, and
would have accepted states outside it. Now `floor`/`ceil` by direction —
obstacle `[5,10]`, goal `[-2,2]`.

### 5.6 The plant never stopped at the horizon

`environment_update` runs every tick regardless of the tree, so `hold` did not
freeze anything: the robot kept flying and the specifications ranged over an
infinite suffix the benchmark says nothing about.

Gating on the step counter does not work in either direction — `step < horizon`
drops the 30th control period so `t = 6` is never reached, and `step <= horizon`
never stops because `step` saturates. The fix gates on whether *this tick
applied a control*: an `apply_plant` flag set true by `advance` and false by
`hold`, with every plant assignment a no-op when it is false.

### 5.7 The reach property was encoded as unbounded `AF`

The generator emitted `AF(goal)`, which is not the benchmark property. With the
plant frozen at the horizon, the benchmark property is

```
INVARSPEC (step = horizon) -> goal
```

`AF(goal)` is **weaker**, not stronger: after the freeze it says only that some
sample in `[0,6]` was in the goal, so a trajectory that clips the goal at
`t = 4` and leaves satisfies `AF` and fails the invariant. It is still emitted,
labelled as the weaker "visited the goal" check.

**Falsification test.** At horizon 10 (`t = 2 s`) the robot is at
`(1.38, 2.39)`, far from the goal. The model returns `goal@horizon = false` and
`visited = false`, while `avoid` and `containment` stay true. The encoding
discriminates.

---

## 6. Results, and what they license

Faithful box `x1,x2 in [-1.0,3.4]`, `x3 in [-2.4,0.4]`, `x4 in [-1.2,3.6]`,
resolution 0.2, single initial state `(3.0, 3.0)`. These are the committed
defaults in `core/nav_domain_config.yaml`, so the reproduce steps below
regenerate exactly this table.

| | `set` | `point` |
|---|---|---|
| Table entries | 198,375 | 198,375 |
| SMV | 39 MB | 39 MB |
| nuXmv (stack raised) | **72 s** | **61 s** |
| Avoid obstacle | true | true |
| **Goal at horizon** | **true** | **true** |
| Box containment | true | true |
| Visited goal (weaker) | true | true |

Box containment holding is what stops the clamping in `environment_update` from
silently rewriting the plant: the state never reaches the boundary, so the box
is a verified assumption rather than a hidden one.

**What these four TRUEs license.** For one discrete explicit-Euler trajectory,
from the single lattice point `(3.0, 3.0)`, under the wrapped three-level
controller, with the tabulated `cos`/`sin`: no sampled state occupies an
obstacle cell, the state at `t = 6` is in the (inward-rounded) goal, and the
state never leaves the modelled box.

**What they do not license.** Anything about the ARCH-COMP initial set
`[2.9,3.1]^2`; anything about the continuous plant, or about times between
samples; anything about the tanh controller as opposed to its three-level
quantisation; and anything about other lattice points — **two of nine collide**.

Margins are thin enough that this matters. Closest approach to the obstacle,
with control held constant over each period:

| Network | Start | Euler `dt=0.2` | Sub-stepped ODE |
|---|---|---|---|
| `set` | (3.0, 3.0) | 0.2889 | **0.1887** |
| `set` | (2.9, 3.0) | 0.2466 | 0.1715 |
| `point` | (3.1, 2.9) | 0.2482 | **0.0157** |

One lattice cell at resolution 0.2 is 0.2000; the half-cell diagonal is 0.1414.
An earlier draft quoted 0.2889 as "the real trajectory's margin" — that is the
Euler figure, not the flow. The ODE margin at the midpoint is already **under
one cell**, and the `point` network passes within 0.0157 of the obstacle.

### Reproduce

`ulimit -s unlimited` is **required**, not optional: nuXmv recurses over the
table's nested `case` and overflows the default 8192 KB stack, segfaulting
during `go` before any specification is checked (§5.1).

```bash
cd REPRODUCIBILITY/2026_TBA/examples/NAV

# 1. emit the .tree (prints the table size; expect 198,375)
python3 -m core.nav_tree_generator --network set

# 2. build the SMV (~7 s, ~39 MB)
mkdir -p results/monolithic
python3 ../../src/dsl_to_nuxmv.py ../../metamodel/behaverify.tx \
    tree/nav_set.tree results/monolithic/nav_set.smv --recursion_limit 10000

# 3. model check (~70 s) -- the stack limit must be raised first
ulimit -s unlimited
../../nuXmv_DL/bin/nuXmv -source \
    ../../commands/nuxmv_commands/command_combo_invar_ctl \
    results/monolithic/nav_set.smv
```

Substitute `--network point` for the second controller. The command file checks
`check_invar` then `check_ctlspec`, so the four verdicts print in the order
avoid, goal-at-horizon, containment, visited.

Regenerating the ONNX wrappers is documented in `networks/README.md`; they are
committed, so step 1 works without it.

---

## 7. Planned next, in order

**A–C are done** (§§5.4–5.7). The remainder is what would turn this into a
result about faithfulness.

**D. Decouple the position, velocity, and heading scales.** Control fidelity is
governed by `Sv dt`, table size by `Sp^2 Sv Stheta`. Sharing one `S` conflated
"the robot cannot step" with "the controller is three-level". Run
`Sv in {5, 10, 25}` at fixed `Sp` as a measurement: `Sv = 25` gives `Sv dt = 5`,
so 11 control levels. If the table then blows up, that is a genuine monolithic
cost rather than a rounding bug. Three things must change together, or this
recreates the wrapper problem: `S` no longer cancels, so the position update
needs `Sp/Sv` rather than the current `dt_den`; heading `Stheta` is a third knob
that is both a network input and the trig index, and coarse `Stheta` is a plant
error rather than a control error; and the wrapper plus the `u1, u2` domain are
baked at `S dt = 1`.

**E. Make the successor an over-approximation.** Replace the single rounded
position increment with a nondeterministic assignment over the interval it
spans. This is what would make `INVARSPEC = true` mean something. Scope
honestly: `{floor, ceil}` of the increment at the cell *centre* is sound only
for "explicit Euler from the reconstructed centre, with the tabulated `cos`,
snapped to the lattice". It does **not** cover the cell around that centre — at
`S = 5` the cell is 0.2 wide and variation in `x3, x4` inside it already moves
the increment by a few tenths of a cell, so covering the cell needs roughly
+/-1 cell of fattening. It also does not close the sampled-vs-continuous gap. If
a nondeterministic successor would leave the box, it must not be clamped back
in — the strict containment invariant is what catches that.

Whether control quantisation must also become nondeterministic depends on the
claim: for the three-level controller, leave `Round` deterministic; for the tanh
network at the centre, `{floor, ceil}` of `u Sv dt` as well; for the tanh
network on the *cell*, neither suffices and a range for the network over a box
is required — Lipschitz or CROWN. That last case is precisely the compositional
opening.

Expect FALSE to be the likely and correct first outcome.

**F. Cover the initial set.** Sound covering means every lattice cell
intersecting `[2.9,3.1]^2`, not every rounded image of a 3x3 sample. At `S = 5`
that set is **one cell** (centre 15, `[2.9, 3.1]`), so F adds nothing at the
current resolution — nondeterministic init over `{14,15,16}^2` would be a
0.6x0.6 box of centres including `(2.8, 3.0)`, which is outside the benchmark
and is one of the colliding traces. F becomes a real product only once `Sp` is
raised enough that the initial set spans several cells. Until then the 9-point
sweep is a quantisation diagnostic, not a verdict.

---

## 8. Relationship to earlier work

**Structurally faithful to the existing benchmarks.** Same `config {table}`
mechanism as grid world and ACAS Xu; same discretisation habit (ACAS Xu already
stores position as integer magnitudes and heading as an index 0–39); same
trig-avoidance; a selector-shaped tree mirroring grid world's. The one departure
is `regression` instead of `classification`, which the benchmark forces.

**Motivation.** Grid world and ACAS Xu are discrete, so a finite table always
exists and monolithic can always run; matching it there was never evidence that
compositional does something monolithic cannot. NAV is the first benchmark here
with no finite table absent discretisation. The opening it exposes is **not**
that monolithic fails — it does not — but that monolithic must *approximate*,
and that making the approximation faithful is expensive in a way the property
margin does not shrink to match.

## 9. Provenance

The corrections in §§2, 3, 5.4–5.7 and 6 follow an independent review by Grok
(xAI), which identified the unsound resolution-floor argument, the mixed
rounding conventions, the three-level controller, the unbounded-`AF` encoding,
the unfrozen plant, the Euler-vs-ODE margin confusion, and — most importantly —
that the lattice is not a conservative abstraction. Every falsifiable claim in
that review was independently reproduced here before being adopted; the
architecture correction (`64 -> 32`, not `64 -> 64`) and the artifact-vs-report
disagreements in §1.1 came from the same review, and were re-checked here
against the ONNX graphs, the `.mat` weights, and `dynamics.m`.
The `apply_plant` gating in 5.6 and the direction-aware region rounding in 5.5
are refinements that emerged from implementing its recommendations.
