# Monolithic NAV (First Run)

**Date:** 2026-09-03
**Feature:** Our first run of monolithic NAV. There may be some flaws, but we have some stuff working.

**Baseline (before):** `316067a` — *Match ACAS Xu safety invariant for discrete*
**Scope:** `examples/NAV/` (new): `core/`, `networks/` wrappers, `tree/`.

**Status (final):** monolithic NAV **works**. Both networks verify all three
specifications on the faithful box in about a minute.

The first run of the day did not look like that: it produced a 145 MB SMV and a
nuXmv segfault. §§1–4 describe that first attempt as it happened, §5 lists the
caveats we refused to skip, and **§7 records what the caveats turned up** — two
of them were real defects in this work, and both earlier conclusions were
wrong. Read §7 before quoting anything from §4.

---

## 1. The benchmark

ARCH-COMP 2025 AINNCS §3.11. A robot navigates to a goal region while avoiding
an obstacle. State is `(x1, x2, x3, x4)` = position, position, speed, heading.

```
x1' = x3 cos(x4)      x3' = u1
x2' = x3 sin(x4)      x4' = u2
```

Control period 0.2 s, horizon 6 s (30 steps), `u in (-1, 1)^2` from a
`4 -> 64 -> 64 -> 2` tanh network. Two controllers ship: `point` (standard RL)
and `set` (set-based robust training).

**Properties** (both are safety in the formal sense; the reach property is
*bounded* reachability, refutable by a finite prefix):

```
INVARSPEC  never inside the obstacle   x1,x2 in [1,2]^2
CTLSPEC    reaches the goal region     x1,x2 in [-0.5,0.5]^2
```

---

## 2. Why a lattice is unavoidable

The state space is continuous. The monolithic table method enumerates every NN
input and records the output, so it needs finitely many inputs. Discretization
is therefore not a modelling preference here — it is a precondition for
monolithic to run at all.

State is stored as `round(value * scale)`; the model is integer arithmetic
throughout.

**Resolution is not free to choose.** One control period changes speed and
heading by at most `|u| * dt = 0.2`. On a lattice coarser than that, a single
step moves less than one cell and rounds back to where it started:

| Resolution | Behaviour of the discretised closed loop |
|---|---|
| 0.5 | robot never leaves its initial cell — model is meaningless |
| **0.2** | **coarsest resolution that still reaches the goal** |
| 0.1 | reaches the goal |
| 0.05 | reaches the goal |

Verified by direct simulation on the integer lattice for both networks. So
0.2 is monolithic's **best case**, not a pessimistic choice.

---

## 3. Three obstacles that had to be solved

**1. `int()` truncation makes regression networks inert.**
`src/dsl_to_nuxmv.py:1105` stores regression outputs as `int(output)`. NAV's
`tanh` outputs lie in `(-1, 1)`, so every control truncates to **0**. Fixed by
wrapping the ONNX so it emits the integer state delta directly (`Mul` by
`S * dt`, then `Round`); see `networks/README.md`. `Round` needs opset 11+, so
the opset was raised 8 -> 13. No weights changed.

NAV is the repository's **first `regression` network** — grid world and ACAS Xu
are both `classification`. That code path had never been exercised.

**2. No trigonometry in the model.**
`dsl_to_nuxmv` will emit `sin`/`cos` into SMV, but ACAS Xu avoids them entirely
by discretising heading and using integer updates. NAV follows that precedent:
`cos`/`sin` are precomputed into static `DEFINE` arrays indexed by the heading
lattice and scaled by 1000.

**3. Meta-function inputs added a dimension.**
Writing network inputs as `(rdiv, x1, scale)` produced a rank-3 tensor and an
onnxruntime shape error. Resolved by moving the input scaling into the wrapper
(`Div` by `S`), so the `.tree` passes plain variables `{x1, x2, x3, x4}`,
exactly as grid world does.

---

## 4. Result

Instance: `set` network, resolution 0.2, box `x1,x2 in [-1,4]`,
`x3 in [-3,1]`, `x4 in [-2,3]`.

| Quantity | Value |
|---|---|
| Lattice sizes (x1, x2, x3, x4) | 26, 26, 21, 26 |
| **Table entries** | **369,096** |
| SMV generation | **15.4 s** |
| SMV file | **145 MB / 1,476,568 lines** |
| nuXmv | **signal 11 (segfault) at 6.9 s, peak RSS 1.2 GB** |
| Specifications reported | **none** |

Both specifications translate correctly into the SMV — obstacle `[1,2]^2` maps
to lattice `[5,10]^2`, goal `[-0.5,0.5]^2` to `[-2,2]^2`.

For comparison, ACAS Xu's full compositional run also died with signal 11
during BDD construction, at 8,982 injected INVAR lines.

**Reproduce:**

```bash
cd REPRODUCIBILITY/2026_TBA/examples/NAV
python3 -m core.nav_tree_generator --network set
python3 ../../src/dsl_to_nuxmv.py ../../metamodel/behaverify.tx \
    tree/nav_set.tree results/monolithic/nav_set.smv --recursion_limit 10000
../../nuXmv_DL/bin/nuXmv -source \
    ../../commands/nuxmv_commands/command_combo_invar_ctl \
    results/monolithic/nav_set.smv
```

Wrapper regeneration is in `networks/README.md`; the generating snippet lives in
this report's git history for this commit.

---

## 5. What this does NOT establish

Four checks stand between this run and any claim.

1. **The segfault is not confirmed to be size-driven.** NAV is the first
   `regression` model in the repository, so an untested code path could produce
   an identical symptom. Shrinking the box at fixed resolution until nuXmv
   survives would locate a ceiling and distinguish the two.

2. **The ONNX wrappers are unvalidated.** They have not been compared against
   the originals over a random input sample. Every number above depends on them
   behaving as intended.

3. **Only the `set` network was run.** `point` is untouched. ARCH-COMP reports
   NNV verifying `set` but not `point`, so the two are expected to differ.

4. **There is no compositional counterpart yet.** Without one, this is a
   monolithic failure with nothing to compare it to.

Also unresolved: the box was chosen from measured trajectories with margin, not
derived. A smaller box at the same resolution is the other knob, and it has not
been tried.

---

## 6. Relationship to earlier work

**Structurally faithful to the existing benchmarks.** Same `config {table}`
mechanism as grid world and ACAS Xu; same discretisation habit (ACAS Xu already
stores position as integer magnitudes and heading as an index 0–39); same
trig-avoidance; one `INVARSPEC` plus one `CTLSPEC`; a selector-shaped tree
mirroring grid world's. The single departure is `regression` instead of
`classification`, which the benchmark forces.

**Motivation.** Grid world and ACAS Xu are both discrete, so a finite table
always exists and monolithic can always run. Matching it there was never
evidence that compositional does something monolithic cannot. NAV is the first
benchmark in this repository where no finite table exists without
discretisation — which is what makes it the test case for that claim.


---

## 7. What the caveats turned up (same day)

Every check in §5 was run. Two found real defects, and both of §4's
conclusions were wrong. The corrected results are in §8.

### 7.1 Wrapper equivalence — passed

2,000 random lattice points per network, comparing the wrapper against
`round(u * S * dt)` computed from the untouched original.

**0 / 2000 mismatches for both networks.** Caveat 2 discharged.

### 7.2 The segfault was the shell's stack limit, not a scaling wall

`ulimit -s` defaults to 8192 KB. nuXmv builds the model with deep recursion over
the table's nested `case` expression and overflows that stack. The symptom —
signal 11 during `go`, before any specification is checked — is identical to
running out of capacity, which is why it was mistaken for one.

With `ulimit -s unlimited`, every box completes:

| Box | Entries | SMV | nuXmv wall |
|---|---|---|---|
| tiny | 4,356 | 2 MB | 4 s |
| small | 23,409 | 9 MB | 16 s |
| med | 51,623 | 20 MB | 38 s |
| tight | 198,375 | 75 MB | 227 s |

The claim in §4 that nuXmv could not build the model is **withdrawn**. It builds
it fine. This also explains the anomaly we should have caught immediately:
NEUS verified a 6.25 M-entry grid world, so a 23 k-entry model failing was never
consistent with a capacity story.

### 7.3 `idiv` truncation was breaking the dynamics

`idiv` truncates toward zero. The position update
`x1 + idiv(x3 * cos(x4) * dt, D)` therefore rounds *against* motion at every
step, and the bias accumulates:

| Position update | Initial states reaching the goal |
|---|---|
| truncation (as first written) | **0 / 9** |
| round half away from zero | 6 / 9 |

Fixed in the generator by shifting the numerator half a divisor in its own
direction before dividing, using a `case` on the sign.

### 7.4 The model evaluated the network on a state that never exists

This is the defect that produced §4's obstacle "violation".

The generated table conditioned on **mixed stages**:

```
x1_stage_1, x2_stage_1, x3_stage_0, x4_stage_0
```

That is position *after* the environment update and velocity *before* it — a
state the system is never in.

The comparison that establishes it is a defect, not normal encoding: grid world
has the same shape (a NEURAL variable whose inputs are `env` position variables
that `environment_update` writes), and its table conditions on

```
destination_x_stage_0, destination_y_stage_0, drone_x_stage_0, drone_y_stage_0
```

— all `stage_0`. ACAS Xu likewise conditions on all `stage_0`, though it is a
weaker control since its network inputs are derived `DEFINE`s rather than the
state variables themselves.

**Cause.** NAV read the NEURAL variable directly from `environment_update`.
`x1`'s update reads `x3` and `x4`, which are themselves network inputs, so
BehaVerify resolved the dependency through staging. Grid world never hits this
because its `drone_x` update reads only `drone_x` and a blackboard variable —
never another network input.

**Fix.** Latch the control into blackboard variables during the tick, exactly as
grid world latches `network` into `current_action`, and let
`environment_update` read the latched values. After the fix the table conditions
on `x1_stage_0, x2_stage_0, x3_stage_0, x4_stage_0`.

This is a modelling error in this example, not a BehaVerify bug. It is worth
recording because nothing warns you: the model generates, builds, verifies, and
returns a plausible counterexample.

---

## 8. Corrected results

Faithful box `x1,x2 in [-1.0, 3.4]`, `x3 in [-2.4, 0.4]`, `x4 in [-1.2, 3.6]`,
resolution 0.2, latched control, rounded position update.

| | `set` | `point` |
|---|---|---|
| Table entries | 198,375 | 198,375 |
| SMV | 39 MB | 39 MB |
| nuXmv wall (stack raised) | **71 s** | **61 s** |
| Obstacle avoidance | **true** | **true** |
| Box containment | **true** | **true** |
| Reach goal | **true** | **true** |

The SMV is *smaller* than §4's (39 MB vs 145 MB) and nuXmv is faster, because
the box is tighter and the mixed-stage encoding had inflated the model.

Box containment holding is what makes the other two meaningful: the environment
update clamps at the box edge, and clamping would silently keep an escaping
robot inside. Since the state never reaches the boundary, the box is a verified
assumption rather than a hidden one.

**Reproduce:**

```bash
cd REPRODUCIBILITY/2026_TBA/examples/NAV
python3 -m core.nav_tree_generator --network set
ulimit -s unlimited          # required; see 7.2
python3 ../../src/dsl_to_nuxmv.py ../../metamodel/behaverify.tx \
    tree/nav_set.tree results/monolithic/nav_set.smv --recursion_limit 10000
../../nuXmv_DL/bin/nuXmv -source \
    ../../commands/nuxmv_commands/command_combo_invar_ctl \
    results/monolithic/nav_set.smv
```

---

## 9. What this now means

**Monolithic handles NAV.** The morning's framing — "the first benchmark where
monolithic cannot run" — is wrong, and no claim in that direction should be
made from this work.

What survives is quantitative, and it is about **discretisation**, not capacity:

1. **A lattice is mandatory.** The plant is continuous; the table needs finitely
   many inputs.
2. **Resolution is bounded below by the physics.** One control period changes
   speed and heading by at most 0.2, so a coarser lattice freezes the robot.
   0.2 is the floor.
3. **Faithfulness costs more than the floor.** At 0.2 only 6 of 9 initial states
   reach the goal on the lattice, though the continuous system reaches from all
   9. Restoring all 9 needs position resolution 0.1 for `set` (259,200 entries)
   and 0.05 for `point` (1,893,360). The verified run above uses the single
   midpoint initial state, which is one of the 6 that works.
4. **The margins are thin.** The real trajectory's closest approach to the
   obstacle is 0.2889, barely more than one lattice cell. A slightly coarser
   lattice snaps the trajectory onto the obstacle corner.

Point 3 is the real opening for the compositional approach, and it is a
different argument from the one we started the day with: monolithic does not
fail here, it **approximates**, and the cost of making the approximation
faithful grows as `resolution^-4` while the property margin stays fixed.

Whether compositional avoids that trade is untested.

### Still open

- The discretised model is verified for **one** initial state, not the
  ARCH-COMP initial *set* `x1,x2 in [2.9,3.1]`. Covering the set needs either a
  nondeterministic initial condition or the finer lattice from point 3.
- No compositional counterpart exists yet.
- `ulimit -s unlimited` is required and is not recorded anywhere a user would
  see it before hitting the crash.
