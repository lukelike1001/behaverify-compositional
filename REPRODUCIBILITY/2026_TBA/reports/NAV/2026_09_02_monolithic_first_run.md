# Monolithic NAV (First Run)

**Date:** 2026-09-02
**Feature:** Our first run of monolithic NAV. There may be some flaws, but we have some stuff working.

**Baseline (before):** `316067a` — *Match ACAS Xu safety invariant for discrete*
**Scope:** `examples/NAV/` (new): `core/`, `networks/` wrappers, `tree/`.

**Status:** the pipeline runs end to end and produces a 145 MB SMV in 15.4 s.
**nuXmv then dies with signal 11 (segfault) after 6.9 s, reporting no
specifications.** Whether that is a scaling wall or a bug is **not yet
established** — see §5. Nothing here is a claim yet.

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
