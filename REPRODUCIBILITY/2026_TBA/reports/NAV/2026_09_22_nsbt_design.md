# NAV: Designing the NSBT (Part 1 — reasoning and design audit)

**Date:** 2026-09-22
**Baseline (before):** `c2490df82` — *Provide preliminary Lipschitz bounds for monolithic NAV*
**Scope:** Design reasoning for the NAV NSBT, and the results of a throwaway
spike that exercised every risky mechanism against the real network before any
repository code was written.

**Status: half a report.** Part 2 will be written once the NSBT generator
exists, and will cover the implementation and the measured results. Nothing
here is committed code — the spike lived in a scratchpad and has been
discarded. Findings that can be checked against the repository today are marked
as such; the rest become tests when the generator is written (§7).

**Headline:** the design is settled and every mechanism it depends on has been
demonstrated end to end. Two defects in BehaVerify's regression path were found
and worked around without patching `src/`. A second argument against the
monolithic case emerged, independent of the soundness argument rather than a
replacement for it: **the coarsest grid at which the model tracks the robot at
all still needs a table roughly 15× larger than the one nuXmv segfaults on.**

---

## 1. Terms

| term | meaning |
|---|---|
| **generator** | the Python module that *writes* the `.tree` file. The tree cannot be hand-written — the trig tables alone are one line per θ value, and every constant changes with resolution. ACAS Xu's 4,029-line tree was produced the same way. |
| **h** (cell size) | the discretization resolution: how wide one cell is in physical units. `h = 0.5` means the model represents `x` only in steps of 0.5 m. Smaller `h` = finer grid = more table entries. |
| **`-dynamic`** | a nuXmv flag enabling dynamic BDD variable reordering. A BDD's size depends enormously on variable order; nuXmv can reorder on the fly when one grows. Off by default. |
| **integration scheme** | how the continuous ODE becomes discrete steps. Explicit Euler uses pre-update values for the position update; semi-implicit uses the newly updated velocity and heading. |

## 2. The tree

One action node. It is both root and only leaf.

```
Root (Act)
  reads:  net
  writes: u1, u2
  returns: success
```

Every 0.2 s the tree wakes, reads the network's output for the current state,
copies the two control values onto the blackboard, and reports success. The
environment then moves the robot. There are no check nodes and no branching.

**Why no obstacle or goal check.** The controller takes only the 4-D state; the
obstacle and goal are baked into its weights, not supplied as inputs. A tree
that branched on obstacle membership would model a robot with a sensor that
does not exist. Detection belongs in the specification.

**Why minimal is the right answer, not an apology.** The tree is deliberately
minimal so that the monolithic/compositional comparison isolates the neural
encoding. Any structure added here would be fiction absent from the benchmark,
and would be a confound — a difference between the two pipelines that is not
the thing under study.

For what it is worth, neither existing benchmark uses anything a `case`
statement could not express either: grid world's
`selector[sequence[NeedGoal, NewGoal], Act]` is a two-way branch on "am I on the
goal", and ACAS Xu's five-way `selector` is a dispatch on `command`. NAV having
one leaf is not a step down in kind.

## 3. State variables: integers in units of `h`

An SMV variable's representable values *are* the lattice, so "scale factor" and
"cell size" are one knob, not two. Each state variable is therefore an integer
in units of its own cell size, with physical value `idx · h`.

Working the arithmetic through, with `cos`/`sin` tabulated at ×1000:

```
v'   = v  + idiv(u1_s, 5000 * h_v)
th'  = th + idiv(u2_s, 5000 * h_th)
x'   = x  + idiv(v * cos_t[th + th_off], 5000)      <- h cancels
y'   = y  + idiv(v * sin_t[th + th_off], 5000)
```

The position update falling out independent of `h` is a useful property:
changing resolution touches the domains and the trig tables, not the update
logic.

**Per-dimension resolution.** There is no reason `x`, `y`, `v` and `θ` should
share a cell size — they are different units, and their quantization floors
differ (§6). Position can be coarser than velocity and heading. The cost is
that the position divisor becomes `5000·h_x/h_v`, integer-friendly only for
convenient ratios. Deliberate choice, not an inherited default.

**θ is not wrapped.** ACAS Xu wraps heading because its heading is cyclic there.
NAV's θ has a bounded domain already, so the trig tables cover the whole range
directly — no `mod`, no second representation. This matters because the network
is *not* periodic in θ: feeding it `θ + 2π` yields a completely different
control (verified against both shipped networks), so wrapping before the network
sees it would be wrong.

## 4. The clock and the horizon

`k ∈ [0, 30]` increments unconditionally, and the specification is a single
invariant:

```
AG( !overflow
  & (k <= 30 -> (x,y) not in Obs)
  & (k = 30  -> (x,y) in Goal) )
```

Both obstacle avoidance and "in the goal at t = 6" are safety properties in the
Alpern–Schneider sense — each is violated by a finite prefix — so no CTL, no
`AF`, and no fairness. The grid-world CTL gap does not recur here.

**The horizon must actually stop.** If `k` saturates while the physics keeps
running, many states have `k = 30` and the specification would demand the robot
*stay* in the goal forever. So the model freezes at the horizon, via
`tick_prerequisite { (lt, k, 30) }`.

**This is not the ACAS Xu freeze, and the distinction matters.** ACAS Xu freezes
on a *state* condition (`distance < max_dist`) that can fire at any time and
silently truncates exploration — that is what produced the spurious CTL
counterexample in `2026_07_15_liveness_parity.md`. Freezing on a *clock* at the
end of a bounded horizon is the benchmark's own semantics. Same mechanism,
entirely different justification.

## 5. Bounds as a proof obligation, not an assumption

The declared state box comes from `core/nav_boundary_finder.py`'s analytic
bounds, which are sound for any discretization of these dynamics because they
use only `|u| ≤ 1` and the integrator. They are also loose — roughly 11× wider
per position axis than the measured envelope.

Rather than assume a tighter box, `overflow` is set whenever an unclamped
successor would leave the declared domain, and `!overflow` is a conjunct of the
invariant. If nuXmv returns true, the box was valid *for the discretized model*
— proven, not asserted. If it returns false with an overflow counterexample,
the trace names the state where the box was too small.

This is strictly better than ACAS Xu, where `overflow` feeds `tick_prerequisite`
to enforce the bound rather than appearing in any specification.

## 6. What the spike established

A spike is a throwaway prototype built only to answer technical questions. This
one exercised every mechanism above against the real `nn-nav-set.onnx`.

### 6.1 Two defects in BehaVerify's regression path

Both are verifiable by reading the repository today.

**Outputs are truncated to integers.** `dsl_to_nuxmv.py:1107` does
`output = int(current_outputs[0][0][index])`. NAV's `u ∈ [-1, 1]`, and `int()`
truncates toward zero, so *every* table entry becomes 0. Demonstrated: a 36-cell
table emitted 72 zeros plus two `66` defaults — a controller that never
accelerates and never turns.

**Input expressions are broken.** The regression and classification input
builders are identical except for one character:

```python
current_input.append(atom((constants, cur_ref)))   # regression   (line 1101)
current_input += atom((constants, cur_ref))        # classification (line 1129)
```

Meta functions return lists, so the regression path nests one and onnxruntime
rejects the rank-3 input. **Regression networks currently accept only bare
variable references — no arithmetic in `inputs {}` at all.** ACAS Xu escapes
this only because it is classification; its five normalization expressions would
be impossible for a regression network.

**Workaround, chosen over patching `src/`:** fold both conversions into the ONNX
graph — `Mul(input, [h,h,h,h]) → original network → Mul(output, 1000)`. The
table then feeds raw index integers and receives milli-units. This keeps the
monolithic baseline stock, and carries an exact identity that belongs in a test:

```
wrapped(idx) == 1000 * original(idx * h)
```

verified exactly at several points. It also does the same job ACAS Xu's
`inputs { (rdiv, (sub, distance, distance_mean), distance_range), ... }` does in
the DSL — just in the graph, because the DSL path is unavailable.

### 6.2 Mechanisms confirmed

- `idiv` works at runtime, emitted as `/`. nuXmv truncates toward zero
  (`-7/2 = -3`), matching BehaVerify's meta `int(x/y)`. No meta/runtime
  mismatch, but a systematic bias toward zero — and `v` is negative for most of
  this trajectory.
- `DEFINE` arrays indexed by a variable work, emitted as a `case` chain with one
  arm per index. Cost is linear in array size, so the trig tables are cheap —
  unlike the NN table, they are one-dimensional and not part of the 4-D product.
- `tick_prerequisite` becomes `root.active`, and every update is guarded
  `!(active) : <unchanged>`. The clock freeze behaves exactly as designed.
- Negative θ indexing is correct: with `θ ∈ [-7, 7]` the emitted `cos_t` values
  and the `(th_stage_1 + 7)` offset both check out.
- `overflow` fires on a too-narrow `v` box and on a too-narrow `θ` box, and
  stays false on an adequate box. Both directions tested; an untriggered guard
  is an untested guard.
- Statement order sets the integration scheme. Writing `v` before `x` makes the
  position update read `v_stage_1` — the updated value — i.e. semi-implicit
  Euler. Nothing else in the file reveals this, so it must be a commented
  decision.

### 6.3 The discrete model does track the robot

The single most important audit, and the one a reviewer would ask for first:
does running the integer dynamics for 30 steps produce anything like the real
trajectory? Implemented in Python, reproducing exactly the arithmetic the tree
encodes.

```
RK4 ground truth: (0.155, -0.056)   in goal

       h  cells moved/step   final x   final y  in goal     err
     0.2               1.0     3.000     3.000    False   4.175    <- frozen
    0.15               1.3     0.150     0.600    False   0.656    <- moves, misses goal
     0.1               2.0     0.300     0.300     True   0.384
    0.05               4.0     0.300     0.000     True   0.155
   0.025               8.0     0.225     0.050     True   0.127
  0.0125              16.0     0.238     0.000     True   0.099
 0.00625              32.0     0.200     0.000     True   0.071
```

The encoding converges monotonically. That establishes the arithmetic is right,
whatever else is true.

**The error decomposes cleanly.** Single-step Euler on real numbers lands at
(0.104, −0.109) — error 0.073 against RK4 — and the integer model bottoms out at
0.071. So Euler is not the problem; quantization dominates everywhere above
`h ≈ 0.01`, and the integrator is the floor below it. This retires "why not
RK4?" in one line.

### 6.4 The quantization floor

At `h = 0.5` the model is frozen solid: `x`, `y`, `v`, `θ` all constant for 30
steps, `overflow` false. One control period changes `v` by at most
`|u|max · dt = 0.2` in physical units, so any cell wider than that truncates
every step to zero.

```
cells moved per maximal step = |u|max * dt / h    must be >= 1
```

`h < 0.198` is therefore a hard floor, and the table above shows the goal is
only reached from `h ≤ 0.1`. This is a **second lower bound on resolution,
entirely independent of the Lipschitz argument** in
`2026_09_21_monolithic_lipschitz.md`, which put the informative range at
`h < 0.117`. Two unrelated arguments converging on the same region is
considerably stronger than either alone.

### 6.5 nuXmv

`-dynamic` is effectively mandatory. A 2,835-cell model ran past 900 s without
it and finished inside 180 s with it. Any timing number that does not state
which was used is meaningless.

Valid rows from the scaling sweep (three others were discarded — see §8):

```
    cells     lines     nuXmv  outcome
   14,175    28,475      40.4  false
   22,253    44,631      11.4  SEGFAULT
   28,611    57,351       0.4  SEGFAULT
```

nuXmv segfaults between 14,175 and 22,253 cells — the same exit-139 failure
recorded for ACAS Xu, but at a small fraction of ACAS's 456,775 entries,
because NAV's transition relation is far denser (four real state variables, all
updating through arithmetic).

### 6.6 A second, independent argument

Two separate claims are in play, and they should not be conflated.

**(A) Soundness.** Even when the table returns the right verdict it has proved
nothing, because an entry stores a point where the truth is a range — it says
nothing about states between lattice points. This is the thesis, argued in
`2026_09_21_monolithic_lipschitz.md`, and it holds at any compute budget. It is
the claim that cannot be engineered around, and therefore the reason a
compositional approach is the answer rather than a larger machine.

**(B) Feasibility.** Independently of (A), the table cannot be built and checked
at a resolution where the model tracks the robot. Establishing this needs only
the fidelity requirement — that a discrete model should behave like the system
it models, which nobody disputes — plus the cell count and the point where
nuXmv fails. Combining §6.3 and §6.5:

- The model only reaches the goal from `h ≤ 0.1`. Above that it misses (0.15) or
  is frozen (0.2).
- At `h = 0.1` the table needs **321,408 cells** on the *optimistic* measured
  envelope, and 1.9 billion on the box that can actually be justified.
- nuXmv segfaults at ~20,000 cells.

So even granting the unsound centre-method table entirely, monolithic NAV cannot
produce a meaningful verdict: the coarsest resolution that tracks the robot at
all needs a table ~15× larger than the one nuXmv dies on, on the most generous
accounting, and ~10⁵× on the honest one.

**How the two fit together.** (B) does not depend on (A): a reader who rejects
the entire soundness analysis must still accept it. But (B) is the weaker claim,
because it is a tooling limit — it invites "so use a better model checker," and
in principle a smarter encoding could push the wall further out. (A) is the
durable claim, and (B) forecloses the obvious escape route for this benchmark
specifically.

Neither argument excuses a token attempt. A claim about what the monolithic
approach cannot do is worth nothing unless the attempt behind it was a real one,
which is why §7 specifies a generator that gives the table its best available
shot — correct arithmetic, a certified state box, and the tightest verifier-free
control bound available.

## 7. Requirements this places on the generator

Each becomes a test when the generator is written.

1. Emit a per-resolution ONNX wrapper, with `wrapped(idx) == 1000·original(idx·h)`
   asserted.
2. Refuse resolutions where `|u|max·dt / h < 1`; the model would be frozen and
   its verdict meaningless. Report the ratio as a diagnostic.
3. Assert the initial state lies inside the declared box (§8).
4. Emit `overflow` coverage for **all four** state variables — the spike only
   covered `v` and `θ`.
5. Pass `-dynamic`; check nuXmv's exit code; treat "no verdict" as an error,
   never a data point.
6. Record the integration scheme as an explicit, commented statement ordering.

## 8. What is not established

1. **None of this is reproducible from the repository yet.** The spike was
   throwaway. The two BehaVerify defects (§6.1) can be confirmed by reading
   `src/dsl_to_nuxmv.py`; everything else becomes reproducible when the
   generator and its tests land, which is Part 2.
2. **Three of six scaling rows were discarded**, and the reason is worth
   recording because the failure mode is nasty. The spike harness hardcoded the
   initial position as `3.0/h` while declaring boxes too small to contain it.
   nuXmv rejects the assignment, *aborts the command script*, and then waits at
   its interactive prompt — so a pipeline that captures output and moves on
   records a fast, clean-looking run with no verdict. That is precisely how a
   bad number reaches a table. Hence requirements 3 and 5.
3. **nuXmv's scaling is dominated by BDD variable ordering**, not by table size.
   The surviving rows are consistent but sparse, and a published curve would
   need a controlled experiment (fixed ordering, build and check timed
   separately) rather than the sweep run here.
4. **The verdict at any feasible resolution is `false`**, and always for a
   quantization reason rather than a property of the controller. No monolithic
   NAV run to date says anything about the robot.
5. **Per-dimension cell sizes are proposed but untested.** Everything measured
   used a single `h` across all four dimensions.
6. **Nothing here touches compositional NAV**, which does not exist yet. The
   comparison this sets up is still hypothetical on one side.

## 9. Relationship to earlier reports

**`2026_09_21_monolithic_lipschitz.md`.** Unchanged and complementary. That
report argues the table cannot be made *sound* at any feasible resolution; this
one argues it cannot be made *meaningful* at any feasible resolution. The two
are independent — one is about what a table entry guarantees, the other about
whether the model moves — and they converge on the same resolution range
(`h ≈ 0.1` versus `h < 0.117`).

**2025 NEUS.** The conclusion names regression networks and reals as an open
problem: *"our attempts to use reals with BehaVerify have yielded very poor
performance results. As such, we are still exploring how to improve our support
for regression networks (and reals in general)."* The two defects in §6.1 are
what that sentence looks like from the inside — the regression path exists but
has not been exercised.

**`2026_07_15_liveness_parity.md`.** Its freeze diagnosis is why §4 distinguishes
a clock freeze from a state freeze rather than avoiding `tick_prerequisite`
altogether.

---

# Part 2 — the NSBT, and what it produced

**Date:** 2026-09-22 (appended)
**Scope:** `examples/NAV/core/nav_domain.py` (new),
`examples/NAV/core/nav_tree_generator.py` (new),
`examples/NAV/core/monolithic/nav_table_pipeline.py` (new),
`tests/NAV/test_nav_domain.py`, `tests/NAV/test_nav_tree_generator.py` (new).
103 tests pass.

**Headline:** the NSBT is built and the monolithic pipeline runs end to end.
**No configuration exists that is both fine enough to represent the dynamics and
small enough for nuXmv to check** — not even on the optimistic, unsound box. The
smallest legal configuration is 14,641 cells and it segfaults. A control
experiment isolates the cause: with the table removed and everything else held
fixed, the same models check in 2.4 seconds.

## 10. What was built

Three modules, matching the split in §7.

| module | owns |
|---|---|
| `core/nav_domain.py` | the lattice, the integer dynamics, quantization diagnostics. Also the *reference* implementation — the tree and this class must agree, and tests pin that rather than relying on inspection. |
| `core/nav_tree_generator.py` | the `.tree` text. One generator, two modes; only the control declaration differs, and a test asserts the `environment_update` and `specifications` blocks are byte-identical across modes. |
| `core/monolithic/nav_table_pipeline.py` | ONNX wrapping, translation, model checking, reporting. |

The requirements from §7 are all implemented and tested. Two are worth
restating because they refuse work rather than doing it:

```
h = 0.5  ->  ValueError: cell size too coarse on: v, theta. A maximal one-step
             change truncates to zero there, so the robot cannot move.
             cells per step = {'x': 2.4, 'y': 2.4, 'v': 0.4, 'theta': 0.4}
```

and the same for an initial state outside the declared box. Both encode spike
findings where the failure mode is silent rather than loud.

## 11. The legal region

Two constraints bound the cell sizes from opposite directions.

**From below** (§6.4), a maximal one-step change must move at least one cell:

```
h_v, h_theta  <=  |u|max * dt      = 0.200
h_x, h_y      <=  |v|max  * dt      = 0.346   (|v|max from the measured envelope)
```

**From above**, nuXmv must survive the resulting table.

On the **sampled** box — the measured envelope, which is *not* sound and is used
here only to give the monolithic approach the most generous possible footing —
the coarsest legal configuration is `h_x = h_y = 0.345`, `h_v = h_th = 0.2`,
giving **14,641 cells**. On the **analytic** box, the one that can actually be
justified, the same configuration is **42,601,729 cells**.

## 12. Monolithic results

All runs: `nn-nav-set.onnx`, sampled box, `-dynamic`, explicit Euler, horizon 30.

| h_x = h_y | h_v = h_th | cells | SMV lines | SMV MB | gen (s) | nuXmv (s) | verdict |
|---|---|---|---|---|---|---|---|
| 0.345 | 0.20 | 14,641 | 29,415 | 3.4 | 1.0 | 8.0 | **segfault** |
| 0.345 | 0.19 | 15,972 | 32,077 | 3.7 | 1.0 | 8.9 | **segfault** |
| 0.340 | 0.20 | 17,424 | 34,981 | 4.1 | 1.2 | 10.0 | **segfault** |
| 0.330 | 0.19 | 19,008 | 38,149 | 4.5 | 1.2 | 11.2 | **segfault** |
| 0.300 | 0.20 | 20,449 | 41,031 | 4.8 | 1.2 | 11.6 | **segfault** |
| 0.330 | 0.18 | 20,736 | 41,607 | 4.9 | 2.2 | 17.7 | **segfault** |
| 0.250 | 0.18 | 32,400 | 64,935 | 7.6 | 1.7 | 0.5 | **segfault** |
| 0.200 | 0.20 | 39,204 | 78,541 | 9.2 | 1.9 | 0.5 | **segfault** |
| 0.150 | 0.15 | 103,684 | 207,507 | 24.5 | 4.5 | 1.2 | **segfault** |

Every figure above is transcribed from the pipeline's own `run_report.json`.

The first row is the smallest configuration the generator will legally emit.
Every row fails, on a box that is already indefensible. The window between
"fine enough to move" and "small enough to check" is empty, not merely narrow.

**Translation is not the bottleneck.** BehaVerify enumerates 103,684 ONNX
inputs and writes a 24 MB SMV in 4.5 seconds. Extrapolating, even a 5-million
cell table would translate in minutes. The table can be *built*; it cannot be
*checked*.

**A caveat on the earlier spike number.** Part 1 recorded a 14,175-cell model
completing in 40.4 s. That was the spike's simpler tree — overflow on `v` and
`theta` only, no `x`/`y` raw DEFINEs — so it is not comparable to what the
production generator emits, which carries a denser transition relation. The
production pipeline has no surviving configuration at all, and the 14,175 figure
should not be read as one.

## 13. The control experiment

The obvious objection to §12 is that the *model* is too big, not the table. The
generator answers it directly, because it emits both pipelines from the same
code: identical physics, clock, box, and specification, with only the control
declaration changed from a NEURAL table to a free nondeterministic variable.

| cells | monolithic SMV lines | monolithic nuXmv | compositional SMV lines | compositional nuXmv |
|---|---|---|---|---|
| 14,641 | 29,415 | segfault | **127** | false, 2.3 s |
| 20,736 | 41,607 | segfault | **129** | false, 2.3 s |
| 32,400 | 64,935 | segfault | **129** | false, 2.4 s |
| 103,684 | 207,507 | segfault | **133** | false, 2.4 s |

The compositional SMV is **essentially constant** — 127 to 133 lines while the
grid grows sevenfold — because the only thing that scales with resolution is the
trig table, which is one-dimensional. The 4-D product never appears.

So the table is the cause, not the state space. The same lattice, the same
dynamics and the same property are checked in 2.4 seconds once the control is a
free variable instead of an enumerated function.

**The `false` verdict is expected and is not a result about the controller.**
With no contracts injected, the abstract robot may choose any control at every
step, so it can obviously reach the obstacle. This row is the zero-contract
baseline: the starting point compositional verification improves on, and the
quantity contracts have to buy back.

## 13a. Why `table` is the only applicable encoding

An obvious objection to §12 is that it indicts one encoding. BehaVerify's grammar
offers five (`metamodel/behaverify.tx`, `store_as` at lines 166-174): `table`,
`real`, `float`, `fixed_direct`, `fixed`. Only `table` enumerates the **joint**
input space. `fixed` writes the network's arithmetic into the SMV instead, and
converts each *input* by enumerating that axis alone -- at the smallest legal
configuration that is 11+11+11+11 = **44** case arms, against **14,641** table
entries. It therefore sidesteps precisely the 4-D product that §12 dies on, and
NAV is the regime where it should win: ~2,368 multiply-accumulates against an
input space of 14,641 (optimistic box) to 42,601,729 (sound box).

**The blocker is the activation function.** All four non-table encodings carry
the same guard, and in each block it precedes the regression branch:

| encoding | block starts | op guard | regression `raise` |
|---|---|---|---|
| `float` | 1141 | **1194** | 1380 |
| `fixed` | 1381 | **1433** | 1538 |
| `fixed_direct` | 1539 | **1585** | 1685 |
| `real` | 1686 | **1721** | 1830 |

```python
if network_node.op_type not in {'Relu', 'Gemm'}:
    raise BTreeException([], 'Network is not being converted to table but has operation other than Relu or Gemm: ' + network_node.op_type)
```

NAV's graph is `MatMul -> Add -> Relu -> MatMul -> Add -> Relu -> MatMul -> Add
-> Tanh`. The `MatMul + Add` pairs **do** fuse into `Gemm`: rewriting them with
the weights transposed to `[out, in]` and `transB=1` gives a network that is
numerically identical to the original (checked at two input points) and that
BehaVerify accepts -- it processes the Gemm and Relu layers and then fails
specifically on

```
Network is not being converted to table but has operation other than Relu or Gemm: Tanh
```

`Tanh` is transcendental and cannot be expressed in fixed-point integer
arithmetic. A piecewise-linear approximation is feasible in principle -- ~79
segments puts the error below 1e-3, which is under the model's own milli-unit
control quantum -- but it means verifying a different controller, which forfeits
the ARCH-COMP comparison the benchmark exists for, and it would additionally
need the approximation error carried through the model as nondeterminism to stay
sound.

So the op check rejects NAV's network before the regression branch is ever
reached, for `fixed`, `fixed_direct`, `float` and `real` alike. `table` accepts
it only because the table path never inspects `op_type` at all -- it calls
onnxruntime.

**`table` is therefore not a choice, it is the only option**, and §12 shows it
cannot be checked. Worth noting for the paper: the tanh layer is what bounds the
control to [-1, 1], which is what makes the analytic state box in §5 derivable
at all. The same layer that makes the system tractable to bound is the one that
excludes every encoding except the one that blows up.

### 13a.1 How this was checked, and what remains inference

Verified directly:

- the five `store_as` values, read from the grammar;
- the four guard/raise line numbers and their ordering, read from the source;
- that the table path contains no `op_type` test;
- the Gemm fusion's numerical equivalence, and that BehaVerify accepts it and
  then rejects `Tanh` (probe: NAV's network declared as a 2-class
  classification under `config { fixed 100 35 }`, so that the fixed path's
  arithmetic is exercised without needing a regression branch);
- `toint` on a signed word, against nuXmv 2.1.0 -- so the word-to-int direction
  the source comment warns about does work. `swconst` does require a constant,
  which is why inputs are enumerated per axis.

**Not verified, and stated as inference only:** that a regression branch for
`fixed` would otherwise work. No such branch was written. `toint` availability
makes it plausible, but the claim is untested and is moot for NAV, since the op
guard fires first. It would matter only for a ReLU-only regression network.

**Not attempted:** the piecewise-linear tanh above; the ~79-segment figure is
arithmetic (linear interpolation error `(b-a)^2/8 * max|tanh''|`, with
`max|tanh''| = 4/(3*sqrt(3)) ~ 0.770`), not a measurement.

## 14. What this establishes, and what it does not

**Established.**

1. The NSBT design of Part 1 is implementable and the monolithic pipeline runs
   end to end against the shipped network.
2. No legal configuration of the monolithic table survives nuXmv, on a box
   chosen to flatter it.
2a. `table` is the only encoding BehaVerify can apply to this network at all;
   the other four are ReLU-only and NAV's controller ends in tanh (13a).
3. The table, not the state space, is the cause — demonstrated by holding
   everything else fixed.
4. Translation cost is not the obstacle; model checking is.

**Not established.**

1. **Every result here uses the sampled box**, which is unsound. It was chosen
   deliberately: failing on the generous box is the stronger claim. The sound
   box makes the smallest legal configuration 42.6 million cells, which was not
   attempted.
2. **No monolithic verdict about the robot has been obtained at any
   resolution.** Every run either refuses to generate, or segfaults. Nothing
   here says the controller is safe or unsafe — CORA and CROWN-Reach already
   answer that, and this pipeline does not.
3. **The segfault has not been diagnosed.** It is reproducible and the exit code
   is consistent (rc = -11, matching the ACAS Xu experience), but whether it is
   a nuXmv limitation, a BDD blowup, or an artifact of the emitted encoding is
   unknown. "nuXmv cannot check a table of this size" is the observation; the
   mechanism is not.
4. **No compositional contracts exist.** §13's compositional rows are the
   zero-contract baseline only. Whether CROWN-derived contracts recover a `true`
   verdict at a useful region count is the open question, and the whole point of
   what follows.
5. **Per-axis cell sizes are used but their trade-offs are unexplored.** The
   choices in §12 were driven by the quantization floor, not by any search for a
   good aspect ratio.

## 15. What follows

The comparison is now set up with one side measured. The compositional side
needs contracts: regions over the same lattice, CROWN-discharged guarantees of
the form `state in region -> u in box`, injected as INVAR constraints on the
free control variable that §13 already exercises.

The relevant number will not be wall time — §13 already shows the symbolic side
costs seconds either way. It is **how few regions suffice**, which is the
`min-cut` question from `2026_09_01_min_cut_contracts.md` transposed from a
finite graph to a continuous domain.
