# The geofence NSBT: a pilot benchmark

The smallest honest neuro-symbolic behavior tree with continuous state, built to
test one question: **does behavior-tree structure make compositional
verification worth its cost?**

On NAV the answer was no — plain per-step interval reachability beat the
compositional architecture (step 10 with 10 contracts and no solver, against
step 8 with ~1,300 contracts and Z3). NAV's tree is a single action node, so
nothing ever branches and one box tracks the system exactly.

**On this benchmark the ordering reverses.**

## The system

```
selector
  |-- sequence
  |     |-- OutsideFence?   check:  x >= 1.0
  |     +-- Brake           action: a := -1.0
  +-- NeuralCruise          action: a := net(x, v)

x' = x + v*dt      v' = v + a*dt      dt = 0.1
```

`net` is a 2→8→8→1 ReLU network with a `tanh` output, trained by imitation of a
PD law (final MSE 0.0035; the closed loop settles from x=1.5 to x=0.009).

**Safety property:** `x` never exceeds 1.5.

## Why this shape, and not something else

**It is the architecture a CDC paper proposed.** Sprague & Ögren, *"Adding
Neural Network Controllers to Behavior Trees without Destroying Performance
Guarantees"* (IEEE CDC 2022, pp. 3989–3996) shows how a behavior tree carrying
performance guarantees can be extended with machine-learned components without
losing them — a hand-written safety envelope wrapping a learned controller. That
is exactly this tree, so its shape is not chosen to suit our method.

Their guarantees are enforced **at runtime**: the network is an untrusted black
box, and a pre-certified classical controller takes over when a boundary is
breached. We verify the closed loop **offline**, over the network's actual
weights, before execution.

**And it is the minimal system a box cannot represent.** The two branches
accelerate in *opposite* directions, so a state set straddling the fence splits
into disjoint velocity clusters. `box_must_fail.py` measures it: from x ∈
[0.95, 1.05], after one step the reachable velocity is the two-point set
{−0.2, +0.2} while the bounding box is the interval [−0.2, +0.2] — admitting
v = 0, which the system can never have.

droneNew lacked this. Its ±1 Manhattan walk has a reachable set that *is* an
interval, so a box is exact there (measured 1.00× for 25 steps). **The branch
must flip the sign of a derivative, not merely change a setpoint.**

## The operating envelope is derived, not asserted

The first initial set made the safety property **false** (max x = 2.55 — it
admitted states already past the fence). Rather than guess a smaller one,
`find_envelope.py` measures the largest safe one:

| x₀ ≤ | \|v₀\| ≤ | max x | |
|---|---|---|---|
| 0.9 | 0.8 | 1.2622 | safe, margin 0.238 |
| 0.9 | **1.0** | **1.4510** | safe, margin **0.049** |
| 0.9 | 1.2 | 1.6809 | **violated** |

This matches the physics exactly: braking at 1.0 with 0.5 of room past the fence
gives a maximum survivable approach velocity of √(2·1.0·0.5) = 1.0. The
benchmark is physically coherent rather than numerically tuned, and the
0.049-margin case gives it a **difficulty dial** — true, but only just.

## Result

Same system, same contracts, same property. The only difference is whether the
branch is kept as a **disjunction** or collapsed into a **hull**.

| method | verdict | |
|---|---|---|
| per-step interval reachability | **violated** (max x 1.5298) | spurious — the true max is 1.2622 |
| compositional (Z3) | **safe** | proved in 15.9 s at 20 steps |

**The per-step failure is structural, not precision.** Refining the grid
12×12 → 24×24 → 48×48 (144 → 576 → 2,304 contracts, 16×) leaves its answer
*identical to four decimal places*: 1.5298 every time. Once the box spans the
fence it must admit both branches, and a finer grid gives proportionally more,
smaller cells covering the same straddle.

This mirrors the NAV feedback loop (claim #64) with the methods **reversed** —
there it was compositional that could not escape by refining; here it is
per-step.

### The baseline is not a strawman

The obvious objection: `per_step_box` takes the **hull** of both branches when
the box straddles. A competent hybrid-reachability tool would **split at the
guard** and propagate each piece under its own dynamics — what Flow*, CORA and
SpaceEx actually do at mode transitions. If splitting recovered the proof, the
headline would have to be restated.

`splitting_box.py` implements it. **It also fails**, and also does not converge:

| grid | contracts | verdict | max x | peak sets |
|---|---|---|---|---|
| 12×12 | 144 | violated | 1.5498 | 59 |
| 24×24 | 576 | violated | 1.5497 | 64 |
| 48×48 | 2,304 | violated | 1.5497 | 64 |

Splitting is *slightly worse* than the hull (1.5498 vs 1.5298), which looks
backwards until you see why: the hull's lower acceleration bound is
`min(crown_lo, −BRAKE) = −1`, which drags velocity **down**, while the split
inside-piece uses `crown_lo` alone, typically greater than −1 and so permitting
more speed. Neither over-approximation dominates; both are sound and both fail.

The fragmentation cost is visible too — 59–64 distinct sets by step 20, on a
system with **one** guard.

**The proof is not vacuous.** `compare.py`'s `sanity()` confirms the model has
executions and that state can actually change. An UNSAT safety result on a
frozen or infeasible model proves everything and means nothing — Gemini caught
exactly that failure on NAV's goal property.

## Reproduce

```bash
cd COLLAB/experiments/geofence
python3 box_must_fail.py          # why a box must lose here
python3 train_controller.py       # the learned controller
python3 nsbt.py                   # both specs on the true system
python3 find_envelope.py          # the derived operating envelope
python3 compare.py 20 0.8 12      # the head-to-head
```

## v2: hysteresis, and where the expressiveness argument actually bites

In the tree above, the active node is a function of `x` alone — `x >= FENCE`
means `Brake`. A reachability tool could therefore express *"Brake is active"*
as the state predicate `x >= FENCE`, so the expressiveness claim does not yet
bite.

Real safety envelopes do not work that way. Releasing a brake the instant the
boundary is cleared makes the controller **chatter**, so engineers add
hysteresis — braking continues until the state is comfortably back inside. That
turns the envelope into a latch:

```
selector
  |-- sequence
  |     |-- ShouldBrake?   x >= FENCE  OR  (braking AND x >= FENCE - 0.3)
  |     +-- Brake          a := -1.0 ;  braking := True
  +-- sequence
        |-- ReleaseLatch   braking := False
        +-- NeuralCruise   a := net(x, v)
```

Sweeping 1,485 starts × 30 steps over 10,917 distinct (x, v) cells finds states
where **the same continuous state carries different active nodes**:

| at (x=1.0, v=0.12) | active node | latch |
|---|---|---|
| from start (0.65, 0.80) | `Brake` | set |
| from start (0.73, 0.70) | `NeuralCruise` | clear |

A tool tracking only (x, v) cannot tell these apart — **not imprecisely, but in
principle**, because the distinguishing information is not in (x, v) at all.

This also gives the benchmark a property of the right shape: *"Brake never stays
active for more than N consecutive ticks"* (measured worst case: 12) is a
statement about the **tree's** state, not the drone's position — and is what a
safety engineer actually writes.

**The obvious objection, and why it is the point.** One could answer that the
latch is just another state variable, so track (x, v, braking) in 3-D. True —
and that means encoding the tree into the state space *by hand*. That
translation is exactly what nobody has automated, and what BehaVerify does from
the DSL for free.

### Seed robustness

Retraining the controller from four independent seeds and re-running the whole
comparison on each:

| seed | true max x | per-step | compositional |
|---|---|---|---|
| 0 | 1.2622 | violated | **safe** |
| 1 | 1.2603 | violated | **safe** |
| 2 | 1.2609 | violated | **safe** |
| 3 | 1.2639 | violated | **safe** |

The ordering is a property of the **system** — a branch that flips the sign of
the acceleration — not of one set of weights.

### What this prototype has survived

| attack | outcome |
|---|---|
| baseline is a strawman | guard-splitting also fails (1.5498), also does not converge |
| artifact of one network | four seeds, same verdict |
| the proof is vacuous | guard confirms executions exist and state can change |
| just needs a finer grid | 16× refinement changes neither reachability variant |
| the envelope was chosen to fit | derived after the first choice made the property *false*; matches braking physics |
| the horizon is too short | true worst case at step 11; 20- and 40-step proofs both succeed |

## What this is and is not

**Is:** a pilot. One tree, one branch, one network, two continuous variables. It
demonstrates that the thesis holds on *some* system, which after NAV was an open
question.

**Is not:** a benchmark anyone should adopt yet. The dynamics are a double
integrator, the tree has one branch, and the property is a single invariant. A
benchmark worth publishing needs a domain-native plant (quadrotor, Dubins car),
a tree with real mission structure, and properties over node status — which is
the category no continuous-state tool can express at all.
