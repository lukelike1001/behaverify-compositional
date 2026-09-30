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

## What this is and is not

**Is:** a pilot. One tree, one branch, one network, two continuous variables. It
demonstrates that the thesis holds on *some* system, which after NAV was an open
question.

**Is not:** a benchmark anyone should adopt yet. The dynamics are a double
integrator, the tree has one branch, and the property is a single invariant. A
benchmark worth publishing needs a domain-native plant (quadrotor, Dubins car),
a tree with real mission structure, and properties over node status — which is
the category no continuous-state tool can express at all.
