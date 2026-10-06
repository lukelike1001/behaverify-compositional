# `divert`: contracts over a trained network, and where the horizon wall is

2026-10-06. Overnight session. Gemini (Antigravity) reviewed the plan before it
was built and the conclusions after.

## a) High-level goal

Get contracts working on a neuro-symbolic behavior tree in **continuous** state
space - the thing monolithic BehaVerify cannot do at all, because
`check_grammar.py:643` requires neural network inputs to be `INT`.

The property is the **finite-escape hypothesis** of Ögren's behavior-tree
convergence theorem, as stated in Rietz et al. (arXiv:2602.06525v2, Thm 1):
`x_t in Omega_i` implies `x_{t+h} not in Omega_i`. Rietz et al. enforce the
theorem's *other* hypothesis (invariance) by masking actions at runtime, and
leave this one to reward shaping; their Table I reports 7% outright failure.
The property is bounded, so by Alpern and Schneider it is a safety property,
not liveness - but it is the missing precondition of a published liveness
theorem, which is a better thing to be than a liveness claim we cannot support.

## b) Why a reviewer should care

**The headline is a proved bound over a trained network.** `net_divert` is a
4-64-64-64-2 ReLU policy; its contracts come from `auto_LiRPA`'s
`compute_bounds(..., return_A=True)` and are the real linear relaxations, with
`lA != uA` and cross-axis coefficients. Nothing here is a placeholder.

**The negative half is quantified, and that is what makes it useful.** Unrolled
SMT - the method this project has used throughout - cannot verify this
benchmark, and two independent measurements say why. They do not meet by about
three orders of magnitude.

**The night also overturned four of our own results.** Three were soundness
defects that all pushed the same way: they let the model finish *earlier* than
the plant, so a proof under them would not have been a proof. The fourth made
every earlier refinement experiment vacuous.

## c) How this relates to compositional BehaVerify

Directly: the proof is decomposed **along the tree's own sequence node**, which
is the clearest instance so far of the tree structure doing verification work.
Brake and Return are proved separately, with Brake's handover box as the
contract interface between them. Brake is analytic and carries no network
contracts at all; every contract in the proof sits inside Return, over a much
smaller initial set than the trip region.

Gemini's correction, adopted: **do not claim latching modes are beyond a hybrid
automaton.** A latching tree node is an acyclic finite-state graph with linear
guards, which SpaceEx, CORA, JuliaReach and NNV have handled for a decade. The
differentiator is the assume-guarantee *decomposition* the tree syntax dictates,
not the modes themselves.

---

## The result

`python3 compose.py`, about 41 seconds end to end:

| | method | contracts | result |
|---|---|---|---|
| Lemma 1 (Brake) | Z3, exact LRA | **none** - Brake is analytic | `n1 = 10` ticks, 0.3s |
| B1 (handover box) | `z3.Optimize`, exact LP faces | - | x [0.193, 1.257], y [-0.157, 0.707], v ±0.0354 |
| Lemma 2 (Return) | zonotope propagation | CROWN over trained `net_divert` | `n2 = 23` ticks, 40.4s |
| **Composed** | | | **N = 33** |

Measured longest recovery over a 26,244-start sweep of the *same* initial set
the proof quantifies over: **31 ticks**. So the proved bound is two ticks loose.

Two independent cross-checks, both of which caught real defects first:

- **B1 against simulation.** The LP-optimal handover box agrees with an
  independent simulated corridor to three decimals on the `y` face
  ([-0.157, 0.707] both ways).
- **Containment.** Simulate the true closed loop - run the network, clip,
  step - and ask at every tick whether the state lies in the propagated set.
  This found **10 misses in 4,972 states** and so caught a soundness bug: the
  contract bounds the *network's* output, but the actuator clips it, so where
  CROWN's bound sits entirely above `A_MAX` the value the plant applies is
  outside the interval. After the fix: **0 misses in 4,972 states over 23
  ticks.**

## The negative half, quantified

Unrolled SMT - the method used throughout this project - does not verify this
benchmark, and the two obstructions miss each other by about 10^3.

**Contracts need small cells.** CROWN's bound gap over `net_divert` against
cell half-width, with `A_MAX = 2.0`:

| half-width | 0.012 | 0.030 | 0.060 | 0.150 | 0.300 | 0.600 |
|---|---|---|---|---|---|---|
| gap | 0.011 | 0.029 | 0.078 | 1.114 | 4.443 | 14.799 |

Useful contracts need half-width ≈ 0.06. The Return corridor is 0.6 wide in
four dimensions: order **10^4 cells**.

**The solver cannot take small cells.** Per-cell contracts make the solver
choose *which* cell applies at *every* tick, so the search grows like
`cells^horizon`:

| cells | horizon | verdict | time |
|---|---|---|---|
| 256 | 6 | `unknown` | 180s |
| 16 | 14 | `unknown` | 180s |
| 16 | 17 | TIMEOUT | 401s |
| 28 | 35 | TIMEOUT | 1201s |
| 1153 (CEGAR, 40 rounds) | 35 | still `sat` | ~45 min |

Shrinking the benchmark until the true horizon was 15 did not rescue it
either: TIMEOUT at both 16 and 256 cells.

## The wrapping effect, settled by experiment

Interval reachability then diverged: 1,280 → 33,024 → 204,055 regions, and
outward grid-snapping only slowed it to 14,144 → 94,533. Having misdiagnosed
twice already this session, this one was settled by measurement rather than
argument. Same linear map, same 24 ticks, from a width of 0.02:

- interval arithmetic: 0.02000 → **0.29786** (diverging)
- correlations kept: 0.02000 → **0.00028** (contracting)

The closed loop has eigenvalues 0.795 ± 0.148i - modulus 0.809, contracting -
but `||A||_inf = 1.23`. The set *rotates*, and no axis-aligned box can follow
it.

**This is also the real argument for affine contracts, and a better one than
tightness.** That matrix exists only because the contract carries a slope.
Concretise CROWN's output to an interval - which the earlier pipeline did - and
the slope is zero, `A` becomes `[[1, dt], [0, 1 - drag*dt]]`, spectral radius
1, and the closed loop does not contract at all. An interval contract cannot
express a converging loop. `lA` and `uA` matter because they **compose** with
an affine set representation.

## Four of our own results overturned

Three were soundness defects, and all pushed the same way - they let the model
finish *earlier* than the plant, so a proof under them would not have been one.

1. Brake was `a = -3v` asserted alongside `|a| <= A_MAX`. Jointly unsatisfiable
   whenever `|v| > A_MAX/3`, which silently deleted every fast initial state
   from the set the proof quantified over.
2. `hypot(v) <= 0.05` was encoded as the box `|vx|,|vy| <= 0.05`, which
   *contains* that circle, so the model came to rest at speeds the plant still
   calls moving.
3. The success region was the start box padded by 0.1; `Tree._home` is already
   a box.

The fourth was a completeness defect and explains a whole earlier experiment.
Cell guards were half-open on the upper face, so the **top face of every
dimension belonged to no cell**. The solver parked the state on `vy = 0.25`,
every contract fell away, and only `|a| <= A_MAX` applied - free control. The
earlier "no proof at 576/1024/1600/2304 cells" sweep was refining a **vacuous
model**, which is exactly why it failed monotonically while getting more
expensive.

Also retracted: the claim that interval contracts *cannot* work. With a
Schur-stable closed loop the ultimate bound shrinks with the cell, so
refinement does converge. The honest claim is a rate - zeroth-order error falls
as `O(M^(-1/4))` in four dimensions, so the cell count needed is out of reach.

## Benchmark changes, each with its measurement

- **Actuator is now a per-axis box, not a direction-preserving disc.** Forced
  by defect 1: per-axis clipping is exactly encodable in LRA as a nested `If`.
  Box input sets are also what CORA, NNV and JuliaReach accept.
- **`return_law` lost its obstacle term.** Measured, not assumed: with `avoid`
  disabled the worst clearance is identical at 0.0929 and recovery is one tick
  *shorter*, because the worst approach happens while Brake still has the
  stick. It cost a great deal - `side` flips on a cross product, making the law
  discontinuous, and cloning stalled at a worst error of 3.64. The obstacle
  remains the *cruise* branch's problem, where it is load-bearing.
- **`envelope.py` now sweeps the per-axis velocity box** that `verify.py`
  actually encodes rather than the speed circle it does not, and its sweep
  count is returned by the sweep instead of hard-coded, where it had gone stale
  at 6,760 for a 26,244-start sweep.

## Open, for the morning

1. **Run CORA on Lemma 2 directly from B1.** Gemini's sharpest point: a
   reviewer will say the zonotope engine is a toy reimplementation of CORA. If
   CORA discharges Lemma 2 given B1, that *strengthens* the paper - it shows
   the decomposition is the portable contribution and the engine is incidental.
   This is the single most valuable next experiment.
2. **The ranking-function route.** A one-step inductive check `V(s') <= λV(s)`
   per cell costs `O(cells)` and is *independent of horizon*. It would remove
   the horizon wall rather than route around it. Needs a polyhedral `V`,
   because the eigenvalues are complex and no axis-aligned norm contracts.
3. **Tightness.** N = 33 vs 31. Where do the two ticks go - the two-tick
   handover conservatism in the encoding, or `reduce`?
4. **Train a second network** for the cruise branch, so the tree has two
   networks rather than one plus an analytic law.
