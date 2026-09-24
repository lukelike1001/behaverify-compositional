# NAV: Compositional Verification — What Worked, What Didn't, and Why I'd Stop

**Date:** 2026-09-24
**Baseline:** `0970562f9` — *Document why monolithic NAV likely won't work*
**Scope:** Exploratory. All code lived in a scratchpad and is **not committed** (see §7).

**Headline:** compositional verification of NAV produced a working *mechanism* and no
*sound result*. Contracts scale with reachable states rather than grid cells —
1,386 contracts on a 5,080,320-cell grid where the monolithic table segfaults at
17,424 — but that result rests on a plant model that rounds rather than covers,
so it describes a discrete system, not the benchmark. Every sound variant either
explodes (covering) or exceeds nuXmv's constraint capacity (exact reals).

**Recommendation: stop NAV.** For a one-node tree with hard continuous dynamics,
CORA verifies this benchmark soundly and we do not. The mechanism that worked
needs a benchmark whose *discrete* structure carries the cost.

---

## 1. What was built

Four pieces, each of which turned out to be load-bearing.

**CROWN as a bound query, not a verdict query.** Grid world and ACAS Xu ask
"can class *k* be selected?" — a SAT/UNSAT question needing branch-and-bound,
which cost those pipelines 30–47 minutes per network. NAV needs "what interval
does `u` occupy over this region?", which is bound propagation:

| query | cost |
|---|---|
| `compute_bounds(method="CROWN")`, all 14,641 cells, batched | **0.4 s** |
| `compute_bounds(method="CROWN-Optimized")`, same | **65 s** |
| grid world's 38 contracts via branch-and-bound (`2026_04_07`) | **30–47 min** |

Same tool, four orders of magnitude, purely from asking the right question.

**Region generalization.** On a spurious transition, grow a box around the cell
while its bound still excludes the offending control. Measured **400–1,764 cells
blocked per contract** instead of 1.

**alpha-CROWN, for a non-obvious reason.** Plain CROWN's bounds are 1.21–1.27× of
the true range; alpha-CROWN's are 1.08×. That sounds like a minor difference and
is not:

| cell | true moves | CROWN | alpha-CROWN |
|---|---|---|---|
| (9,9,−3,2) | 1 dv / 1 dth | 1 / **2** | 1 / 1 |
| (9,9,−5,3) | 2 / 1 | 2 / **2** | 2 / 1 |
| 4 others | 1 / 1 | 1 / 1 | 1 / 1 |

What matters is not bound *width* but whether the slack crosses a quantization
boundary and admits a successor the real robot cannot reach. Plain CROWN does on
a third of sampled cells; alpha-CROWN on none. CEGAR with plain CROWN stalls at
15 regions; with alpha-CROWN it converges.

**CEGAR.** Zero contracts, take the counterexample, block the first transition
whose control lies outside the region's certified bound, repeat.

## 2. The lattice result — real, reproducible, and not sound

Grid `h_x = h_y = 0.05`, `h_v = h_th = 0.1`, box = measured envelope + 0.4 padding.

```
grid                       5,080,320 cells
contracts                      1,386   (0.027% of the grid)
SMV                            1,548 lines
nuXmv obstacle-safety           TRUE   (193 s)
```

Verified against the failure modes that produced two earlier false positives:

| check | outcome |
|---|---|
| independent nuXmv re-check | **true** |
| bound soundness, 600 samples per contract | **0 / 1,386 failed** (831,600 samples) |
| robot actually moves (`robot_x = initial` always?) | **false** — it travels |
| obstacle reachable with **0** contracts | **yes** — property is non-trivial |
| true discrete run | 0 obstacle hits, reaches goal at (−0.35, 0.05) |

The construction is an inductive invariant, not coverage: the contracted set
contains the initial state, is closed under the abstract transition relation by
construction, and excludes the obstacle.

**Scaling is the interesting part.** Refining the grid costs the table everything
and costs contracts almost nothing:

| grid | cells | reachable states | contracts |
|---|---|---|---|
| h_x 0.2 | 63,504 | 130 | — |
| h_x 0.05, h_v 0.1 | 1,664,000 | 747 | — |
| h_x 0.05, pad 0.4 | 5,080,320 | 1,386 | **1,386** |

The compositional SMV is **145 lines at 1.66M cells**, because only the
one-dimensional trig table scales; the 4-D product never appears. The monolithic
table needs one `case` arm per cell and segfaults at 17,424.

**And it is not sound.** The plant rounds: the model takes the cell index,
computes one successor, and assigns it. The true continuous successor from a
point in that cell spans 3–4 cells (measured over all 14,641 cells). So this is a
statement about a discrete system, exactly as monolithic BehaVerify has always
been — NEUS says the same of its closed-loop ACAS Xu model, which "cannot be used
to argue for the correctness of ACAS Xu."

## 3. Two results retracted along the way

Both were vacuous in the same way, and both passed a guard that was too weak.

**"21 contracts prove obstacle safety."** At `h_x = 0.345` the true discrete
trajectory never leaves (3.1, 3.1): position requires `|v| >= 9` index units and
the trajectory only reaches −7. Safe because paralysed.

**The analytic quantization guard was too permissive, twice.** It uses declared
bounds, and the network never attains them:

| occasion | declared | attained | consequence |
|---|---|---|---|
| control bound | 1.0 (tanh's limit) | ~0.99 | `h_v = 0.2` passes, model frozen |
| speed bound | 1.728 (box envelope) | 1.39 | `h_x = 0.345` passes, position frozen |

The lesson generalizes: **an analytic floor computed from declared bounds is
necessary, never sufficient.** The check must be "does the real closed loop
move", and specifically *does position move* — the guard in
`NavDiscreteDynamics.trajectory_moves` checks whether any state component
changes, which speed and heading satisfy while position stays put.

## 4. Sound variants, and why each fails

### 4.1 Covering successors — dead

The textbook sound construction (feedback refinement relation): successor = every
cell any point in the source cell can reach.

```
h_x 0.05  h_v 0.10   5,080,320 cells | COVERING reach 135,561 | obstacle HIT step 7
h_x 0.05  h_v 0.05  18,895,680 cells | COVERING reach 214,565 | obstacle HIT step 8
```

The second satisfies the growth-bound ratio rule and still explodes. The reason
is structural: **quantization error compounds.** Every step re-rounds and injects
fresh uncertainty on top of what is already carried, so the reachable set grows
geometrically regardless of resolution.

Growth bounds (the SCOTS device) formalize the spread but do not reduce it. For
NAV the Jacobian is nilpotent (`L^2 = 0`), so `beta(r, tau) = (I + L*tau) r`
exactly — the tightest possible linear growth bound is what interval arithmetic
already gives. It yields a design rule, `h_x > 0.546 * h_v`, which the failing
configuration satisfied. Sub-stepping does not help either: with nilpotent `L`
the continuous growth is unchanged while each extra step adds a re-quantization.

### 4.2 Exact real state (LRA) — sound, runs, exceeds capacity

Make `x, y, v, th` real and updated exactly; over-approximate only `cos`/`sin`
per theta-band by constants, so the transition stays **linear real arithmetic**:

```
TRANS next(v)  = (run ? v + u1 * 0.2 : v);                      exact
TRANS next(x) >= (run ? x + dxl : x) & next(x) <= (run ? x + dxh : x);
DEFINE dxl := (v >= 0.0 ? v * cl : v * chh) * 0.2;              cl, chh constants per band
TRANS (x >= a & x <= b & ...) -> (u1 >= lo & u1 <= hi & ...);   contracts over real boxes
```

Three properties make this the right shape:

- **Sound.** No rounding at any step. The only over-approximation is the trig
  interval, and it does not compound — it is recomputed from the exact `th` each
  step rather than carried forward.
- **No wrapping effect.** MathSAT keeps the 31-step unrolling symbolic and never
  re-boxes the reachable set. This is what CORA needs zonotopes to mitigate.
- **BMC is complete, not bounded.** The horizon is bounded and the model freezes
  at `k = 30`, so depth-31 BMC is a proof. `msat_check_invar_bmc -a een-sorensson`
  returns `is true` / `is false`, not "no counterexample up to k".

Contracts seeded on the reachable tube are genuinely tight — **mean width 0.391
out of 4.0 vacuous**, against 3.0 for uniform refinement over the operating box.

The blocker is capacity:

| boxes | BMC | outcome |
|---|---|---|
| 1,343–1,978 | 173–301 s | `FALSE` |
| ~5,000 | > 800 s | timeout |
| 11,627 | 2 s | **segfault** |
| 34,924 | 4 s | **segfault** |

Tight contracts need small boxes; a tube that stays closed under a
nondeterministic plant needs a wide margin. Both drive constraint count up, and
nuXmv's ceiling here is ~2,000 `TRANS` constraints. The window where the model is
both tight enough to prove something and small enough to check appears empty —
the same squeeze that killed the monolithic table, one level up.

## 5. What generalizes, and what is NAV-specific

**Generalizes:**

1. Contract count tracks the **reachable set**, not the state space. Grid world
   had 1,519 reachable of 2,401 (63%) and compositional never beat the table
   there; NAV had 1,386 of 5,080,320 (0.027%) and the table cannot run at all.
   That ratio is a usable a-priori predictor of whether compositional can win.
2. For regression networks, **bound propagation is the right CROWN primitive**,
   and it is orders of magnitude cheaper than the branch-and-bound queries the
   existing pipelines use.
3. **Bound tightness matters discretely, not continuously** — what counts is
   whether slack changes the successor set.
4. Analytic feasibility guards computed from declared bounds are necessary and
   not sufficient; the empirical check is mandatory.

**NAV-specific:** everything about the verdict. NAV's tree is one node, so the
discrete structure that a model checker is good at is absent, and the continuous
dynamics that a reachability tool is good at are the entire problem.

## 6. Recommendation

Stop NAV. The honest position is:

- CORA verifies this benchmark **soundly, in continuous time, and faster**.
- Our sound formulations do not verify it at all.
- Our unsound formulation verifies a discrete model with 1,386 contracts where
  the table approach segfaults — which is a real scaling result about *encodings*
  and not a safety claim about the robot.

Reporting a timing comparison against CORA would lose on both axes at once and
invite attention to the soundness gap. The defensible axes are **what can be
expressed** (temporal logic over behaviour-tree execution) and **how cost scales**
(reachable states, not cells) — and NAV exercises neither, because it has no tree.

The next benchmark should have genuine discrete structure. **Closed-loop ACAS Xu**
is in the ARCH-COMP AINNCS suite (five networks plus the switching logic between
them, ten closed-loop properties), CORA numbers are published for it, and the
compositional infrastructure already exists in
`2026_TBA/examples/AcasXu_closed_loop/` — min-cut contract selection, reachability
pruning, the SMV patcher. The NEUS version is discretized; making it continuous
would give a head-to-head on a system where the discrete state space is the
expensive part.

## 7. Reproducibility — the weak point of this report

**None of the code in this report is committed.** It lived in a session
scratchpad, which was wiped once mid-investigation, taking the verified
`final_contracts.smv` with it. The contract shown in §2 was reconstructed; the
procedure is deterministic (two independent CEGAR runs produced byte-identical
21-contract sets), but the artifact itself is gone.

What *is* committed and reproducible: `examples/NAV/core/` (domain, tree
generator, boundary finder, Lipschitz bound) and `tests/NAV/` (105 tests).

Before any of this is built on, the CEGAR loop, the reachability computation, the
LRA generator and the alpha-CROWN contract generation need to move into the
repository with tests. Given that two headline results in this line of work were
retracted after verification, a result that cannot be re-run is not a result.
