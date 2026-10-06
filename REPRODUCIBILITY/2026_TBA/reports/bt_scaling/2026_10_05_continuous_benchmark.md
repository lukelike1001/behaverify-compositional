# A Continuous-Plant NSBT Benchmark, and Why One Has To Be Built

**Date:** 2026-10-05
**Scope:** `COLLAB/experiments/divert/`, `COLLAB/experiments/cora_expressiveness/`,
`COLLAB/experiments/encoding_scaling/`, plus a survey of every published behavior-tree
verification benchmark available to us.
**Status:** plan and evidence. No verification result yet — the benchmark exists and is
validated, the networks and contracts do not.

---

## 1. The high-level goal

**Show that compositional BehaVerify (α,β-CROWN contracts + a symbolic solver) verifies a
neuro-symbolic behavior tree over a continuous state space, where no other NSBT verifier
can.**

The contribution is an existence proof with real tools and a template others can follow.
The underlying theory may not be new — behavior-tree convergence results already exist
(§6) — but nothing has ever been made to work end to end on a continuous plant, and that
is the claim.

Deliberately **not** the goal: beating CORA. Six attempts to find something CORA cannot
express all collapsed (§5), and CORA is absent from all four BehaVerify papers anyway.

## 2. Why a reviewer should care

**Behavior trees are standard in robotics; neuro-symbolic behavior trees are not. There is
no way to verify them, which is a reason not to build them.** Removing that obstacle is the
point.

And the gap is documented rather than asserted. **No published BT verification benchmark
has a plant:**

| source | state |
|---|---|
| BT2BIP — `MarsRover`, `TrainControl` | booleans and one counter; `environment_update {}` empty |
| MoVe4BT | synthetic binary trees over an integer `vv`; empty environment update |
| BT2Fiacre — `drone3`, `droneNew` | integer lattice `[-30,30]`, ±1 steps, 3-level enum battery |
| BehaVerify — `DrunkenDrone`, `NetworkExample`, ACAS closed-loop | 20×20 grid, 10×10 grid, 9° heading bins |
| Colledanchise & Ögren, Sprague & Ögren | continuous vector fields appear only in pencil-and-paper proofs and MATLAB simulations, never as a model-checking artifact |

`use_reals` appears in **zero** `.tree` files across the repository. The BT verification
literature treats trees as discrete coordination logic over pre-abstracted predicates,
because the tools behind it — nuXmv, Tina, BIP, UPPAAL — could only ever do that. The
continuous BT control theory and the BT verification tooling have never met in an artifact.

That is also the answer to *"why not adapt an existing benchmark?"*. We tried: §4.

## 3. How this relates to compositional BehaVerify

The architecture is a three-layer stack, and only the third layer is ours:

| layer | job | tools |
|---|---|---|
| network | bound the NN's output over a region | α,β-CROWN, Marabou, **or CORA's `nn.verify`** — interchangeable |
| dynamics | propagate state over time | Z3 BMC, nuXmv, or a reachability engine |
| **structure** | ticks, node status, priority, memory, temporal logic over control flow | **BehaVerify only** |

Compositional BehaVerify does not compete with α,β-CROWN — it **consumes** it. The contract
is the interface between layers 1 and 3, and it is agnostic to which network verifier fills
layer 1. That is worth stating as a design property rather than hiding.

**The property that matters is bounded progress** — *"Recovery completes within N ticks"* —
the shape we have already proved tight to one tick (claim #93: refuted at N=12 and 16,
proved at 19 and 22, against a true maximum of 18).

**Two corrections, both from Gemini on 2026-10-05, that this report originally got wrong:**

- **Bounded progress is a SAFETY property, not liveness.** `G(triggered → F_{≤N} completed)`
  is negated by a finite prefix of length N+1, and every bounded-time property is a safety
  property (Alpern & Schneider). Calling it liveness in front of a formal-methods reviewer
  is a red flag. The genuine liveness result we hold is grid world's `AG AF goal`
  (`2026_08_31_liveness_ctl.md`), which is unbounded — and discrete.
- **Reachability CAN soundly establish "must arrive".** This report claimed it structurally
  cannot; that was the sixth expressiveness-shaped claim of the week and it died like the
  other five. At least four techniques do it: forward over-approximation with an absorbing
  goal (if the over-approximated non-terminal set is empty at step N, every trajectory has
  arrived); backward reachability and capture basins (Hamilton-Jacobi); viability kernels
  and control barrier functions — which this very report cites in §7; and under-approximate
  reachability via Kaucher interval arithmetic or Taylor models.

So what actually defeats CORA on `divert` is **representation and scalability** — wrapping
error and fragmentation when convex sets are split repeatedly across guards — **not a
structural inability to prove progress.** That is a weaker and more honest claim, and it is
a cost argument, which has died twice this week and must be measured rather than asserted.

## 4. What every alternative cannot do, measured

| | why it falls short | evidence |
|---|---|---|
| **monolithic BehaVerify** | `check_grammar.py:643` refuses non-INT network inputs. Quantising to fit memory (19.6 KB/case → ~1.6M cases → 30 per axis → 0.067-wide cells) destroys a 0.0929 clearance margin. | `timing-1.txt`, `timing-2.txt`; claims #114, #120 |
| — its other encodings | `fixed` is **921×** slower than `table`, `float` **times out**, on the published 7×7 at its real size with the pinned 2025_NEUS source | claim #120 |
| **CORA** | 0 mentions of behavior trees in 1,775 files; 6 of 10 encoding paths crash; the only working path is a hand-written outer loop | `COLLAB/CORA_NSBT_MATRIX.md` |
| **BT2Fiacre / BT2BIP / MoVe4BT** | no neural leaves at all | claim #77, from the group's own FM 2026 related work |

**Adapting a published BT benchmark was tried and does not work.** `TrainControl` and
`MarsRover` have real tree structure (10 and 8 composites) and `MarsRover` even has a
published node-status property — `storm → success(Hibernate)` — but both are purely
discrete with no dynamics and only safety specs. Making them fit would mean adding a plant,
a liveness spec and a neural leaf, at which point only the tree shape is still theirs and
the provenance defence evaporates.

## 5. What this plan deliberately does not claim

Six expressiveness and cost claims died between 2026-09-28 and 2026-10-05. They are
recorded so nobody rebuilds them:

- **hysteresis as a barrier** — it is the canonical 2-location hybrid automaton
- **parallel nodes as 3^N** — they execute in fixed sequential order; the policy is a fold
- **node status as unstatable in CORA** — it is a clock variable, `τ̇ = 1`
- **"every reachability baseline fails"** (geofence) — a 32×32 initial-set split proves it at
  1.2625 against a true 1.2622, *tighter than our own floor of 1.270*
- **"CORA cannot branch on an ambiguous argmax"** — `nn.verify` decides it by refining the
  input; 7 of 20 regions where CORA's own VCAS loop gives up
- **"the branching loop is exponential"** — ours, for not merging: merged by advisory it
  holds at **1 set over 8 ticks**

The pattern in all six: a mechanism was tested, found limited, and generalised to the whole
tool. The correction each time came from enumerating what exists rather than reasoning from
what was opened first.

## 6. Related theory we are not inventing

Sprague & Ögren, *An Extended Convergence Result for Behaviour Tree Controllers*
(arXiv:2308.08994) states BTs as `T_i : ℝⁿ → ℝᵐ × {R,S,F}` — controllers over continuous
state, partitioning `ℝⁿ` into running/success/failure **regions**. Theorem 1 gives
finite-time arrival at the goal under Assumptions 1 and 2, where Assumption 2 is *bounded
escape from a region* — the shape we already discharge.

Its §D covers data-driven controllers, citing Sprague & Ögren CDC 2022, and **Assumption 5
assumes** the learned region is eventually left; the guarantee comes from the fallback
architecture, with the network an untrusted black box. **Neither paper has benchmarks or
tool experiments.**

This is related work and possible framing, not a pivot. Four ways it could fail to apply are
recorded in claim #124, including that the theorem is continuous-*time* while our machinery
is discrete-time.

## 7. The benchmark: `divert`

Built and validated; see `COLLAB/experiments/divert/`.

```
state   x, y, vx, vy        (4-D planar double integrator with drag)
plant   x' = x + vx·dt      vx' = vx + (ax − cd·vx)·dt     dt = 0.1
tree    Root selector
          ├── Recovery sequence
          │     ├── HazardMonitor  parallel success_on_one with_partial_memory
          │     └── DivertProcedure sequence with_partial_memory  (Brake → Return)
          └── NeuralCruise
```

Provenance: the plant is the double integrator of the run-time-assurance tutorial (Hobbs,
Mote, Abate, Coogan & Feron, *IEEE Control Systems Magazine* 2023,
doi:10.1109/MCS.2023.3234380); the architecture — a BT with guarantees extended by learned
components — is Sprague & Ögren, IEEE CDC 2022, pp. 3989–3996. The geofence's derived
envelope matched that tutorial's viability kernel to four decimals.

**Envelope derived, not asserted**, over 6,760 recovery starts (13×13 positions × 5 speeds ×
8 headings): worst clearance **0.0929**, longest recovery **33 ticks**, **0** failures to
complete. Hence `δ_safe = 0.074` and `N = 36`, both with margin.

**Properties**, after Gemini's tautology audit cut two of four:

- **N1** — `Recovery` is never active more than N ticks *(bounded progress)*
- **N4** — `Brake` completes within M ticks of triggering
- *(cut: N2 followed from `with_partial_memory` semantics; N3 checked that an SMV boolean
  retains its value)*

## 8. Plan, in kill-condition-first order

**Step 0 — the horizon gate, before anything else.** N1 needs `K ≥ 37`. Z3's measured wall
is **40 steps proved at 1192.8s, timeout at 45 and 50** — on a *2-state* system with *one*
network. `divert` is 4-state with two. Test with the analytic controllers as placeholder
contracts; no training needed. *If Z3 cannot reach K=37 on this structure, no amount of
network work rescues it, and we learn that in an hour.*

**Step 1 — train the two networks** by imitation, export ONNX. *Kill: the trained loop
violates N1 or N4 → re-derive the envelope, as the geofence required.*

**Step 2 — contracts with CEGAR from the start.** A dense 4-D grid is 20,736 contracts at
12/axis and will not load. Demand-driven refinement gave **1,327 instead of 16,562** on NAV
(claim #62). *Kill: the active set does not stay in the hundreds.*

**Step 3 — Z3 encoding with the vacuity guard built in.** A poisoned `Init` returns "proved
safe" in **0.149s**, perturbation-tested. *Kill: the guard does not fire on a poisoned model.*

**Step 4 — prove N1 and N4**, and fill the comparison table of §4.

**If step 0 or step 4 fails**, the result is §6.1.1 of the dissertation measured — *"models
would likely need Bounded Model Checking, which weakens the guarantees"* — which is
publishable and is not the result we want. Say so plainly rather than reframing.

## 9. Venue, honestly

Gemini's read, which Luke should weigh before committing weeks:

- **FMAS — strong contender.** Case studies, neuro-symbolic architectures and new
  benchmarks are its scope. An end-to-end NSBT verification over continuous dynamics fits.
- **FM, ATVA — reject.** Both want theoretical or algorithmic novelty. A paper whose stated
  contribution is *"an existence proof with real tools, not new theory"* tells an ATVA PC it
  is in the wrong room.
- **ICRA / IROS / RA-L — reject in this form.** Robotics reviewers want physical experiments
  or high-fidelity simulation, not SMT unrolling on an isolated plant.

To reach a top formal-methods venue the contribution would have to be an **algorithmic**
one — for instance, how BT priority and memory semantics enable a specialised compositional
CEGAR loop that beats monolithic SMT or a general hybrid engine. That is a different and
larger claim than this plan makes, and we do not have it.

## 10. Rules this report follows

Every figure above carries its benchmark and source. Baselines come from pinned sources
(`2025_NEUS/src`, `2026_TBA/src`), never the repository's current `src/`, which is
post-overhaul. Numbers measured on a model built with `--no_checks` are excluded.
