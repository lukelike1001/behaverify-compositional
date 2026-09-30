# Every avenue attempted for compositional BehaVerify

One sentence per point. Compiled by Claude from claims #49–72 and the September
work; March–August reconstructed by Gemini from the reports and cross-checked.

## The baseline and why we left it

1. Monolithic BehaVerify can encode a network in `table`, `fixed`, `fixed_direct`, `float`, or `real` mode.
2. Monolithic `table` was the approach used in the 2025 NeuS paper.
3. Monolithic `table` enumerates one row per input cell, so it cannot scale to continuous state.
4. Compositional BehaVerify replaces that table with a neural verifier's contracts, discharged by a symbolic model checker.
5. The aim was to keep monolithic's guarantees while dropping its enumeration cost.

## Making the contracts work at all (Mar–Apr)

6. Branch-and-bound CROWN timed out on 221 of 490 continuous ACAS contracts after 146M domains, because it was trying to prove false properties.
7. Adding PGD adversarial attack resolved all 221 to UNSAT in 17.3 s by finding concrete counterexamples, making PGD a mandatory pipeline stage.
8. PGD on 100%-accurate grid world networks over continuous goals gave a ~50/50 SAT/UNSAT split.
9. Those counterexamples were at *non-integer* coordinates, showing the networks were safe on their integer training points and unsafe off-lattice.
10. That was the first real finding: compositional verification exposes continuous out-of-distribution bugs a lookup table cannot see.
11. Pinning grid world to discrete point queries (ε=0) verified all 38 contracts and replicated the 2025 NeuS discrete baseline.
12. Discrete ACAS contracts across five networks gave 1,659 SAT and 791 UNSAT (~32% UNSAT).
13. The 791 UNSATs were proved to be unreachable states, by contradiction with the monolithic proof.
14. Injecting all 1,659 SAT contracts expanded to 8,982 `INVAR` lines and segfaulted nuXmv during BDD construction.

## Fighting the contract explosion (Apr–Jul)

15. Luke tried UCLID5 with BMC(50) on discrete grid world and abandoned it: 99–1,142 s versus nuXmv's 0.094 s, and no CTL support.
16. A pipeline-generalization analysis concluded contract generation and SMV patching must stay example-specific.
17. A BFS proposal suggested filtering unreachable contracts before calling CROWN or nuXmv.
18. Computing the reachable set R (9,428 of 96,800 states) showed 790 of 791 UNSATs lay outside R.
19. But **zero** of the 1,659 SAT contracts intersected R, so naive reachability pruning left nothing to inject and produced a spurious counterexample.
20. Reformulating safety as an inductive invariant over the viability kernel V showed contracts are only needed on the boundary ∂V.
21. ∂V met R at exactly **one** physical state, and a single CROWN-verified contract severed the 3-step crash corridor.
22. That one contract verified ACAS in 8.6 s against monolithic's 49.3 s — ~6× faster, ~11× less memory, and no segfault.
23. Applying the same kernel argument to grid world proved the "hover theorem" and derived its 38 contracts from theory rather than an ad-hoc obstacle walk.

## Liveness (Jul–Aug)

24. ACAS liveness was obtained by pinning the abstract controller along its 52-state lasso with ε=0 point queries, proving `AG AF(ρ≥1000)`.
25. The farther spec `AG AF(ρ≥1400)` failed because BehaVerify freezes the model at ρ≥1000, halting in a self-loop.
26. ACAS liveness was then **deleted** (4,208 lines) because ACAS has no canonical liveness property and trajectory pins were concretization, not abstraction.
27. The same day, grid world liveness **succeeded** by replacing single-action pins with set-valued progress contracts from a discrete Lyapunov ranking `Dec(s,g)`.
28. 930 progress contracts (3,596 CROWN sub-obligations, 14.5 min) let nuXmv prove `INVARSPEC` and `CTLSPEC` both true in 0.33 s.
29. Ablation proved safety and liveness contracts are mutually necessary — neither alone suffices.

## The architectural bottleneck (Sep 1)

30. ACAS corridor selection was reformulated as max-flow min-cut and independently picked the same single contract in 1.9 s, removing the circularity.
31. The same report diagnosed the fatal flaw: the pipeline still built the full 8,020-line monolithic table before regex-stripping it, so it saved nothing and blocked continuous domains outright.

## The continuous attempt — NAV (Sep)

32. NAV was chosen as the continuous benchmark, but its BehaVerify `.tree` is itself discretized to a ~103,684-state integer grid with fixed-point trig tables.
33. `check_grammar.py:643` forbids non-INT inputs to `NEURAL` nodes, so no `.tree` with a neural node can be continuous by construction.
34. `nav_tree_generator.py` emits one tree for both pipelines, differing only in whether the control is a `NEURAL` table or a free `VAR` for contracts.
35. `dsl_with_abstract_control_to_nuxmv.py` was built to free the control to `REAL` and inject CROWN contracts without building a table first.
36. Every September NAV experiment wrote raw SMV and Z3 straight from the ONNX and `dynamics.m`, bypassing the `.tree` pipeline entirely.
37. UCLID5 was retried at NAV scale and produced **no verdict** in 236 minutes of CPU on 16,384 contracts.
38. nuXmv was found to support uninterpreted functions and to beat UCLID5 12–82×, so the UCLID5 detour had the right architecture and the wrong tool.
39. nuXmv segfaults between 5,184 and 9,604 `INVAR` constraints regardless of model complexity.
40. Finer uniform grids, anisotropic reallocation (nv > nx), and non-uniform grids all gave the identical k=8 horizon.
41. CROWN was measured to be within 1.32× of the exact range over a cell, so no tighter bounding method can help.
42. Affine (linear) CROWN contracts are 2.5–8× tighter per contract, but nuXmv cannot decide them at any size, down to 514 contracts.
43. A direct Z3 encoding decides affine in 14.6 s and is ~7× faster than nuXmv on intervals, so the ceiling was one tool's, not the approach's.
44. NAV is finite-horizon, so BMC to depth 30 is a **complete** proof and no inductive invariant is needed.
45. Demand-driven contract refinement (CEGAR) proved k=8 with ~1,300 contracts on a grid 128× past nuXmv's capacity.
46. CEGAR cell *splitting* was added and escalated through 2⁹ refinement without converging.
47. The abstraction's drift is resolution-independent: 4× network refinement (64× the cells) buys 10%.
48. The cause is a feedback loop — once the abstract state spans more than one cell, the admitted control is the **union** of the spanned cells' contracts.
49. The **plant** abstraction (`v·cos θ`) is binding, not the neural contracts, which is the opposite of where all the effort went.
50. Plain per-step interval reachability reaches k=10 with 10 contracts and **no solver**, beating the compositional architecture on every axis.

## Why NAV was the wrong test

51. NAV's tree is one action node with zero composites.
52. That is **by design** — the controller is sensorless, so a branch on obstacle membership would model a robot that does not exist.
53. Every attempted branching NAV variant either never straddled the branch or made the true system crash.
54. A survey of all 358 `.tree` files found nothing with BT structure *and* a continuous reachable set *and* a neural network.
55. `real.tree` looked like the exception but is grid world with `REAL` type annotations and a 49-point reachable set.
56. nuXmv cannot check CTL on it at all — "Impossible to build a BDD FSM with infinite precision variables".
57. nuXmv has no CTL command for infinite-state models; its SMT family covers invariants and bounded LTL only.

## Not yet exhausted

58. Relational contracts bounding closed-loop displacement `(x' − x)` directly, which would dissolve the union-over-cells feedback loop.
59. Pairing CROWN bounds with a dedicated reachability engine (CORA, Flow*, JuliaReach) instead of an SMT solver.
60. Applying CEGAR splitting to the **plant** approximation rather than the network, or changing coordinates to linearise it.
61. Region-merging grid world's progress contracts into boxes to cut the 3,596 CROWN calls below monolithic cost.
62. ~~Benchmarking on ANSR_ONNX's 10 composites~~ — **dead**: ANSR_ONNX is in no reproducibility folder, so it is an unpublished artifact of a DARPA grant.
63. Barrier certificates or invariant clusters, to get infinite-horizon properties without a bounded k.


---

## Correction to #62, and what it implies

`ANSR_ONNX` appears in **no** `REPRODUCIBILITY/` folder — it lives only in
`examples/`, making it an unpublished internal artifact. DARPA ANSR is a funding
grant, not a venue. (ANSR-derived examples *did* reach NFM 2024, but as the
`ANSR_no_net_*` variants, i.e. with the networks removed.)

Ranking every **submitted** `.tree` containing a `NEURAL` node by BT structure (see the correction below — folders are submissions, not acceptances):

| composites | networks | benchmark |
|---|---|---|
| **6** | **5** | `2025_NEUS/acasxu_*` |
| 3 | 1 | `2024_FMAS_BTM/template_network` |
| 2 | 1 | `2026_FM/using1000`, `using9995` |
| 2 | 1 | `2025_NEUS/grid_world_big` |

Two consequences.

**First, grid world is published too** — `grid_world_big.tree`,
`template_fake_network_SCALE.tree` and `counter_1/2.tree` are all in
`REPRODUCIBILITY/2025_NEUS`, alongside the ACAS family.

**Second, and more important: the two benchmarks where compositional already
wins are the two most structured published neural benchmarks that exist here.**
ACAS Xu (~6× faster, ~11× less memory, one contract) and grid world (safety and
liveness, 0.33 s). There is nowhere more structured to move to.

So the remaining gap is not "find a better benchmark." It is that **no published
NSBT benchmark has continuous state** — the field does not have one. Which makes
proposing that benchmark a contribution in its own right, rather than a
prerequisite we are missing.


---

## Correction: `REPRODUCIBILITY/` folders are SUBMISSIONS, not acceptances

Luke checked the NFM 2024 proceedings: **BehaVerify was not accepted.** So every
use of "published" above should read "submitted." The venues CLAUDE.md confirms
as actually published are only four:

- **NeuS 2025** — Neuro-Symbolic Behavior Trees and Their Verification
- **FMAS 2024** ×2 — Formalizing Stateful BTs; BTs with Contingency Monitors
- **SEFM 2022** — BehaVerify

Everything else — `2023_SEFM`, `2024_ANSR`, `2024_ATVA`, `2024_NFM`,
`2024_SEFM`, `2025_CAV`, `2025_FMCAD`, `2026_ATVA`, `2026_FM` — is a submission
folder of unknown outcome.

**So the only confirmed-published neural benchmarks are the 2025_NEUS ones:**
the `acasxu` family (6 composites, 5 networks) and grid world (2 composites,
1 network). Both are benchmarks compositional already wins on.

## The successor search: `droneNew.tree`

| | composites | neural | spatial state |
|---|---|---|---|
| `ANSR_no_net_5` (2024_NFM, rejected) | 10 | 0 | — |
| **`droneNew`** (2025_FMCAD → 2026_ATVA → 2026_FM) | **19** | 0 | `INT` |

`droneNew` is the most structured tree in the repository and an actively
maintained lineage, carried across three successive submissions. Its domain is a
real drone mission — `droneStatus`, `camStatus`, `battery {good, low,
critical}`, `move {none, land, takeoff, fly}`, a waypoint index, and spatial
state `x_a, y_a ∈ [−30,30]`, `z_a ∈ [0,80]`.

It has **zero neural nodes** and **integer** spatial state.

So the artifact that does not exist is precisely: **`droneNew` + a learned
controller + `REAL` coordinates.** Three of the four ingredients are already
built and already submitted; what is missing is the network and the type change
— and `dsl_with_abstract_control_to_nuxmv.py` exists to handle exactly that
type change.

## What this does to the "build a benchmark" question

It stops being "author a benchmark from nothing," which is a large and risky
thing for a first-year, and becomes "add a learned controller to a drone mission
BT the group has already built, submitted three times, and can hand to BT2Fiacre
and BT2BIP in its existing network-free form."

---

# The night of 09/30: the first positive result on continuous state

## 64–70: the literature gap, checked adversarially

64. Gemini audited **all 11 works citing** Sprague & Ögren's continuous-time BT semantics (IEEE L-CSS 2022) and found **none** build verification on them.
65. The only formal-verification papers citing it are this group's own FMAS 2024 papers.
66. Nobody has verified a behavior tree in CORA, JuliaReach, Flow*, SpaceEx, KeYmaera X or NNV.
67. NNV and BehaVerify are both Taylor Johnson's lab and have been kept strictly partitioned.
68. The closest work is **Sprague & Ögren, CDC 2022** — a BT as a *runtime* fallback around an **unverified** network, with safety from switching to a certified classical controller online.
69. VCAS is the only ARCH-COMP AINNCS benchmark with discrete supervisory logic, but it is flat — no hierarchy, no fallback, no node activation.
70. Honest probability the gap is already filled: **under 5%**.

## 71–79: the geofence pilot

71. NAV failed because its tree never branches, so the obvious fix is a benchmark whose branch a box cannot represent.
72. A box only loses when the branch makes the reachable set **non-convex** — droneNew's ±1 walk gives an interval, so a box is exact there.
73. A **geofence** does it: the branches accelerate in opposite directions, so a straddling set splits into disjoint velocity clusters.
74. Measured: from x ∈ [0.95, 1.05], after one step the reachable v is the two-point set {−0.2, +0.2} while the box admits v = 0, which the system can never have.
75. The architecture is **not invented** — a safety envelope wrapping a learned controller is exactly Sprague & Ögren's CDC 2022 design.
76. The operating envelope was **derived**: my first initial set made the property *false*, so the largest safe one was measured, and it matches the braking physics (√(2·1·0.5) = 1.0).
77. **Result: per-step reachability reports violated (1.5298, spurious); compositional Z3 proves safe in 15.9 s** — with the same contracts and the same property.
78. It holds at the razor-thin envelope too (margin 0.049): per-step 1.6499, compositional safe.
79. Adding **hysteresis** makes node status stop being a function of (x, v): at (x=1.0, v=0.12) one execution is braking with the latch set and another is cruising with it clear.

## 80–85: attacks on our own result, all survived

80. ~~**Strawman?** Guard-splitting reachability — what Flow*/CORA actually do — also fails.~~ **Refuted by Gemini:** interval boxes lose x–v correlation on coupled linear dynamics regardless of guards, so that was a strawman. Rebuilt with **zonotopes** (exact under linear maps, no wrapping): still fails, at *exactly* 1.5298 — so the loss is the **branch**, not the representation. Not yet tested against constrained zonotopes or Taylor models.
81. **Resolution?** 16× refinement leaves *both* reachability variants unchanged to four decimals.
82. **One lucky network?** Four independent training seeds, same verdict every time.
83. **Vacuous proof?** The guard confirms the model has executions and that state can change.
84. **Horizon too short?** True worst case at step 11; 20- and 40-step proofs both succeed.
85. Fragmentation is visible in the splitting baseline — 59–64 distinct sets by step 20, on a system with **one** guard.

## Still not exhausted

86. A domain-native plant (quadrotor, Dubins car) rather than a double integrator.
87. A tree with real mission structure rather than one branch.
88. Properties over node status discharged *automatically* from the DSL, which is the thing no other tool can do at all.
89. Whether compositional still wins once the tree has many guards and the disjunction itself becomes expensive.


---

# Corrections from the 09/30 night

Three claims retracted, all mine, all caught by Gemini.

**94 & 95 — the "nuXmv inc_coi is unsound" finding was a misreading.**
`msat_check_invar_inc_coi` is an abstraction-refinement loop. At COI(0) it logs
an *abstract candidate* with the words `is false, a counter-example of size 10
exists`, then rejects it as spurious, refines to COI(1), and finishes with
`could not determine if the property is either true or false`. Its real verdict
is **inconclusive**. I grepped for `is false`, took the first match as the
verdict, and never read the tail — then "confirmed" the error by tracing the
discarded candidate in more detail. *A self-confirmation deserved more suspicion
than a self-refutation.*

**86 — the splitting baseline was a strawman.** Axis-aligned intervals lose the
x–v correlation on coupled linear dynamics independently of any guard. CORA uses
zonotopes, Flow* Taylor models, SpaceEx support functions, precisely to avoid
that.

**What replaced it.** A zonotope baseline — exact under linear maps, no wrapping
— still fails, at 1.5298 with a branch hull (identical to intervals) and 1.5861
with guard splitting. Removing the wrapping effect changes nothing, so the
failure is the **branch**. The surviving claim is *"compositional beats every
cheap reachability baseline we could build, including one with exact linear
dynamics"* — **not** *"compositional beats continuous reachability tools."*
