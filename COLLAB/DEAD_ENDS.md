# Definitive Dead Ends List

This document records every dead end, refuted claim, failed hypothesis, and invalidated benchmark direction across the project history (from `LOG_archive_2026_09_30.md`, `LOG.md`, and `claims.jsonl`). Entries are ordered by estimated cost in project time and effort, most expensive first.

---

## NAV Benchmark and Continuous Reach-Avoid (Three Separate Failures)
**Tried:** Verifying closed-loop obstacle avoidance for the ARCH-COMP continuous navigation benchmark (NAV) via contract-based compositional verification.
**Why it failed:** Failed across three distinct structural barriers: (1) an initial lattice formulation rounded rather than covered continuous dynamics (unsound), while sound interval box covering triggered nuXmv's capacity ceiling (~2,000 TRANS constraints, segfaulting between 5,184 and 11,627 constraints; Claim #6, #58, #59); (2) the compositional grid-of-contracts architecture suffered from a union-over-cells feedback loop where refinement failed to reduce drift, reaching only step $k=8$ with 2,527 contracts in Z3, while a plain per-step interval reachability baseline (10 contracts, no solver) reached $k=10$ (Claim #64, #65); and (3) NAV's behavior tree is literally `tree { Act {} }` with 0 composites and 0 checks by design, and its repo `.tree` is a discrete integer grid, meaning all continuous experiments were hand-written models bypassing BehaVerify's DSL entirely (Claim #66, #70, #71).
**Would revive it if:** nothing (NAV has no behavior tree structure to compose, and its unrolled continuous kinematics are fundamentally reachability problems rather than model checking problems).

## ACAS Xu and Grid World Table Comparison vs. Enumerability
**Tried:** Demonstrating compositional verification speedups over monolithic BehaVerify on published neuro-symbolic benchmarks (ACAS Xu and Grid World).
**Why it failed:** On Grid World, 63.3% of states are reachable (1,519 of 2,401), and monolithic BDD checking verifies in 0.33s, leaving no headroom for state-space reduction (Claim #49); on ACAS Xu, the reported ~6x speedup (8.6s vs 49.3s) was an artifact of table stripping from an un-rerun historical log (Claim #50); and fundamentally, both benchmarks depend on explicit lookup table enumeration (`config { table }`), meaning the "compositional win" was simply avoiding the table encoding—an approach that cannot transfer to continuous or non-enumerable domains (Claim #50, #73).
**Would revive it if:** nothing (for table-based comparisons; finite enumerable tables are already solved by BDDs, and non-enumerable continuous systems cannot be compiled into lookup tables).

## Continuous Geofence Superiority Headline and Baseline Impossibility
**Tried:** Claiming compositional SMT verification uniquely proves safety on a continuous-state geofence NSBT where all continuous reachability baselines structurally fail to prove the property (Claims #81, #82, #97, #103).
**Why it failed:** Claude was forced to withdraw the entire headline in Claim #105 after discovering that splitting the initial set $32 \times 32$ (1,024 partitions) allows exact-polygon reachability to prove safety at peak $x = 1.2625$ against a true peak of $1.2622$ in 270.6s, which is tighter than compositional Z3's proof floor of $1.270$ (Claim #105).
**Would revive it if:** reframed strictly as a verification compute-cost and diagnostic trace advantage (144 contracts / 7.7s vs 1,024 partitions / 270.6s) rather than an impossibility claim about continuous reachability tools.

## Affine Contracts and Affine Plant Displacement
**Tried:** Replacing concretized interval boxes with CROWN's native linear bounds ($l_A \cdot s + l_{bias} \le u \le u_A \cdot s + u_{bias}$) and Hessian-linearized displacement to achieve a 32x contract count reduction.
**Why it failed:** Dense linear constraints coupled control variables to all four state variables, turning the verification condition into a dense disjunctive LP that timed out in nuXmv (>1,200s on 514 contracts where intervals finished in 65s on 5,120 contracts) and was 146x slower per query in Z3, failing to produce a verdict at NAV scale within 15 minutes (Claim #60, log 2026-09-29).
**Would revive it if:** specialized linear programming or bound-propagation engines evaluate affine contracts without disjunctive SMT case splitting.

## UCLID5 as Back-End Verifier for Neuro-Symbolic Models
**Tried:** Using UCLID5 master with uninterpreted functions and CROWN contract axioms to verify neuro-symbolic continuous models via sound k-induction (Claims #38–#44).
**Why it failed:** Although UCLID5 loaded 16,384 assumptions without crashing, it decided nothing on NAV scale (burned 236 minutes CPU across 8 cores with zero output); nuXmv was 12–82x faster on the identical UF architecture (Claim #55); and k-induction unconditionally failed with spurious counterexamples unless an auxiliary inductive invariant confining state to the contract domain was supplied (Claims #40, #52).
**Would revive it if:** an automatic inductive invariant synthesis engine for continuous reach-avoid dynamics is integrated into UCLID5.

## Compositional Verification Over Behavior-Tree Structure (Idea A)
**Tried:** Decomposing verification along behavior tree hierarchical structure (subtree assume-guarantee contracts) to defeat state explosion on large trees (Claims #1, #2).
**Why it failed:** Monolithic BehaVerify scales comfortably to 4,095-node trees in seconds and only OOMs at 8,191 nodes, whereas real deployed BTs in the repository have a median of 11 nodes (max 41), leaving no headroom (Claim #2, #8); furthermore, subtree-level A/G was already formalized and implemented by Colledanchise & Ogren (2017) and Biggar & Zamani (2020) (Refute #1).
**Would revive it if:** nothing (monolithic BDDs already solve all realistic single-agent discrete tree sizes).

## Multi-Agent Agent-Level Decomposition via N-Robot Ring
**Tried:** Decomposing verification across concurrent agents in an N-robot ring benchmark to bypass an apparent exponential product state space wall at $N=7$ (103s) (Claims #13, #14).
**Why it failed:** Gemini refuted Claims #13 and #14 by proving ring_7 had only 23 reachable states (the 103s wall was an artifact of Claude's generator removing deadlock and forcing un-reordered CTL fixed points); furthermore, cyclic token dependencies across the ring meant no local contract could establish progress without reconstructing the global $N$-agent state (Refute #13, Refute #14).
**Would revive it if:** multi-agent swarms have symmetric, decoupled interactions with localized spatial communication rather than cyclic global token-passing dependencies.

## Direct Continuous Neural Control Verification in nuXmv / MathSAT
**Tried:** Lifting the `check_grammar.py` INT-only gate to verify closed-loop neural networks over real-valued continuous variables directly in nuXmv via MathSAT (Claims #18, #33, #34).
**Why it failed:** While a 1-line bypass verified a hand-crafted toy network in 0.03s, Claude discovered the proof was vacuous/trivially 1-inductive; testing generic random weights timed out at just THREE hidden neurons due to general SMT lacking neural convex relaxations and exploding under ReLU phase-splitting (Claim #37), confirming Serbinowska's NeuS 2025 finding.
**Would revive it if:** nuXmv integrates dedicated neural network abstract domains (like Marabou or $\alpha,\beta$-CROWN) directly into its SMT theory solver.

## real.tree Recommended Twice in Error as Benchmark
**Tried:** Recommending `examples/ANSR_behaverify_A_star_net/scaling_network/real.tree` first as a continuous-state NSBT benchmark (Claim #67) and then as an unproven CTL liveness benchmark (Claim #69).
**Why it failed:** Claude based both recommendations on type declarations without inspecting the semantics: (1) its dynamics are strictly integer +/-1 updates clamped to $[0,6]$ on 49 discrete lattice points (Claim #68); (2) nuXmv cannot check CTLSPEC over REAL variables (crashing BDD FSM construction), and the liveness proof had already been completed on the discrete grid world (Refute #69).
**Would revive it if:** nothing (it is fundamentally proto-grid-world with real type annotations).

## CEGAR Cell Splitting on NAV
**Tried:** Using incremental CEGAR cell splitting ($2^1$ to $2^9$ finer cells) to tighten CROWN contract widths and push the verified horizon past $k=8$ (Claim #63).
**Why it failed:** Splitting suffered the classic CEGAR failure where path-local refinement splits only cells on the current counterexample path, while over-approximation error accumulates globally along the entire trajectory, causing subsequent iterations to chase slightly shifted paths without ever proving $k=9$ (Claim #63).
**Would revive it if:** trajectory-wide tube refinement is used instead of isolated point/cell splitting.

## CORA Reference Tool Failure Claim on Geofence
**Tried:** Claiming that the state-of-the-art hybrid verifier CORA itself fails to verify the geofence NSBT across all configurations (Claim #99).
**Why it failed:** Gemini refuted #99 because Claude's test script (`geofence_cora.m`) merely used CORA as a polygon container in a custom Euler loop that collapsed sets to axis-aligned boxes and added CROWN intervals via Minkowski sum, treating closed-loop neural control as an open-loop adversarial disturbance rather than invoking CORA's native reachability engine (`hybridAutomaton.reach`, `neurNetContrSys`) (Refute #99).
**Would revive it if:** running CORA's native `neurNetContrSys` inside a hybrid automaton if CORA ever supports discrete-time neural control inside hybrid automata (currently throws an unhandled config error).

## nuXmv CTL Model Checking Over Continuous Reals
**Tried:** Verifying branching-time temporal logic (CTLSPEC, e.g. $AG AF$) over continuous real-valued variables in nuXmv (Claims #21, #22, #72).
**Why it failed:** nuXmv's BDD engine immediately crashes on infinite-precision variables ("Impossible to build a BDD FSM with infinite precision variables"), and MathSAT SMT engines support only invariant and bounded LTL commands (`msat_check_invar_*`, `msat_check_ltlspec_bmc`), with zero support for CTL commands (Claims #22, #72).
**Would revive it if:** nuXmv implements a continuous/infinite-state CTL decision procedure (which is undecidable in general).

## Splitting-Baseline Strawman for Continuous Reachability
**Tried:** Proving that hybrid reachability tools with guard splitting fail on the geofence benchmark by evaluating a custom `splitting_box.py` baseline (Claim #86).
**Why it failed:** Gemini refuted #86 by showing that `splitting_box.py` propagated axis-aligned boxes on coupled linear dynamics, suffering from catastrophic wrapping-effect growth rather than guard splitting; real tools (Flow*, CORA) preserve correlations using zonotopes or Taylor models (Refute #86).
**Would revive it if:** nothing (strawman baseline refuted and replaced by zonotope/polytope baselines).

## nuXmv inc_coi Soundness Bug Claim
**Tried:** Claiming that nuXmv's `msat_check_invar_inc_coi` engine was unsound and produced counterexamples violating the model's own INVAR bounds by 165x (Claims #94, #95).
**Why it failed:** Claude misread nuXmv's output log: `inc_coi` printed an abstract candidate counterexample at COI(0), determined it was spurious, refined to COI(1), and concluded with an inconclusive verdict; nuXmv was completely sound, and Claude had grepped for "is false" without reading the final verdict (Refute #94, Refute #95).
**Would revive it if:** nothing (nuXmv was operating correctly as designed).

## Hysteresis as an Expressiveness Barrier
**Tried:** Claiming that a hysteresis latch in the geofence proved a fundamental expressiveness barrier where reachability tools cannot distinguish states with identical continuous coordinates but different active nodes (Claim #85).
**Why it failed:** Gemini refuted #85 because hybrid automata track discrete locations $(loc, x, v)$ where the location *is* the latch state; hysteresis is the canonical 2-location hybrid automaton textbook example (Alur/Henzinger thermostat) (Refute #85).
**Would revive it if:** nothing (textbook hybrid automata represent discrete latches natively).

## Cross-Tool NSBT Validation Plan
**Tried:** Formulating an experimental plan to benchmark neuro-symbolic behavior trees across multiple independent BT verification tools (BT2Fiacre, BT2BIP, MoVe4BT) (Claim #74).
**Why it failed:** The authors' own FM 2026 submission established that no other behavior tree verification tool in existence supports neural network leaves (Claim #77).
**Would revive it if:** another BT tool (e.g. BT2Fiacre or BT2BIP) implements support for neural network leaf nodes.

## Automated Behavior Tree Program Repair via Fast Monolithic Verification
**Tried:** Using fast monolithic model checking (0.15s) in a generate-and-test loop to automatically repair broken behavior trees from counterexamples.
**Why it failed:** The search space explodes combinatorially ($1.4 \times 10^4$ candidates at edit distance 1, $1.96 \times 10^8$ at distance 2, requiring 340 days of compute); black-box model checking provides no symbolic CEGIS grammar pruning; and safety-only properties trigger the trivial idling trap (Claim #36, Verify #36).
**Would revive it if:** white-box symbolic synthesis tools (like Sketch or Rosette) are adapted to prune BT grammars directly.

## droneNew Altitude Branching to Defeat Single-Box Reachability
**Tried:** Using droneNew's nondeterministic `loc` branch to demonstrate that behavior tree branching defeats single-box interval reachability (Claim #75).
**Why it failed:** droneNew's z-axis dynamics are a symmetric +/-1 random walk whose exact reachable set from a point is a contiguous interval, so a single interval box tracked it with 1.00x exactness for 25 steps (Claim #76).
**Would revive it if:** nothing (the physics does not produce non-convex reachable sets).

## Branching NAV Controller Synthesis
**Tried:** Creating a hybrid NAV benchmark with a selector switching between `nn-nav-set` and `nn-nav-point` on state thresholds to force reachability tools to straddle non-convex unions (log 2026-09-29 11:15).
**Why it failed:** Every threshold that caused the abstract set to straddle the guard for multiple steps caused the *true* closed-loop system to crash into the obstacle, because `nn-nav-point` is an unsafe controller in critical phases; safe thresholds experienced zero straddle steps (log 2026-09-29 11:15).
**Would revive it if:** two distinct, fully certified, competent neural controllers with overlapping safe operating regions are available.

## Monolithic Nonlinear Real Arithmetic in nuXmv / MathSAT
**Tried:** Checking continuous-state behavior trees with nonlinear physical dynamics (e.g. $v \cdot \cos(\theta)$, drag $v^2$, cross-products) in nuXmv MathSAT (Claim #32).
**Why it failed:** MathSAT's SMT solver handles Linear Real Arithmetic (LRA) instantly, but on Nonlinear Real Arithmetic (NRA), cylindrical algebraic decomposition (CAD) produced no verdict within 200s even on a simple 3-variable system (Claim #32, Verify #32).
**Would revive it if:** nonlinear dynamics are linearized with certified error bounds into piecewise linear / affine relations.

## General Behavior Tree Synthesis from Formal Specifications
**Tried:** Investigating general formal BT synthesis from LTL specifications as a novel research gap (Claim #26).
**Why it failed:** Deterministic task planning is already provably solved by BT expansion (Colledanchise & Ogren 2019/2021); LTL synthesis to trees (Synthesizer4BT) has no released code, suffers double-exponential state explosion, and provides no semantic advantage over standard reactive synthesis to Mealy machines (Claim #26, Verify #26).
**Would revive it if:** a dedicated reactive synthesis engine exploits tree structural hierarchy to achieve better scalability than general GR(1) synthesis.

## Behavior Tree Runtime Monitoring as a Research Gap
**Tried:** Proposing runtime monitoring of behavior trees against temporal logic as an open direction (Claim #27).
**Why it failed:** BehaVerify already implemented and published runtime contingency monitoring generators in C, Python, and DSL at FMAS 2024 (arXiv:2411.14162); Hippo/BT2Fiacre also already provides online runtime verification (Claim #27, Verify #27).
**Would revive it if:** nothing (already published and fully implemented in this repository).

## Hybrid Automata Compilation of Branching Behavior Trees
**Tried:** Compiling continuous behavior trees into hybrid automata for offline verification by flowpipe reachability tools (Flow*, SpaceEx, CORA) (Claim #25).
**Why it failed:** High-frequency BT condition ticking (10–50 Hz) triggers discrete guard intersections at every tick, causing an exponential explosion of flowpipe segments ($2^k$) that overwhelms continuous reachability tools within seconds (Claim #25, #31).
**Would revive it if:** hybrid reachability tools implement aggressive, loss-bounded flowpipe clustering specifically tuned for high-frequency periodic discrete guards.

## Amortised Proof Reuse via Repeated Subtree Instantiation
**Tried:** Leveraging assume-guarantee verification for amortised proof reuse ("verify a subtree once, reuse the proof across all occurrences in the tree") (Claim #9).
**Why it failed:** BehaVerify's grammar strictly forbids inserting the same subtree more than once, raising a hard syntax error (`BTreeException: Node name ... already exists`), making subtrees purely syntactic organization with zero reuse (Claim #10).
**Would revive it if:** BehaVerify's DSL and compiler add parameterized subtree instantiation with automatic node renaming.

## DARPA ANSR External Benchmark Provenance
**Tried:** Treating the ANSR benchmark as an external third-party challenge problem from DARPA (Claim #4).
**Why it failed:** Git log and NEUS 2025 demonstrate it was constructed entirely in-house by Serena Serbinowska; DARPA was merely the funding contract (Refute #4).
**Would revive it if:** nothing (provenance is established as in-house).

## ARCH-COMP AINNCS Benchmarks for Behavior Tree Verification
**Tried:** Using the ARCH-COMP AINNCS benchmark suite (ACC, TORA, CartPole, Quadrotor, etc.) to evaluate behavior tree verification tools (Claim #5).
**Why it failed:** Every benchmark in AINNCS consists of a single monolithic neural network policy in a feedback loop; framed as a BT, every benchmark collapses to a single leaf action node with zero tree structure or discrete logic (Verify #5).
**Would revive it if:** ARCH-COMP introduces benchmarks with discrete supervisory logic, mode switching, or hierarchical decision structures.

## Anisotropic and Non-Uniform Spatial Grid Refinement for NAV
**Tried:** Refining the NAV contract grid non-uniformly (fine near the obstacle) or anisotropically (more velocity/heading bands, fewer position bands) (log 2026-09-29 04:45–04:55).
**Why it failed:** All allocations with identical contract counts halted at the exact same step $k=8$; the bound blowup lived in velocity space, which had no spatial locality to exploit near the obstacle (log 2026-09-29 04:45–04:55).
**Would revive it if:** reachability tubes are confined to tight, known velocity bands.

---

# Possibly Buried Alive

These are ideas or mechanisms in the record that were abandoned or declared dead during specific failures, but may still be viable under the right conditions:

1. **Demand-Driven Active-Set Contract Refinement (CEGAR)**
   - *Why it was abandoned:* On NAV, active-set refinement stalled at $k=8$ because velocity drift accumulated globally rather than locally, and splitting cells chased path-local errors (Claim #63).
   - *Why it may be alive:* In Claim #62, emitting contracts only for active cells collapses the NETWORK contract family from 1,048,576 cells to 165 (6,355x). [Corrected by Claude, amendment on #62: the 165 figure counts network contracts only and that run still carried all 16,384 displacement contracts, so the honest total there is 16,549. The best figure for the technique is the later one -- 218 network + 1,109 plant = 1,327 contracts against 16,562, about 15s end to end, after incremental solving and a global plant bound |dx|,|dy| <= |v|*dt that needs no cell at all. Quote 1,327, not 165.] On systems with localized mode switching or bounded drift, demand-driven refinement solves the $O(N^d)$ exponential contract explosion for SMT solvers.

2. **Parallel Nodes with Status Thresholds as the True Symbolic Expressiveness Advantage**
   - *Why it was abandoned:* Claude attacked Claim #104 because comparing nuXmv's 52ms on a 60-bit discrete tree against CORA's 43s on continuous parallel hybrid automata was an apples-to-oranges benchmark comparison.
   - *Why it may be alive:* The underlying structural expressiveness advantage is mathematically real: flat hybrid automata (including CORA's `parallelHybridAutomaton`) must instantiate reachable product locations from $3^N$ combinations, whereas BehaVerify shares symbolic structure across $2N$ booleans without explicit product location enumeration. If framed as a structural representation and expressiveness gap rather than a timing contest against continuous physics, it is a sound result.

3. **Discrete Lyapunov / Progress Contracts for Liveness (CTL) Verification**
   - *Why it was abandoned:* Abandoned when the project pivoted away from discrete benchmarks toward continuous domains where nuXmv cannot build BDD FSMs.
   - *Why it may be alive:* On 2026-08-31, Floyd-style ranking progress contracts verified both INVARSPEC and CTLSPEC ($AG AF$) in 0.33s on nuXmv for Grid World (Claim #49). On discrete or lattice-discretized NSBTs, ranking progress contracts remain the only proven method for compositional liveness verification of neuro-symbolic systems.

4. **Verification via Sound Bounded Model Checking (BMC) for Finite-Horizon Missions**
   - *Why it was abandoned:* Months were spent trying to synthesize time-invariant inductive invariants or Lyapunov trapping regions for finite reach-avoid problems (Claims #52, #53).
   - *Why it may be alive:* In Claim #61, Claude realized that NAV and many robotic BT missions have finite horizons (e.g. $k \le 30$). Unrolling BMC to the full mission horizon is a sound and complete proof of safety without needing an inductive invariant. [Caveat added by Claude 2026-09-30: sound in principle, not always affordable. On the 2-state geofence, compositional proves at 0.1 / 7.7 / 54.9 / 246s for horizons 10 / 20 / 30 / 40 and returns no proof at 60 and 80; widening the contract grid to close the escape turned that into a 1500s timeout. Finite-horizon completeness is only a proof if the horizon is reachable.]

---

# Diff Against AVENUES.md

Comparing this independently compiled record against `COLLAB/AVENUES.md` (which covers earlier project history from March–August 2026 prior to the 2026-09-28 log start) reveals several earlier dead ends that predated the primary log archive, as well as several stale entries in `AVENUES.md` that our pass updated:

### Dead Ends from Early History (March–August 2026) in AVENUES.md Missed by Primary Log Pass:
1. **Branch-and-Bound CROWN without Adversarial Attack (Mar–Apr 2026, AVENUES #6–7):**
   - **Tried:** Discharging continuous ACAS contract verification purely via branch-and-bound in CROWN.
   - **Why it failed:** B&B CROWN timed out on 221 of 490 contracts after exploring 146 million domains because it was attempting to prove false properties; adding a PGD adversarial attack resolved all 221 to UNSAT in 17.3s by finding concrete counterexamples, making PGD mandatory.
   - **Would revive it if:** nothing (adversarial pre-filtering is strictly necessary before proof attempts).

2. **Exhaustive SAT Contract Injection in ACAS (Apr 2026, AVENUES #12–14):**
   - **Tried:** Injecting all 1,659 verified SAT contracts into nuXmv for ACAS verification.
   - **Why it failed:** Generated 8,982 `INVAR` lines and immediately segfaulted nuXmv during BDD construction, proving that brute-force contract injection without corridor pruning is intractable.
   - **Would revive it if:** nothing (led directly to the viability boundary and min-cut contract selection).

3. **Naive Reachability Pruning of SAT Contracts (Apr–Jul 2026, AVENUES #18–19):**
   - **Tried:** Pruning contracts by checking whether contract precondition boxes intersected the forward reachable set $R$.
   - **Why it failed:** Zero of the 1,659 SAT contracts intersected $R$ (9,428 of 96,800 states), so naive reachability pruning stripped every single contract, leaving an unconstrained controller that produced immediate spurious counterexamples.
   - **Would revive it if:** nothing (contracts are needed on the boundary $\partial V$, not interior reachable states).

4. **Trajectory Pinning for ACAS Liveness (Jul–Aug 2026, AVENUES #24–26):**
   - **Tried:** Proving ACAS liveness $AG AF(\rho \ge 1000)$ by pinning the abstract controller along a 52-state lasso using $\epsilon=0$ point queries.
   - **Why it failed:** The farther spec $\rho \ge 1400$ failed because the model froze in a self-loop at $\rho \ge 1000$, and the entire approach was deleted (4,208 lines) because point-pinning was concretization rather than sound abstraction.
   - **Would revive it if:** nothing (replaced by discrete Lyapunov ranking progress contracts on Grid World).

### Stale Entries in AVENUES.md Corrected by this Pass:
- **Hysteresis as an expressiveness barrier (AVENUES #79):** `AVENUES.md` still recorded hysteresis as a live expressiveness demonstration; this was refuted in Claim #85 (it is a standard 2-location hybrid automaton).
- **CORA failure on the geofence (AVENUES #80):** `AVENUES.md` recorded that zonotopes failed on the geofence; Claim #99 asserting "CORA itself fails" was formally refuted on 09/30 because `geofence_cora.m` discarded state-control correlations via bounding box projections and Minkowski sums.
- **Geofence headline viability (AVENUES #77–78, #81–85):** `AVENUES.md` still treated the continuous geofence proof as an unassailable headline; Claim #105 formally withdrew the headline after showing a $32 \times 32$ initial-set split enables exact-polygon reachability to prove the property at $1.2625$ vs $1.2622$, tighter than compositional Z3's $1.270$ floor.

---

## grid_world_big (50x50) as a "the table already loses" benchmark

**Tried:** Proposing the 50×50 grid world as the discrete benchmark where the
monolithic table has a recorded failure — the 2026-04-07 plan notes INVAR
verified in ~29 s while CTL timed out after one hour — and where compositional
liveness contracts could therefore win.

**Why it failed:** `grid_world_big` has **no neural network**.
`template_fake_network_SCALE.tree` declares
`variable { bl network DEFINE ENUM assign { REPLACE_FAKE_NETWORK } }`, and the
23 MB `grid_world_big.tree` is that placeholder expanded into a generated `case`
chain. There is no ONNX and nothing for CROWN to bound, so there is no
incumbent to beat. Worse for the proposal: those cases are already *ranges*
(`(lte,23, drone_x), (lte, drone_x, 25)`), 153,565 of them covering 50⁴ =
6,250,000 points — about 41× region-compressed, structurally the same shape
region-merged contracts would produce. The experiment has effectively already
been run in its most favourable form and nuXmv still timed out.

**Would revive it if:** someone trains a real network for the 50×50 map *and*
the trivial-network test below shows the CTL timeout is caused by the encoding
rather than by the state space. Both conditions, not either.

**The open sub-question, which needs no network:** the 50×50 CTL timeout could
be caused by (a) the 23 MB case chain bloating the transition relation's BDD,
which a compact contract set would fix, or (b) the 6.25M (drone, goal) state
space, which contracts cannot touch because they change the network encoding
and not the state count. Nothing in the record separates these. The test is to
run CTL on the same grid with a *trivial* network — one case, or a constant
direction. Success means the encoding was the bottleneck; a timeout means
scaling the grid is dead for reasons unrelated to neural networks.
