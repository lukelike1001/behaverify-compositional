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
62. Benchmarking on a discrete domain with **deep** BT structure, e.g. ANSR_ONNX's 10 composites, where monolithic BDDs should blow up first.
63. Barrier certificates or invariant clusters, to get infinite-horizon properties without a bounded k.
