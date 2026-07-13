# ACAS Xu Hiatus Return: Optimizing the ACAS Xu Contracts

**Date:** 2026-07-11

## (a) The invariant and the closed loop

The system state is $(x_{\text{mag}}, y_{\text{mag}}, x_{\text{sign}}, y_{\text{sign}}, h, a_{\text{prev}})$ with $x_{\text{mag}}, y_{\text{mag}} \in \{0,\dots,10\}$ (units of 100 raw distance), signs in $\{-1,+1\}$, heading index $h \in \{0,\dots,39\}$ (9° steps), and $a_{\text{prev}}$ ranging over the five advisories. Each tick: $a_{\text{prev}}$ selects one of the five networks, the network emits an advisory, the heading updates first ($\pm 1$ or $\pm 2$ heading steps, or none), and then position updates using the *new* ownship heading against a fixed intruder (heading 225°, speed 30) with ownship speed 20 and 6 seconds per tick. The invariant is

$$\rho = \operatorname{round}\!\left(\sqrt{x_{\text{mag}}^2 + y_{\text{mag}}^2}\right) \times 100 \;\geq\; 200 \quad \text{at every reachable state.}$$

One geometric consequence worth keeping in mind for everything below: per-step relative displacement is bounded by $6 \times (20 + 30) = 300$ raw units, i.e., 3 grid cells, and $\rho < 200$ requires $\operatorname{round}(\sqrt{x^2+y^2}) \leq 1$. So a state can only be dangerous if it sits within roughly 4 to 5 cells of the origin. Your dangerous set is a small annulus around the origin, and the contract bounding boxes should all be tight in the distance dimension. That is good news for CROWN and it also means the contract population is intrinsically small relative to the 19,360 physical states.

## (b) How contracts are generated

My understanding of the pipeline, stated back for correction:

1. **Enumerate dangerous pairs.** For every physical state with $\rho \geq 200$, simulate all five advisories one tick (heading first, then position, matching the sequential `environment_update` order). If advisory $F$ yields $\rho' < 200$, emit the dangerous pair $(s, F)$. This produces 2,830 pairs.
2. **Group.** Pairs are keyed by $(h, x_{\text{sign}}, y_{\text{sign}}, F)$. Fixing this key pins NN inputs 3, 4, 5 exactly (intersect angle is a function of $h$ alone; the speeds are constants), so only inputs 1 and 2 (normalized distance and relative angle) vary within a group. This yields 490 non-empty groups.
3. **Box.** For each group, take the componentwise min/max of the covered states' NN input vectors, pad by $\varepsilon = 10^{-4}$, and emit the box $[l, u] \subset \mathbb{R}^5$.
4. **Cross with networks.** Because a dangerous physical state could in principle co-occur with any $a_{\text{prev}}$, each group is emitted once per network: $490 \times 5 = 2{,}450$ contracts, each of the form

$$C_k: \quad \big(a_{\text{prev}} = k \;\wedge\; \text{inputs} \in [l,u]\big) \implies \text{NN}_k \neq F.$$

Discrete mode evaluates the covered points individually; continuous mode gives CROWN the whole box. Downstream, each SAT contract is expanded into one INVAR constraint per covered state, averaging 5.4, which produces the 8,982 lines and the segfault.

Two observations before optimizations. First, the $\times 5$ in step 4 is where the paper's unreachability theory lives: a contract pairing group $G$ with network $k$ is only meaningful if some state in $G$ is actually reachable *with* $a_{\text{prev}} = k$, and nothing in the generator checks that. Second, the box in step 3 covers only the dangerous states' input images, but the continuous box between those images contains input points corresponding to no integer state at all (the normalization involves `round`, so realizable inputs form a lattice). Continuous-mode UNSATs at off-lattice points are spurious with respect to this closed loop, exactly the phenomenon you saw in grid world. That doesn't affect Table IV (discrete mode), but it matters for interpreting the continuous ACAS results.

## (c) Optimizations, ranked by value-to-risk ratio

**Optimization 0 (do this first): classify the 791 UNSATs.** This costs an afternoon and tells you which of the following actually matters. In discrete mode every UNSAT is an exact state $s^*$ where $\text{NN}_k(s^*) = F$. For each one, check two things: is $(s^*, F)$ genuinely in your dangerous set (it should be, by construction), and is $(s^*, a_{\text{prev}}=k)$ reachable from any valid initial condition? If the overwhelming majority fail the reachability check, your theory from the paper is confirmed and Optimization 1 will eliminate them. If some are reachable, you have found real safety-relevant NN behavior and that is a finding, not a bug.

**Optimization 1: reachability pruning of the state × $a_{\text{prev}}$ product.** This is your BFS idea from the future-work section, and I want to stress how cheap it is here: the augmented state space has $11 \times 11 \times 2 \times 2 \times 40 \times 5 = 96{,}800$ nodes. Build the transition relation *without ever querying a network* by letting the advisory be nondeterministic: $(s, a) \to (s', a')$ iff $s' = \delta(s, a')$ for any $a' \in \mathcal{A}$. BFS from the initial conditions gives an over-approximate reachable set $R$ in seconds. Then drop every contract whose covered states, paired with its $a_{\text{prev}}$, miss $R$ entirely, and shrink the remaining contracts' state sets to their intersection with $R$.

The soundness argument is short enough to write out. Let $\hat{M}$ be your abstract compositional model. At every state, $\hat{M}$'s network chooses some advisory from a subset of $\mathcal{A}$ (whatever the injected constraints permit), while the relation defining $R$ permits all of $\mathcal{A}$. Therefore $\text{Reach}(\hat{M}) \subseteq R$, and this containment survives *removing* constraints, since removal only ever enlarges the permitted advisory set toward $\mathcal{A}$, never past it. A pruned contract $C_k$ satisfies $A_k \cap R = \emptyset$, so no reachable state of $\hat{M}$ ever satisfies its assumption, and the constraint was vacuous on every behavior nuXmv can explore. INVARSPEC quantifies over reachable states only, so the verdict is unchanged. You are not weakening the guarantee; you are deleting dead weight.

Expected payoff: fewer CROWN calls (my guess is a large fraction of the 2,450, given the 32% UNSAT rate and near-uniform output scores you observed), fewer INVAR constraints, and the UNSAT noise disappears from your correctness story simultaneously. One bonus check falls out for free: scan for reachable states where *all five* advisories are dangerous. If any exist, no network can save you there and the invariant's truth depends entirely on unreachability, which is worth a sentence in any writeup.

**Optimization 2: rectangle-assumption INVAR encoding, keeping your CROWN results untouched.** This attacks the 8,982-line blowup directly and is deliberately designed so the 1,659 existing SAT verdicts remain valid, since re-running CROWN is your expensive step. The idea: for each SAT contract, instead of emitting one INVAR line per covered state, decompose the contract's covered cell set $D_k \subset \mathbb{Z}^2$ (in $(x_{\text{mag}}, y_{\text{mag}})$ space) into a small number of maximal rectangles $Q_1, \dots, Q_m$ with $\bigcup_j Q_j \cap \mathbb{Z}^2 = D_k$, and emit one implication per rectangle:

$$\text{INVAR} \;\big(h = h^* \wedge x_{\text{sign}} = \sigma_x \wedge y_{\text{sign}} = \sigma_y \wedge a_{\text{prev}} = k \wedge x_{\text{mag}} \in [l_x, u_x] \wedge y_{\text{mag}} \in [l_y, u_y]\big) \implies \text{adv} \neq F$$

Soundness is immediate because the rectangles cover exactly $D_k$ and nothing more: every state satisfying the antecedent is a state whose NN input image lies inside the box CROWN certified, so the consequent is exactly what CROWN proved. There is no over-approximation introduced at all; this is purely a re-encoding of the same set. With 5.4 states per contract on average and dangerous states clustering contiguously near the origin, I would expect 1 to 3 rectangles per contract, putting you somewhere around 2,000 to 3,000 constraints instead of 8,982, and interval predicates over small bounded integers build far more compact BDDs than scattered point disjunctions. Whether that alone clears the exit-139 buffer issue is empirical, but it is the right first move, and if it is insufficient you can additionally merge all rectangles sharing $(h, \sigma_x, \sigma_y, k)$ into one INVAR line as a disjunction to cut the declaration count further.

A more aggressive variant, for later: define the assumption as the bounding rectangle $Q$ of $D_k$ in state space *up front*, compute the input box as the image $\iota(Q)$, and hand that to CROWN. For fixed $(h, \sigma_x, \sigma_y)$ the input map is componentwise monotone: $\rho$ is nondecreasing in both magnitudes, and within a fixed quadrant the relative angle is monotone in each coordinate via $\arctan(y/x)$, with `round` preserving monotonicity since it is nondecreasing. So $\iota(Q)$ is computable exactly from corner evaluation, modulo your $x=0$/$y=0$ special cases which need explicit handling. The trade-off is real, though: $\iota(Q) \supseteq$ your current tight box, so CROWN's job gets harder and some SATs may degrade to TIMEOUT. Given your 45% TIMEOUT rate already, I would hold this in reserve and lead with the CROWN-preserving version.

**Optimization 3: merge forbidden advisories per group.** States dangerous for multiple advisories currently generate multiple contracts with overlapping boxes. CROWN can verify a guarantee of the form $\text{output} \notin \{F_1, F_2\}$ in a single call by conjoining output constraints. Grouping states by their full forbidden *set* rather than by individual advisory would cut calls where overlaps are common. I rank this third because the win is modest next to Optimization 1 and it complicates the JSON schema and the retry-merge logic your README warns about.

**A caution on the objective itself.** Contract count is a proxy, and sometimes the right move increases it. Your own two-pass timeout data shows contracts either verify in under a second or stall near the decision boundary, so splitting a stalling box into halves (more contracts, each easier) can reduce total wall time even as the count rises. The quantities that actually matter are total CROWN wall time, INVAR constraint count on the symbolic side, and whether nuXmv terminates. I would frame any paper evaluation around those three, with contract count reported as a descriptive statistic.

My suggested sequence: Optimization 0 this week since it is pure analysis of data you already have, then 1 and 2 in either order since they are independent, then reassess whether 3 or box splitting is worth it. If you want to go concrete on Optimization 2, I could use the SMV-patching portion of `run_acas_compositional_pipeline.py` and a snippet of the patched SMV (say, 20 lines around the injected INVAR block) to sketch the exact rectangle-decomposition encoding against your real syntax. Which would you like to start with?

## (d) Bonus audit of ACAS Xu compositional representation

I checked this rather than taking the report's framing on faith. Here's what I verified, and the verdict.

**1. Are the networks actually the same?** Yes — MD5 hashes match exactly between the two folders:

2025_NEUS	| 2026_TBA| 	match
| --- | --- | --- |
ACASXU_run2a_1_1_batch_2000.onnx |	aprev_clear.onnx |	✓ 44c40...
ACASXU_run2a_2_1_batch_2000.onnx |	aprev_weak_right.onnx |	✓ d46d4...
ACASXU_run2a_3_1_batch_2000.onnx |	aprev_weak_left.onnx |	✓ 8ceec...
ACASXU_run2a_4_1_batch_2000.onnx |	aprev_strong_right.onnx |	✓ de5f9...
ACASXU_run2a_5_1_batch_2000.onnx |	aprev_strong_left.onnx |	✓ d237e...
No swapped or substituted weights.

**2. Is the closed-loop model itself the same?** I diffed 2025_NEUS/.../acasxu_template_360.tree against 2026_TBA/.../acas_template_360.tree (96 lines differ, out of ~350). Every single diff line is a pure rename — x_var→x_mag, x_mult→x_sign, y_mult→y_sign, the ONNX source paths, and acasxu→acas as the subtree name. Physics constants (distance_mean=19791.091, distance_range=60261, speeds, seconds-per-update, degree multiplier), the update order (heading first, then position), and the specification block are byte-identical. There's no semantic drift — it's the same model.

**3. Is the "49.3s / 9.2 GB / INVARSPEC=true" figure real, or an unsourced claim?** I found the actual log: 2025_NEUS/examples/AcasXu_closed_loop/invar.txt. It's a genuine nuXmv 2.1.0 run, User time 49.289 seconds, Maximum resident size = 9631076K (≈9.19 GB), and system.distance_stage_1 >= 200 : Invar True. This matches every number cited in the 2026_TBA README and reports, down to the decimal. It's not a fabricated or misremembered figure.

Verdict
Your approach from months back holds up. The base comparison is faithful: same weights, same closed-loop dynamics, same invariant, and a traceable ground-truth artifact for the monolithic side. This isn't a case of comparing two different systems dressed up to look alike.

The thing that might look like a flaw — 32% of discrete contracts UNSAT while the monolithic run proves the invariant true — isn't evidence the contracts are wrong. It's the expected consequence of one specific, already-identified gap: enumerate_dangerous_pairs() iterates over every syntactically possible state with ρ≥200 (the full 19,360-state domain × 5 advisories), with no reachability filter, while nuXmv's INVARSPEC check is implicitly restricted to states reachable from the single fixed initial condition via its own fixed-point computation. Those are different state sets by construction, and CROWN faithfully reports UNSAT for states in the first set that were never going to be reached anyway. That's not a faithfulness violation between the two benchmarks — it's a known scope mismatch within the compositional contract generator, and it's exactly what Optimization 0/1 are designed to close.
