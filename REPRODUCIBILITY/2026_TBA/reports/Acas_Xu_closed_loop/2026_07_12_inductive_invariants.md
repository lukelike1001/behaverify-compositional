# ACAS Xu: Inductive-Invariant Contracts

**Date:** 2026-07-12

**Scope:** Answer the two stress-test questions (how nondeterministic is $Allowed(s)$;
how small is the boundary relative to the reachable set), verify every resulting claim
by direct computation, and reduce the remaining verification work to an explicit,
minimal CROWN worklist.

**Provenance note.** All computations here were performed by Claude in a sandbox using
`generate_acas_contracts.py` imported directly (the module's own `simulate_step`,
`compute_distance`, `compute_nn_inputs`), with `acas_model_params.yaml` reconstructed
from documented values. Four independent cross-checks against previously committed
results all matched exactly: $|R| = 9{,}428$, the per-tick bound
$|\Delta x_{\text{mag}}|, |\Delta y_{\text{mag}}| \leq 3$, the number of dangerous pairs
(2,830), and the identity of the single unsafe pair in $R$. **Action item: diff the
reconstructed yaml (committed alongside this report) against the repo's committed yaml
before trusting anything downstream.**

---

## 1. Concepts, briefly

**Inductive invariant.** A set of states $I$ proves the safety invariant if three
obligations hold:

$$\text{(i)}\ s_0 \in I \qquad \text{(ii)}\ \forall s \in I:\ \delta(s, \mathrm{NN}(s)) \in I \qquad \text{(iii)}\ I \subseteq \{\rho \geq 200\}$$

Read: the system starts inside $I$, can never leave $I$ in one tick, and everything in
$I$ is safe. By induction on ticks, the system is safe forever. Obligations (i) and
(iii) are set-membership checks. Obligation (ii) is the only place the networks appear,
and therefore the only place CROWN is needed.

**Allowed set.** $Allowed_I(s) = \{a \in \mathcal{A} : \delta(s,a) \in I\}$: the
advisories that keep the system inside $I$ from state $s$. Obligation (ii) at $s$ says
$\mathrm{NN}(s) \in Allowed_I(s)$. If $Allowed_I(s) = \mathcal{A}$ (all five
advisories), the obligation holds no matter what the network does, so **no contract is
needed there**. Contracts are needed only on the **boundary**
$\partial I = \{s \in I : Allowed_I(s) \neq \mathcal{A}\}$.

**Viability kernel $V$.** The *largest* possible choice of $I$ ignoring the networks:
the set of safe states from which *some* advisory sequence stays safe forever. Computed
as a greatest fixpoint: start with all safe states, repeatedly delete states where
every advisory leads outside the current set, stop when nothing is deleted. Deleted
states are "doomed": currently safe, but no policy can keep them safe. $V$ is
network-free, so it is computed once per physics configuration and shared across all
five networks (and any retrained variants).

**The lattice point (the key design insight).** Valid inductive invariants live between
two extremes: $\mathrm{Reach}_{\text{true}}$ (the exact set the trained networks visit)
at the bottom, and $V$-shaped sets at the top. The stress-test questions adjudicate
between them:

- With $I = \mathrm{Reach}_{\text{true}}$: successors under *non-chosen* advisories are
  typically not in $I$, so $Allowed_I(s)$ collapses toward the singleton
  $\{\mathrm{NN}(s)\}$, nearly every state is boundary, and the contracts degenerate
  into a pointwise determinism table. This is the failure mode the stress test
  hypothesized, and it is real, at the bottom of the lattice.
- With $I \approx V$: the measurements below show 92.9% of states are interior (no
  contract needed), and the reachable boundary is a **single state**.

Conclusion: **choose $I$ as large as possible, not as small as possible.** This
reverses the earlier "Route A: use $\mathrm{Reach}_{\text{true}}$" recommendation.

---

## 2. Measured results

All reproducible via `acas_inductive_analysis.py` (every line below is a `[CHECK]`
printout of that script).

```
07/23 Note: `acas_inductive_analysis.py` has since been renamed to
`scripts/discrete/run_acas_inductive_invariant_check.py`
```

| Quantity | Value |
|---|---|
| Physical states | 19,360 (11 × 11 × 2 × 2 × 40) |
| Augmented states (× $a_{\text{prev}}$) | 96,800 |
| Safe physical states | 18,720 |
| Viability kernel $\lvert V \rvert$ | 17,455 (fixpoint in 8 rounds; 1,265 safe-but-doomed states deleted) |
| Boundary $\lvert \partial V \rvert$ | 1,234 (7.07% of $V$) |
| $\lvert Allowed_V \rvert$ histogram over $V$ | 5 advisories: 16,221 · 4: 345 · 3: 300 · 2: 311 · 1: 278 |
| Reachable pairs $\lvert R \rvert$ | 9,428 (matches committed result exactly) |
| Unsafe pairs in $R$ | 1: $((0,1,+,-,h{=}4), \text{strong\_right})$, $\rho = 100$ |
| $\partial V \cap R_{\text{phys}}$ | **1 state**: $(3,0,+,+,h{=}6)$, the contract-319 state |
| Candidate invariant $I_0 = \{(s,p) \in R : s \in V\}$ | 9,427 pairs; seed inside; all safe |
| Pairs of $I_0$ where obligation (ii) needs network info | **1**: $((3,0,+,+,h{=}6), \text{strong\_right})$ |

Answers to the two stress-test questions:

1. **$\lvert Allowed \rvert$:** for $I = V$, only 1.6% of the kernel (278 states) is
   single-advisory-forced; 92.9% allows everything. The "invariant encodes
   near-deterministic behavior" collapse does **not** occur at the top of the lattice.
   It does occur at the bottom ($\mathrm{Reach}_{\text{true}}$), which is why the
   bottom is the wrong default.
2. **Boundary vs. size:** $\lvert \partial V \rvert / \lvert V \rvert = 7\%$, and after
   intersecting with reachability the boundary is one state. The scaling story is not
   just "boundary is smaller"; on this benchmark it is minimal.

The boundary forms an annulus (radius 2 to 14 cells, densest around 8 to 9), which is
the shadow of the doomed region cast by the faster intruder (30 vs. 20). (A figure
visualizing this annulus was produced during the original analysis but had no
reproducing script and was not retained; regenerating it is a candidate future item.)

## 3. The corridor and the CEGAR endgame

Phase B already established that the single required contract **fails**: the
`strong_right` network genuinely outputs `strong_right` at $(3,0,+,+,h{=}6)$. So $I_0$
must shrink, and the question is how far the damage propagates. Answer: it is a
corridor with unique links, not a branching tree. Computed predecessor sets within $R$
(each is a singleton, verified directly):

$$\underbrace{(7,6,+,+,10)}_{a_{\text{prev}}=\text{clear (seed)}}
\ \xrightarrow{SR}\ \underbrace{(5,3,+,+,8)}_{a_{\text{prev}}=SR}
\ \xrightarrow{SR}\ \underbrace{(3,0,+,+,6)}_{a_{\text{prev}}=SR,\ \text{contract 319}}
\ \xrightarrow{SR}\ \underbrace{(0,1,+,-,4)}_{\rho=100,\ \text{crash}}$$

Each corridor physical state is reachable under exactly one $a_{\text{prev}}$. This is
the unique spurious counterexample of the abstract model, and it can be severed at
either remaining link:

**Q1 (primary).** Verify $\mathrm{NN}_{\text{strong\_right}} \neq \text{strong\_right}$
at $(5,3,+,+,h{=}8)$. If SAT, inject **one INVAR line** and the abstract model satisfies
the invariant. Verified by direct model check: blocking this edge yields reachable size
9,225 with zero unsafe states, i.e. `INVARSPEC TRUE`.

**Q2 (fallback).** If Q1 is UNSAT, verify
$\mathrm{NN}_{\text{clear}} \neq \text{strong\_right}$ at the seed $(7,6,+,+,h{=}10)$.
Also verified sufficient by direct model check (`INVARSPEC TRUE`, reachable size 9,220).

**Theorem (termination in at most two queries).** At least one of Q1, Q2 is satisfied
by the networks. *Proof.* Suppose both fail. Then $\mathrm{NN}_{\text{clear}}$ outputs
`strong_right` at the seed, $\mathrm{NN}_{\text{strong\_right}}$ outputs `strong_right`
at $(5,3,+,+,8)$, and (Phase B) $\mathrm{NN}_{\text{strong\_right}}$ outputs
`strong_right` at $(3,0,+,+,6)$. The real closed loop, started from its initial
condition, then deterministically follows the corridor and reaches $\rho = 100 < 200$
in three ticks, contradicting the monolithic `INVARSPEC = true`. $\blacksquare$

(The proof assumes the monolithic model and this analysis share the same physics; the
four cross-checks above are the evidence for that, and the yaml diff is the remaining
verification.)

Note the closure with March: the *hypothetical* one-step contract at the 319 state
would also have sufficed (verified: blocking that edge gives `INVARSPEC TRUE`). The
original contract vocabulary identified exactly the right state; the network simply
does not satisfy the contract there, so the constraint must move one step upstream.
That is the entire content of CEGAR on this benchmark.

## 4. What this replaces

The previous pipeline injected 8,982 per-state INVAR constraints (all of which, per the
07-11 analysis, constrain unreachable states) and segfaulted nuXmv during BDD
construction. The pipeline implied by this analysis injects **one constraint** (two in
the fallback branch). The scale problem does not get solved; it gets deleted.

## 5. Results

**Q1 is UNSAT; Q2 is the operative contract — confirmed three independent ways.**

- *ONNX forward pass* (real network weights, no CROWN): `strong_right` selects
  `strong_right` at $(5,3,+,+,h{=}8)$ — Q1 fails, margin $-0.0011$ against the forbidden
  class, a near-tie consistent with the `03-25` "no meaningful preference" finding.
  `clear` selects `clear` at the seed $(7,6,+,+,h{=}10)$ — Q2 holds, margin $+0.0146$.
- *Ground-truth trajectory* (`acas_lasso_trajectory.py`, real ONNX inference, independent
  of the reachability/viability-kernel computation in Sections 1–2): the true closed loop
  from the seed is a lasso — 39-tick stem, 13-tick cycle, 52 distinct augmented states,
  minimum distance **exactly 200**. The seed never recurs after tick 0, confirming Q2's
  antecedent fires exactly once, matching the termination theorem's assumptions exactly.
- *CROWN certificate*: `verify_single_state.py` verified Q2 **SAT** at the initial bound
  (no PGD, no branching needed). Q1 was not run through CROWN — the termination theorem
  guarantees at least one of Q1/Q2 holds, and Q2 already does, so it is the one used.

**Provenance and consistency checks all passed.** The reconstructed
`acas_model_params.yaml` used for Sections 1–4 matches the committed one on every value
(the only discrepancy found anywhere was a typo in the NeuS 2025 paper's text, 60,621
vs. the correct 60,261 — not a bug in this repo). Advisory index ordering in
`corridor_contracts.json` was confirmed against `verify_acas_contracts.py`'s argmax
semantics. One real bug was found and fixed along the way:
`verify_acas_contracts_config.yaml`'s `smv_variables` section referenced pre-rename
symbol names that don't exist in the current SMV (`x_var_stage_0`/`x_mult_stage_0` — real
names are `x_mag_stage_0`/`x_sign_stage_0`); fixed there and in
`run_acas_compositional_pipeline.py`, verified against the generated SMV directly.

**The patched abstract model verifies, matching the monolithic baseline:**

| | Monolithic | Compositional (corridor) |
|---|---|---|
| INVARSPEC | true | **true** |
| INVAR constraints | 0 (full NN table) | **1** |
| Wall time, full pipeline | 49.3 s | **~8.6 s** (discovery 0.5s + CROWN 7.3s + patch/nuXmv 0.8s) |
| Peak RSS | 9.19 GB | **~1.64 GB** (CROWN's process; nuXmv alone uses 52 MB) |

No segfault, same verdict as monolithic, from one CROWN-verified constraint instead of the
8,982-line table the old pipeline could never actually check. **~6x faster, ~5.6x less
memory, end to end** (an earlier draft of this comparison overstated this as 60x/180x by
comparing only the patch+nuXmv step against monolithic's full cost — corrected here). The
CROWN call dominates the total and is mostly process cold-start (torch/CUDA), not
verification math. The patched model itself is legitimately fast for a mechanistic
reason, not luck: `_remove_nn_defines` strips 8,020 lines of `case`-expression lookup
tables from the SMV, replaced by one free nondeterministic variable and one INVAR line —
a much smaller transition relation producing much smaller BDDs.

---

## 6. What's still open

- **Not a general speedup.** $\lvert \partial V \cap R \rvert = 1$ is a property of this
  seed, physics configuration, and set of trained weights — not of the method. An
  instance needing dozens of CROWN calls at ~7s cold-start each could erase the
  advantage and return to the monolithic-wins pattern seen on every other benchmark
  since January 2026. What generalizes is the viability-kernel construction collapsing
  the naive per-state contract count to an enumerable boundary; the ~6x number does not.
- **Continuous-mode extension.** Everything here is the discrete closed loop. The same
  $V$/boundary construction lifts to continuous mode with boxes over boundary regions —
  where this framework would actually outgrow the monolithic table approach, since the
  table approach has no continuous analogue. Not started.
- **Which distance-range constant the monolithic table generation used** (60,261 vs. the
  NeuS paper's 60,621 typo) was never directly confirmed — only that this repo's own
  value is internally consistent. Worth checking before treating parameter provenance as
  fully closed.
- **CROWN's cold-start cost is unmeasured as a warm process.** The 7.3s for Q2 is a
  fresh CLI invocation; a long-lived process would likely cut this substantially,
  relevant if a future instance needs many CROWN queries instead of one.
- **Paper framing.** The publishable claim is the method and the termination bound
  (contract count $= \lvert \partial V \cap R \rvert$ plus CEGAR refinements), with this
  benchmark as the instance where that count is 1 — not a general "compositional is
  fast" claim.
