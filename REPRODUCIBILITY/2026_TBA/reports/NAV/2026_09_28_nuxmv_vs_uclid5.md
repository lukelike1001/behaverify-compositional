# nuXmv vs UCLID5: contracts as axioms, and why it changes the NAV picture

**Date:** 2026-09-28
**Status:** **Preliminary.** Toy plant, synthetic contracts. Not yet NAV.
**Branch:** `collaboration` · **Artifacts:** `COLLAB/experiments/uclid5/`
**Work split:** Claude (measurement) and Gemini (literature, soundness analysis), cross-checked via `COLLAB/claims.jsonl`.

**Headline:** the architecture was wrong, not the idea. Every previous attempt
injected contracts as `INVAR`s into a model that *encodes* the neural network.
Making the network an **uninterpreted function** and each contract an **axiom**
gives a sound k-induction proof over continuous state, and scales to 1,400
contracts in 7.4 s — past the point where nuXmv segfaults.

**This does not yet verify NAV.** See §6 before citing anything here.

---

## 1. Why nuXmv could not do this

Three measured walls, all from this week (`COLLAB/experiments/`):

| limit | measurement |
|---|---|
| neural inputs must be `INT` | `check_grammar.py:643`, all five encodings — a network can never read continuous state |
| regression networks | `table` encoding only, which needs an enumerable input space |
| CTL over reals | impossible — BDD cannot build an FSM over reals; MathSAT has no `check_ctlspec` |
| symbolic neural encoding | **times out at three hidden neurons** with generic weights |

That last one is the fatal one. Lifting the `INT` guard is a one-line change and
the model then verifies — but only for networks whose weights make the property
trivially 1-inductive. With random weights, a 2-3-2 network times out at 60 s.
The cause is ReLU phase-splitting: the solver must branch on every ReLU at every
unrolling step.

## 2. The architecture that works

```ucl
module main {
  type act_t = enum { up, down, stay };
  function ctrl(x : real, y : real) : act_t;        // the network, ABSTRACTED
  var px : real;  var py : real;

  // each CROWN contract becomes an assumption; no weights, no ReLUs
  assume c0 : ((px >= 10.0) && (px <= 13.0)) ==> (ctrl(px, py) != up);
  ...
  control { induction; check; print_results; }      // SOUND, not bmc(N)
}
```

The solver never sees a ReLU. Phase-splitting cannot occur because the network
is not in the model — only its certified consequences are.

## 3. Results

| contracts | UCLID5 k-induction |
|---|---|
| 10 | 1.64 s |
| 50 | 2.13 s |
| 200 | 2.76 s |
| 500 | 4.19 s |
| 1000 | 6.03 s |
| **1400** | **7.39 s** |

1,400 exceeds NAV's 1,386. For contrast, nuXmv stalled at 2,595 `TRANS`
constraints and segfaulted at 11,627.

**Quantified vs quantifier-free.** `axiom (forall (x:real) :: ...)` and
module-level `assume` both work (7.78 s vs 7.46 s at n=1400). The 4% is not the
reason to prefer `assume`: the quantifier-free form sits in **QF_LRA + UF**,
which is decidable, rather than relying on Z3's E-matching heuristics. Adopt it.

## 4. Validation

Near-linear scaling on real-valued axioms deserves suspicion, and the first
version of this experiment *was* an artifact — disjoint boxes with a property
matching `c0` alone, so Z3 found one axiom and ignored 1,399. It measured
parsing, not reasoning. Corrected and then checked four ways:

| check | result |
|---|---|
| property spans every contract (contiguous boxes) | all *n* axioms required |
| remove **one** mid-range axiom | `1 passed / 1 failed` |
| remove **all** axioms | proof fails |
| properties not implied by any axiom | correctly fail |

## 5. Is the abstraction sound?

Yes, conditionally. Any concrete network satisfying the contracts is a model of
the axioms, so a property proved for all models holds for the concrete network.
Under-constrained contracts produce **spurious counterexamples**, not false
proofs — the failure mode is imprecision, not unsoundness.

Four conditions must hold, and all four are threats to a NAV result:

1. **Contract soundness.** Inherited from CROWN; the proof is only as good as the bound.
2. **Inter-sample behaviour.** k-induction sees ticks, not continuous time between them.
3. **Float vs real.** The deployed network is IEEE-754; UCLID5 reasons over mathematical ℝ.
4. **Plant determinism.** A UF enforces `x = y ⟹ f(x) = f(y)`. If the controller has hidden state or sensor noise, the abstraction is **unsound**, not merely imprecise.

## 6. What this does NOT show

- The plant is a toy: 1-D motion, `px' = px ± 1.0`.
- Contracts are **synthetic boxes**, not α,β-CROWN output.
- NAV has not been run.
- No comparison against CORA or JuliaReach on the same property.

## 7. The honest limitation

Complexity does not vanish; it **moves from network size to state dimension**.

- d = 2 (NAV): K = 40 → ~1,600 contracts. CROWN 0.4 s, UCLID5 ~6 s. Comfortable.
- d ≥ 4 (CartPole, Quadrotor): K = 20 → 20⁴ = 160,000 contracts. Both tools choke.

So the claim is bounded and must be stated that way: **arbitrary network size,
low-dimensional plants**. It is not a general neural verification result.

## 8. Prior art

| work | relation | distinction |
|---|---|---|
| Vehicle (FM/CAV 2023, Daggitt et al.) | NN specs → Marabou bounds → Rocq/Agda proofs; uses UFs and contracts | interactive theorem proving, not automated k-induction over BTs |
| Perception Contracts (Seshia, Ghosh, Mitra 2022–24) | input-output error bounds as A/G contracts | reachability and CBFs, not SMT k-induction over discrete BT logic |
| CoVeNN (NeurIPS 2023) | A/G contracts for NN verification | decomposes *within* a network, not closed-loop with a plant |
| PREMAP, Rober & Johnson | input-space partitioning for NNCS | set reachability with wrapping effect, not SMT model checking |

**Verdict.** UF abstraction with pre/post-conditions is textbook formal methods —
pitching it as a novel abstraction invites rejection. The unpublished part is the
**end-to-end composition**: CROWN piecewise contracts → quantifier-free SMT
assumptions → k-induction over behaviour-tree execution. Pitch it as an
integration architecture that dissolves the phase-splitting wall, not as a new
abstraction technique.

## 9. Most promising directions, in order

1. **Run NAV with real contracts.** Generate α,β-CROWN contracts, emit them as
   `assume` statements, run k-induction. This is the experiment that decides
   whether any of the above matters; everything else is contingent on it.

   **Use `nn-nav-set.onnx`, not `nn-nav-point.onnx`.** The benchmark ships two
   networks with identical architecture (4-64-32-2, ReLU/ReLU/Tanh) but different
   training: `point` is standard RL, `set` is set-based training that folds
   reachability into the training loop for verifiable robustness
   (arXiv:2401.14961). The 09/21 Lipschitz measurements show why this matters:

   | network | Lipschitz | critical cell side |
   |---|---|---|
   | `nn-nav-set.onnx` | 17.09 | h = 0.0585 |
   | `nn-nav-point.onnx` | 51.21 | h = 0.0195 |

   A 3x smaller Lipschitz constant permits a 3x larger cell, which in 4-D is
   roughly 81x fewer cells and therefore ~81x fewer contracts. `nn-nav-set` is
   also the `nav_table_pipeline.py` default and what the 09/21 and 09/22 reports
   used. Report **both** in the end - ARCH-COMP ships both, and showing the method
   on the harder `point` network as well is the stronger claim - but `set` is the
   correct primary.
2. **Add a vacuity guard to the pipeline.** Before each proof, check the contract
   set is satisfiable (assert `false` and require UNSAT). Six times this session a
   result looked right for the wrong reason; this makes the check structural.
3. **Compare against CORA/JuliaReach on the same property.** We have never done a
   head-to-head. Without it, "we can verify NAV" invites "so can CORA, faster."
4. **Characterise the dimensional ceiling.** §7 predicts a wall between d=2 and
   d=4. Measuring exactly where it falls turns a limitation into a contribution.
5. **Only then**: a direct BehaVerify → UCLID5 backend. The April 2026 translator
   (`src/dsl_with_contracts_to_uclid5.py` on `main`) goes via nuXmv SMV and so
   inherits every limitation in §1. A direct emitter is the right design, but it
   is worth building only if (1) succeeds.

## 10. Reproduce

```bash
python3 COLLAB/experiments/uclid5/gen_contracts.py COLLAB/experiments/uclid5 10 50 200 500 1000 1400
./COLLAB/experiments/uclid5/run_scaling.sh 10 50 200 500 1000 1400
```

Install (no sudo; UCLID5 accepts JDK 8–11):
`REPRODUCIBILITY/2026_TBA/uclid5/` is a clone of `uclid-org/uclid` with Z3 4.12.2
from its own `get-z3-linux.sh`. The shipped 2020 prebuilt has **no `real` type** —
an `sbt universal:packageBin` build is required.
