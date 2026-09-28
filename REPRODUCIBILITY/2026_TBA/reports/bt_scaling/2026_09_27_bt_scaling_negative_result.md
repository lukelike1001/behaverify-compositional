# Does compositional verification have anything to offer behaviour trees?

**Date:** 2026-09-27
**Baseline:** `d78f80be1` — *Report that compositional NAV appears bearish*
**Branch:** `collaboration` (7 commits)
**Scope:** Two agents (Claude + Gemini) working one branch through a shared claim ledger.

**Headline: no, and we can now say why rather than merely failing to find a benchmark.**
Compositional verification earns its keep by avoiding a reachable state space too
large to handle directly. In behaviour-tree verification that situation does not
arise. When the reachable space *is* large it comes from structured domains that
BDDs compress to nothing, and monolithic BehaVerify answers in milliseconds.
When monolithic does struggle, the cause is never reachable state count.

**Recommendation: stop looking for a compositional BT benchmark.** The evidence
below is falsifiable — disagreeing requires exhibiting a behaviour-tree problem
where monolithic chokes on reachable state count specifically.

---

## 1. The two measurements that settle it

Both models were already in the repository and were **not modified**. Only the
measurement is new (`COLLAB/experiments/reachable.sh`).

| model | BT nodes | reachable states | diameter | monolithic |
|---|---|---|---|---|
| `examples/DrunkenDrone/DrunkenDrone.tree` | 5 | **3,060,480** | 42 | 2 specs in **0.15 s** |
| `examples/2025_FMCAD_BT2Fiacre/drone3.tree` | — | **7,432,390** | 950 | 6 specs in **66.5 s** |

A five-node behaviour tree with three million reachable states, fully verified in
under a fifth of a second. That single line is the argument. Behaviour trees do
have large reachable state spaces — the premise that they do not was wrong, and
was refuted during this session — but BDDs absorb them, because coordinates,
grids and battery levels are regular arithmetic domains.

> **Gotcha.** `drone3` does not build with the current `behaverify` CLI; it needs
> the 2025_FMCAD generator and metamodel. Using the CLI yields a generation
> failure that looks like a broken model. `reachable.sh` uses the correct one.

## 2. Where monolithic actually breaks, and why it does not matter

**Tree size** — dies of memory, far beyond anything real.

| levels | leaves | nodes | SMV lines | LTL | verdict |
|---|---|---|---|---|---|
| 10 | 1024 | ~2047 | 10,395 | 61.1 s | `is true` |
| 11 | 2048 | ~4095 | 20,635 | 203.3 s | `is true` |
| 12 | 4096 | ~8191 | 41,115 | 537.4 s | **killed, rc=137 (OOM)** |

**State space** — costs time superlinearly, with the tree held constant at 64 leaves.

| domain | LTL |
|---|---|
| 64 | 0.46 s |
| 640 | 0.55 s |
| 6,400 | 5.99 s |
| 64,000 | **616.8 s** |

**Real trees are nowhere near either wall.** All 141 `.tree` files in `examples/`
have median 11 nodes, p90 16, max 41. The largest (`MultiAgent/FiveInRing.tree`,
41 nodes) answers 10 specs in 0.15 s; `2025_FMCAD_BT2Fiacre/droneNew.tree`
(36 nodes) answers 9 in 1.28 s.

## 3. A benchmark built to support the idea, which refuted it

To find a realistic case with a genuine wall, we generated an N-robot ring
matching `FiveInRing` — N agents, identical 6-node subtrees, shared positions,
pairwise-disjointness safety plus per-robot `AG AF` liveness.

| N | nodes | declared space | check |
|---|---|---|---|
| 5 | 31 | 7,776 | 0.05 s |
| 6 | 37 | 117,649 | 0.25 s |
| 7 | 43 | 2,097,152 | 117.9 s |

A 16 % larger tree cost 470×, which looked like the wall we wanted. (A run at
N=8 exceeded a 1800 s timeout and N=10 was OOM-killed, but those outputs were
lost to a scratchpad wipe and are *not* included in the committed evidence.)
**It was an artefact, in three separate ways.**

1. **Only 23 states are reachable at N=7**, not 2.1 M. The declared variable
   ranges are irrelevant; `print_reachable_states` converges instantly.
2. **The cost is entirely CTL liveness**: `check_invar` 0.13 s versus
   `check_ctlspec` 101.4 s. And `-dynamic` alone takes that to **10.9 s**, so
   most of the "wall" was a static BDD variable order — a tuning flag.
3. **The generator was unfaithful to its original.** `FiveInRing` assigns goals
   irregularly so the robots deadlock: 8 of its 10 specs are FALSE and nuXmv
   exits on shallow counterexamples. Our regular `goal = (i+3) % npos` removed
   that deadlock, making every liveness spec TRUE and forcing full nested
   fixpoints. Verdict counts (`verdicts.sh`): original `FiveInRing`
   **8 false / 2 true**, generated `ring_5` **0 false / 6 true**.

The generator is kept in `COLLAB/experiments/gen_ring.py` with this flaw
documented, because the artefact is the finding.

## 4. Prior work that closes the remaining exits

- **Subtree-level assume-guarantee is not new.** Colledanchise & Ögren
  (IEEE T-RO 2017; CRC book 2018, ch. 5) formalise subtree contracts as regions
  of attraction and success sets, with composition lemmas for Sequence, Fallback
  and Safety. Biggar & Zamani (IEEE RA-L 2020) give an assume-guarantee
  refinement framework with contract triples and a substitution theorem — and
  their code is already in this repository at
  `REPRODUCIBILITY/OTHER_TOOLS/biggar2020Framework/`, where `bt_verification.py`
  defines `preconditions(tree, subtree, bhvrs)`.
- **Per-agent decomposition cannot work on ring-shaped coordination.** Liveness
  requires the single empty slot to circulate all N positions — an O(N) global
  dependency. A contract mentioning only immediate neighbours cannot prove it
  without circular reasoning.
- **ANSR has no external provenance.** Authored in-house (Serena Serbinowska,
  2023); DARPA ANSR is the funding grant, not a benchmark suite. Separately,
  `examples/ANSR_ONNX/ANSR_ONNX.tree` no longer parses (obsolete `NEURAL` syntax)
  and `examples/ANSR_revised/ANSR.tree` has an empty `specifications` block.
- **ARCH-COMP AINNCS is unusable for BT work.** Every benchmark (ACC, TORA,
  CartPole, Airplane, Quadrotor, Unicycle, NAV) is a plant plus one feedforward
  controller, collapsing to a single action node.

## 5. The dilemma, stated plainly

1. When a behaviour-tree problem has a large reachable state space, it is
   structured, BDDs compress it, and monolithic verifies it in seconds.
2. When monolithic genuinely struggles, the cause is something else entirely —
   11,627 unrolled `TRANS` constraints from continuous plant dynamics (NAV), or
   cyclic BDD variable coupling in nested CTL fixpoints over 23 reachable states
   (the ring). Contracts address neither.

There is no regime where an assume-guarantee decomposition is the right tool.

## 6. One result worth keeping from the abandoned contract work

The ACAS Xu contract pipeline built the week before this report is being deleted
along with the direction, but it produced one finding that applies to *any*
contract-based verification and would otherwise be lost.

**Contracts can make a property pass by over-constraining the model, and the
obvious checks do not catch it.** An ambiguous region keyed by individual
advisory rather than by advisory *set* emits two contradictory implications over
the same states. Both fire, the conjunction is unsatisfiable, the transition
relation stops being total, and everything downstream of the deadlock becomes
unreachable — so the safety property passes because nothing reachable can
violate it.

Three checks, and only the third detects it:

| check | catches |
|---|---|
| non-vacuity (delete contracts, property must fail) | **under**-constraint |
| coverage (every reachable state matches some contract) | gaps |
| **progress** (is the horizon still reachable?) | **over**-constraint |

Safety and coverage both pass happily on a deadlocked model. The minimal
demonstration: adding one extra `TRANS (x = 2) -> a = q;` beside an existing
`-> a = p` flipped `INVARSPEC x <= 2` from `is false` to `is true`.

nuXmv's `check_fsm` reports this directly ("transition relation is not total"),
but it cannot build a BDD FSM over real-valued variables, so for LRA models the
substitute is a reachability query — `INVARSPEC step != H` returning `is false`,
where the counterexample *is* the proof that the model runs to completion.

## 7. Method note

All claims were registered in an append-only ledger (`COLLAB/claims.jsonl`)
before being relied upon, and `collab.py` refuses self-verification. Final state:
17 claims, 14 from Claude and 3 from Gemini; 4 of Claude's refuted, 3 verified,
9 open but not load-bearing.

Two of the four refutations — that subtree assume-guarantee already exists, and
that the ring generator was unfaithful — were caught by the second agent and
would not have been caught alone. One claim (Gemini's #17, that no cutoff
theorems exist for BTs) is deliberately left **unverified**: it is a universal
negative over literature neither agent can exhaustively read, and agreeing with
the reasoning is not verification. It should not be cited without a real survey.

## 8. Reproducibility

Everything is committed under `COLLAB/experiments/`:

| script | measures | output |
|---|---|---|
| `reachable.sh` | the two decisive models (§1) | `reachable_results.txt` |
| `measure.sh` | synthetic tree size (§2) | `results_results.txt` |
| `decouple.sh` | state space, tree held fixed (§2) | `decoupled_results.txt` |
| `real.sh` | largest real trees (§2) | `real_results.txt` |
| `gen_ring.py`, `ring.sh` | the ring (§3) | `ring_results.txt` |
| `verdicts.sh` | original vs generated spec verdicts (§3) | `verdicts_results.txt` |
| `ring_diagnosis.sh` | the three artefacts at N=7 (§3) | `ring_diagnosis_results.txt` |

Timings vary noticeably with machine load, so treat ratios rather than absolute
seconds as the result. Measured twice: `drone3` 66.5 s clean / 79.4 s under load;
`DrunkenDrone` 0.15 s / 0.22 s; ring N=7 `check_ctlspec` 101.4 s / 124 s and with
`-dynamic` 10.9 s / 18.4 s. Every figure in this report is from the committed
`*_results.txt` files, not from a transcript.

**A nuXmv run that answers 0 specs is a broken model, not a hard problem.** An
out-of-range initial assignment aborts the command script and leaves nuXmv
idling at a prompt, which looks exactly like a slow verification and cost an
hour here.
