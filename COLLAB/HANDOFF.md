# Handoff — Gemini takes the lead, 2026-09-30 ~02:50 CST

Claude is out of budget. Gemini leads until Luke returns. Luke's steer, in his
words: *keep the scope small at first and focus on CORA, then more monolithic vs
compositional tests, before worrying about other benchmarks.*

And his standing permission: **failing is fine if the reason is written down.**
Every patched dead end is evidence the project was taken seriously.

---

## Priority 1 — CORA (the decisive experiment)

**The question:** does a real constrained-zonotope tool prove the geofence
property that our compositional encoding proves and every cheap baseline fails?

- **If CORA fails** → the pilot's headline is about as strong as it can get.
- **If CORA succeeds** → the benchmark does not discriminate, and the headline
  must be **withdrawn**. That outcome is worth more than a confirmation, and it
  is far better to find it now than in review.

**Everything needed is already in place:**
- MATLAB R2020b works headless — verified: `~/MATLAB/R2020b/bin/matlab -batch "disp(1)"`, license 299681.
- The system, contracts and baselines: `COLLAB/experiments/geofence/`
- Our own zonotope baseline to compare against: `zonotope_box.py`
  (hull 1.5298, guard-split 1.5861, both violated; limit is 1.5)
- Compositional proves safe in 5.9 s (`compare.py 20 0.8 12`)
- True closed-loop maximum: **1.2622** — so anything above 1.5 is spurious

**The one thing our baseline cannot do**, and the reason CORA matters: a
zonotope is not closed under halfspace intersection, so our guard split re-fits
an *axis-aligned* zonotope to each piece and loses correlation through the cut.
CORA's constrained zonotopes keep it. That is the only remaining way the
baseline could win.

## Priority 2 — monolithic vs compositional

Once CORA has answered, go back to the comparison the project actually rests
on. The ATVA reviews (now deleted; lessons below) said the NSBT support was
**claimed but never evaluated** — that is the gap to close, and it needs
monolithic-vs-compositional numbers on NSBTs, not more benchmarks.

## Priority 3 — only then, benchmarks

Extending the geofence to other verifiers is allowed but is not the priority.

---

## What is retracted — do not build on these

Three claims were refuted tonight, all Claude's, all caught by Gemini:

- **#94, #95 — "nuXmv inc_coi is unsound."** Wrong. `msat_check_invar_inc_coi`
  is an abstraction-refinement loop: at COI(0) it *logs* an abstract candidate
  containing the words `is false`, then rejects it as spurious and finishes with
  `could not determine`. Its verdict is **inconclusive**. Claude grepped for
  `is false`, took the first match, and never read the tail.
- **#86 — "the splitting baseline is not a strawman."** It was. Axis-aligned
  intervals lose x–v correlation on coupled linear dynamics regardless of any
  guard.

**The surviving claim, stated at the right width:** compositional beats every
*cheap* reachability baseline we could build, including one with exact linear
dynamics — **not** "compositional beats continuous reachability tools." CORA
decides whether the stronger sentence is ever available.

## Discipline that earned its keep tonight

1. **Check the property is true of the real system before trying to prove it.**
   The first geofence envelope made safety *false* (max x 2.55). The envelope in
   use now was derived, not guessed, and matches the braking physics.
2. **Run the vacuity guard.** An UNSAT on a frozen model proves everything.
   `compare.py`'s `sanity()` does this.
3. **Be more suspicious of a self-confirmation than a self-refutation.** #95 was
   Claude doubting #94 and then talking himself back into it with more detail —
   rigour applied to the wrong object.
4. **Read a tool's full output before concluding it is broken.**

## Lessons from the ATVA 2026 reviews (deleted; never commit review text)

Scores 4/3/4/2 — two leaning accept. **Not a novelty rejection.** The criticisms
with teeth were methodological:

- Cross-tool comparisons judged unfair because PyTrees and BehaviorTree.cpp
  semantics differ — and the paper conceding "exact recreation is not possible"
  made it worse, not better.
- Each competing tool compared on only one experiment; reviewers wanted a
  matrix.
- **"No evaluation of neuro-symbolic BTs support"**, listed as a Con by the only
  high-confidence reviewer. Another asked what makes NSBT support novel beyond
  an off-the-shelf piecewise-linear encoding.

**Why this argues for our direction:** the cross-tool fairness problem
*evaporates* for NSBTs, because no other BT tool has neural leaves at all. The
comparison becomes against NN verifiers and reachability tools — which is
exactly what the geofence work is building.

## State

Clean tree, 57 tests passing, geofence modules at pylint 10.00/10, 96 claims
logged with 3 refuted. Reports: `reports/misc/2026_09_30_why_should_we_care.md`
(for a first-year audience) and `COLLAB/experiments/geofence/README.md`.
