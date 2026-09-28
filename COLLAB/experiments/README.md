# Scaling experiments — evidence for claims #2, #8, #13

Scripts and raw output behind the 2026-09-28 measurements. Kept because the
conclusion is negative, and a negative result is only worth anything if it can
be re-run.

| script | what it measures | output |
|---|---|---|
| `measure.sh` | synthetic tree size (FMCAD binary-tree benchmark), generation and checking timed separately | `results_results.txt` |
| `decouple.sh` | state-space size with the tree held FIXED — the benchmark confounds the two axes | `decoupled_results.txt` |
| `real.sh` | the largest real trees in `examples/` | `real_results.txt` |
| `gen_ring.py` + `ring.sh` | generated N-robot ring, a realistic multi-agent scaling axis | `ring_results.txt` |

## Read this before trusting `gen_ring.py`

**It is not a faithful generalisation of `examples/MultiAgent/FiveInRing.tree`.**
The hand-written original assigns goals irregularly so the robots deadlock: 8 of
its 10 CTL specs are FALSE and nuXmv terminates on shallow counterexamples in
0.15s. `gen_ring.py` uses `goal = (i + 3) % npos`, which removes the deadlock,
making every liveness spec TRUE and forcing full nested `AG AF` fixpoints. The
apparent wall at N=7 is a consequence of that choice, not a property of
BehaVerify or of multi-agent behaviour trees.

Kept anyway, because the artefact is the finding.

## What the numbers actually showed

- Tree size is cheap: 4095 nodes, LTL in 203s.
- Real trees are tiny: 141 files in `examples/`, median 11 nodes, max 41.
- The ring's reachable state space is **23 states** at N=7, not the 2,097,152
  the declared variable ranges suggest.
- At N=7: `check_invar` 0.20s, `check_ctlspec` 124s — the cost is entirely CTL
  liveness fixpoints, and `-dynamic` alone takes that to 18.39s.

## Gotcha that cost an hour

A nuXmv run answering **0 specs** is a broken model, not a hard problem. An
out-of-range initial assignment aborts the command script and leaves nuXmv
idling at a prompt, which looks exactly like a slow verification.
`examples/ANSR_revised/ANSR.tree` does this, and also has an empty
`specifications` block.
