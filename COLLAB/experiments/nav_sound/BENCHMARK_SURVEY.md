# Which BehaVerify example is the right benchmark?

NAV turned out to have **zero** behavior-tree structure (`tree { Act {} }`), so a
method that composes nothing loses to per-step reachability there. This surveys
all 358 `.tree` files in the repo against the three properties compositional
NSBT verification actually needs.

| example | composites | neural | continuous state |
|---|---|---|---|
| `ANSR_ONNX` | **10** | 2 | ✗ `INT [0, 200]` |
| `acas_closed_loop` | 6 | 5 | ✗ |
| `grid_world/counter_template` | 2 | 1 | ✗ |
| **`ANSR_behaverify_A_star_net/scaling_network/real.tree`** | **2** | **1** | **✓ `use_reals`** |
| NAV (`nn-nav-set.tree`) | **0** | 2 | ✓ |

**`real.tree` is the only file in the repository with all three.**

## What it contains

- `use_reals`; `drone_x`, `drone_y` declared `VAR REAL`
- `network NEURAL classification {left, right, up, down, no_action}` with those
  reals as inputs — discrete action over continuous state, the NSBT case
- a tree that is a `sequence` containing a nested `selector` (2 composites,
  18 leaves) — genuine branching
- an **18-obstacle** safety `INVARSPEC`
- a **`CTLSPEC` with `AF`** — liveness: `AG(AF(at destination) ∨ destination
  unreachable)`

A pre-generated `real.smv` sits beside it with 6 real variables, so it has been
built before. Generating it fresh hits the `check_grammar.py` INT-only gate —
the same gate `dsl_with_abstract_control_to_nuxmv.py` was written to bypass.

## Why the liveness spec is the point

Per-step interval reachability — the baseline that beat the compositional
architecture on NAV — **cannot express `AG AF` at all**. It computes a forward
reachable tube and answers safety. It has no way to say "from every reachable
state the destination is eventually reached."

So the honest reading of the NAV result is not that compositional verification
loses, but that we were competing on the one property class where a cheaper
method suffices. Safety over a single-mode continuous loop is reachability's
home ground; liveness over a branching tree is not, and cannot be. This fits the
project's history — grid world's compositional win (2026-08-31) was a
CTL/liveness result, not a safety one.

## Open risks on this candidate

1. Only **2 composites** — modest structure.
2. Not yet checked whether the safety property is actually **true**, or the
   network accurate enough. Both killed earlier candidates, including every
   branching NAV variant attempted (§14).
3. `AF` over reals needs a BDD-free CTL path; nuXmv's BDD engine rejects real
   variables (Gemini's claim #33), so the liveness check may need a different
   route than the safety one.

## Why not ARCH-COMP AINNCS or VNN-COMP

Those are single-controller closed loops by construction: one network, one loop,
no branching. They are all NAV-shaped, and per-step reachability would beat
compositional verification on every one of them, for the reason established in
§12–13. More of them would add cases, not evidence.
