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

**`real.tree` is the only file with all three — but only on a type technicality.**

> **Correction, after reading its transition relation rather than its
> declarations.** `drone_x` is `VAR REAL`, but it starts at 3 and updates only as
> `(max, min_val, (sub, drone_x, 1))` / `(min, max_val, (add, drone_x, 1))` —
> ±1, clamped to `[0,6]`. Destinations come from `(loop, loop_var, [min_val,
> max_val])`, i.e. integers. Obstacles have `max_obstacle_size := 0`, so they are
> lattice points. The non-determinism clause is commented out.
>
> **The reachable set is 49 integer positions in a 7×7 grid.** `use_reals` is a
> type declaration that makes nuXmv use real arithmetic — which matters for the
> network's `config { real }` symbolic encoding — not a continuous state space.
> `real.tree` is grid world wearing REAL annotations.
>
> **Corrected conclusion: nothing in the 358 trees has BT structure AND a
> genuinely continuous reachable set AND a neural network.** The gap is real and
> unfilled.

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


---

# Second correction: `real.tree` IS grid world, already solved

Luke's instinct ("oddly familiar to proto-grid world") was right, and stronger
than "similar."

| | `real.tree` | 2026-08-31 liveness report |
|---|---|---|
| grid | `min_val 0, max_val 6` → 7×7 | 7×7 |
| obstacles | `number_of_obstacles := 18`, `max_obstacle_size := 0` | 18, size 0 |
| network | `1000__6_18_0__0100_1.onnx` | `1000__6_18_0__0200_1.onnx` |

The filename decodes as *grid 6, 18 obstacles, size 0* — the same family, the
same domain. A structural diff against `grid_world_big/template_fake_network_SCALE.tree`
shows the same action enum, the same `obstacles`/`obstacle_sizes` arrays, the
same `destination = drone position` initialisation, and the same action names
(`new_destination`, `next_action`).

`real.tree` differs only by declaring positions `VAR REAL` instead of
`VAR [min_val, max_val]`, inverting the selector/sequence nesting, and adding
`read_position` / `cell_changed` / `send_action` leaves.

**And the 2026-08-31 report already proves `INVARSPEC = true` AND
`CTLSPEC = true` on it, in 0.33 s**, via Floyd-style progress contracts.

So pivoting to `real.tree` would re-derive an existing result with a type
annotation changed. Claims #67 and #68 are superseded by #69; the
recommendation is fully withdrawn.

## Terminology note

I described the reachable set as a "lattice." That is sloppy and actively
confusing in this field: in abstract interpretation a *lattice* is the
order-theoretic structure of an abstract domain (with meets and joins), which is
a different object entirely. The correct phrase is **"the reachable set is a
finite set of integer grid points"** or simply **"the state space is discrete."**
