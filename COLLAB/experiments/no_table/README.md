# Skipping the monolithic SMV entirely

`dsl_with_contracts_to_nuxmv.py` (and the April UCLID5 translator) both generate
the **full monolithic SMV first** — network encoded as a lookup table — and then
regex the table back out. That inherits every limitation of the table path:

- neural inputs must be `INT` (`check_grammar.py:643`) → no continuous state
- regression networks work only via `table`
- the table must enumerate the input space before being discarded

## The redesign: never declare a NEURAL variable

If the control is an **ordinary blackboard variable** rather than a `NEURAL` one,
no table is generated and the INT-only check never fires — so the state can be
`real`. `abstract_nav.tree` declares `u1` as a plain `REAL` blackboard variable.

Then a small patch (vs. deleting `case` blocks):

1. drop `init(u1_stage_0)` and its assignment, and point `u1_stage_1` at a fresh
   free variable `u1_free` — the control becomes unconstrained
2. inject CROWN contracts as `INVAR` at `MODULE main` scope

## Result — nuXmv, continuous state, no table

```
with contracts       -- invariant (v >= -10.0 & v <= 10.0)   is true
contracts stripped   -- invariant (v >= -10.0 & v <= 10.0)   is false
```

Non-vacuous: the contracts are load-bearing.

**So the NAV blocker was substantially self-inflicted.** Routing through the
NEURAL/table path imposed the INT-only restriction; not declaring a NEURAL
variable avoids it, and this works in **nuXmv** — UCLID5 is not required for
this part.

## What this does NOT yet show

- Dynamics here are **linear** (`v + dt*u1`, `x + dt*v`). NAV's real plant is
  `dx = v*cos(θ)`, which is nonlinear and still outside MathSAT's reach. The
  displacement-as-UF trick from `../nav_uclid5/` may transfer, untested here.
- This is a **hand-patched SMV**, not a generator. The redesign is not implemented.
- Three contracts, not 16,000. Scale in this form is untested.
