# Real CROWN contracts for NAV, as UCLID5 assumptions

Uses **`nn-nav-set.onnx`**, not `nn-nav-point` — identical architecture, but
set-based training gives Lipschitz 17.09 vs 51.21 and a 3x larger admissible
cell (`reports/NAV/2026_09_21_monolithic_lipschitz.md`).

## Soundness: directed outward rounding

CROWN returns floats; SMT needs decimal literals. Rounding a lower bound up, or
an upper bound down, silently narrows the certified interval and makes the
contracts **unsound**. `gen_nav_contracts.py` floors every lower bound and ceils
every upper bound (`floor_to` / `ceil_to`, 6 dp). Naive `f"{v:.6f}"` would not be
safe. Contracts are also intersected with `[-1, 1]`, which `tanh` guarantees.

## Resolution vs contract count

| grid (x,y,v,θ) | contracts | mean width | max width |
|---|---|---|---|
| 4×4×2×2 | 64 | 2.36 | 4.10 |
| 8×8×4×4 | 1,024 | 0.95 | 3.35 |
| 16×16×8×8 | 16,384 | 0.40 | 2.36 |
| 24×24×12×12 | 82,944 | 0.24 | 1.75 |

Vacuous is 2.0 (the free `tanh` bound). At 64 boxes the contracts are **worse
than useless** — θ cells are π wide. Informative bounds need ~10⁴–10⁵ contracts.

**So UCLID5 removes the solver bottleneck, not the contract-count one.** Contract
count is set by the network's Lipschitz constant, not by the checker.

## Nonlinear dynamics stay in QF_UFLRA

NAV's plant is `dx = v·cos(θ)` — a product of a variable and a trig term, which
MathSAT cannot handle (measured: `var*var` gives no verdict in 200 s). Fix:
abstract the **displacement** as a UF too, with interval contracts from interval
arithmetic over `(v, θ)`. Same trick as the network. See
`nav_displacement_as_uf.ucl` — it parses and runs with no nonlinearity error.

## The open problem: a certified trapping region

k-induction needs an inductive invariant. For `|v|` to be bounded, the network
must be certified to decelerate at high speed. Queried over the whole operating
box, CROWN gives **sign-undetermined** bounds at every speed band:

```
v in [ 0.8, 1.0]   u1 in [-2.4095, +2.1079]   SIGN UNDETERMINED
v in [-0.2, 0.2]   u1 in [-2.6489, +2.0756]   SIGN UNDETERMINED
```

Those queries are deliberately coarse (all x, y and all θ at once). Whether a
*fine-grained* certified trapping region exists is the unresolved question, and
it is what decides whether NAV is provable by this route.
