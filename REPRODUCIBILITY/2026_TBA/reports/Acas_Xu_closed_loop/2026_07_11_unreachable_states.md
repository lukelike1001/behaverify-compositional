# ACAS Xu: Classifying the UNSAT Contracts

**Date:** 2026-07-11

**Scope:** Optimization 0 from `2026_07_11_hiatus_revival.md` — classify each of the 791
discrete-mode UNSAT contracts (Table IV) as explained by unreachability or genuinely
unresolved, then resolve whichever remains by direct CROWN point-query (Phase B). Extended
partway through to also classify the 1,659 SAT contracts, since Optimization 1 (reachability
pruning) only ever prunes SAT contracts and needs that number to size the work — this
extension is what caught an error in the original contract-319 finding (see below).

---

## Method

**Reachability (`acas_reachability.py`).** BFS over the augmented state
$(x_{\text{mag}}, y_{\text{mag}}, x_{\text{sign}}, y_{\text{sign}}, h, a_{\text{prev}})$,
seeded from the closed loop's fixed initial condition
($x_{\text{mag}}=7, y_{\text{mag}}=6, x_{\text{sign}}=y_{\text{sign}}=1, h=10, a_{\text{prev}}=\text{clear}$,
now recorded in `acas_model_params.yaml` rather than hardcoded). Each tick's advisory is
treated as nondeterministic — any of the 5 — which only enlarges the reachable set
relative to the real system, so containment holds and the result is a sound
over-approximation (soundness argument in the hiatus report, part (c), Optimization 1).

**How reachable states are computed.** BFS is a fixed-point computation, not a
bounded-depth search: from the single seed state, each round applies `simulate_step`
under all 5 advisories to every state found so far, adds any newly-seen states to the
frontier, and repeats until a full round adds nothing new. So the result is the *complete*
forward-reachable set — every state reachable in 1 tick, 2 ticks, ..., or any number of
ticks, not a fixed-horizon snapshot. (This answers "is it just the next tick?" — no, it's
reachability over the whole unbounded-time closed loop; the state space is finite, so the
fixed point is guaranteed to exist and BFS finds it in seconds.)

The single biggest driver of how fast the frontier spreads per round is the per-tick bound
$|\Delta x_{\text{mag}}|, |\Delta y_{\text{mag}}| \leq 3$ (verified by exhaustively
simulating all 96,800 (state, advisory) pairs — see part 3 of the reachability discussion
above). That bound is why the reachable set converges to 9,428 states rather than staying
near the seed or covering the full 96,800.

**Other factors that shift the reachable set:**
- **Initial state.** A different seed produces a different forward closure through the
  same transition graph — reachability is defined relative to where you start (see Q1/Q2
  above).
- **Advisory nondeterminism.** Every tick is allowed to pick freely among all 5 advisories,
  not just whatever the trained network would actually output. This is a deliberate
  over-approximation, sound because $R \supseteq$ the true reachable set — so 9,428 is an
  upper bound, and the real closed loop's reachable set could be smaller.
- **Fixed intruder/ownship speeds and intruder heading** (30, 20, 225° respectively) are
  held constant in this benchmark. Changing any of them changes the per-tick velocity
  difference and therefore the $\Delta x_{\text{mag}}/\Delta y_{\text{mag}}$ bound itself,
  not just the seed — a different physics configuration would need this whole analysis
  re-run, not just re-seeded.

**Reachability results.** No network is queried. Runs in under a second: **9,428 of 96,800** augmented states reachable (~9.7%).

**Classification (`classify_contracts_by_reachability.py`, originally named
`classify_unsat_contracts.py` when scoped to UNSAT only).** For each contract, checked
whether any of its `dangerous_xy` states, paired with the network's `a_prev`, fall in the
reachable set. A contract with no reachable dangerous state is `unreachable_explained`;
otherwise `reachable_dangerous_state` -- meaning the state needs direct verification to know
what the network actually does there (Phase B, below). Initially run over the 791 UNSAT
contracts only; later extended to all 2,450 (SAT included) once it was clear the SAT side
mattered for Optimization 1 sizing.

## Results

| Network | Total UNSAT | Unreachable-explained | Reachable dangerous state |
|---|---|---|---|
| clear | 179 | 179 | 0 |
| weak_right | 135 | 135 | 0 |
| weak_left | 174 | 174 | 0 |
| strong_right | 155 | 154 | 1 |
| strong_left | 148 | 148 | 0 |
| **Total** | **791** | **790** | **1** |

**790/791 (99.9%) of discrete UNSATs are explained by unreachability**, confirming the
theory from `2026_03_25_pgd_unsat_acas_report.md` and part (d) of the hiatus report:
the 32%-UNSAT-vs-INVARSPEC=true tension is a scope mismatch between the contract
generator (enumerates the full syntactic domain) and nuXmv's reachable-state semantics,
not a defect in the contracts or the networks.

## Extended check: the 1,659 SAT contracts

Only SAT contracts are ever turned into INVAR constraints (`_load_sat_contracts()` in
`run_acas_compositional_pipeline.py` filters to SAT before injection), so Optimization 1's
actual payoff — the projected INVAR count after pruning, replacing the 8,982 that segfault
nuXmv — depends entirely on how many SAT contracts' dangerous states intersect the reachable
set. Ran `classify_contracts_by_reachability.py` over all 2,450 contracts (not just the 791
UNSAT) to get this number before writing any pruning code.

**Result: 0 of 1,659 SAT contracts have any reachable dangerous state.** Projected INVAR
ceiling after pruning: **0**. Combined with the UNSAT-side result, exactly **one** dangerous
`(state, advisory)` pair out of all 14,150 checked across the full contract set intersects
the reachable set at all — and it's on the UNSAT side (contract 319, below).

## The Exception: contract 319 (and a correction)

**Correction:** an earlier pass of this report claimed contract 319's one reachable state was
$(x_{\text{mag}}, y_{\text{mag}}) = (2, 3)$ and that CROWN verified SAT there. That claim was
never actually computed — it was asserted without running the code. The SAT-side extension
above required directly enumerating every contract's states against the reachable set, which
caught the error. The corrected results follow.

The one exception — `strong_right`, contract id 319
($h=6, x_{\text{sign}}=1, y_{\text{sign}}=1$, forbids `strong_right`, 14 covered states) —
has exactly one reachable dangerous state among its 14: $(x_{\text{mag}}, y_{\text{mag}}) = (3, 0)$,
confirmed by two independent checks (a full cross-check over all 2,450 contracts, and
enumerating contract 319's 14 states one by one). Discrete verification short-circuits on
the first violating state and does not record which one, so the original contract-level
UNSAT verdict could have come from any of the 14.

Ran a single CROWN point-query (`verify_single_state.py`) at the actual reachable state:

```
state: x_mag=3, y_mag=0, x_sign=1, y_sign=1, heading_own_var=6
network: strong_right, forbidden: strong_right
status: UNSAT   (verified_status: unsafe-pgd -- a real PGD counterexample found)
```

**The network genuinely selects `strong_right` at this state.** This is not a timeout or an
inconclusive result; it's a confirmed violation of contract 319's guarantee at a state inside
the over-approximated reachable set `R`.

### Why this isn't a real safety bug

`R` is a *sound over-approximation* built without ever querying a network (nondeterministic
advisory choice at every tick), so `R \supseteq` the true reachable set, not `R =` the true
reachable set. The monolithic run already proved `INVARSPEC=true` unconditionally over its
own (exact) reachable set. If $(x_{\text{mag}}=3, y_{\text{mag}}=0, x_{\text{sign}}=1,
y_{\text{sign}}=1, h=6, a_{\text{prev}}=\text{strong\_right})$ were truly reachable, and the
network truly picks `strong_right` there (both now confirmed), the real closed loop would
violate $\rho \geq 200$ -- contradicting the monolithic proof. Since that proof is trustworthy
(verified against `2025_NEUS/.../invar.txt` earlier in this effort), the only consistent
conclusion is that **this exact state is in $R$ but not in the true reachable set** -- a
concrete witness that $R$ is not tight here, not evidence of a real violation.

This argument generalizes: the monolithic `INVARSPEC=true` proof alone already guarantees
none of the 791 UNSAT contracts describes a real safety bug, since a real one would have shown
up as a monolithic counterexample. That's a cleaner reason for the 790/791 (and now 791/791)
result than doing state-by-state reachability classification -- the classification work here
mainly serves Optimization 1's pruning, not re-establishing safety that the monolithic proof
already settled.

### A real wrinkle for Optimization 1

Contract 319 is UNSAT, so it was never a candidate for injection regardless of pruning. But
its dangerous state *is* in $R$. `_add_command_free_var()` replaces each network's output
with a nondeterministic free variable in the pruned SMV -- the same kind of transition rule
`acas_reachability.py`'s BFS uses. If nuXmv's own (exact, BDD-based) reachability computation
over the pruned model also reaches this state, and nothing constrains the free variable to
avoid `strong_right` there, nuXmv will find that trace and report `INVARSPEC=false` -- a
spurious counterexample caused by $R$'s looseness, not a real one, but one that would make the
pruned compositional pipeline disagree with the monolithic verdict. Naive pruning ("keep SAT
contracts intersecting $R$, drop the rest, ignore UNSAT contracts entirely") does not by
itself guarantee the pruned model reproduces `INVARSPEC=true`. This is the open design
question for Optimization 1's implementation, not yet resolved.

## Conclusion

**Two results, not one.** The SAT side is clean and strong: 0 of 1,659 SAT contracts touch
the reachable set at all, so Optimization 1's pruning has a real, near-total payoff on the
INVAR count that actually matters (0 projected constraints from SAT contracts, replacing
8,982). The UNSAT side has exactly one exception, and it isn't a safety finding -- it's a
demonstration that $R$ is a strict (non-tight) over-approximation at one specific point, which
is expected of any sound abstraction that avoids querying the real networks. What it does
change is the plan: Optimization 1 can't just prune by "$R$-intersection, SAT contracts only"
and assume the pruned model matches the monolithic verdict -- contract 319's gap needs to be
accounted for, or `INVARSPEC=false` on the pruned model should be expected and explained
rather than treated as a surprise.

This is close to, but not exactly, the outcome the hiatus report anticipated ("if the
overwhelming majority fail the reachability check, your theory is confirmed") -- 790/791 fail
the check outright, and the 791st is resolved by the monolithic proof rather than by direct
reachability. Optimization 0's original question (do these UNSATs indicate real safety
problems?) is answered: no. Optimization 1's scope grew by one edge case in the process.

## Infrastructure notes

Running the July 11th Optimization 0 required an environment fix not exercised by prior committed work:

- **`auto_LiRPA` is a git submodule** of `alpha-beta-CROWN`, not a pip dependency —
  a plain `git clone` leaves it empty, and `import abcrown` fails several layers deep
  (`ModuleNotFoundError: No module named 'auto_LiRPA'`) with no indication why. Needs
  `git submodule update --init` plus its own `pip install -e auto_LiRPA`.
- **The `abcrown` package itself was never being installed** — the README's prior
  `pip install -r .../requirements.txt` step only installs dependencies, not the package
  (`import abcrown` fails with `ModuleNotFoundError`). Needs `pip install -e .` from
  `alpha-beta-CROWN/`.
- `pip install -e alpha-beta-CROWN` downgrades `torch`/`torchvision` at the user level
  (abcrown pins `torch<2.9.0`) — a real side effect outside this repo's scope, noted in
  both READMEs.

Both `README.md` and `examples/AcasXu_closed_loop/README.md` updated with the corrected
install sequence and a warning about the torch downgrade.
