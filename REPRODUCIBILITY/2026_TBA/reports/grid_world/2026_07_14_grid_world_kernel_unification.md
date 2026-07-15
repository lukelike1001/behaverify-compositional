# Grid World as the Degenerate Viability Kernel

**Date:** 2026-07-14

**Scope:** Retrofit the inductive-invariant / viability-kernel framework (developed
on ACAS Xu, 2026-07-11–12) onto the 1-NN grid-world benchmark, and make the kernel
the single source of truth for A/G contract generation.

**Reproduce:**

```bash
cd REPRODUCIBILITY/2026_TBA/examples/grid_world
python3 grid_world_inductive_proof.py
```

Every `[CHECK]` line is an assertion from `GridWorldInductiveProof`.

---

## 1. Claim

Grid world has a **hover** action (`XX` / `no_action`). Staying in a safe cell is
always safe. Therefore the viability kernel is the entire safe set:

$$
V = \mathrm{Safe}, \qquad \text{safe-but-doomed} = \emptyset,
$$

and the greatest fixpoint converges in **zero** deletion rounds.

The boundary then collapses to the obstacle-adjacent cells:

$$
\partial V = \{ s \in V : Allowed_V(s) \neq \mathcal{A} \},
$$

and $Allowed_V(s)$ is exactly “all actions except those that step into an
obstacle.” Kernel-boundary contracts

$$
s \in \partial V,\quad a \notin Allowed_V(s)
\quad\Longrightarrow\quad
\bigl(s \;\Rightarrow\; \mathrm{NN} \neq a\bigr)
$$

are therefore the same one-step crash-avoidance contracts the compositional
pipeline already used — derived, not hand-enumerated.

**Unification with ACAS Xu.** The same framework covers both benchmarks:

| | Grid world | ACAS Xu closed loop |
|---|---|---|
| Hover / stay action? | Yes (`XX`) | No |
| Safe-but-doomed count | **0** | 1,265 |
| Fixpoint rounds | **0** | 8 |
| $V$ vs Safe | $V = \mathrm{Safe}$ | $V \subsetneq \mathrm{Safe}$ |
| $\partial V$ | obstacle-adjacent cells | annulus around doomed core |
| Contracts needed | $\|\partial V\|$ crash edges (38) | $\partial V \cap R$ after CEGAR (1 on this seed) |

One-step crash avoidance is inductive on grid world because the doomed region is
empty. On ACAS Xu it is not: the kernel boundary detaches from the unsafe set,
and contracts must sit on $\partial V$ (refined by reachability / CEGAR), not
on every unsafe-adjacent state of the full syntactic domain.

---

## 2. Measured results (this map)

| Quantity | Value |
|---|---|
| Grid cells | 49 ($7 \times 7$) |
| Obstacles = Unsafe | 18 |
| $V = \mathrm{Safe}$ | 31 |
| Fixpoint rounds | 0 |
| Safe-but-doomed | 0 |
| Interior($V$) | 6 ($\|Allowed\| = 5$) |
| Boundary $\partial V$ | 25 |
| $\partial V$ contracts | **38** |
| $Allowed_V$ histogram over $V$ | 5: 6 · 4: 12 · 3: 13 |

Independent cross-check: `domain.obstacle_adjacent_cells()` (pure geometry)
equals `kernel.boundary` (from $Allowed_V$).

---

## 3. Code structure

Intentionally class-oriented (single responsibility, named collaborators):

| Type | Responsibility |
|---|---|
| `GridWorldDomain` | Bounds, obstacles, `simulate_step`, `obstacle_adjacent_cells` |
| `GridWorldContract` | One A/G triple (source, forbid, obstacle) |
| `GridWorldViabilityKernel` | Partition + `Allowed` + `contracts_from_boundary()` |
| `GridWorldInductiveProof` | Orchestrates the claims above; prints `[CHECK]` lines |
| `GridWorldContractVerifier` | CROWN discharge of kernel contracts (separate module) |
| `generate_contracts()` | Thin public entry for the CROWN / nuXmv pipeline |

Deleted: `generate_grid_world_contracts.py` (obstacle-walk generator). The pipeline
obtains contracts only through the kernel; `grid_world_contract_verifier.py`
imports `generate_contracts` / `load_config` from `grid_world_viability.py`.
Committed CROWN JSON results under `contracts/crown/` are unchanged and remain
valid: the 38 contract identities match what the kernel emits.

---

## 4. What this is *not*

- Not a claim that compositional verification is faster on grid world (it is not;
  see `2026_04_08_monolithic_vs_compositional.md`).
- Not continuous-mode work. Continuous goals still produce genuine UNSATs for
  100%-accurate nets; that is a training-distribution issue, not a kernel issue.
- Not a deletion of discrete pipeline results. Discrete SAT/UNSAT tables still
  stand; only the *derivation* of the contract list is unified.

---

## 5. Paper framing (one sentence)

> The assume-guarantee contracts used for grid-world NSBTs are the boundary
> contracts of the viability kernel in the special case where a stay action makes
> $V = \mathrm{Safe}$; ACAS Xu is the non-degenerate case of the same framework,
> where a nonempty doomed region forces contracts onto $\partial V \cap R$
> rather than onto the full syntactic crash set.

---

## 6. Limitation: CTLSPEC still fails under kernel contracts

The April 2026 compositional report (and `2026_04_08_monolithic_vs_compositional.md`)
recorded a standing gap: for the grid-world CTLSPEC

$$
\mathrm{AG}\bigl(\text{goal in obstacle}\ \lor\ \mathrm{AF}(\text{drone} = \text{goal})\bigr),
$$

the **monolithic** table model proves true on 100%-accurate networks, while the
**compositional** model proves false on every network.

**Kernel / inductive-invariant contracts do not close this gap.** They are still
safety contracts: on $\partial V$, forbid actions that leave $V$ (here, step into
an obstacle). In the patched SMV the NN remains a free variable constrained only
by those forbidden actions — an over-approximation of the real controller.

On this benchmark the hover theorem makes the over-approximation especially
loose: $Allowed_V(s)$ always includes stay (`XX`), and on interior cells it is
the full action set. nuXmv can therefore build infinite paths that never reach
the goal (stutter with stay, or wander among non-crashing moves). Those paths
are allowed by $V$ and by every $\partial V$ contract, so CTLSPEC remains false
under the compositional model even when the trained NN is a perfect progress
policy and the monolithic run says true.

| Spec | Monolithic (100% nets) | Old obstacle-walk contracts | Kernel $\partial V$ contracts (this work) |
|---|---|---|---|
| INVAR (no collision) | true | true (if all SAT) | true (same 38 contracts) |
| CTL (eventual goal) | true | **false** | **still false** |

What would be needed for compositional CTL is outside this framework: progress
or ranking contracts, near-deterministic pinning of the NN, fairness assumptions,
or a hybrid (safety compositionally, liveness monolithically). Viability answers
“can some policy stay safe forever?”; the CTLSPEC asks “does *this* network always
eventually reach the goal?” Those are different questions.

---

## 7. TL;DR

- **Hover theorem:** grid world has stay (`XX`), so $V = \mathrm{Safe}$, doomed
  region empty, fixpoint in 0 rounds; $\partial V$ = obstacle-adjacent cells.
- **Same 38 contracts, new derivation:** A/G crash contracts are exactly the
  kernel-boundary contracts; pipeline now emits them from
  `GridWorldViabilityKernel`, not a hand-rolled obstacle walk.
- **Unifies with ACAS Xu:** grid world = degenerate kernel (prior compositional
  safety approach recovered); ACAS = nonempty doomed set, contracts on
  $\partial V \cap R$ (corridor / CEGAR), not the full syntactic crash set.
- **Code:** `GridWorldDomain` / `GridWorldContract` /
  `GridWorldViabilityKernel` / `GridWorldInductiveProof` /
  `GridWorldContractVerifier`; deleted hand-rolled contract generator and the
  old function-bundle verifier / batch shell.
- **Not a speedup here:** still 38 contracts and the same discrete timing story
  as April; runtime wins live on ACAS corridor, not this map.
- **CTL still broken compositionally:** inductive invariants are safety-only;
  CTLSPEC (eventual goal) remains false under the free-NN abstraction, same as
  the April report.
