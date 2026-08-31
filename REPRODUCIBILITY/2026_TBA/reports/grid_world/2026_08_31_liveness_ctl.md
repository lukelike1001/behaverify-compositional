# Grid World: Progress Contracts Close the CTL Gap

**Date:** 2026-08-31
**Baseline (before):** `7196233` — *Flatten `pipeline/` and `contracts/` folders*
**Scope:** `examples/grid_world/core/` (new `liveness/` product line, `safety/` split),
`src/dsl_with_contracts_to_nuxmv.py` (set-membership injection), `tests/`.

**Headline:** the compositional pipeline now proves **CTLSPEC = true** on the 7×7
grid world for a 100 %-accurate network, matching the monolithic result. This
closes the divergence recorded in `2026_04_08_monolithic_vs_compositional.md`
and stated as Limitation 3 in `limitations_and_todos.md`.

**Reproduce:**

```bash
cd REPRODUCIBILITY/2026_TBA/examples/grid_world

# 1. discharge progress contracts with CROWN (~15 min)
python3 -m core.liveness.grid_world_liveness_contract_verifier \
    --onnx networks/1000__6_18_0__0200_1.onnx \
    --output contracts/discrete/liveness/1000__6_18_0__0200_1_liveness.json

# 2. inject + model check
python3 run_liveness_pipeline.py \
    --onnx networks/1000__6_18_0__0200_1.onnx \
    --output results/liveness/1000__6_18_0__0200_1 \
    --liveness-contracts contracts/discrete/liveness/1000__6_18_0__0200_1_liveness.json \
    --safety-contracts contracts/discrete/safety/1000__6_18_0__0200_1_discrete.json \
    --skip-contracts
```

---

## 1. The problem this addresses

Since April 2026 the compositional pipeline has reproduced the monolithic
INVARSPEC verdict exactly while returning **CTLSPEC = false for every network**,
including the five that are 100 % accurate and for which monolithic proves
CTL = true.

The cause was structural, not a bug. A safety contract has the shape

```
Assume:    (x_d, y_d) == source  AND  (x_g, y_g) IN G
Guarantee: argmax NN(x_d, y_d, x_g, y_g) != forbidden_dir
```

The guarantee is a **disjunction** — some action other than `forbidden_dir`
wins, and nothing says which. Under the hover theorem
(`core/safety/grid_world_viability.py`), `XX` maps every safe cell to itself, so
the greatest fixpoint deletes nothing, `V = Safe`, and `∂V` is exactly the
obstacle-adjacent cells. Stay is therefore never a contract antecedent: it is
allowed everywhere. The abstract model consequently contains a legal infinite
path on which the drone hovers forever — never crashing, never arriving. `AF
goal` is false of that model, correctly. The real network does not hover, which
is why monolithic says true.

No quantity of safety contracts fixes this. Hovering is already inside
`Allowed`, so adding crash edges on `∂V` constrains nothing about it. Closing
the gap requires a guarantee that says what the network **must** do, not only
what it must not.

The obvious candidate — pinning the network to one required action, as the
retired ACAS Xu lasso work did — is not that guarantee. Pinning
leaves the abstract network exactly one legal choice, so the abstraction has
nothing left to abstract; it is concretization rather than composition. The
design constraint adopted here follows from that: **the guarantee must be
set-valued**, leaving a choice wherever more than one action is acceptable, so
the abstract model stays genuinely non-deterministic.

---

## 2. The construction

### 2.1 Ranking function

Define, from grid geometry alone — no network involved:

```
dist(s, g)  shortest number of ticks from cell s to goal g through safe cells
Dec(s, g)   { a : dist(step(s, a), g) == dist(s, g) - 1 }
```

`dist` is computed by one backward breadth-first search per goal cell
(`core/liveness/grid_world_goal_distance.py`). Predecessors are built from
`GridWorldDomain.simulate_step` rather than raw grid adjacency, so border
clamping is handled correctly: a move that clamps is a self-loop and yields no
predecessor, and therefore can never masquerade as progress.

`dist` is a ranking function. Every action in `Dec` drops it by exactly one, it
is bounded below by zero, and the state space is finite. A controller confined
to `Dec` at every state therefore reaches `g` in at most `dist(s, g)` ticks.
That is the well-founded descent argument the CTL specification needs.

`XX` is never in `Dec`, since `simulate_step(s, "XX") == s` leaves `dist`
unchanged. Excluding stay is precisely what eliminates the hovering path.

This is a rediscovery of established ideas, and the report should say so:
`dist` is a Floyd-style ranking function, equivalently a discrete
control-Lyapunov function, and the backward BFS is the least-fixpoint attractor
dual to the greatest-fixpoint viability kernel already computed on the safety
side. Constraining a learned controller to a permitted action *set* is the
shielding / permissive-strategy formulation from supervisory control. What is
new is not the ranking — it is discharging the decrease condition as an A/G
contract on a neural network via a NN verifier, then composing the result into
a model checker. The citation gap flagged as Limitation 6 now applies to the
liveness side as well as the viability side.

### 2.2 Contract shape

```
Assume:    (x_d, y_d) == source  AND  (x_g, y_g) == goal
Guarantee: argmax NN(x_d, y_d, x_g, y_g) IN Dec(source, goal)
```

Two departures from the safety contract, both deliberate:

**The goal is pinned, not quantified.** A safety contract holds for every goal
("never crash, wherever the target is") and carries `goal_region = None`.
Progress is goal-relative: the same action that makes progress toward one
target moves away from another. Every liveness contract therefore names one
goal cell. `goal_region` lives on the shared `GridWorldContract` base rather
than on either subclass, because it is the half of the assumption both kinds
constrain — differently.

**The guarantee is set-valued.** `|Dec|` ranges from 1 to 4 on this map. Where
it exceeds one, the abstract network keeps a real choice.

### 2.3 Discharging membership without a new verifier primitive

alpha-beta-CROWN has no "argmax in set" query. It does not need one:

```
argmax NN IN Dec   ⟺   argmax NN != a   for every a not in Dec
```

Membership is the conjunction of never-select obligations over the complement.
One liveness contract therefore expands into `5 - |Dec(s, g)|` calls to
`certify_network_never_selects_class` — the same primitive the safety side
already uses. The verifier short-circuits on the first UNSAT, since one
forbidden action the network might still select is enough to break descent at
that state.

Both halves of the assumption are pinned to lattice points (`eps = 0`), because
the SMV guard matches exact cells. These are point queries: as strong as the
monolithic table at integer inputs, and no stronger.

### 2.4 SMV injection

`build_invar_lines` in `src/dsl_with_contracts_to_nuxmv.py` now dispatches on
record shape — `forbidden_dir` (safety, drone-guarded) versus `forbidden_dirs` +
`goal` (liveness, drone- and goal-guarded):

```
INVAR (system.drone_x_stage_0 = 4 & system.drone_y_stage_0 = 0
     & system.destination_x_stage_0 = 3 & system.destination_y_stage_0 = 5)
     -> system.network_stage_0 != left;
```

**Guarding on the goal is a soundness requirement, not a nicety.** Omitting it
would forbid an action at that drone cell for *every* target, when the same
action is required elsewhere. The function raises `ValueError` if a liveness
record arrives without goal variable names, and a test asserts all four
coordinates appear in every emitted line. The safety path is unchanged and
still guards on the drone cell only.

---

## 3. Instance numbers (7×7 grid, 18 obstacles)

| Quantity | Value |
|---|---|
| Safe cells | 31 of 49 |
| Progress pairs (`source != goal`, reachable) | **930** |
| Unreachable (source, goal) pairs | 0 |
| `\|Dec\|` range | 1 – 4 |
| Maximum `dist` | 12 |
| CROWN obligations, unmerged | **3596** |
| Monolithic table entries, for comparison | 2401 (`7^4`) |

Free space on this map is fully connected: zero unreachable pairs, so the CTL
specification's "target inside an obstacle" escape clause never fires for a
*safe* goal. It does fire for obstacle goals — see §5.

**The 3596 figure is the honest cost, and it exceeds the monolithic table.**
Full-coverage liveness contracts approach the table because liveness requires
the model to encode what the network *will* do, and pinning direction per
(position, goal) is what the table already is. Contracts with `|Dec| = 1`
forbid four of five actions and are the most concretization-shaped; they are
also where region merging will help least. Reducing this to a sub-table cost is
open work (§7).

---

## 4. Results

### 4.1 Headline: `1000__6_18_0__0200_1` (100 % accurate)

| Stage | Result |
|---|---|
| CROWN, liveness contracts | **930 SAT / 0 UNSAT / 0 TIMEOUT** |
| CROWN calls / wall time | 3596 / **14.5 min** |
| INVAR constraints injected | 968 (930 liveness + 38 safety) |
| SMV generation | 1.13 s, peak RSS 829 MB |
| nuXmv | **INVARSPEC = true, CTLSPEC = true** in **0.33 s** |

First compositional CTL = true on grid world; it agrees with the monolithic
verdict for this network. `1000__6_18_0__0100_1` replicates it end to end
(968 INVARs, INVARSPEC = true, CTLSPEC = true). The CROWN results are committed at
`contracts/discrete/liveness/1000__6_18_0__0200_1_liveness.json`; `results/` is
gitignored repo-wide, so the verdicts above are recorded here rather than as
artifacts.

The nuXmv time deserves a note. 968 INVAR constraints are checked in a third of
a second, against a monolithic CTL run that NEUS terminated after an hour on
the larger grid. That contrast is suggestive but is a property of this
instance, not a general speedup claim — the same caution as Limitation 5.

### 4.2 Forward-pass pre-check (all seven networks)

`--check-onnx` evaluates a network's argmax at all 930 pairs in seconds. It is
a prediction, not a proof — one forward pass per pair, no bound propagation.

| Network | Accuracy | Obeys `Dec` | Violations | of which stalls (`XX`) |
|---|---|---|---|---|
| `1000__…__0100_1` | 100 % | 930 / 930 | 0 | 0 |
| `1000__…__0150_1` | 100 % | 930 / 930 | 0 | 0 |
| `1000__…__0200_1` | 100 % | 930 / 930 | 0 | 0 |
| `1000__…__0250_1` | 100 % | 930 / 930 | 0 | 0 |
| `1000__…__0300_1` | 100 % | 930 / 930 | 0 | 0 |
| `0996__…__200_1` | 99.6 % | 902 / 930 | 28 | **24** |
| `0995__…__200_1` | 99.5 % | 925 / 930 | 5 | 2 |

CROWN confirmed the prediction exactly on all three networks run so far:

| Network | Forward-pass prediction | CROWN verdict |
|---|---|---|
| `1000__…__0200_1` | 930 obey `Dec` | **930 SAT / 0 UNSAT** |
| `1000__…__0100_1` | 930 obey `Dec` | **930 SAT / 0 UNSAT** |
| `0996__…__200_1` | 902 obey, 28 violate | **902 SAT / 28 UNSAT** |

The remaining four rows are still predictions.

**The imperfect networks are the interesting column.** 24 of `0996`'s 28
violations are the network outputting `XX` where progress was required — the
drone stalls. That is the "gets stuck" failure mode illustrated in NEUS Figure 5
with its counterexample trace in Figure 6, here surfacing as a *contract
violation naming a state* rather than as a model-checker trace to be read
backwards. CROWN certifies exactly those 28 as UNSAT, e.g.

```
106   source (4,0)  goal (0,5)  dist 9   require We     UNSAT
315   source (3,5)  goal (2,5)  dist 1   require We     UNSAT
```

Contract 315 is worth reading: the drone is one step from the goal and the
network will not take it.

This gives the two contract families complementary diagnostic roles — safety
contracts localize crashes, liveness contracts localize stalls.

### 4.3 Ablation: liveness contracts alone

| Configuration | INVARSPEC | CTLSPEC |
|---|---|---|
| liveness + safety (968 INVARs) | **true** | **true** |
| liveness only (930 INVARs, `--liveness-only`) | false | false |

The nuXmv counterexample for the ablation has **goal = (3,3), an obstacle
cell**, with the drone stepping (3,0) → (3,1) → (4,1) under an unconstrained
network. §5 explains why.

---

## 5. Coverage: what liveness contracts cannot reach

`dist(s, g)` is undefined when `g` is not a safe cell, so no progress contract
is generated for such goals. The environment picks a new target
nondeterministically over the whole grid — `destination_x ∈ [min_val, max_val]`
in `counter_template.tree`, obstacles included. Hence:

| (drone, goal) states the model can occupy | 1519 |
|---|---|
| Covered by liveness contracts | 930 |
| **Goal inside an obstacle — no contract** | **558** |
| Drone already at goal — no contract needed | 31 |

At those 558 states the abstract network is entirely free. The CTL
specification excuses them through its "target in Obs" disjunct; **INVARSPEC
has no such escape clause**. This is why liveness-only cannot establish the
invariant, and why its CTL also fails: once the drone crashes into an obstacle
it stands on a cell with no liveness contract at all, so it can wander or hover
forever without reaching any goal.

The decomposition (930 + 558 + 31 = 1519) is pinned in
`tests/grid_world/test_grid_world_liveness_contracts.py` so it cannot drift
silently.

### The structural claim

Safety and liveness contracts are neither alternatives nor redundant. They
compose in one direction:

> Liveness contracts are only stated on safe cells with reachable goals.
> Safety contracts are what keep the system inside the region where the descent
> argument applies.

This is assume-guarantee reasoning in the textbook sense — safety establishes
the invariant that liveness's assumption depends on. Neither family alone
yields both specifications. This is the most transferable finding in this
report, and a better framing for the paper than "we added a second contract
type."

It is also why `run_liveness_pipeline.py` injects both families by default and
keeps `--liveness-only` solely as an ablation, with its help text stating that
INVARSPEC is *expected* to fail there for coverage reasons unrelated to
progress.

---

## 6. What this does not establish

1. **One instance, one map.** 930 pairs, `|Dec| ∈ [1,4]`, fully connected free
   space. Nothing here shows how the construction degrades on a map with
   disconnected regions or a much larger grid, where the pair count grows as
   `(safe cells)²`.

2. **The obligation count exceeds the monolithic table.** 3596 CROWN calls
   versus 2401 table entries. This is a *correctness* result, not yet an
   efficiency one. Region merging — grouping contiguous `(source, goal)` blocks
   that share a `Dec` set into one bounding-box CROWN call, the technique ACAS
   uses to get 490 contracts instead of one per lattice point — is the path to
   sub-table cost and is not implemented.

3. **Discrete only.** `eps = 0` point queries. Continuous-mode liveness is
   untouched, and Limitation 4 (continuous is the strategic hole) is unchanged.

4. **Two networks end to end.** `0200` and `0100` have completed CROWN runs and
   nuXmv verdicts; `0996` has a CROWN run but no nuXmv verdict (its 28 UNSAT
   contracts are dropped at injection, so the run would measure coverage loss,
   not liveness). The remaining four rows of Table 4.2 are forward-pass
   predictions.

5. **No timing comparison against monolithic** for this configuration. The
   0.33 s nuXmv figure is not a like-for-like benchmark.

---

## 7. Relationship to earlier reports

Earlier reports are snapshots and stay as written. What follows records where
this report supersedes them.

**`2026_04_08_monolithic_vs_compositional.md` — CTL divergence.** That report
observed compositional CTL = false for all seven networks against monolithic
CTL = true for the five accurate ones, and attributed the divergence to
over-approximation. The diagnosis was correct. Its conclusion that the
divergence is inherent to compositional A/G contracts is **superseded**: it is
inherent to *never-select* contracts specifically. A set-valued progress
guarantee recovers CTL = true without determinizing the abstraction.

**`limitations_and_todos.md` — Limitation 3** ("Framework is safety-inductive;
CTLSPEC stays open compositionally"). **Resolved for grid world, discrete
mode.** Its diagnosis — that kernel/`∂V` contracts only constrain forbidden
actions and that "progress/ranking contracts or another abstraction are
required" — was exactly right, and this work implements the first branch of
that prediction. The limitation stands for continuous mode and for ACAS Xu.

**Limitation 6** (classical viability / controlled-invariance citation gap)
**widens**: it now covers ranking functions, discrete control-Lyapunov
functions, attractor computation, and shielding, as noted in §2.1.

**ACAS Xu liveness retirement.** `2026_08_25_liveness_removal.md` is a stub —
the decision is made, its write-up is deferred until after this report, and the
ACAS `core/liveness/` files are still on disk. Nothing here depends on that
write-up. What carried over from that line of work is negative and
methodological — the observation that equality pins are concretization rather
than composition is what motivated the set-valued guarantee here (§1). Its
`certify_network_always_selects_class` primitive in `pipeline/` remains
uncalled; the grid-world construction needed set-membership, which decomposes
into never-select instead.

**`2026_07_14_grid_world_kernel_unification.md`.** The `GridWorldDomain` /
`GridWorldContract` / `GridWorldViabilityKernel` decomposition it introduced is
intact, redistributed into `core/` and extended with a
`GridWorldSafetyContractGenerator` so that `contracts_from_boundary` no longer
lives on the kernel and the safety and liveness sides are symmetric. Partition
numbers unchanged: `|V| = 31`, `interior = 6`, `∂V = 25`, 38 contracts, 0
fixpoint rounds.

---

## 8. Code map

```
core/
  grid_world_domain.py                     GridWorldDomain, action tables, physics
  grid_world_contract.py                   GridWorldContract (base, owns goal_region)
                                           GridWorldSafetyContract   (never_selects)
                                           GridWorldLivenessContract (in_set)
  safety/
    grid_world_viability.py                GridWorldViabilityKernel  (gfp)
    grid_world_safety_contract_generator.py
    grid_world_safety_contract_verifier.py
  liveness/
    grid_world_goal_distance.py            dist / Dec                (lfp)
    grid_world_liveness_contract_generator.py
    grid_world_liveness_contract_verifier.py
    grid_world_liveness_params.yaml        eps, timeouts, expected shape
run_liveness_pipeline.py                   CROWN → merge → SMV → nuXmv → report
```

`grid_world_viability.py` and `grid_world_contract_verifier.py` at the example
root are gone: the first split into domain / contract / viability modules, the
second moved to `core/safety/` as `GridWorldSafetyContractVerifier` so that both
contract kinds keep generator and verifier side by side under `core/<kind>/`.
`contracts/` was reorganized to mirror ACAS Xu — `continuous/{enabled,disabled}_pgd/`
and `discrete/{safety,liveness}/`. `src/dsl_with_contracts_to_nuxmv.py`:
`build_invar_lines` gained the set-membership branch and optional `goal_x` /
`goal_y` parameters; the never-select path is unchanged.

**Tests** (`tests/`, run with `python3 -m pytest tests/ -q` from `2026_TBA/`; not
collected by the repository-root suite, whose `testpaths` is the top-level
`tests/`): 42 passing. The descent argument is executed over all 930 pairs —
`Dec` totality, exact-one decrease, greedy descent arriving in `dist` steps,
`XX` exclusion, clamping never counting as progress. `tests/example_imports.py`
resolves a collision between the two examples' identically-named `core`
packages, which would otherwise make the two test suites mutually exclusive in
one pytest session.
