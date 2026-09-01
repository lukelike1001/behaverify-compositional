# ACAS Xu: Minimum Contract Set as a Min-Cut

**Date:** 2026-09-01
**Baseline (before):** `7196233` — *Flatten `pipeline/` and `contracts/` folders*
**Scope:** `core/safety/acas_contract_min_cut.py` (new),
`core/safety/acas_network_oracle.py` (new),
`scripts/discrete/run_acas_corridor_pipeline.py` (invocation fix).

**Headline:** the compositional pipeline satisfies the ACAS Xu safety invariant
and **matches the monolithic claim**, with the contract set now derived by
max-flow rather than by hand — and without consulting the monolithic verdict.

**Reproduce:**

```bash
cd REPRODUCIBILITY/2026_TBA/examples/AcasXu_closed_loop
python3 -m core.safety.acas_contract_min_cut          # 1.9 s
python3 scripts/discrete/run_acas_corridor_pipeline.py
```

---

## 1. Result

```
reachable states   = 9428
unsafe states in R = 1
abstract edges     = 47140
  real             =  9428   (capacity ∞)
  blockable (B)    = 37712   (capacity 1)
MIN CUT            = 1 contract
  a_prev=clear (7,6) signs=(+1,+1) h=10  forbid strong_right  (network picks clear)
```

| | Monolithic | Compositional |
|---|---|---|
| INVARSPEC | true | **true** |
| Contracts injected | — | **1** |
| nuXmv wall | 8.88 s verify (49.28 s cumulative with build) | 0.17 s |
| Peak RSS | 9.63 GB | 809 MB |

Both check the identical property string `system.distance_stage_1 >= 200`.

## 2. Why it matches (proof)

Abstract graph $G=(V,E)$: $V$ = augmented states reachable from the seed under a
free network, edge $u \xrightarrow{a} v$ for each advisory $a$ with $v=\text{step}(u,a)$;
$U$ = unsafe states. Split $E$ into $E_{\text{real}}$ (one edge per state, what the
network selects) and $B = E \setminus E_{\text{real}}$ (each a true never-select
property, hence CROWN-dischargeable).

**Theorem.** Injecting contract $(u,a)$ deletes edge $u \xrightarrow{a} v$, so
$A(C) \models \texttt{INVARSPEC}$ iff $C \subseteq B$ is an edge cut separating the seed
from $U$.

**Corollary (existence).** If monolithic satisfies the invariant, $C=B$ suffices:
deleting $B$ leaves out-degree 1 everywhere, so the only path from the seed is the
real trajectory, which never enters $U$. $|B| \approx 4|R|$ — that is the table
rebuilt as contracts, which is why the naive 8982-line run was correct in
principle and hopeless in practice.

**Corollary (minimality).** Capacity $\infty$ on $E_{\text{real}}$, $1$ on $B$;
max-flow min-cut returns the smallest dischargeable set. Infinite capacity on real
edges encodes "you cannot forbid what the network actually does", so a finite cut
never selects an unverifiable contract. The guard fires as a `ValueError` if an
$\infty$-capacity augmenting path exists — i.e. the real loop is genuinely unsafe.

**What this replaces.** The July corridor reached the same contract, but its
"at most two queries" theorem was proved by contradiction *against the monolithic
result*, making it circular as a standalone compositional proof. Max-flow gets
existence from the theorem above instead. The corridor's $|\partial V \cap R| = 1$
was a min-cut of size 1; max-flow finds it in 1.9 s and certifies it minimal.

## 3. Cross-check against 2025_NEUS

| Check | Outcome |
|---|---|
| ONNX networks | byte-identical MD5s to `2025_NEUS/.../networks/`, same tree positions |
| Tree template | 12 diff lines, all cosmetic (`x_var`→`x_mag`, filenames, tree name) |
| INVARSPEC | identical string in both models |
| `a_prev` → network | YAML matches tree selector (clear→1, weak_right→2, weak_left→3, strong_right→4, strong_left→5) |
| Output class order | tree classification == YAML `advisories`, so oracle argmax→advisory agrees with the SMV |
| Monolithic verdict | `000 :system.distance_stage_1 >= 200` → `[Invar True N/A N/A]` |
| Monolithic timing | `elapse: 40.40` build + `elapse: 8.88` verify = `total: 49.28` — the paper's own figures |

## 4. Open issues

**1. The pipeline builds the NN table, then deletes it.**
`dsl_with_contracts_to_nuxmv` calls `dsl_to_nuxmv` to generate the complete
monolithic SMV — ONNX enumeration included — and only then strips it
(`"nn_lines_removed": 8020` in this run's report). So compositional cost is
build-the-table **plus** delete it **plus** CROWN **plus** nuXmv. It avoids the
*model-checking* cost of the table, not its construction. Consequences: the 0.17 s
figure is nuXmv-only and not like-for-like against 49.28 s; and the architecture
**cannot run on a continuous domain at all**, which is the stated motivation for
the compositional approach. The memory result (9.63 GB → 809 MB) is unaffected and
stands. Fixing this means emitting the base SMV with the NEURAL variable free from
the start — a change in `src/`, not in this example.

**2. The monolithic baseline was never re-run here.**
`results/monolithic/pipeline_report.json` carries `"skipped_live_nuxmv": true` and
`smv_path: symbolic/smv/acas_360.smv`, which does not exist (the real artifact is
`symbolic/smv/acas_closed_loop.smv`). The verdict and timings are transcribed from
a genuine `2025_NEUS` nuXmv log, so nothing is fabricated, but the baseline is not
reproducible from this repo's own pipeline. One clean re-run would fix it.

**3. The oracle is forward-pass, not verified — and that is fine.**
`AcasNetworkOracle` uses ONNX argmax to label $E_{\text{real}}$. Monolithic's table is
built the same way (`dsl_to_nuxmv.py:1082-1132`), so the comparison is fair. The
compositional side is in fact stronger: those forward passes only *choose* which
contract to attempt, while the injected INVAR is licensed by a CROWN certificate.
A wrong oracle yields an UNSAT contract, not an unsound verdict.

**4. One unsafe state.** $|U \cap R| = 1$ out of 9428. The min-cut is 1 because the
instance has a single unsafe state behind a single blockable edge. This is
Limitation 9 restated: the method has not been shown to degrade gracefully where
the cut is large.

## 5. Relationship to earlier reports

**`2026_07_12_inductive_invariants.md`** — the corridor and its Q1/Q2 analysis.
Its *sufficiency* checks were already self-contained (direct model checks giving
`INVARSPEC TRUE` at $|R| = 9{,}225$ and $9{,}220$). Only its **termination theorem**
used the monolithic result; that step is **superseded** by the min-cut corollary.
The operative contract is unchanged — max-flow independently selects Q2.

**`2026_07_11_unreachable_states.md`** — reachability pruning via
`AcasReachableSet` is retained and is what makes the graph small enough to cut. Its
open question, whether pruning preserves `INVARSPEC=true`, is answered for this
instance by the cut being computed on the unpruned reachable graph.

**`2026_08_31_liveness_ctl.md` (grid world)** — unaffected. Contract counts there
are set by coverage, not by a cut; whether min-cut also reduces the 930 progress
contracts is untested.
