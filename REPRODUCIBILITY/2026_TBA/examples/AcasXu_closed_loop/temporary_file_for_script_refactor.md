# Design: ACAS compositional drivers (safety vs liveness vs symbolic)

## Recommendation: hybrid of Option 1 + Option 2 — not Option 3

### Why not pure Option 3 (one mega `run_acas_compositional_pipeline`)

- Safety and liveness share **process** (generate → CROWN → patch → nuXmv) but not **algorithms** (never_selects vs always_selects, corridor/full vs trajectory pins, optional CTL).
- One file with mode flags re-creates the April god-script: hard to read, hard to test, easy to break one product line fixing the other.
- Reviewers and you both benefit from intention-revealing entrypoints (`…safety…` vs `…liveness…`).

### Why not pure Option 1 alone (only safety e2e, no shared symbolic story)

- End-to-end safety is correct **as a product driver**.
- Duplication risk with liveness is real **only if** both re-implement SMV surgery / nuXmv wiring.
- That risk is already mitigated by **libraries**: `AcasSmvContractPatcher`, `NuxmvVerifier`, generators/verifiers.

### Why not pure Option 2 alone (only symbolic script)

- Correct **architectural** split (neural ahead of time vs symbolic).
- Incomplete **product** story: “how do I go from scratch?” still needs something that runs generate + verify + symbolic.
- Symbolic-only is a **mode** (`--skip-contracts` / pre-verified inputs), not the only entrypoint.

### Best practice shape for this repo

```text
Libraries (no “run the paper” claim):
  AcasSafetyContractGenerator / Verifier
  AcasLivenessContractGenerator / Verifier
  AcasSmvContractPatcher
  NuxmvVerifier, PipelineReportWriter

Product drivers (scripts/, thin):
  run_acas_safety_pipeline.py     # generate → CROWN → patch → nuXmv → report
  run_acas_liveness_pipeline.py   # already close; align naming/stages

Optional convenience (same safety script flags, not a third philosophy):
  --skip-tree / --skip-smv / --skip-contracts
  # --skip-contracts ≈ today’s “patch-only compositional” use case
```

Optional separate `run_acas_symbolic_verification.py` **only if** both drivers grow fat calling patcher; until then **Option 2 is a layer, not a mandatory extra file**.

### What to do with today’s `run_acas_compositional_pipeline.py`

| Action | Rationale |
|--------|-----------|
| **Delete or replace** after a real safety e2e exists | Name lies; only stage 3 + tree/smv |
| **Do not** keep “compositional” as the name of patch-only | Compositional = full NSBT method |
| Tree/SMV ensure | Shared helper or copy short blocks from mono script |

### Naming

- Prefer **`run_acas_safety_pipeline.py`** for the full safety product line.
- Prefer **`run_acas_liveness_pipeline.py`** for liveness (keep; slim to call patcher only for SMV).
- Reserve **“compositional”** for docs (“compositional verification”) or a future master `reproduce_all` that invokes both—not one ambiguous CLI.

### Summary

| Option | Verdict |
|--------|---------|
| 1 Safety e2e rename | **Yes** as product driver shape |
| 2 Symbolic-only file | **Yes as concern**; optional as extra script |
| 3 Merge safety+liveness | **No** |
| Hybrid | **Best**: two product drivers + libraries; flags for skip stages |
