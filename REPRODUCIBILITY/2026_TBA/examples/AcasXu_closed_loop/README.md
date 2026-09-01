# ACAS Xu Closed-Loop Compositional Verification

Compositional verification for a closed-loop, 5-NN ACAS Xu Neuro-Symbolic
Behavior Tree (NSBT). Ownship selects one of five networks from the previous
advisory (`a_prev`), applies the advisory to heading, and steps the relative
geometry. The safety invariant is `distance >= 200` (raw units).

Background: *Neuro-Symbolic Behavior Trees and Their Verification*,
Serbinowska et al., NeuS 2025.

All commands below assume:

```bash
cd REPRODUCIBILITY/2026_TBA/examples/AcasXu_closed_loop
```

---

## Quick start — interactive contract explorer

No CROWN or nuXmv required:

```bash
pip install gradio   # once, if needed
python3 figures/image_scripts/acas_contract_explorer.py
# → http://localhost:7860
```

See [`figures/README.md`](figures/README.md) for the Gradio app.

---

## Architecture

Two product lines share the same process (generate → verify → patch → nuXmv)
with different algorithms:

| Line | Driver | Contracts | CROWN property |
|------|--------|-----------|----------------|
| **Safety** | `scripts/discrete/run_acas_safety_pipeline.py` | range A/G (never-selects) | forbidden advisory not selected |

ACAS Xu's canonical properties are input-output safety conditions; the
benchmark has no liveness specification, and this project does not invent one.
The empty `liveness/` folders exist for symmetry with grid world — see their
`DISCLAIMER.md`.

Shared plant and tooling live under `core/`. Drivers live under `scripts/`.

```
tree template  ──AcasTreeGenerator──►  tree/acas_closed_loop.tree
                                              │
                                         dsl_to_nuxmv
                                              ▼
                               symbolic/smv/acas_closed_loop.smv
                                              │
         contracts (JSON) ──AcasSmvContractPatcher──►  patched SMV ──nuXmv──► report
```

---

## Directory layout

```
AcasXu_closed_loop/
├── tree/
│   ├── acas_closed_loop_template.tree   # Template (REPLACE_* tables)
│   └── acas_closed_loop.tree            # Expanded tree (committed; regenerable)
├── symbolic/smv/
│   └── acas_closed_loop.smv             # Base SMV (generated; gitignored)
├── core/                                # Libraries (import from here)
│   ├── paths.py                         # EXAMPLE_ROOT anchor
│   ├── acas_domain.py                   # Plant physics (AcasDomain)
│   ├── acas_state.py                    # State / augmented state
│   ├── acas_reachability.py             # Reachability kernel R
│   ├── acas_viability.py                # Viability kernel V
│   ├── acas_contract.py                 # Safety contract types
│   ├── acas_smv_contract_patcher.py     # Inject SAT contracts into SMV
│   ├── acas_tree_generator.py           # AcasTreeGenerator
│   ├── acas_tree_parameter_extractor.py # YAML refresh from template constants
│   ├── acas_model_params.yaml           # Single source of truth (physics + catalogs)
│   ├── acas_verifier_params.yaml        # CROWN + SMV variable names
│   ├── safety/                          # Generator, CROWN verifier, min-cut, ONNX oracle
│   └── liveness/                        # DISCLAIMER.md only (see above)
├── scripts/
│   ├── run_acas_monolithic_pipeline.py  # Baseline (NNs stay in the SMV)
│   ├── discrete/
│   │   ├── run_acas_safety_pipeline.py  # Safety product line
│   │   ├── run_acas_corridor_pipeline.py      # Inductive corridor end-to-end
│   │   └── run_acas_inductive_invariant_check.py
│   └── continuous/
│       └── run_all_continuous_pipelines.sh    # Deferred continuous batch
├── contracts/
│   ├── continuous/                      # Frozen continuous CROWN results
│   └── discrete/safety/                 # Discrete specs + results (committed)
├── results/{discrete,monolithic,continuous}/  # Pipeline outputs (gitignored)
├── networks/                            # 5 ONNX models
└── figures/                             # Gradio explorer + figure docs
```

---

## Prerequisites

### BehaVerify + extras

From the repository root:

```bash
pip install -e .
pip install -r REPRODUCIBILITY/2026_TBA/requirements.txt
```

### nuXmv 2.1.0

Cannot be redistributed. Place under `REPRODUCIBILITY/2026_TBA/nuXmv_DL/`:

```bash
wget "https://nuxmv.fbk.eu/theme/download.php?file=nuXmv-2.1.0-linux64.tar.xz" \
    -O nuXmv_DL.tar.xz
tar -xf nuXmv_DL.tar.xz --one-top-level=nuXmv_DL --strip-components 1
chmod +x nuXmv_DL/bin/nuXmv
```

Pipelines default to `../../nuXmv_DL/bin/nuXmv` relative to this directory.

### alpha-beta-CROWN (only to re-verify contracts)

Pre-computed specs and many results are already under `contracts/`. Install CROWN
only to re-run or extend verification (see NeuS / TBA notes for pin `6b8bbcf`).

---

## Model overview

| Variable | Domain | Meaning |
|---|---|---|
| `x_mag`, `y_mag` | integers [0, 10] | Position magnitude (× distance_modifier = raw units) |
| `x_sign`, `y_sign` | {−1, +1} | Quadrant |
| `heading_own_var` | integers [0, 39] | Ownship heading index (× 9° = degrees) |
| `command` / `a_prev` | 5 advisories | Selects which NN runs |

`heading_int` is fixed at 225°; speeds are 20 (own) and 30 (intruder) from
`core/acas_model_params.yaml`.

```
a_prev = clear        → NN_1  (aprev_clear.onnx)
a_prev = weak_right   → NN_2
a_prev = weak_left    → NN_3
a_prev = strong_right → NN_4
a_prev = strong_left  → NN_5
```

Safety invariant:

```
INVARSPEC (distance >= 200)
```

with `distance = round(sqrt(x_mag² + y_mag²)) × distance_modifier`.

---

## Artifacts: tree and SMV

Both trees are **committed** (deterministic expansion). The base SMV is
**generated and gitignored** under `symbolic/smv/` (large NN lookup tables).

```bash
# Optional: regenerate expanded tree
python3 -m core.acas_tree_generator

# Optional: refresh tree-sourced YAML fields from the template
python3 -m core.acas_tree_parameter_extractor

# Base SMV (also done automatically by pipelines unless --skip-smv)
# Prefer a pipeline run, or generate via any driver without --skip-smv.
```

---

## Pipelines

### Monolithic baseline

Full closed-loop SMV with NN tables still inside. ~9.6 GB RSS if run live.

```bash
# Use 2025_NEUS reference timing/verdict (no huge RAM)
python3 scripts/run_acas_monolithic_pipeline.py --skip-monolithic

# Live nuXmv (~12 GB free RAM recommended)
python3 scripts/run_acas_monolithic_pipeline.py
```

Reference outcome: `INVARSPEC: true`, ~49 s user time, ~9.6 GB peak RSS.

### Discrete safety

```bash
# Symbolic only on pre-verified corridor contracts
python3 scripts/discrete/run_acas_safety_pipeline.py \
    --specs contracts/discrete/safety/safety_corridor_contracts.json \
    --results contracts/discrete/safety/safety_corridor_contract_results.json \
    --output results/discrete/safety/corridor \
    --skip-contracts --skip-tree --skip-smv

# Full discrete safety for one network (generate + CROWN + symbolic)
python3 scripts/discrete/run_acas_safety_pipeline.py \
    --output results/discrete/safety/nn1 \
    --network-idx 1 --discrete --run-crown
```

Patched SMV: `results/.../acas_closed_loop_safety.smv`.

### Inductive corridor (report-style)

```bash
python3 scripts/discrete/run_acas_corridor_pipeline.py
# → results/discrete/safety/corridor/pipeline_report.json
```

Runs discovery (`run_acas_inductive_invariant_check.py`), a seed CROWN check,
then the safety pipeline with corridor JSON.

### Continuous (deferred)

`scripts/continuous/run_all_continuous_pipelines.sh` batches continuous-domain
work. Frozen continuous CROWN results live under
`contracts/continuous/`. Prefer discrete pipelines for day-to-day runs.

---

## Library entry points

Prefer importing classes; modules also support `python3 -m` where noted.

```python
from core.acas_domain import AcasDomain
from core.acas_tree_generator import AcasTreeGenerator
from core.acas_tree_parameter_extractor import AcasTreeParameterExtractor
from core.acas_smv_contract_patcher import AcasSmvContractPatcher
from core.safety.acas_safety_contract_generator import AcasSafetyContractGenerator
from core.safety.acas_safety_contract_verifier import AcasSafetyContractVerifier
```

| Task | Entry |
|------|--------|
| Expand template → tree | `python3 -m core.acas_tree_generator` |
| YAML from template constants | `python3 -m core.acas_tree_parameter_extractor` |
| Safety specs / CROWN | `core/safety/...` (via safety pipeline or class API) |

---

## Interpreting results

| Field | Meaning |
|---|---|
| `steps.smv_patch.sat_contracts` | SAT contracts injected as INVARs |
| `steps.smv_patch.invar_lines` | INVAR lines added |
| `steps.smv_patch.nn_lines_removed` | NN table lines stripped from base SMV |
| `steps.nuxmv.invarspec` | `"true"` holds / `"false"` counterexample |
| `total_wall_sec` | End-to-end wall time |

**`INVARSPEC=false` with TIMEOUT contracts is usually spurious:** holes in the
abstraction let nuXmv invent paths. The verdict is meaningful when the relevant
contracts are SAT (same pattern as the grid-world example).

Compositional patched models are much smaller than the monolithic SMV
(~1.6k lines vs ~8k+ with full NN tables).

---

## Common issues

### Tree / SMV missing

```bash
python3 -m core.acas_tree_generator
# Then run any pipeline without --skip-smv, or use a driver that builds SMV.
```

### Monolithic OOM

`symbolic/smv/acas_closed_loop.smv` embeds five NN tables. Prefer
`--skip-monolithic` on machines with less than ~12 GB free RAM.

### CROWN `--retry-from` mismatches

Retry merges by contract `id`. If specs were regenerated, re-verify from
scratch rather than merging against an old results file.

### `INVARSPEC=None`

Check `results/**/nuxmv_output.txt`. Typical causes: SMV type errors (reuse a
known-good base with `--skip-smv`) or a missing nuXmv binary.

---

## Contract design (safety)

Range-based A/G contracts: for each non-empty group of
`(heading_own_var, x_sign, y_sign, forbidden_advisory)`, one CROWN call covers
the bounding box of dangerous `(x_mag, y_mag)` states.

| Grouping | Max contracts / NN | vs per-state |
|---|---|---|
| Per state | ~2,830 | baseline |
| Per heading+sign+advisory (default) | ~490 | ~6× reduction |
| Per sign+advisory only | ~100 | ~28× reduction |
