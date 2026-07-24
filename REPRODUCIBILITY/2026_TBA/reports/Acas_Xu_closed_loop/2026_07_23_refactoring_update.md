# ACAS Xu: Refactoring Update (Safety vs Liveness)

**Date:** 2026-07-23  
**Baseline (before):** `91b22f9` — *Pin abstract NN to closed-loop lasso for ACAS Xu*  
**After:** `005c263` — *Finish major refactor that splits ACAS Xu between safety and liveness*  
**Scope:** `REPRODUCIBILITY/2026_TBA/examples/AcasXu_closed_loop/`

---

## Why this happened

- Immediate trigger: plan to build a **liveness-lasso visualization** (Gradio / figures).
- On inspection, the example was a **flat dump** of scripts, shells, YAML, and analysis modules mixed at the root.
- Safety vs liveness shared vocabulary (“contract”, “pipeline”, “pin”) without clear product boundaries.
- Hard to know what to import, what was dead, or where new viz code should live → **structure first**, viz tomorrow.

---

## Design targets (what we aimed for)

- **Two product lines**, same process shape: generate → verify → patch SMV → nuXmv.
  - **Safety:** range A/G, CROWN *never_selects*, inductive corridor optional.
  - **Liveness:** equals contracts on a **lasso trajectory**, CROWN *always_selects*.
- **OOP over free-function dumps** — named classes (`AcasDomain`, generators, verifiers, patcher).
- **`core/` = libraries**, **`scripts/` = drivers** (not continuous/discrete under `core/`).
- **Strict YAML** for config knobs (no soft defaults in code for research params).
- **Shell → Python** for example drivers (master paper reproducer still future).
- No permanent “pin stack”; **lasso** kept only as the FM trajectory term.

---

## Before vs after — folder structure

### Before (`91b22f9`)

Everything co-located at the example root; contracts mixed by “goals” naming; template at root.

```
AcasXu_closed_loop/
├── README.md
│
├── # Plant / analysis (root free functions)
├── acas_model_params.yaml
├── acas_reachability.py
├── acas_viability.py
├── acas_inductive_analysis.py
├── acas_lasso_trajectory.py
├── acas_lasso_trajectory.json
├── acas_lasso_pins.py
│
├── # Tree
├── acas_template_360.tree
├── generate_acas_tree.py          → tree/acas_360.tree
├── extract_acas_constants.py
│
├── # Safety contracts (procedural)
├── generate_acas_contracts.py
├── verify_acas_contracts.py
├── verify_acas_contracts_parallel.py
├── verify_acas_contracts_config.yaml
├── verify_single_state.py
├── classify_contracts_by_reachability.py
│
├── # Pipelines / shells (root)
├── run_acas_compositional_pipeline.py
├── run_acas_liveness_pipeline.py
├── run_acas_monolithic_pipelines.sh
├── run_acas_corridor_pipeline.sh
├── run_all_continuous_pipelines.sh
├── verify_all_continuous_contracts.sh
├── verify_all_discrete_contracts.sh
├── retry_all_discrete.sh
│
├── contracts/crown/
│   ├── continuous_goals/{enabled_pgd,disabled_pgd,contract_specs_…}
│   ├── discrete_goals/aprev_*_crown_results.json
│   ├── corridor_contracts.json
│   ├── corridor_crown_results.json
│   ├── lasso_pin_specs.json
│   └── lasso_pin_crown_results.json
│
├── networks/          # 5 ONNX
├── figures/image_scripts/acas_contract_explorer.py
├── tree/              # generated acas_360.tree (gitignored)
└── symbolic/smv/      # generated acas_360.smv (gitignored)
```

### After (`005c263`)

Libraries under `core/`, drivers under `scripts/`, contracts by **safety | liveness**, trees committed under `tree/`.

```
AcasXu_closed_loop/
├── README.md
│
├── tree/
│   ├── acas_closed_loop_template.tree   # REPLACE_* geometric tables
│   └── acas_closed_loop.tree            # expanded (committed; deterministic)
│
├── symbolic/smv/
│   └── acas_closed_loop.smv             # base SMV (generated; still gitignored)
│
├── core/                                # importable libraries
│   ├── paths.py                         # EXAMPLE_ROOT
│   ├── acas_domain.py                   # AcasDomain (plant)
│   ├── acas_state.py
│   ├── acas_reachability.py
│   ├── acas_viability.py
│   ├── acas_contract.py                 # typed safety / liveness contracts
│   ├── acas_smv_contract_patcher.py     # shared SAT → INVAR patch
│   ├── acas_tree_generator.py           # AcasTreeGenerator
│   ├── acas_tree_parameter_extractor.py
│   ├── acas_model_params.yaml
│   ├── acas_verifier_params.yaml
│   ├── safety/
│   │   ├── acas_safety_contract_generator.py
│   │   ├── acas_safety_contract_verifier.py
│   │   └── acas_contract_reachability_classifier.py
│   └── liveness/
│       ├── acas_lasso_trajectory.py + .json
│       ├── acas_liveness_params.yaml
│       ├── acas_liveness_contract_config.py
│       ├── acas_liveness_contract_generator.py
│       └── acas_liveness_contract_verifier.py
│
├── scripts/
│   ├── run_acas_monolithic_pipeline.py  # baseline (NNs stay in SMV)
│   ├── discrete/
│   │   ├── run_acas_safety_pipeline.py
│   │   ├── run_acas_liveness_pipeline.py
│   │   ├── run_acas_corridor_pipeline.py
│   │   └── run_acas_inductive_invariant_check.py
│   └── continuous/
│       └── run_all_continuous_pipelines.sh   # deferred batch
│
├── contracts/crown/
│   ├── continuous/{enabled_pgd,disabled_pgd}/
│   └── discrete/
│       ├── safety/
│       │   ├── safety_full_contracts.json
│       │   ├── safety_corridor_*.json
│       │   └── archive/                 # old full-table discrete CROWN
│       └── liveness/
│           ├── liveness_contracts.json
│           └── liveness_contract_results.json
│
├── networks/
├── figures/
│   └── image_scripts/
│       ├── acas_contract_explorer.py
│       └── acas_visualization_common.py   # shared viz helpers (lasso viz later)
└── results/{discrete,monolithic,continuous}/   # pipeline outputs (gitignored)
```

---

## Mapping: old root → new home

| Before (root) | After |
|---|---|
| `acas_model_params.yaml` | `core/acas_model_params.yaml` |
| `acas_reachability.py` / `acas_viability.py` | `core/` (same names) |
| (implicit plant constants scattered) | `core/acas_domain.py`, `core/acas_state.py` |
| `generate_acas_contracts.py` | `core/safety/acas_safety_contract_generator.py` |
| `verify_acas_contracts.py` | `core/safety/acas_safety_contract_verifier.py` |
| `classify_contracts_by_reachability.py` | `core/safety/acas_contract_reachability_classifier.py` |
| `verify_acas_contracts_config.yaml` | `core/acas_verifier_params.yaml` |
| `acas_lasso_trajectory.*` | `core/liveness/` |
| `acas_lasso_pins.py` + pin JSON | **removed** (equals contracts + generator/verifier) |
| `generate_acas_tree.py` | `core/acas_tree_generator.py` (`AcasTreeGenerator`) |
| `extract_acas_constants.py` | `core/acas_tree_parameter_extractor.py` |
| `acas_template_360.tree` | `tree/acas_closed_loop_template.tree` |
| `tree/acas_360.tree` | `tree/acas_closed_loop.tree` (**committed**) |
| `symbolic/smv/acas_360.smv` | `symbolic/smv/acas_closed_loop.smv` |
| `run_acas_compositional_pipeline.py` | split → `scripts/discrete/run_acas_safety_pipeline.py` (+ shared patcher) |
| `run_acas_liveness_pipeline.py` | `scripts/discrete/run_acas_liveness_pipeline.py` |
| `run_acas_corridor_pipeline.sh` | `scripts/discrete/run_acas_corridor_pipeline.py` |
| `acas_inductive_analysis.py` | `scripts/discrete/run_acas_inductive_invariant_check.py` |
| `run_acas_monolithic_pipelines.sh` | `scripts/run_acas_monolithic_pipeline.py` |
| `run_all_continuous_pipelines.sh` | `scripts/continuous/` |
| `verify_all_*.sh`, `retry_all_*.sh`, parallel verifier | **removed** (drivers own CROWN stages) |
| SMV patch logic inside mega pipeline | `core/acas_smv_contract_patcher.py` |

---

## Structural changes (bullets)

### Product lines

- Safety and liveness share **process**, not one mega compositional script.
- Shared **`AcasSmvContractPatcher`**: strip NN tables, inject SAT equals/range INVARs.
- Monolithic path stays separate (no contract injection).

### OOP / config

- Domain, contracts, generators, verifiers, trajectory, tree expand as **classes**.
- Liveness: `AcasLivenessContractConfig.from_yaml` only; `nn_input_eps` (not pin_eps).
- Fail-fast YAML: required keys, no silent research defaults in code.

### Contracts layout

- `continuous_goals` / `discrete_goals` → `continuous/` / `discrete/{safety,liveness}/`.
- Corridor + full safety specs under `discrete/safety/`.
- Lasso pin files → liveness contract JSON naming.
- Old per-NN discrete full-table results → `discrete/safety/archive/`.

### Tree / SMV naming

- Dropped “360” branding for closed-loop artifact names.
- Template + expanded tree **committed** (deterministic; no “force generate” contradiction).
- `.gitignore`: keep `REPRODUCIBILITY/**/tree/**`, with **exceptions** for this example’s `tree/`.
- Base SMV still gitignored (large NN tables); regenerate via pipelines / dsl_to_nuxmv.

### Deletions (intentional)

- Pin stack / `acas_lasso_pins.py`.
- Parallel verifier + batch shell retry wrappers.
- Monolithic shell mega-benchmark (Python mono driver instead).
- Thin CLI shims that only wrapped a class.

---

## How you run things now

```bash
cd REPRODUCIBILITY/2026_TBA/examples/AcasXu_closed_loop

# Tree (optional; committed)
python3 -m core.acas_tree_generator

# Safety (corridor symbolic)
python3 scripts/discrete/run_acas_safety_pipeline.py \
  --specs contracts/crown/discrete/safety/safety_corridor_contracts.json \
  --results contracts/crown/discrete/safety/safety_corridor_contract_results.json \
  --output results/discrete/safety/corridor \
  --skip-contracts --skip-tree --skip-smv

# Liveness
python3 scripts/discrete/run_acas_liveness_pipeline.py --skip-tree --skip-smv

# Corridor end-to-end (discovery + seed CROWN + symbolic)
python3 scripts/discrete/run_acas_corridor_pipeline.py

# Monolithic baseline
python3 scripts/run_acas_monolithic_pipeline.py --skip-monolithic
```

---

## What this unlocks next

- **Liveness lasso viz app** — trajectory + equals contracts live under `core/liveness/`; shared plotting hooks can sit in `figures/image_scripts/` (e.g. `acas_visualization_common.py`).
- Clear import surface for notebooks / Gradio without root-path soup.
- Room for continuous product-line code later without re-flattening discrete.

---

## Explicitly deferred / out of scope

- Continuous full pipeline productization (shell batch only).
- Single master paper reproducer shell.
- Deep e2e pytest for full nuXmv/CROWN runs (JSON contract load tests exist under `REPRODUCIBILITY/2026_TBA/tests/`).
- Rewriting dated notes under `reports/` that still name old files (leave as history).

---

## One-line summary

**From a flat script pile to `core/{safety,liveness}` libraries + `scripts/{discrete,continuous}` drivers, with shared plant/patcher, renamed closed-loop tree/SMV artifacts, and committed deterministic trees — so liveness visualization (and everything else) has a place to land.**
