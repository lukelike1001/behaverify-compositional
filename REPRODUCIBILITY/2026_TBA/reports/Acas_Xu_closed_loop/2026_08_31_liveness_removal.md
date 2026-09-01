# ACAS Xu: Removing the Liveness Files

**Date:** 2026-08-31

**Scope:** This report overrides the liveness decision introduced in the `2026_07_15_liveness_parity.md` report and all future reports that talk about ACAS Xu liveness. The ACAS Xu benchmark does not have a liveness specification. Therefore, we should not invent one.

Files Affected: (TBA should fill in)


---

## Files Affected

**Baseline:** `7196233`. Net: **4208 deletions, 45 insertions** across 23 files.

### Deleted outright

| File | Lines | Role |
|---|---|---|
| `core/liveness/acas_lasso_trajectory.py` | 268 | ONNX closed-loop walk to the 52-state lasso |
| `core/liveness/acas_lasso_trajectory.json` | 522 | Committed lasso dump |
| `core/liveness/acas_liveness_contract_verifier.py` | 299 | CROWN `always_selects` discharge |
| `core/liveness/acas_liveness_contract_generator.py` | 231 | Equals-pin emission |
| `core/liveness/acas_liveness_contract_config.py` | 151 | YAML loader |
| `core/liveness/acas_liveness_params.yaml` | 18 | eps, expected stem/total, timeouts |
| `core/liveness/__init__.py` | 0 | package marker |
| `scripts/discrete/run_acas_liveness_pipeline.py` | 282 | Liveness driver |
| `contracts/discrete/liveness/liveness_contracts.json` | 1880 | 52 equals specs |
| `contracts/discrete/liveness/liveness_contract_results.json` | 374 | CROWN verdicts |

`results/discrete/liveness/` (gitignored) was also emptied of its SMV and nuXmv
artifacts.

### Edited

| File | Change |
|---|---|
| `core/acas_contract.py` | `AcasLivenessContract` removed (−91); `AcasSafetyContract` untouched |
| `core/acas_smv_contract_patcher.py` | `guarantee_type` dispatch dropped; every INVAR is now `not_equals` |
| `figures/image_scripts/acas_contract_explorer.py` | `Reach_true` lasso overlay removed — trace, checkbox, wiring, JSON load |
| `figures/image_scripts/acas_visualization_common.py` | stale reference to a nonexistent `acas_lasso_explorer.py` |
| `core/paths.py`, `core/acas_artifact_builder.py`, `core/safety/*.py` | comment references |
| `README.md`, `contracts/discrete/safety/archive/README.md` | product-line table, tree, run instructions, class API |
| `tests/AcasXu_closed_loop/test_acas_contract.py` | two liveness tests removed; `crown_input_bounds` repointed at the safety corridor spec |

### Retained deliberately

- **`liveness/` folders with `DISCLAIMER.md` only** — `core/`, `contracts/discrete/`,
  `results/discrete/`. They preserve symmetry with grid world, where liveness is
  real, and they document the absence rather than leaving a reader to wonder.
- **`pipeline/crown_verifier.py::certify_network_always_selects_class`** —
  untouched. It lives in the shared verifier adapter, not in this example, and
  is generic infrastructure whose only caller happened to be ACAS liveness.

### Verification

40 tests pass (`python3 -m pytest tests -q` from `2026_TBA/`), down from 42 by
exactly the two deleted liveness tests. Safety-side imports, the SMV patcher's
INVAR emission, and the Gradio explorer all still work.

### Note on the grid world contrast

Removing liveness here is not a retreat from liveness in general. Grid world
gained working progress contracts the same day
(`reports/grid_world/2026_08_31_liveness_ctl.md`), and the negative lesson from
this line of work — that equality pins are concretization rather than
composition — is what shaped their set-valued guarantee. ACAS Xu is the wrong
benchmark for liveness because its canonical properties φ1–φ10 are input-output
safety conditions and its authority comes from being a fixed, agreed-upon set.
Grid world has a published CTL specification to target; ACAS Xu does not.
