# ACAS Xu: Refactoring Addendum (Flattening Paths)

**Date:** 2026-07-24
**Baseline (before):** `0925d1a` — *Add refactoring report to document why we refactored and what changed*
**Scope:** `REPRODUCIBILITY/2026_TBA/pipeline/`, both examples' `contracts/`, and the
three ACAS pipeline drivers under `scripts/`.

Follow-on to `2026_07_23_refactoring_update.md`. That refactor built the
`core/` vs `scripts/` split; this one removes directory levels that were
holding space for capabilities this branch no longer has, and finishes one
piece of duplication the 07/23 pass left behind.

---

## 1. `pipeline/` flattened

```
pipeline/neuro/crown/crown_verifier.py     →  pipeline/crown_verifier.py
pipeline/symbolic/nuxmv/nuxmv_verifier.py  →  pipeline/nuxmv_verifier.py
```

The `neuro/` and `symbolic/` levels existed to hold alternatives — `neuro/nnv/`
and `symbolic/uclid5/` — that were cut when this research direction narrowed to
the CROWN + nuXmv pair. What remained was three directory levels with exactly
one file each. Seven import sites updated (including two in `grid_world/`).

`pipeline/__init__.py` now records why it is flat and what would justify
re-nesting, so the mix-and-match intent survives the deletion:

> A compositional pipeline is a neural verifier paired with a symbolic one.
> This branch supports exactly one of each, so they sit flat. Earlier branches
> nested them under `neuro/` and `symbolic/` to hold alternatives (NNV,
> UCLID5); restore that nesting if a second verifier of either kind comes back.

## 2. `contracts/crown/` → `contracts/`

Same reasoning, same cause: the `crown/` level distinguished CROWN artifacts
from other neural verifiers' artifacts that no longer exist.

```
AcasXu_closed_loop/contracts/        grid_world/contracts/
├── continuous/{disabled,enabled}_pgd/    ├── continuous_goals/{disabled,enabled}_pgd/
└── discrete/{safety,liveness}/           └── discrete_goals/
```

**40 files moved** (all recorded as renames by git), **19 files** with path
references rewritten across Python, YAML, shell, README, and tests.

grid world's `*_goals` naming is deliberately untouched — renaming those to
match the ACAS `{safety,liveness}` split is a structural change, and grid world
is frozen. Only the path prefix moved.

## 3. `AcasArtifactBuilder` (finishing 07/23)

`_ensure_tree` and `_ensure_smv` had been copy-pasted near-verbatim into all
three drivers — `run_acas_safety_pipeline.py`, `run_acas_liveness_pipeline.py`,
`run_acas_monolithic_pipeline.py`. Extracted to `core/acas_artifact_builder.py`
(168 lines), removing 171 lines from `scripts/`.

Both stages are unchanged in behavior: generate unless the artifact exists and
reuse is requested. Drivers now open with

```python
builder = AcasArtifactBuilder(tree_path=DEFAULT_TREE, smv_path=DEFAULT_BASE_SMV)
tree_metrics = builder.ensure_tree(reuse_existing=args.skip_tree)
smv_metrics = builder.ensure_smv(reuse_existing=args.skip_smv)
```

Three things the merge surfaced:

- **Metrics had drifted.** Only the safety copy collected `tracemalloc`,
  `peak_rss_kb`, and `smv_lines`; the other two collected none of it despite
  being copies of the same function. The unified stage always collects the
  superset, so liveness and monolithic reports now carry SMV-generation figures
  they were silently missing. Comparisons against pre-07/24 reports should not
  read the appearance of these fields as a change in the pipeline.
- **`peak_rss_kb` is process-wide.** It is
  `ProcessMemory.peak_self_rss_kilobytes()` — the whole driver's high-water mark
  at the moment of the call, not the SMV stage's cost, despite sitting in the
  SMV stage's metrics dict. Preserved as-is to keep the report schema stable;
  worth fixing when the reporting is next touched.
- **A latent path bug.** `tree_path.relative_to(EXAMPLE_ROOT)` raises
  `ValueError` for any path outside the example root — harmless while these
  were module constants, a real trap now that they are constructor parameters.
  Switched to `os.path.relpath`, identical for in-tree paths.

## 4. Finding: the base SMV is semantically, not byte, deterministic

Regenerating `symbolic/smv/acas_closed_loop.smv` from the committed tree
produces a file that differs from the existing one in exactly three hunks:

| Hunk | Difference |
|---|---|
| line 12 | `CONSTANTS` enum member order |
| line 8154 | `command_stage_0 : {…}` advisory domain order |
| lines 8218–8249 | `composite_sequence_without_memory_2` MODULE block, byte-identical, emitted 19 lines later |

Both files are 1,686,089 bytes and 8,285 lines — conclusively a permutation,
not a content change. SMV enum domains are unordered and `MODULE` declaration
order is irrelevant, so the two models are the same model.

Cause is unsorted set iteration in the vendored `src/dsl_to_nuxmv.py` (it does
`sorted(list(set(...)))` in places but not for the enum/constant collections),
combined with `PYTHONHASHSEED` randomization for strings. The exact line was
not pinpointed.

**The tree is byte-deterministic** — regenerating reproduced the committed
`acas_closed_loop.tree` exactly, so the 07/23 determinism claim holds for the
artifact it was making the claim about.

**Unmeasured caveat.** Enum declaration order can influence nuXmv's BDD
variable ordering, so the monolithic 49.3 s / 9.19 GB figures could drift
between SMV regenerations. Not measured, and not claimed to occur — but it is
the first thing to check if those numbers ever move without a model change, and
it is an argument for keeping `2025_NEUS/.../invar.txt` as the citable baseline
rather than re-deriving it.

## 5. Verification

- 6/6 tests pass. `test_acas_contract.py` built its path from segments
  (`_EXAMPLE / "contracts" / "crown"`), so the string-level rewrite missed it
  and six tests failed immediately — the one failure mode a `sed`-driven path
  refactor reliably has, caught at once. Fixed, and `_CROWN` renamed
  `_CONTRACTS`.
- All ACAS drivers plus grid world's `--help` cleanly; pyflakes clean.
- Both YAML-configured liveness paths resolve; all 15 committed contract JSONs
  parse.
- Base SMV regenerated end to end through the new builder (540 s, 8,285 lines).

## 6. Deliberately unchanged

- `reports/` still names `contracts/crown/` and the nested `pipeline/` paths.
  Dated reports are history and are not rewritten.
- `results/discrete/safety/corridor/pipeline_report.json` records the old
  absolute contract paths. It is a record of a run that happened when those
  paths were real; rewriting it would falsify the record.
- grid world's `*_goals` contract naming (see §2).
- `core/safety/acas_safety_contract_verifier.py` (589 lines, largest file in
  the tree) — splitting it is plausible but discrete experiments are still
  running through it, so churn there is unwelcome.

## 7. Open items

Artifact-evaluation gaps, in rough value order — these matter more than further
refactoring, since ACM/EAPLS *Reusable* badging is about structure and setup,
and the structure is now largely in place:

1. **No container.** `2025_NEUS/` has a `Dockerfile`; `2026_TBA/` does not. The
   07/11 install pain (uninitialized `auto_LiRPA` submodule, `abcrown` never
   actually installed, silent user-level torch downgrade) is exactly what a
   container exists to absorb.
2. **Dependencies uncaptured.** `requirements.txt` covers gradio and plotly,
   unpinned. The versions that determine whether the numbers reproduce — torch,
   the CROWN commit, nuXmv 2.1.0 — are pinned nowhere.
3. **Master reproducer script**, still deferred from 07/23.
4. **The two empty CTL rows** in the 07/15 three-way table (monolithic, and
   safety-only corridor) — single reruns of existing code.
5. SMV byte-reproducibility, if wanted: either generate in a subprocess with
   `PYTHONHASHSEED=0`, or sort the collections upstream in `dsl_to_nuxmv.py`.

---

## One-line summary

**Removed two directory levels that were holding space for verifiers this
branch no longer has (`pipeline/neuro|symbolic/`, `contracts/crown/`), finished
the driver de-duplication 07/23 left behind as `AcasArtifactBuilder`, and
recorded that the base SMV regenerates equivalently but not byte-identically.**
