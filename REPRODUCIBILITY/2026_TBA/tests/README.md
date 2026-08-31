# Compositional BehaVerify Tests

Fast, dependency-light tests for the pieces the pipeline's correctness rests on:
contract records, the viability partition, and the liveness ranking function.
Nothing here calls CROWN or nuXmv — those stay in the pipeline scripts.

## Running

These are **not** collected by the repository-root `pytest` (root
`pyproject.toml` sets `testpaths = ["tests"]`, which is the top-level suite).
Run them from `REPRODUCIBILITY/2026_TBA/`:

```bash
python3 -m pytest tests/ -q
```

## Layout

```
tests/
├── example_imports.py    # activate_example(): picks which example's core/ package is live
├── conftest.py           # puts tests/ on sys.path so example_imports is importable
├── grid_world/
│   ├── test_grid_world_safety_contracts.py   # viability partition, ∂V contracts, JSON parity
│   └── test_grid_world_goal_distance.py      # dist(s,g) / Dec(s,g) descent argument
└── AcasXu_closed_loop/
    └── test_acas_contract.py                 # typed contract specs, JSON round-trips
```

## The `core` package collision

Both examples ship their own top-level `core` package
(`examples/grid_world/core/` and `examples/AcasXu_closed_loop/core/`). Python
caches the first one imported under the name `core`, so without intervention
whichever test module imported first would decide what `core.*` means for the
whole session, and the other example's tests would fail with
`ModuleNotFoundError`.

Every test module therefore calls `activate_example("<name>")` **before** its
`core.*` imports. It drops the cached package and re-points `sys.path`, so test
file ordering stops mattering. Add the same call at the top of any new test
module that imports from an example.

## Conventions

- Assertions that pin a specific number (930 progress pairs, 38 contracts,
  3596 unmerged obligations) are pinning claims that reports and the paper
  cite. If a legitimate change moves one, update the test and the report in
  the same commit.
- Docstrings say *why the property matters*, not what the code does — these
  tests double as the executable form of the descent and hover arguments.
