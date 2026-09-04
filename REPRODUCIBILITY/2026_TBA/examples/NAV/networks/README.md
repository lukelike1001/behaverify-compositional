# NAV Networks

## Upstream (unmodified, from ARCH-COMP 2025)

| File | Training |
|---|---|
| `nn-nav-point.onnx` / `.mat` | standard point-based reinforcement learning |
| `nn-nav-set.onnx` / `.mat` | set-based training for verifiable robustness |

Both are `4 -> 64 ReLU -> 32 ReLU -> 2 Tanh` (the AINNCS report says
64 each; the artifacts disagree -- see `reports/NAV/2026_09_03_paper_mismatch.md`). Inputs are the real-valued state
`(x1, x2, x3, x4)`; outputs are the two controls `u in (-1, 1)^2`.

## Lattice wrappers (generated)

`nn-nav-{point,set}-p{P}v{V}h{H}.onnx` wrap the upstream networks so
BehaVerify's table method can consume them. `P`, `V`, `H` are the position,
speed, and heading lattice scales; the scales are baked into the graph, so a
wrapper and a `.tree` from different configurations must never be paired, and
encoding them in the filename makes a mismatch a missing-file error rather than
a silent wrong answer.

Build them with `python3 -m core.nav_wrapper_builder`, which reads the scales
from `core/nav_domain_config.yaml` and verifies each wrapper against the
original over 1000 random lattice points.

They exist because of one hard constraint:

> `dsl_to_nuxmv.py` stores regression outputs as `int(output)`. NAV's `tanh`
> outputs lie in `(-1, 1)`, so **every control would truncate to 0** and the
> lookup table would be inert.

Each wrapper adds two things around the untouched original graph:

1. **Input** `Div` by `[P, P, V, H]`, so the model takes integer lattice
   coordinates and feeds the original network real state.
2. **Output** `Mul` by `[V*dt, H*dt]` then `Round`, so the model returns the
   **integer state delta** on each lattice. `int()` is then exact. The two
   factors differ because speed and heading live on separate lattices.

The output factors also set the **control resolution**: the delta ranges over
`{-round(S*dt), ..., +round(S*dt)}`, so `V*dt = 1` gives a three-level
controller. See the `D` measurement in
`reports/NAV/2026_09_03_monolithic_first_run.md`.

The wrapper therefore reads "lattice in, lattice delta out", and the `.tree`
passes plain variables (`x1, x2, x3, x4`) exactly as grid world does.

The opset is raised from 8 to 13, because `Round` requires opset 11 or later.
No weights are modified.

Regenerate with `python3 -m core.nav_wrapper_builder`.

## Equivalence

`core.nav_wrapper_builder` checks each wrapper against the original over 1000
random lattice points on every build and prints the mismatch count. All
committed wrappers report 0.
