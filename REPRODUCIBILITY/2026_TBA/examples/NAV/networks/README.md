# NAV Networks

## Upstream (unmodified, from ARCH-COMP 2025)

| File | Training |
|---|---|
| `nn-nav-point.onnx` / `.mat` | standard point-based reinforcement learning |
| `nn-nav-set.onnx` / `.mat` | set-based training for verifiable robustness |

Both are `4 -> 64 ReLU -> 64 ReLU -> 2 Tanh`. Inputs are the real-valued state
`(x1, x2, x3, x4)`; outputs are the two controls `u in (-1, 1)^2`.

## Lattice wrappers (generated)

`nn-nav-{point,set}-lattice-s5.onnx` wrap the upstream networks so BehaVerify's
table method can consume them. They exist because of one hard constraint:

> `dsl_to_nuxmv.py` stores regression outputs as `int(output)`. NAV's `tanh`
> outputs lie in `(-1, 1)`, so **every control would truncate to 0** and the
> lookup table would be inert.

Each wrapper adds two things around the untouched original graph:

1. **Input** `Div` by the lattice scale `S = 5`, so the model takes integer
   lattice coordinates and feeds the original network real state.
2. **Output** `Mul` by `S * dt = 1.0` then `Round`, so the model returns the
   **integer state delta** on the lattice. `int()` is then exact.

The wrapper therefore reads "lattice in, lattice delta out", and the `.tree`
passes plain variables (`x1, x2, x3, x4`) exactly as grid world does.

The opset is raised from 8 to 13, because `Round` requires opset 11 or later.
No weights are modified.

Regenerate with the snippet recorded in
`reports/NAV/2026_09_02_monolithic_first_run.md`.

## Caveat

The wrappers have **not** yet been checked for output equivalence against the
originals over a random sample. Until they are, treat any verdict obtained
through them as provisional.
