# NAV: Where the ARCH-COMP Paper Disagrees With Its Own Artifacts

**Date:** 2026-09-03
**Scope:** `examples/NAV/` — comparison of ARCH-COMP 2025 AINNCS §3.11 (the
report text) against the files in the competition repository.
**Companion:** `2026_09_03_monolithic_first_run.md` §1.1 summarises this; here
is the full check with evidence.

**Summary.** The NAV files in `examples/NAV/` came directly from the ARCH-COMP
2025 repository and are not the problem. The *report prose* disagrees with the
shipped artifacts in three places, one of which silently breaks the controller
if you follow the text. **The artifacts win in all three.**

| # | Report says | Artifacts say | Severity |
|---|---|---|---|
| 1 | state is `(x, y, θ, ν)` — angle third | speed third, angle fourth | **breaks the controller** |
| 2 | "two hidden layers with 64 neurons each" | `64 -> 32` | reshapes the network |
| 3 | "adversarial training" | "set-based training" | naming only |

---

## 1. State variable order — the one that matters

**Report §3.11:** "The state is four-dimensional consisting of the horizontal
and vertical position `x, y`, the angle `θ` of the robot, and velocity `ν`."
So: angle is the third component, velocity the fourth. Equation (15) is an
unnamed 4-vector

```
ṡ = [ ν cos θ ;  ν sin θ ;  u(1) ;  u(2) ]
```

Bound to the report's English order, the third component is `θ`, so
**`θ' = u(1)` and `ν' = u(2)`**. Bound to `dynamics.m` it is the plant we run
(`ν' = u(1)`). The two bindings are not reconcilable by relabelling: they
disagree about which control drives which state.

**`dynamics.m`, shipped in the repository:**

```matlab
dx = [ x(3)*cos(x(4)); x(3)*sin(x(4)); u(1); u(2) ];
```

`x(3)` multiplies both `cos` and `sin`, so `x(3)` is the **speed** and `x(4)` is
the **angle** — the opposite of the prose. Consequently **`ν' = u(1)` and
`θ' = u(2)`**, also the opposite.

### Settled empirically

The initial set has `θ = ν = 0`, and the obstacle and goal constrain only
`x, y`, so no static reading of the specification distinguishes the two. Running
the closed loop does, immediately:

| Network | Ordering used | Final position at t = 6 | In goal |
|---|---|---|---|
| `set` | **code** (speed 3rd) | **(+0.10, −0.11)** | **yes** |
| `set` | paper (angle 3rd) | (+6.33, +2.81) | no |
| `point` | **code** (speed 3rd) | **(−0.14, +0.10)** | **yes** |
| `point` | paper (angle 3rd) | (+9.22, +0.61) | no |

Under the paper's ordering both controllers fly away from the goal. The networks
were trained against `dynamics.m`'s convention.

**Verdict: follow `dynamics.m`.** Our `.tree` does. Do not "fix" the network
input order to match the report's English.

**Likely cause.** The Unicycle benchmark (§3.3) in the same document *does* use
angle-third / speed-fourth:

```
ẋ1 = x4 cos(x3),  ẋ2 = x4 sin(x3),  ẋ3 = u2,  ẋ4 = u1 + w
```

NAV's prose appears to have inherited Unicycle's naming while its equation and
artifacts kept the other order.

---

## 2. Network architecture

**Report §3.11:** "Both networks have two hidden layers with 64 neurons each and
ReLU activation, with a final layer with tanh activation."

The layer *structure* is right — two ReLU hidden layers then a tanh output, as
the ONNX op sequence confirms:

```
MatMul, Add, Relu, MatMul, Add, Relu, MatMul, Add, Tanh
```

The *widths* are not. Both networks are `64 -> 32`:

| Source | Evidence |
|---|---|
| `nn-nav-{point,set}.onnx` | `fc_1_MatMul_W [4,64]`, `fc_2_MatMul_W [64,32]`, `fc_3_MatMul_W [32,2]` |
| `nn-nav-{point,set}.mat` | `W[0] (64,4)`, `W[1] (32,64)`, `W[2] (2,32)` |

ONNX and MATLAB agree with each other and with the set-based RL paper that
introduces this navigation task (arXiv 2408.09112: "two hidden layers of 64
and 32 neurons"). arXiv 2401.14961 is the supervised method paper the ARCH
README links; it does not state these widths.

**Verdict: `4 -> 64 -> 32 -> 2`.** This is cosmetic for us — we consume the
ONNX, so nothing in our pipeline depends on the report's number — but it is the
kind of figure that gets copied into a related-work table.

---

## 3. Training method naming

**Report §3.11:** "We used **adversarial training** to obtain a second, more
robust controller: During training, uncertainties of a given state are modeled
using sets, and the weights are updated based on the entire input set rather
than individual points."

**Repository `README.md`:** "The second network is trained **set-based** to
improve its verifiable robustness by integrating reachability analysis into the
training process."

The report's *description* is set-based training; only its opening label says
"adversarial". The file is named `nn-nav-set.onnx`. No technical
disagreement — worth noting only so the two names are not mistaken for two
different networks.

---

## 4. What agrees

Checked and consistent between report, README, and artifacts:

- initial set `x1, x2 ∈ [2.9, 3.1]`, `x3 = x4 = 0`
- obstacle `[1,2]²`, goal `[-0.5, 0.5]²`, both unconstrained in `x3, x4`
- control period 0.2 s, horizon 6 s
- two controllers, `point` (standard RL) and `set` (robust)
- output activation `tanh`, so `u ∈ (-1, 1)²` — the report writes the closed
  interval `[-1, 1]²`, which `tanh` never attains; immaterial

---

## 5. Impact on our work

**Discrepancy 1 would have been silent and fatal.** Our tree follows
`dynamics.m`, so the model is correct — but only because the MATLAB was read
before the prose. Had the input order been taken from the report, the network
would have received a permuted state, the controller would have driven the robot
away from the goal, and the model would still have generated, built, and
verified. The specifications would simply have come back `false`, and the
natural next move would have been to blame the discretisation.

**Discrepancy 2** affects nothing computational, since we load weights from
ONNX. It matters only for how the benchmark is described in writing.

**Discrepancy 3** affects nothing.

---

## 6. Recommendation

Report discrepancies 1 and 2 upstream to the ARCH-COMP AINNCS organisers.
Discrepancy 1 in particular is a trap for any tool that builds its plant from
the report text rather than from `dynamics.m`, and the failure is silent — a
wrong-but-plausible verdict, not an error.

Vanderbilt has a direct line here: Diego Manzanas Lopez is both a co-author of
the AINNCS report and a co-author of the 2025 NEUS paper this project extends.

---

## 7. Provenance

The architecture and variable-order discrepancies were first raised in an
independent review by Grok (xAI). Both were re-checked here against the ONNX
graphs, the `.mat` weights, and `dynamics.m` before being recorded, and the
variable-order question — which that review noted but did not settle — was
resolved by the closed-loop simulation in §1. The consequent disagreement about
which control drives which state (`θ' = u1` in the report versus `ν' = u1` in
the artifacts) is noted here for the first time.
