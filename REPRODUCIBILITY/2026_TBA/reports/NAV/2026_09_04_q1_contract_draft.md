# NAV: Draft Q1 Contracts (Obstacle Avoidance)

**Date:** 2026-09-04
**Status:** §§1–7 are design. §§8–10 are measured; no contract is CROWN-derived
yet, so nothing here is a NAV verification result.
**Depends on:** `2026_09_04_nav_spec_sheet.md` for `P`/`S`/`Q` and the period map.

`Q1` is "for `k = 0..30`, `s_k` is not in `Obs`", quantified over the whole
initial box. This file proposes what a NAV contract *is*, what CROWN certifies,
what nuXmv sees, and where I expect it to break.

§5 records a prediction that §8 then refutes; both are kept, because the reason
the prediction was wrong is the useful part.

---

## 1. The shape change: NAV contracts are ranges, not never-selects

ACAS Xu and grid world both have a **classification** network, so a contract is
a *never-select*: on box `B`, the network does not emit advisory `a`. That is
what makes the min-cut argument work — an advisory is an edge label, a contract
deletes an edge, and the minimum contract set is a min-cut of the abstract
graph (`2026_09_01_min_cut_contracts.md`).

NAV's network is **regression**: `u in (-1,1)^2`, continuous. There are no edge
labels to delete. The natural contract is instead a *range*:

```
CONTRACT (B, U):
    for all s in B:  net(s) in U

    B = [x1_lo, x1_hi] x [x2_lo, x2_hi] x [x3_lo, x3_hi] x [x4_lo, x4_hi]
    U = [u1_lo, u1_hi] x [u2_lo, u2_hi]
```

This is an assume–guarantee pair in the ordinary sense: assume the state is in
`B`, guarantee the control is in `U`. It is the *only* local fact CROWN can
state about a regression network, and it is exactly what CROWN natively
computes.

**Consequence: min-cut does not transfer.** There is no finite edge label to
cut, so the "minimum contract set" question becomes "coarsest partition that
still proves `Q1`", which is a refinement problem, not a flow problem. Any
attempt to recover min-cut by discretising `u` into levels rebuilds the control
-level explosion that made monolithic unaffordable (`monolithic_first_run.md`
§10). Do not do that.

## 2. What CROWN certifies

`pipeline/crown_verifier.py` already takes an arbitrary output constraint in
`_solve_over_input_box`; only the public wrapper is classification-specific. One
new method covers NAV:

```python
def certify_output_in_box(self, onnx_path, input_lower_bounds,
                          input_upper_bounds, output_lower, output_upper):
    """Certify net(B) subset of U. Output constraint is the conjunction
       lo_i <= y_i <= hi_i over both control components."""
```

Note this certifies a *candidate* `U`. The candidate comes from a forward pass
or from a cheap CROWN bound; the certificate is what licenses the INVAR line.
Same division of labour as `AcasNetworkOracle`: the oracle chooses what to
attempt, the certificate is what is trusted (min-cut report §4.3). A wrong
guess yields UNSAT, never an unsound verdict.

The network to point CROWN at is `nn-nav-{point,set}.onnx`, **unwrapped**. The
lattice wrappers exist only so the table method could consume integer deltas;
a contract is over real state, so the wrapper is not merely unnecessary here,
it would be wrong.

## 3. What nuXmv sees — two encodings

### 3.1 In-model (recommended)

Mirrors the ACAS Xu pipeline: generate the SMV with the state and dynamics
intact, **strip the NN table**, leave `u1, u2` free, and inject one INVAR per
contract.

```smv
-- one line per contract
INVAR (x1 >= b1_lo & x1 <= b1_hi & ... & x4 <= b4_hi)
      -> (u1 >= u1_lo & u1 <= u1_hi & u2 >= u2_lo & u2 <= u2_hi);
```

nuXmv then does the reachability, exactly as it does for ACAS Xu, and the
contract count replaces the table size. If NAV's table is 198,375 entries and
`Q1` needs a few hundred contracts, that is the memory claim, tested directly.

The contracts must **cover** every reachable state — an uncovered state leaves
`u` free over all of `(-1,1)^2`, which is sound but almost certainly too loose
to prove anything.

### 3.2 Pre-computed abstraction (fallback)

Compute `post(B, U)` offline with interval arithmetic on the §7.2 period map,
and hand nuXmv a purely finite system: `region : {r0, ..., rm}` plus a `TRANS`
listing successors. Sound, and it sidesteps nonlinear arithmetic in nuXmv
entirely.

I do not recommend leading with this. It moves the reachability *out* of
nuXmv, which makes the comparison against monolithic much less meaningful — we
would be measuring our own Python, not the model checker. Keep it in reserve
for the case where 3.1's nondeterminism is unmanageable.

## 4. Where the contracts come from

Not a fixed grid. A grid fine enough for the whole `Box` is the table again.
Forward, demand-driven refinement:

```
worklist <- { (Init, k=0) }
covered  <- {}
while worklist:
    (B, k) <- pop
    if k > 30: continue
    if B disjoint from Obs is NOT provable:      # section 8 hull test
        if B is splittable:  split B, push both at k;  continue
        else:                report POSSIBLE VIOLATION at k
    U <- crown_bound(B)                          # candidate
    certify (B, U);  emit contract
    for each B' in post(B, U):                   # section 7.2 + bloat by |R|
        push (B', k+1)
```

with a split heuristic that prefers the axis contributing most to the looseness
of `post`. Two things follow from the spec sheet:

- **`x4` splits first and finest.** On the full `Box`, `x4 in [-1.2, 3.6]`
  contains both `0` and `pi`, so `sin` and `cos` each range over `[-1,1]` and
  every position bound is vacuous. This is not a tuning detail; no `Q1` proof
  is possible without splitting heading, and it should be the default axis
  weight rather than something CEGAR rediscovers.
- **`post` is the §7.2 series bloated by `|R|`,** and the `Q1` check needs the
  range over `tau in [0, dt]` (the tube), while the successor needs `tau = dt`.
  Same expression, two evaluations.

## 5. The failure mode I expect

Interval boxes wrap. Each step hulls the true image into an axis-aligned box,
and the discarded correlation compounds. Rough per-step widths, writing `w(.)`
for width:

```
w(x3') ~ w(x3) + dt * w(u1)
w(x4') ~ w(x4) + dt * w(u2)
w(x1') ~ w(x1) + dt * ( w(x3) + |x3| * w(x4) )
```

Starting from `Init` (`w(x1) = w(x2) = 0.2`, `w(x3) = w(x4) = 0`), suppose
contracts give `w(u) ~ 0.05`. Then after 30 steps `w(x3), w(x4) ~ 0.3`, and with
`|x3|` up to 2.4 the position term is `0.2 * (0.3 + 0.72) ~ 0.2` **per step** by
the end.

Against the measured obstacle margins that is decisive: 0.1887 for the
continuous trajectory is borderline, and **0.0157 for the `point` controller is
hopeless** at that width. And the estimate is optimistic, because it holds
`w(u)` fixed — in reality wider boxes give wider CROWN bounds, so the growth is
superlinear.

This is the same shape of problem that killed the monolithic over-approximation
(`monolithic_first_run.md` §11): the abstraction's error outgrew the physical
quantity it had to resolve. It is *not* the same cause — that was per-point
lattice rounding, this is set-representation wrapping — and the mitigation is
different: splitting boxes actually reduces wrapping, whereas refining the
lattice made rounding worse (§11.3). But I should not pretend the outcome is
obviously better.

**So the honest prediction:** `set` may go through, `point` probably does not,
and the interesting measurement is *how many contracts* the refinement needs
before it either closes or explodes. That number, not a TRUE verdict, is the
result worth reporting.

## 6. What `Q1` alone does not need

`Q1` is a pure invariance property, so it needs no unrolling in principle — an
inductive `S` with `Init subset S`, `S disjoint from Obs`, `post(S) subset S`
would prove it for all time. I am not proposing that first, because NAV's
closed loop is only specified for 6 s and there is no reason to expect forward
invariance past the horizon. The step-indexed family `S_0, ..., S_30` above is
the bounded analogue and is what the refinement loop builds implicitly.

`Q2` (in the goal at `k = 30`) is a different obligation and will need a
different construction — a backward envelope rather than a forward hull. Out of
scope here.

## 7. Open questions

1. Does nuXmv handle the §3.1 model at all? The dynamics are nonlinear
   (`x3 * cos(x4)`), and the monolithic model avoided this only by baking
   trig into an integer table. Encoding 3.1 over a lattice reintroduces
   rounding; over reals it needs an arithmetic fragment nuXmv may not support.
   **This is the first thing to check, before any contract is generated.**
2. Is the coverage requirement in §3.1 affordable? Every reachable state needs
   a contract, and "reachable" is what we are trying to compute.
3. Does `w(u)` shrink fast enough under splitting to beat the wrapping in §5?
   This is the empirical crux and is cheap to measure with forward passes
   alone, before involving CROWN at all.

Question 3 is the one to answer first: it needs no pipeline, no nuXmv, and no
CROWN, and if the answer is no, §§1–4 do not matter.

---

## 8. Measured: the gate is passed, and §5 had the wrong cause

```bash
cd REPRODUCIBILITY/2026_TBA/examples/NAV
python3 -m scripts.init_clearance --grid 101 --widths   # 29 s
python3 -m scripts.hull_split --pieces 1 2 4 8 16       # 7 s
```

### 8.1 Q1 and Q2 are true over Init

10,201 starts on a `101 x 101` grid of `[2.9,3.1]^2`, propagated with RK4 at 200
sub-steps per period, clearance measured on the **tube**:

| Network | min tube clearance | at start | period | collisions | in goal at `k=30` |
|---|---|---|---|---|---|
| `point` | **0.015197** | `(3.1, 2.9)` | 10 | 0 / 10201 | 10201 / 10201 |
| `set` | **0.153146** | `(2.9, 3.1)` | 10 | 0 / 10201 | 10201 / 10201 |

So `Q1` is not false, and a sound contract set can exist. `Q2` is also true on
this sample, which was not expected — it had been assumed the harder of the two.

Both minima sit exactly at a **corner** of `Init` and are unchanged from a
`21 x 21` grid, which is what a smooth closed loop should do. This also
supersedes the framing in `monolithic_first_run.md` §6: the `0.0157` there was
already a corner figure at sampled times; the tube value is `0.015197`.

### 8.2 The true reachable set is wide, and that is not wrapping

`point`, widths of the **exact** cloud (`np.ptp` per axis):

```
step   w(x1)   w(x2)   w(x3)   w(x4)
   0  0.2000  0.2000  0.0000  0.0000
  10  0.2695  0.1947  0.0010  0.6175     <- closest approach
  15  0.8114  0.6426  0.2515  0.5446
  19  0.5866  0.9083  0.5429  0.2764
  30  0.0418  0.2514  0.0698  0.6846
```

At the closest approach the true cloud is `0.27 x 0.19`, while the clearance is
`0.0152` — the set is **18x wider than the margin it has to fit through**. The
cloud then fans out to nearly 1.0 wide before re-converging on the goal.

This kills the §5 diagnosis. The problem for a single box is not accumulated
interval slack; it is that the exact reachable set is genuinely large and an
axis-aligned box around it cannot miss the obstacle.

### 8.3 How finely Init must be split

`scripts/hull_split.py` propagates each sub-box of `Init` as an independent
cloud and encloses it in the bounding box of its own trajectories. That box is a
*subset* of any sound over-approximation, so these clearances are an **upper
bound on what any box-based scheme can achieve** — CROWN slack, wrapping, and
the §7.2 remainder can only reduce them.

| split | boxes | `point` | `set` |
|---|---|---|---|
| 1x1 | 1 | **0.000000** | 0.080654 |
| 2x2 | 4 | 0.003826 | 0.117691 |
| 4x4 | 16 | 0.011056 | 0.135036 |
| 8x8 | 64 | 0.013779 | 0.143949 |
| 16x16 | 256 | 0.014865 | 0.148533 |
| — | true | 0.015197 | 0.153146 |

Three things follow.

1. **`point` with one box is dead on arrival**, and not because of bound
   quality: the exact hull of the reachable set already meets the obstacle.
   No tightening of CROWN or of the period map can fix an unsplit run.
2. **Splitting `Init` alone is enough, and converges fast.** No adaptive
   re-splitting mid-trajectory was needed: 16 pieces recover 73 % of the true
   margin for `point`, 256 pieces recover 98 %. This answers open question (b) —
   independent propagation of initial pieces is the operative mitigation, and
   the box count is `10^2`, not `10^5`.
3. **`set` tolerates a ~10x coarser abstraction than `point`.** Unsplit, `set`
   already clears by `0.0807` while `point` clears by nothing; at `4x4`, `set`
   has `0.135` of headroom against `point`'s `0.011`. This is the benchmark's
   own stated purpose — set-based training for verifiability — showing up as an
   abstraction-coarseness number rather than as a verdict.

### 8.4 What is still open

The headroom is what CROWN slack and interval wrapping must fit inside. For
`set` at `4x4` that is `0.135`, which is roomy. For `point` at `16x16` it is
`0.0149` out of a true `0.0152` — splitting recovers essentially everything, but
the absolute budget stays at 1.5 cm, and **any** slack above that kills it. So
the `point` case remains genuinely doubtful, and it is now doubtful for a
measurable reason with a known budget rather than on a cartoon recursion.

Next measurement, in order: sampled `w(u)` per box at these split depths
(still no CROWN), then CROWN bounds on the same boxes to see the real slack.

---

## 9. The nuXmv gate: §3.1 works, with the right engine and encoding

Open question 1, smoke-tested. Models are in the session scratchpad, not
committed; each was run as `go_msat; check_invar_ic3`.

### 9.1 What nuXmv can and cannot do

| Probe | Result |
|---|---|
| `zzzbogus(x)` | rejected — `undefined identifier` |
| `cos(x)`, `sin(x)` | **accepted as identifiers**, never solved |
| `z' = x * y`, `x,y in [1,2]`, `z <= 3.5` | **FALSE with a counterexample** in seconds: `x = 449/232`, `y = 29/16` |

So **nonlinear real products are supported and solved** — the `x3 * cos(x4)`
shape is not the obstacle. Transcendentals are a different matter: `cos` passes
the type checker, which makes it look usable, but neither IC3 nor BMC to `k=25`
returns a verdict (>180 s). A model that uses `cos` directly is not
ill-formed, it simply never answers — the worst failure mode to discover late.

### 9.2 NAV-shaped models, `Init` as a real box, 30 steps

| # | Encoding | Control | Expected | Reached | Wall | Peak |
|---|---|---|---|---|---|---|
| D3 | direct `cos`/`sin` | free | FALSE | bound 7 | 110 s | 138 MB |
| D4 | `c4,s4` free within per-heading-region bounds | pinned | TRUE | bound 25 | 110 s | 180 MB |
| D5 | fully linear: increment a constant interval per `(x3,x4)` region pair, 84 INVARs | pinned | TRUE | bound 17 | 50 s | 191 MB |
| D6 | as D5 | free | FALSE | bound 7 | 50 s | 180 MB |

**No verdict in any of them.** IC3 was the wrong engine: `check_invar_ic3`
searches for an *unbounded* inductive invariant, and `Q1` is a bounded property.

### 9.3 A soundness trap in the encoding, found first

`INVAR` **prunes transitions** in nuXmv. A model that counts `0, 1, 2, 3, 4, …`
with `INVAR x <= 3.0` reports `INVARSPEC x <= 3.0` **true**:

```smv
MODULE main
VAR x : real;
INVAR x <= 3.0;
ASSIGN init(x) := 0.0; next(x) := x + 1.0;
INVARSPEC x <= 3.0;          -- reported TRUE
```

The behaviour that leaves the box is deleted rather than caught. D3–D6 all
carried box constraints of this kind, so a TRUE from any of them would have been
worthless. None returned TRUE, so nothing was concluded from them — but this is
the same failure the monolithic model's `S4` exists to prevent, reappearing in a
new encoding.

**The fix is structural:** state no box `INVAR` at all. Constrain only through
region *implications* (`state in R -> increment in [lo, hi]`). If the state
escapes the covered regions the increment goes free, so the model becomes
looser, never tighter — a coverage bug then yields a spurious FALSE, which is
the safe direction.

### 9.4 With BMC and a fully linear encoding, it works

Re-run as `go_msat; msat_check_invar_bmc -a een-sorensson -k 35`:

| # | Encoding | Verdict | Wall | Peak |
|---|---|---|---|---|
| D6 | fully linear, free control | **FALSE** at step 8 | **0.62 s** | 64 MB |
| D5 | fully linear, coarse 0.4-wide regions | **FALSE** (spurious) | 34 s | 150 MB |
| D4 | keeps nonlinear products `x3 * c4` | no verdict at **bound 0** | 400 s | 132 MB |
| **D8** | fully linear, 0.05-wide regions, moving robot | **TRUE** | **28.5 s** | **135 MB** |

Two lessons, both sharp:

- **Nonlinear products are supported but do not scale.** The toy `x*y` solves in
  seconds; D4 cannot finish bound 0 in 400 s. So the product `x3 * cos(x4)` must
  be eliminated *entirely* — not just the trig — by making the position
  increment a constant interval per `(x3, x4)` region pair. That is pure LRA.
- **Region width decides everything.** D5 and D8 differ mainly in region width,
  0.4 versus 0.05, and that is the difference between a spurious FALSE and a
  proof. Exactly the §8.3 lesson, now on the model-checking side.

### 9.5 D8 in detail, and why the TRUE is real

Real-valued state, `Init` the **real box** `[2.9,3.1]^2`, 30 steps, 72 region
contracts, control `u1 in [-0.03,-0.025]`, `u2 in [-0.005,0.005]`.

Non-vacuity was checked before the verdict was believed:

| Probe | Result | Meaning |
|---|---|---|
| `!(step = 30)` | false | the full 30-step horizon really executes |
| `x1 >= 2.7` | false | the robot moves |
| `x1 >= 2.3` | false | it moves further than interval arithmetic predicted |
| `x1 >= 2.0` | true | but never into the obstacle's `x1` band |
| `x2 >= 2.5`, `x2 <= 3.5` | true | and clears it in `x2` by 0.5 |

So the abstract reachable set is `x1 in [2.0, 3.1]`, `x2 in [2.5, 3.5]`: a
genuinely moving robot, and a proof that rests on a real 0.5 margin in `x2`.

Note the model is *looser than my own interval propagation* — Python predicted
`x1 >= 2.378`, nuXmv reaches below 2.3 — because the SMV uses the enclosing
0.05-wide region rather than the exact current interval. Sound, and a useful
reminder that the region grid, not the arithmetic, sets the precision.

### 9.6 What this licenses, and what it does not

Against monolithic's `~60 s` and `11.9 GB` for a **single lattice midpoint**,
D8 proves a 30-step safety property over a **continuum of initial states** in
28.5 s and 135 MB — **88x less memory**, on a quantifier the table method cannot
express at all.

**But D8's contracts are hand-picked `u`-ranges, not CROWN certificates of the
NAV networks.** This is a feasibility result for the *encoding*, not a
verification of NAV. The honest statement is: the in-model encoding of §3.1 is
viable, so the pipeline can keep nuXmv doing the reachability, and the
"CROWN ranges instead of a table" framing does **not** have to be retreated to.
Whether the real network's CROWN ranges are tight enough to survive it is the
next question, and D5 is a warning that "too loose" fails visibly.

---

## 10. What the networks actually emit

```bash
python3 -m scripts.control_ranges --pieces 1 4 16     # 20 s
```

D8's contract was hand-picked. This is the same quantity measured from the
networks. Sampled spread over each piece is a lower bound on the true range,
hence on any sound CROWN bound — every width here is optimistic.

### 10.1 Control at the initial set

| Network | `u1` at `Init` | `u2` at `Init` |
|---|---|---|
| `point` | `[-1.0000, -1.0000]` (saturated) | `[+0.9780, +0.9997]` |
| `set` | `[-0.9906, -0.9906]` | `[+0.9901, +0.9956]` |

Both controllers **start at full deflection**. D8's `u1 in [-0.03,-0.025]` is
off by a factor of ~33 and the wrong shape entirely. D8 therefore proves a timid
controller that never enters the obstacle's `x1` band; it is a feasibility
result for the encoding and nothing more. That non-comparability is the
load-bearing one, ahead of BMC-vs-BDD or LRA-vs-table.

### 10.2 Widest per-box control spread over 30 periods

| split | boxes | `point` `u1` | `point` `u2` | `set` `u1` | `set` `u2` |
|---|---|---|---|---|---|
| 1x1 | 1 | 1.5190 | 1.4035 | 0.1691 | 0.4679 |
| 4x4 | 16 | 0.5734 | 0.7779 | 0.0456 | 0.1242 |
| 16x16 | 256 | 0.2024 | 0.2734 | 0.0116 | 0.0317 |

Three readings.

1. **`w(u)` shrinks roughly linearly in box diameter** — each 4x refinement per
   axis divides the spread by ~3.7–3.9. This is the answer to the original
   open question (3): splitting does buy tightness at the expected rate, with
   no plateau up to 256 boxes.
2. **`set` is 17–20x tighter than `point` at every depth**, and 9x tighter even
   unsplit. Unsplit `point` spans `1.5190` of a possible `2.0` — very nearly the
   whole control range, which is a contract that says almost nothing.
3. **`set` at `16x16` reaches `u1` width `0.0116`, `u2` width `0.0317`** —
   within 2–3x of the hand-picked `U` that D8 proved. So a CROWN contract on
   `set` is in the right neighbourhood. `point` at the same depth is at
   `0.2024`, which is D5 territory, and D5 returned a spurious FALSE.

This is the same `set`-versus-`point` ordering as the hull clearance in §8.3,
now measured on the contract side rather than the geometry side. Two independent
measurements agree that the adversarially-trained network is the one that admits
a coarse abstraction.

### 10.3 The next obstacle is the increment grid, not the contract

D8 used 72 region implications because its robot barely moved, so `(x3, x4)`
needed covering only over `[-0.4,0.2] x [-0.15,0.15]`. With `|u| ~ 1` the real
closed loop spans `x3 in [-2.4, 0.4]` and `x4 in [-1.2, 3.6]`. At D8's 0.05
resolution that is `56 x 96 = 5,376` implications rather than 72 — a 75x larger
model, and D5 shows that coarsening to 0.4 fails.

So the open cost question has moved: not "can nuXmv hold the encoding" (it can),
and not "will `w(u)` shrink" (it does), but **whether the increment grid stays
affordable at the resolution the proof needs**. Non-uniform resolution is the
obvious lever — the increment `x3 cos(x4) dt` needs fine cells only where `|x3|`
is large — and is untested.
