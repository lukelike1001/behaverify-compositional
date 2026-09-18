# NAV Specification Sheet

What §3.11 asks, what the monolithic run discharged, and what compositional
should target next. These are three different obligations. They are named `P`,
`S`, and `Q` here so that no sentence can quietly promote one to another.

This file is the reference for anyone encoding the NAV plant. It fixes the
plant, the period map, and the two places where a sound test is not an exact
one.

---

## 1. Notation

```
state       s = (x1, x2, x3, x4)
              x1, x2 = position          (paper's x, y)
              x3     = speed             (paper's nu)
              x4     = heading, radians  (paper's theta)

control     u = (u1, u2) = net(x1, x2, x3, x4),   u in (-1, 1)^2
net         4 -> 64 ReLU -> 32 ReLU -> 2 tanh

dt = 0.2    control period
N  = 30     periods, N * dt = 6 s
k  = 0..N   period index,  t_k = k * dt
```

The state ordering follows `dynamics.m`, **not** the AINNCS prose, which lists
`theta` third. The architecture follows the `.onnx`, **not** the prose, which
says 64/64. Both are settled in `2026_09_03_paper_mismatch.md`.

## 2. Plant

```
d/dt x1 = x3 cos(x4)        d/dt x3 = u1
d/dt x2 = x3 sin(x4)        d/dt x4 = u2
```

with `u` held constant on `[t_k, t_{k+1})`. The hold is the competition's
reading of "`sdot = f(s,u)`" plus "trained with a sampling time of 0.2 s"; §3.11
does not write it as a sentence, and this file does not claim it does.

## 3. Regions

```
Init  = { x1 in [2.9, 3.1], x2 in [2.9, 3.1], x3 = 0, x4 = 0 }
Obs   = { x1 in [1.0, 2.0], x2 in [1.0, 2.0] }      x3, x4 free
Goal  = { x1 in [-0.5, 0.5], x2 in [-0.5, 0.5] }    x3, x4 free
Box   = [-1.0, 3.4]^2 x [-2.4, 0.4] x [-1.2, 3.6]   (ours, not the paper's)
```

## 4. P — the paper obligation

```
for all s(0) in Init, with s(.) the solution of the plant in section 2:

    P1   for all t in [0, 6]  :  s(t) not in Obs
    P2                        :  s(6) in Goal
```

Both are safety properties in the Alpern–Schneider sense: each is refuted by a
finite prefix. `P2` is a bounded-time state constraint, not liveness, and is
*stricter* than `F_{<=N} Goal` — arriving early and leaving does not satisfy it.

## 5. S — the encoding obligation (what monolithic discharged)

```
s(0) = (3.0, 3.0, 0, 0)                       midpoint, not Init
s_{k+1} = s_k + dt * f(s_k, net(s_k))         one forward-Euler step
                                              on the integer lattice

    S1   for k = 0..30 :  s_k not in Obs
    S2                 :  s_30 in Goal
    S4   for k = 0..30 :  s_k strictly inside Box

    S3   exists k in 0..30 : s_k in Goal       DIAGNOSTIC ONLY
```

`S3` is strictly weaker than `S2` and is not a NAV result. `S4` is not in the
paper; it is the price of clamping a finite box, and exists so the clamp cannot
hide a violation. Neither belongs on the ARCH scoreboard.

**Gaps, S vs P:**

| | S | P |
|---|---|---|
| 1. quantifier | one point | all of `Init` |
| 2. time domain | 31 samples | the tube `t in [0,6]` |
| 3. plant | forward Euler | the flow of §2 |

## 6. Q — the compositional target

Identical to `S` except the initial set is all of `Init`, as one real box.

```
    Q1   for k = 0..30 :  s_k not in Obs
    Q2                 :  s_30 in Goal
    Q4   for k = 0..30 :  s_k inside Box, or Box removed
```

**Q closes gap 1 only. Q is not P.** It is nonetheless the first claim the
table method structurally cannot make: a continuum of starts, rather than an
enumeration of lattice points that says nothing about the continuum between
them.

---

## 7. The exact period map

Gap 3 is a modelling choice, not a necessity: NAV's period map is elementary.
On `[t_k, t_{k+1})` the control is held, so with

```
a = x3(t_k)     b = u1      c = x4(t_k)     d = u2      tau in [0, dt]
```

speed and heading are exact,

```
x3(t_k + tau) = a + b*tau
x4(t_k + tau) = c + d*tau
```

and position needs

```
I_cos(tau) = integral_0^tau (a + b*s) cos(c + d*s) ds
I_sin(tau) = integral_0^tau (a + b*s) sin(c + d*s) ds
```

### 7.1 Closed form — correct, and unusable over a box

```
d != 0:
    F_cos(s) =  (a + b*s) sin(c + d*s) / d  +  b cos(c + d*s) / d^2
    F_sin(s) = -(a + b*s) cos(c + d*s) / d  +  b sin(c + d*s) / d^2
    I(tau)   = F(tau) - F(0)

d == 0:
    I_cos(tau) = cos(c) * (a*tau + b*tau^2 / 2)
    I_sin(tau) = sin(c) * (a*tau + b*tau^2 / 2)
```

Both differentiate back to the integrand. The `1/d^2` makes this unusable on
any box containing `u2 = 0`, and NAV's does. Exactness at `d = 0` is not
stability near it. **Use §7.2 instead.**

### 7.2 Series with an explicit remainder — the form to encode

Expanding `cos(c + d*s)` and `sin(c + d*s)` in powers of `d*s` and integrating
termwise. With

```
M_{n+1}(tau) = integral_0^tau (a + b*s) s^n ds
             = a * tau^(n+1) / (n+1)  +  b * tau^(n+2) / (n+2)
```

then for any truncation degree `n`:

```
I_cos(tau) = sum_{j=0..n} (d^j / j!) * cos(c + j*pi/2) * M_{j+1}(tau)  +  R
I_sin(tau) = sum_{j=0..n} (d^j / j!) * sin(c + j*pi/2) * M_{j+1}(tau)  +  R

|R| <= |d|^(n+1) / (n+1)!  *  Mbar_{n+2}(tau)

    where Mbar_{n+2}(tau) = |a| * tau^(n+2)/(n+2)  +  |b| * tau^(n+3)/(n+3)
    is M_{n+2} with |a|, |b| in place of a, b
```

using `|d^j/dc^j cos(c)| <= 1`, so the Lagrange remainder of the integrand is
bounded by `(|d|*s)^(n+1)/(n+1)!` and is then *integrated against* `(a + b*s)`
rather than maximised over the interval. The coarser
`(|a| + |b|*tau) * tau * (|d|*tau)^(n+1)/(n+1)!` is also valid and is about
`n+2` times looser. `n = 0` recovers the `d = 0` formulas. The same bound holds
for `I_sin`.

**This is a theorem, not an `O(.)`.** The remainder is computable, so the
truncated map is a *sound* enclosure of the exact one, not an approximation
of it.

### 7.3 Choosing the degree

The expansion parameter is `d*s`, and on NAV `|d| = |u2| < 1` with
`s <= tau <= 0.2`, so `|d*s| < 0.2` **everywhere on the benchmark** — the
factorial decay is uniform, not a small-`d` special case. On the modelled box
`|a| <= 2.4`, `|b| <= 1`, `|d| <= 1`, `tau <= 0.2`:

| `n` | `|R|` per period | x30 periods | coarse bound, per period |
|---|---|---|---|
| 2 | 1.7e-4 | 5.1e-3 | 6.9e-4 |
| 3 | 6.8e-6 | 2.1e-4 | 3.5e-5 |
| 4 | 2.3e-7 | 6.9e-6 | 1.4e-6 |
| 5 | 6.5e-9 | 2.0e-7 | 4.6e-8 |

Against the measured obstacle margins — **0.1887** for the continuous
trajectory, **0.0157** for the `point` controller — `n = 4` leaves the tighter
margin intact by a factor of ~2300.

> **Caveat.** The right-hand column is 30 x the per-period bound. That is only
> the *directly injected* error. It ignores amplification through the
> controller, since a perturbed state feeds a perturbed `net`. A total bound
> needs a Lipschitz argument on the closed loop and is not established here.
> Take `n = 4` as a well-motivated starting degree, not a proven end-to-end
> error bound.

### 7.4 Why this is cheaper for a region encoding

The closed form is trig of `(c + d*tau)` over `d^2`: awkward to bound over a
box, and singular where the box meets `u2 = 0`. The truncated series is a
**polynomial in `(a, b, d, tau)`** multiplied by `sin(c)` and `cos(c)`, with
`c` in no denominator. Bounding it over a box is polynomial interval arithmetic
plus two monotone trig ranges.

Two costs, neither an objection:

1. Bounding the polynomial and the trig factors independently discards their
   correlation with `u`. CROWN can recover part of this.
2. If the heading box is wide, `sin` and `cos` both range over `[-1, 1]` and
   the enclosure goes slack. This is not hypothetical here: `Box` gives
   `x4 in [-1.2, 3.6]`, which contains both `0` and `pi`, so on the full box
   the trig ranges *are* `[-1, 1]`. **Any region decomposition for NAV must
   split `x4` more finely than the other axes.**

---

## 8. Tube avoidance is a curve–set test

`P1` on one period asks

```
for all tau in [0, dt]:
    NOT ( x1(t_k + tau) in [1,2]  AND  x2(t_k + tau) in [1,2] )
```

a 2-D curve against an axis-aligned square. `P2` needs only `tau = dt` at
`k = 30`; the tube is `P1` alone.

**Sound, and what a region encoding can run.** The hull test: avoided if

```
    range(x1 over [0,dt]) disjoint from [1,2]
 OR range(x2 over [0,dt]) disjoint from [1,2]
```

This is exactly disjointness of the curve's *interval hull* from the square. It
is conservative — it rejects a curve that rounds a corner while both coordinate
ranges straddle `[1,2]`. It must not be relabelled exact.

Two implementation notes. The ranges are over **`tau in [0, dt]`, not the
endpoint** — `P1` is the tube, so a period map evaluated only at `tau = dt`
does not feed this test. And the range must be bloated by the `|R|` of §7.2,
which keeps the whole test sound with a truncated series inside it.

**Geometrically exact, and not available to contracts.** Split `[0, dt]` at the
crossings of `x1 = 1`, `x1 = 2`, `x2 = 1`, `x2 = 2`; between consecutive
crossings continuity alone confines the curve to one of the nine slabs the four
lines cut the plane into. But those crossing times are roots of
`I_cos(tau) = const`, which is transcendental — there is no closed form. This
is 1-D numerical root finding along a single trajectory, so it is a
**simulation and diagnostic** technique, not something CROWN emits.

---

## 9. Summary for whoever encodes the plant

- Target `Q` first. It closes gap 1 and nothing else, and saying otherwise is
  the specific failure this file exists to prevent.
- Closing gaps 2 and 3 means: §7.2 with the remainder as the period map, §8's
  hull (or a tighter curve enclosure) for `P1`, and `tau = dt` only for `P2`.
- Neither the exact period map nor a curve enclosure fits in a lookup table
  over reals. That, not speed, is the argument for compositional.

## 10. Provenance

Derived from `reports/NAV/ARCH25_Navigation.pdf` §3.11 and the ARCH-COMP 2025
AINNCS artifacts (`dynamics.m`, `nn-nav-{point,set}.onnx`). Margins in §7.3 are
measured in `2026_09_04_monolithic_first_run.md` §11. Paper/artifact conflicts
are in `2026_09_03_paper_mismatch.md`. Sections 7 and 8 were reviewed against an
independent check by Grok; the remainder bound, the hull-vs-exact distinction,
and the `x4` splitting requirement come from that exchange.
