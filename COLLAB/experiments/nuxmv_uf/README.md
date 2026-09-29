# nuXmv has uninterpreted functions, and is faster than UCLID5 at using them

The UCLID5 detour was motivated by one belief: that only UCLID5 could abstract
the network as an uninterpreted function constrained by CROWN contracts, because
nuXmv would have to *encode* the network and die of ReLU phase-splitting.

**That belief was wrong.** nuXmv supports `FUN`:

```smv
FUN ctrl : real -> real;
INVAR (px >= 10.0 & px <= 13.0) -> (ctrl(px) >= -1.0 & ctrl(px) <= 0.0);
```

Verified as a genuine uninterpreted function:

| check | expected | result |
|---|---|---|
| congruence `(a=b) -> f(a)=f(b)` | true | `is true` |
| otherwise unconstrained `f(a)=0` | false | `is false` |

## Head-to-head, identical experiment

| contracts | UCLID5 k-induction | nuXmv `msat_check_invar_bmc` | speedup |
|---|---|---|---|
| 10 | 1.64 s | 0.02 s | 82x |
| 200 | 2.76 s | 0.06 s | 46x |
| 1400 | 7.39 s | **0.63 s** | **12x** |

Non-vacuous: dropping **one** of the 1,400 contracts flips the verdict to
`is false`, so every contract is load-bearing.

## What this means

The architecture — contracts constraining an abstract control function, rather
than contracts injected into a model that encodes the network — is what mattered.
**UCLID5 was not.** nuXmv does the same job an order of magnitude faster, and it
already has the CTL support UCLID5 lacks.

The UCLID5 work was still worth doing: it is where the architecture was found,
and it independently confirmed the abstraction is sound. But a direct
DSL-to-UCLID5 converter would be building a slower second path to the same place.
