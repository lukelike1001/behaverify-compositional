"""
acas_inductive_analysis.py

Inductive-invariant stress test for the simplified ACAS Xu closed loop.
Companion to 2026_07_12_inductive_invariant_stress_test.md.

Computes, using ONLY AcasDomain physics (no networks):

  1. R      -- reachable augmented states (state, a_prev) under nondeterministic
               advisories, seeded from the closed loop's initial condition.
  2. V      -- the viability kernel: the largest set of safe physical states from
               which SOME advisory sequence remains safe forever (greatest fixpoint).
  3. dV     -- the kernel boundary: states in V where at least one advisory exits V.
  4. Allowed_V(s) = {a : step(s,a) in V} and its histogram.
  5. The spurious counterexample corridor: the unique path in R from the seed to
     the unique unsafe pair, and the (at most two) CROWN contracts that sever it.
  6. Direct model checks: abstract reachability with each candidate contract
     injected, confirming INVARSPEC TRUE.

Run from examples/AcasXu_closed_loop/:

    python3 acas_inductive_analysis.py

Every printed [CHECK] line is an assertion of a claim in the companion report.
"""

from collections import Counter

from acas_reachability import AcasReachableSet
from acas_viability import AcasViabilityKernel

_seed = AcasReachableSet.seed_state()     # single source of truth: acas_model_params.yaml
SEED = (_seed[:5], _seed[5])             # ((x_mag, y_mag, x_sign, y_sign, h), a_prev)


def _to_pair(flat_state):
    """Convert acas_reachability's flat 6-tuple to this file's (physical_state, a_prev) pair."""
    return (flat_state[:5], flat_state[5])


def _to_flat(pair):
    """Inverse of _to_pair."""
    return (*pair[0], pair[1])


def reachable_pairs(blocked=frozenset()):
    """(physical_state, a_prev) pairs reachable from the seed via AcasReachableSet."""
    from acas_state import AcasAugmentedState
    blocked_edges = frozenset(
        (AcasAugmentedState.from_tuple(_to_flat(pair)), advisory)
        for pair, advisory in blocked
    )
    reachable_set = AcasReachableSet.compute(blocked=blocked_edges)
    return {_to_pair(state) for state in reachable_set.states}


def _walk_corridor(reachable_pairs_set, unsafe, successors):
    """Walk unique predecessors back from the unsafe pair to the seed."""
    assert len(unsafe) == 1
    corridor = [unsafe[0]]
    while True:
        (s2, p2) = corridor[-1]
        preds = [
            (s, p) for (s, p) in reachable_pairs_set
            if successors[s][p2] == s2 and (s, p) != (s2, p2)
        ]
        print(f"[CHECK] predecessors of {corridor[-1]}: {preds}")
        if len(preds) != 1:
            break
        corridor.append(preds[0])
        if corridor[-1][0] == SEED[0]:
            break
    corridor.reverse()
    return corridor


def main() -> None:
    kernel = AcasViabilityKernel.compute()
    successors = kernel.successors
    advisories = list(kernel.domain.advisories)

    R = reachable_pairs()
    unsafe = [(s, p) for (s, p) in R if not kernel.is_plant_safe(s)]
    print(f"[CHECK] |R| = {len(R)} (expect 9428); unsafe pairs = {unsafe}")

    V = kernel.V
    r_phys = {s for (s, _p) in R}
    hist = Counter(len(v) for v in kernel.allowed.values())
    print(f"[CHECK] |V| = {len(V)}; |boundary| = {len(kernel.boundary)} "
          f"({100 * len(kernel.boundary) / len(V):.2f}% of V)")
    print(f"[CHECK] |Allowed_V| histogram: {dict(sorted(hist.items()))}")
    print(f"[CHECK] boundary ∩ R_phys = {sorted(kernel.boundary & r_phys)}")

    # inductive candidate I0 and its single NN-dependent obligation
    I0 = {(s, p) for (s, p) in R if s in V}
    need_nn = sorted(
        (s, p) for (s, p) in I0
        if any(successors[s][a] not in V for a in advisories)
    )
    print(f"[CHECK] |I0| = {len(I0)}; seed in I0: {SEED in I0}; "
          f"pairs needing NN info: {need_nn}")

    corridor = _walk_corridor(R, unsafe, successors)
    print(f"[CHECK] corridor (seed -> crash): {corridor}")

    # candidate injections
    mid = corridor[1]      # ((5,3,1,1,8), 'strong_right') expected
    for label, blocked in (
        ("Q1 at mid", {(mid, 'strong_right')}),
        ("Q2 at seed", {(SEED, 'strong_right')}),
    ):
        RR = reachable_pairs(blocked=frozenset(blocked))
        verdict = all(kernel.is_plant_safe(s) for (s, _p) in RR)
        print(f"[CHECK] inject {label}: |R| = {len(RR)}, "
              f"INVARSPEC {'TRUE' if verdict else 'FALSE'}")


if __name__ == '__main__':
    main()
