"""
acas_check_inductive_invariant.py

CHECK driver for the inductive-invariant stress test (companion to
2026_07_12_inductive_invariant_stress_test.md).

Geometry lives on AcasReachableSet (R) and AcasViabilityKernel (V / dV /
Allowed_V). This file only orchestrates those types and prints report CHECKs.

Uses plant physics only (no networks).

Usage (from AcasXu_closed_loop/):

    python3 scripts/acas_check_inductive_invariant.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_EXAMPLE = Path(__file__).resolve().parent.parent
if str(_EXAMPLE) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE))

from core.acas_domain import AcasDomain
from core.acas_reachability import AcasReachableSet
from core.acas_viability import AcasViabilityKernel


def main() -> None:
    domain = AcasDomain.from_yaml()
    kernel = AcasViabilityKernel.compute(domain)
    reachable = AcasReachableSet.compute(domain)

    unsafe = reachable.plant_unsafe_states(kernel)
    print(
        f"[CHECK] |R| = {len(reachable)} (expect 9428); "
        f"unsafe pairs = {[state.as_tuple() for state in unsafe]}"
    )

    hist = kernel.allowed_size_histogram()
    print(
        f"[CHECK] |V| = {len(kernel.V)}; |boundary| = {len(kernel.boundary)} "
        f"({100 * len(kernel.boundary) / len(kernel.V):.2f}% of V)"
    )
    print(f"[CHECK] |Allowed_V| histogram: {hist}")
    boundary_in_r = kernel.boundary_intersect(reachable.physical_states())
    print(
        f"[CHECK] boundary ∩ R_phys = "
        f"{[state.as_tuple() for state in boundary_in_r]}"
    )

    inductive_pairs = reachable.viable_pairs(kernel)
    need_nn = reachable.pairs_needing_nn_constraint(kernel)
    seed = reachable.seed
    print(
        f"[CHECK] |I0| = {len(inductive_pairs)}; "
        f"seed in I0: {seed in inductive_pairs}; "
        f"pairs needing NN info: {[state.as_tuple() for state in need_nn]}"
    )

    corridor = reachable.corridor_seed_to_unique_unsafe(kernel, verbose=True)
    print(
        f"[CHECK] corridor (seed -> crash): "
        f"{[state.as_tuple() for state in corridor]}"
    )

    mid = corridor[1]
    for label, from_state, forbidden in (
        ("Q1 at mid", mid, "strong_right"),
        ("Q2 at seed", seed, "strong_right"),
    ):
        blocked = frozenset({(from_state, forbidden)})
        reduced = AcasReachableSet.compute(domain=domain, blocked=blocked)
        holds = reduced.all_plant_safe(kernel)
        print(
            f"[CHECK] inject {label}: |R| = {len(reduced)}, "
            f"INVARSPEC {'TRUE' if holds else 'FALSE'}"
        )


if __name__ == "__main__":
    main()
