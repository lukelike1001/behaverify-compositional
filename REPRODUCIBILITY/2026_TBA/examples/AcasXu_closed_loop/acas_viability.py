"""
acas_viability.py

The viability kernel V: the largest set of physical states from which SOME advisory
sequence stays safe forever (greatest fixpoint), computed using ONLY the physics in
generate_acas_contracts.py -- no networks, no a_prev. Orthogonal to acas_reachability.py:
V is about geometry (which states can possibly be kept safe), R is about the controller
(which states the closed loop actually visits).

Physical state: (x_mag, y_mag, x_sign, y_sign, heading_own_var).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import generate_acas_contracts as g

ADVISORIES = g.ADVISORIES

ALL_PHYS = list(itertools.product(
    range(g.MAX_DIST_VAR + 1), range(g.MAX_DIST_VAR + 1),
    (-1, 1), (-1, 1), range(g.MAX_HEADING_VAR)))

SUCC = {s: {a: g.simulate_step(*s, a) for a in ADVISORIES} for s in ALL_PHYS}


def is_safe(s) -> bool:
    return g.compute_distance(s[0], s[1]) >= g.SAFETY_THRESHOLD


@dataclass(frozen=True)
class ViabilityKernel:
    """The safety/viability partition of the physical state space. One responsibility:
    classify every physical state as unsafe, safe-but-doomed, interior(V), or boundary(V)."""
    V: frozenset
    interior: frozenset
    boundary: frozenset
    safe_but_doomed: frozenset
    unsafe: frozenset
    allowed: dict   # s -> Allowed_V(s), only for s in V


def compute_viability_kernel() -> ViabilityKernel:
    """Greatest fixpoint of V = {s safe : exists a with step(s,a) in V}."""
    safe = frozenset(s for s in ALL_PHYS if is_safe(s))
    unsafe = frozenset(ALL_PHYS) - safe

    kernel = set(safe)
    while True:
        doomed = {s for s in kernel
                  if not any(SUCC[s][a] in kernel for a in ADVISORIES)}
        if not doomed:
            break
        kernel -= doomed
    V = frozenset(kernel)

    allowed = {s: [a for a in ADVISORIES if SUCC[s][a] in V] for s in V}
    interior = frozenset(s for s, a in allowed.items() if len(a) == len(ADVISORIES))
    boundary = V - interior
    safe_but_doomed = safe - V

    return ViabilityKernel(
        V=V, interior=interior, boundary=boundary,
        safe_but_doomed=safe_but_doomed, unsafe=unsafe, allowed=allowed,
    )


if __name__ == "__main__":
    kernel = compute_viability_kernel()
    print(f"|V| = {len(kernel.V)} out of {len(ALL_PHYS)} physical states")
    print(f"interior = {len(kernel.interior)}, boundary = {len(kernel.boundary)}, "
          f"safe-but-doomed = {len(kernel.safe_but_doomed)}, unsafe = {len(kernel.unsafe)}")
