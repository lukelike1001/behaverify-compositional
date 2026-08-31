"""
grid_world_viability.py

Viability kernel V for the 1-NN grid-world NSBT: the largest set of safe cells
from which SOME action sequence stays safe forever (greatest fixpoint).
Computed from GridWorldDomain physics only -- no networks.

Hover theorem: for every safe cell s, XX maps s -> s, so the greatest fixpoint
never deletes anything. V = Safe, safe_but_doomed = empty, and ∂V is exactly
the obstacle-adjacent cells. ∂V is what GridWorldSafetyContractGenerator turns
into the one-step crash-avoidance contracts used by the compositional pipeline.

Run from examples/grid_world/:

    python3 -m core.safety.grid_world_viability
"""

from __future__ import annotations

from dataclasses import dataclass

from core.grid_world_domain import ACTIONS, GridWorldDomain


@dataclass(frozen=True)
class GridWorldViabilityKernel:
    """
    Safety/viability partition of the grid.

    Single responsibility: classify every cell as unsafe, safe-but-doomed,
    interior(V), or boundary(V), and expose Allowed_V(s) for s in V.
    Contract emission lives on GridWorldSafetyContractGenerator.
    """

    domain: GridWorldDomain
    V: frozenset[tuple[int, int]]
    interior: frozenset[tuple[int, int]]
    boundary: frozenset[tuple[int, int]]
    safe_but_doomed: frozenset[tuple[int, int]]
    unsafe: frozenset[tuple[int, int]]
    allowed: dict[tuple[int, int], list[str]]
    fixpoint_rounds: int

    @classmethod
    def compute(cls, domain: GridWorldDomain | None = None) -> GridWorldViabilityKernel:
        """Greatest fixpoint of V = {s safe : exists a with step(s,a) in V}."""
        if domain is None:
            domain = GridWorldDomain.from_config()

        succ = domain.build_successor_table()
        all_cells = domain.all_cells

        safe = frozenset(s for s in all_cells if domain.is_safe(s))
        unsafe = frozenset(all_cells) - safe

        kernel: set[tuple[int, int]] = set(safe)
        rounds = 0
        while True:
            doomed = {
                s for s in kernel
                if not any(succ[s][a] in kernel for a in ACTIONS)
            }
            if not doomed:
                break
            kernel -= doomed
            rounds += 1
        V = frozenset(kernel)

        allowed = {s: [a for a in ACTIONS if succ[s][a] in V] for s in V}
        interior = frozenset(
            s for s, acts in allowed.items() if len(acts) == len(ACTIONS)
        )
        boundary = V - interior
        safe_but_doomed = safe - V

        return cls(
            domain=domain,
            V=V,
            interior=interior,
            boundary=boundary,
            safe_but_doomed=safe_but_doomed,
            unsafe=unsafe,
            allowed=allowed,
            fixpoint_rounds=rounds,
        )


def main() -> None:
    domain = GridWorldDomain.from_config()
    kernel = GridWorldViabilityKernel.compute(domain)
    n = domain.side_length ** 2
    print(
        f"grid = [{domain.grid_min},{domain.grid_max}]^2  ({n} cells, "
        f"{len(domain.obstacles)} obstacles)"
    )
    print(
        f"|V| = {len(kernel.V)}  interior = {len(kernel.interior)}  "
        f"boundary = {len(kernel.boundary)}  "
        f"safe-but-doomed = {len(kernel.safe_but_doomed)}  "
        f"unsafe = {len(kernel.unsafe)}"
    )
    print(f"fixpoint rounds = {kernel.fixpoint_rounds}")


if __name__ == "__main__":
    main()
