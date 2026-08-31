"""
grid_world_safety_contract_generator.py

Safety-side contract generation: one never-select contract per forbidden action
on the viability-kernel boundary ∂V.

Produces list[GridWorldSafetyContract] from GridWorldViabilityKernel. No CROWN
(see GridWorldSafetyContractVerifier).

Usage (from examples/grid_world/):

    python3 -m core.safety.grid_world_safety_contract_generator
"""

from __future__ import annotations

from dataclasses import dataclass

from core.grid_world_contract import GridWorldSafetyContract
from core.grid_world_domain import (
    ACTIONS,
    DEFAULT_CONFIG_PATH,
    DIR_IDX,
    GridWorldDomain,
)
from core.safety.grid_world_viability import GridWorldViabilityKernel


@dataclass(frozen=True)
class GridWorldSafetyContractGenerator:
    """
    Build safety (never-select) contracts from a viability partition.

    The kernel owns the partition; this owns the emission policy.
    """

    kernel: GridWorldViabilityKernel

    @classmethod
    def from_domain(
        cls, domain: GridWorldDomain | None = None,
    ) -> GridWorldSafetyContractGenerator:
        return cls(kernel=GridWorldViabilityKernel.compute(domain))

    @property
    def domain(self) -> GridWorldDomain:
        return self.kernel.domain

    def generate_all_contracts(self) -> list[GridWorldSafetyContract]:
        """
        A/G contracts required by inductive obligation (ii) on ∂V:

            Assume drone at source s ∈ ∂V
            Guarantee NN ≠ a  for each a ∉ Allowed_V(s)

        Stay (XX) is never emitted: it always stays in V for s in V.
        """
        contracts: list[GridWorldSafetyContract] = []
        for source in sorted(self.kernel.boundary):
            allowed_here = set(self.kernel.allowed[source])
            for label in ACTIONS:
                if label == "XX" or label in allowed_here:
                    continue
                landing = self.domain.simulate_step(source[0], source[1], label)
                if landing not in self.domain.obstacles:
                    raise AssertionError(
                        f"forbidden action {label} at {source} landed on "
                        f"{landing}, expected an obstacle"
                    )
                contracts.append(GridWorldSafetyContract(
                    source=source,
                    forbidden_dir=label,
                    forbidden_dir_idx=DIR_IDX[label],
                    obstacle=landing,
                ))
        return contracts


def generate_contracts(
    obstacles: list[tuple[int, int]] | None = None,
    grid_min: int | None = None,
    grid_max: int | None = None,
    config_path: str = DEFAULT_CONFIG_PATH,
) -> list[GridWorldSafetyContract]:
    """
    Public entry point for the compositional pipeline.

    Builds the domain (from explicit args or config), computes V, and returns
    kernel-boundary contracts. Prefer this over hand-rolled obstacle walks.
    """
    if obstacles is None or grid_min is None or grid_max is None:
        domain = GridWorldDomain.from_config(config_path=config_path)
    else:
        domain = GridWorldDomain(
            grid_min=grid_min,
            grid_max=grid_max,
            obstacles=frozenset(tuple(o) for o in obstacles),
        )
    return GridWorldSafetyContractGenerator.from_domain(domain).generate_all_contracts()


def main() -> None:
    contracts = generate_contracts()
    print(f"contracts from ∂V = {len(contracts)}")
    for i, c in enumerate(contracts, start=1):
        print(f"{i:<4} {c.description}")


if __name__ == "__main__":
    main()
