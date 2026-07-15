"""
grid_world_inductive_proof.py

Direct-computation checks for the hover / viability-kernel claims on grid world.

Run from examples/grid_world/:

    python3 grid_world_inductive_proof.py
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from grid_world_viability import (
    ACTIONS,
    GridWorldDomain,
    GridWorldViabilityKernel,
    load_config,
)


@dataclass
class GridWorldInductiveProof:
    """
    Orchestrates inductive-invariant claims for this benchmark.

    Holds a domain and its viability kernel; each check_* method is one
    obligation from the companion report. Does not own physics, partition
    logic, or CROWN verification -- those live on GridWorldDomain /
    GridWorldViabilityKernel / GridWorldContractVerifier.
    """

    domain: GridWorldDomain
    kernel: GridWorldViabilityKernel

    @classmethod
    def from_config(
        cls, config_path: str = "grid_world_domain_config.yaml",
    ) -> GridWorldInductiveProof:
        domain = GridWorldDomain.from_config(load_config(config_path))
        kernel = GridWorldViabilityKernel.compute(domain)
        return cls(domain=domain, kernel=kernel)

    # --- individual claims -------------------------------------------------

    def check_hover_theorem(self) -> None:
        """Stay makes every safe cell viable: fixpoint is immediate, V = Safe."""
        k = self.kernel
        safe = frozenset(s for s in self.domain.all_cells if self.domain.is_safe(s))
        print(f"[CHECK] fixpoint_rounds = {k.fixpoint_rounds} (expect 0)")
        assert k.fixpoint_rounds == 0
        print(f"[CHECK] |safe_but_doomed| = {len(k.safe_but_doomed)} (expect 0)")
        assert len(k.safe_but_doomed) == 0
        print(f"[CHECK] V == Safe: |V|={len(k.V)}, |Safe|={len(safe)}")
        assert k.V == safe
        assert all("XX" in k.allowed[s] for s in k.V)

    def check_partition(self) -> None:
        """Unsafe = obstacles; V and Unsafe partition the grid."""
        k = self.kernel
        n = self.domain.side_length ** 2
        print(
            f"[CHECK] |Unsafe| = {len(k.unsafe)} "
            f"(expect {len(self.domain.obstacles)} obstacles)"
        )
        assert k.unsafe == self.domain.obstacles
        assert len(k.V) + len(k.unsafe) == n

    def check_boundary_geometry(self) -> None:
        """
        ∂V equals the pure-geometry obstacle-adjacent set.

        Two independent constructions: Allowed_V (kernel) vs. landing-on-obstacle
        (domain). Equality is the content of the hover specialization.
        """
        geometric = self.domain.obstacle_adjacent_cells()
        print(
            f"[CHECK] |boundary| = {len(self.kernel.boundary)} "
            f"(expect |obstacle-adjacent| = {len(geometric)})"
        )
        assert self.kernel.boundary == geometric

    def check_contracts_are_crash_edges(self) -> None:
        """Every boundary contract forbids a real one-step crash into an obstacle."""
        contracts = self.kernel.contracts_from_boundary()
        print(f"[CHECK] |∂V contracts| = {len(contracts)}")
        assert len(contracts) > 0
        for c in contracts:
            assert c.source in self.kernel.boundary
            assert c.forbidden_dir not in self.kernel.allowed[c.source]
            landing = self.domain.simulate_step(
                c.source[0], c.source[1], c.forbidden_dir,
            )
            assert landing == c.obstacle
            assert landing in self.domain.obstacles
        ids = [c.identity() for c in contracts]
        assert len(ids) == len(set(ids))
        print(f"[CHECK] contract identities unique: {len(ids)}")

    def check_allowed_histogram(self) -> None:
        """Interior allows every action; boundary forbids at least one cardinal."""
        k = self.kernel
        hist = Counter(len(acts) for acts in k.allowed.values())
        print(f"[CHECK] Allowed_V histogram over V: {dict(sorted(hist.items()))}")
        assert all(len(k.allowed[s]) == len(ACTIONS) for s in k.interior)
        assert all(len(k.allowed[s]) < len(ACTIONS) for s in k.boundary)

    # --- driver ------------------------------------------------------------

    def run(self) -> None:
        self.check_hover_theorem()
        self.check_partition()
        self.check_boundary_geometry()
        self.check_contracts_are_crash_edges()
        self.check_allowed_histogram()

        k = self.kernel
        contracts = k.contracts_from_boundary()
        print()
        print("Summary")
        print("-------")
        print(f"  grid cells           : {self.domain.side_length ** 2}")
        print(f"  obstacles (Unsafe)   : {len(k.unsafe)}")
        print(f"  V = Safe             : {len(k.V)}")
        print(f"  interior(V)          : {len(k.interior)}")
        print(f"  boundary(V) = ∂V     : {len(k.boundary)}")
        print(f"  safe-but-doomed      : {len(k.safe_but_doomed)}")
        print(f"  fixpoint rounds      : {k.fixpoint_rounds}")
        print(f"  ∂V contracts         : {len(contracts)}")
        print("All checks passed.")


if __name__ == "__main__":
    GridWorldInductiveProof.from_config().run()
