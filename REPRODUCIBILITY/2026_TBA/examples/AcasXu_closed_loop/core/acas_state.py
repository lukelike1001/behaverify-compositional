"""
acas_state.py

Canonical state types for the ACAS Xu closed-loop example.

AcasState          -- plant configuration (no a_prev): element of Q
AcasAugmentedState -- plant + active network a_prev: element of Q+

Pure data: no dynamics (see AcasDomain.simulate_step) and no NN evaluation.
"""

from __future__ import annotations

from typing import NamedTuple


class AcasState(NamedTuple):
    """
    Physical / plant lattice state (network-free).

    Fields: (x_mag, y_mag, x_sign, y_sign, heading_own_var).
    NamedTuple so values remain tuple-compatible for unpacking and hashing.
    """

    x_mag: int
    y_mag: int
    x_sign: int
    y_sign: int
    heading_own_var: int

    @classmethod
    def from_tuple(cls, values: tuple[int, int, int, int, int]) -> AcasState:
        return cls(*values)

    def as_tuple(self) -> tuple[int, int, int, int, int]:
        return (self.x_mag, self.y_mag, self.x_sign, self.y_sign, self.heading_own_var)


class AcasAugmentedState(NamedTuple):
    """
    Closed-loop configuration: plant state plus which network is active (a_prev).

    Fields: (x_mag, y_mag, x_sign, y_sign, heading_own_var, a_prev).
    """

    x_mag: int
    y_mag: int
    x_sign: int
    y_sign: int
    heading_own_var: int
    a_prev: str

    @classmethod
    def from_tuple(
        cls,
        values: tuple[int, int, int, int, int, str],
    ) -> AcasAugmentedState:
        return cls(*values)

    @classmethod
    def from_physical(
        cls,
        physical: AcasState | tuple[int, int, int, int, int],
        a_prev: str,
    ) -> AcasAugmentedState:
        if isinstance(physical, AcasState):
            return cls(*physical, a_prev)
        return cls(*physical, a_prev)

    @classmethod
    def from_json_pair(cls, pair: list) -> AcasAugmentedState:
        """JSON dump format: [[x_mag, y_mag, x_sign, y_sign, h], a_prev]."""
        state, a_prev = pair
        return cls(
            x_mag=int(state[0]),
            y_mag=int(state[1]),
            x_sign=int(state[2]),
            y_sign=int(state[3]),
            heading_own_var=int(state[4]),
            a_prev=str(a_prev),
        )

    def as_tuple(self) -> tuple[int, int, int, int, int, str]:
        return (
            self.x_mag, self.y_mag, self.x_sign, self.y_sign,
            self.heading_own_var, self.a_prev,
        )

    def physical(self) -> AcasState:
        return AcasState(
            self.x_mag, self.y_mag, self.x_sign, self.y_sign, self.heading_own_var,
        )

    def physical_tuple(self) -> tuple[int, int, int, int, int]:
        return self.physical().as_tuple()

    def pair_key(self) -> tuple[int, int, int, int, int, str]:
        """Identity key for sets/maps (same as as_tuple)."""
        return self.as_tuple()

    def to_json_pair(self) -> list:
        return [list(self.physical_tuple()), self.a_prev]
