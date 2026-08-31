"""
grid_world_contract.py

Typed A/G contract specs for grid world.

GridWorldSafetyContract   -- NN must NOT output forbidden_dir (never_selects)
GridWorldLivenessContract -- NN must output some direction in Dec(s, g) (in_set)

Specs live in separate JSON files (do not mix kinds in one list):
  contracts/discrete/safety/<network>_discrete.json
  contracts/discrete/liveness/<network>_liveness.json

Pure data + serialization. One instance = one JSON object under "contracts".
CROWN execution stays on GridWorldSafetyContractVerifier; generation stays under
core/safety/ (and later core/liveness/).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, kw_only=True)
class GridWorldContract(ABC):
    """
    Shared assumption for one local NN obligation.

    The NN reads (x_d, y_d, x_g, y_g), so an assumption constrains both halves:

        (x_d, y_d) == source  AND  (x_g, y_g) IN goal_region

    goal_region is None when the contract quantifies over every goal on the
    grid (safety: "never crash, wherever the target is"), or a single cell when
    the contract is pinned to one goal (liveness: progress is goal-relative).

    Widths around those points -- the drone eps-ball and the goal
    discretization -- stay on GridWorldSafetyContractVerifier, which is what turns
    this symbolic region into concrete CROWN input bounds.
    """

    source: tuple[int, int]
    goal_region: tuple[int, int] | None = None

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable one-line summary, embedded in the JSON spec."""

    @abstractmethod
    def identity(self) -> tuple[Any, ...]:
        """Canonical key for set comparison across contract sets."""

    def _base_dict(self) -> dict[str, Any]:
        rec: dict[str, Any] = {
            "source": list(self.source),
            "description": self.description,
        }
        if self.goal_region is not None:
            rec["goal"] = list(self.goal_region)
        return rec

    def to_spec_dict(self, contract_id: int | None = None) -> dict[str, Any]:
        """JSON-serializable fields shared with CROWN result records."""
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class GridWorldSafetyContract(GridWorldContract):
    """One A/G safety contract: at `source`, the NN must not output `forbidden_dir`."""

    forbidden_dir: str
    forbidden_dir_idx: int
    obstacle: tuple[int, int]

    @property
    def description(self) -> str:
        ox, oy = self.obstacle
        cx, cy = self.source
        return f"obstacle ({ox},{oy})  source ({cx},{cy})  forbid {self.forbidden_dir}"

    def identity(self) -> tuple[int, int, str, int, int]:
        """Canonical key for set comparison (source, forbid label, obstacle)."""
        return (*self.source, self.forbidden_dir, *self.obstacle)

    def to_spec_dict(self, contract_id: int | None = None) -> dict[str, Any]:
        rec = self._base_dict()
        rec.update({
            "obstacle": list(self.obstacle),
            "forbidden_dir": self.forbidden_dir,
            "forbidden_dir_idx": self.forbidden_dir_idx,
        })
        if contract_id is not None:
            rec["id"] = contract_id
        return rec


@dataclass(frozen=True, kw_only=True)
class GridWorldLivenessContract(GridWorldContract):
    """
    One A/G liveness contract: at `source` with goal `goal_region`, the NN must
    output some direction in `allowed_dirs` = Dec(source, goal).

    Set membership, not equality: the network keeps a choice wherever several
    actions make progress, so the abstract model stays non-deterministic. That
    is what separates this from pinning a trajectory.

    CROWN has no "argmax in set" primitive, but it does not need one --
    membership is the conjunction of never-select obligations over the
    complement, which is `forbidden_dirs`.
    """

    goal_region: tuple[int, int]
    allowed_dirs: tuple[str, ...]
    allowed_dir_idxs: tuple[int, ...]
    forbidden_dirs: tuple[str, ...]
    forbidden_dir_idxs: tuple[int, ...]
    distance: int

    @property
    def description(self) -> str:
        sx, sy = self.source
        gx, gy = self.goal_region
        allowed = "/".join(self.allowed_dirs)
        return (
            f"source ({sx},{sy})  goal ({gx},{gy})  "
            f"dist {self.distance}  require {allowed}"
        )

    def identity(self) -> tuple[int, int, int, int]:
        """Canonical key: one contract per (source, goal) pair."""
        return (*self.source, *self.goal_region)

    def to_spec_dict(self, contract_id: int | None = None) -> dict[str, Any]:
        rec = self._base_dict()
        rec.update({
            "distance": self.distance,
            "allowed_dirs": list(self.allowed_dirs),
            "allowed_dir_idxs": list(self.allowed_dir_idxs),
            "forbidden_dirs": list(self.forbidden_dirs),
            "forbidden_dir_idxs": list(self.forbidden_dir_idxs),
        })
        if contract_id is not None:
            rec["id"] = contract_id
        return rec
