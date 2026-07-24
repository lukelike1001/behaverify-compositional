"""
acas_domain.py

Static ACAS Xu closed-loop plant: numbers from acas_model_params.yaml plus
executable physics that mirrors tree/acas_closed_loop_template.tree
environment_update.

Owns model config and one-step dynamics only -- not viability, contracts,
CROWN settings, or the BehaVerify .tree (those are separate layers).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from core.paths import EXAMPLE_ROOT

DEFAULT_MODEL_PARAMS = EXAMPLE_ROOT / "core" / "acas_model_params.yaml"


@dataclass(frozen=True)
class AcasDomain:
    """
    Closed-loop plant parameters and step semantics.

    Construct with ``AcasDomain.from_yaml()``. Physics matches the sequential
    environment_update in tree/acas_closed_loop_template.tree
    (heading first, then position).
    """

    # Physics (raw units from yaml)
    distance_modifier: int
    max_dist: int
    speed_own: int
    speed_int: int
    seconds_per_update: int
    degree_multiplier: int
    heading_int_degrees: int
    safety_threshold: int

    # NN normalization
    distance_mean: float
    distance_range: float
    speed_own_mean: float
    speed_own_range: float
    speed_int_mean: float
    speed_int_range: float

    # Catalogs
    advisories: tuple[str, ...]
    # a_prev -> (network_idx, onnx relative path)
    a_prev_to_nn: dict[str, tuple[int, str]]
    # Closed-loop seed (template initial_values)
    initial_state: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: Path | str | None = None) -> AcasDomain:
        params_path = Path(path) if path is not None else DEFAULT_MODEL_PARAMS
        with open(params_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cls.from_config(cfg)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> AcasDomain:
        physics = cfg["physics"]
        nn = cfg["nn_normalization"]
        networks = {
            name: (int(net["idx"]), str(net["onnx"]))
            for name, net in cfg["networks"].items()
        }
        return cls(
            distance_modifier=int(physics["distance_modifier"]),
            max_dist=int(physics["max_dist"]),
            speed_own=int(physics["speed_own"]),
            speed_int=int(physics["speed_int"]),
            seconds_per_update=int(physics["seconds_per_update"]),
            degree_multiplier=int(physics["degree_multiplier"]),
            heading_int_degrees=int(physics["heading_int_degrees"]),
            safety_threshold=int(physics["safety_threshold"]),
            distance_mean=float(nn["distance_mean"]),
            distance_range=float(nn["distance_range"]),
            speed_own_mean=float(nn["speed_own_mean"]),
            speed_own_range=float(nn["speed_own_range"]),
            speed_int_mean=float(nn["speed_int_mean"]),
            speed_int_range=float(nn["speed_int_range"]),
            advisories=tuple(cfg["advisories"]),
            a_prev_to_nn=networks,
            initial_state=dict(cfg.get("initial_state") or {}),
        )

    # ------------------------------------------------------------------
    # Derived domains
    # ------------------------------------------------------------------

    @property
    def max_dist_var(self) -> int:
        return self.max_dist // self.distance_modifier

    @property
    def max_heading_var(self) -> int:
        return 360 // self.degree_multiplier

    @property
    def heading_int_var(self) -> int:
        return self.heading_int_degrees // self.degree_multiplier

    @property
    def adv_idx(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.advisories)}

    @property
    def nn_input_speed_own(self) -> float:
        return (self.speed_own - self.speed_own_mean) / self.speed_own_range

    @property
    def nn_input_speed_int(self) -> float:
        return (self.speed_int - self.speed_int_mean) / self.speed_int_range

    @property
    def vel_x_int(self) -> int:
        return self._vel_x(self.heading_int_degrees, self.speed_int)

    @property
    def vel_y_int(self) -> int:
        return self._vel_y(self.heading_int_degrees, self.speed_int)

    def all_physical_states(self) -> list[tuple[int, int, int, int, int]]:
        """Full lattice: (x_mag, y_mag, x_sign, y_sign, heading_own_var)."""
        return list(itertools.product(
            range(self.max_dist_var + 1),
            range(self.max_dist_var + 1),
            (-1, 1),
            (-1, 1),
            range(self.max_heading_var),
        ))

    def seed_physical_and_aprev(self) -> tuple[tuple[int, int, int, int, int], str]:
        """Closed-loop seed from yaml initial_state."""
        initial = self.initial_state
        physical = (
            int(initial["x_mag"]),
            int(initial["y_mag"]),
            int(initial["x_sign"]),
            int(initial["y_sign"]),
            int(initial["heading_own_var"]),
        )
        return physical, str(initial["a_prev"])

    # ------------------------------------------------------------------
    # Safety / advisory / step (tree environment_update mirror)
    # ------------------------------------------------------------------

    @staticmethod
    def _vel_x(heading_degrees: int, speed: int) -> int:
        return round(math.cos(math.radians(heading_degrees)) * speed)

    @staticmethod
    def _vel_y(heading_degrees: int, speed: int) -> int:
        return round(math.sin(math.radians(heading_degrees)) * speed)

    def apply_advisory(self, heading_own_var: int, advisory: str) -> int:
        """Return new heading_own_var after applying advisory."""
        n = self.max_heading_var
        if advisory == "strong_left":
            return (heading_own_var + 2) % n
        if advisory == "weak_left":
            return (heading_own_var + 1) % n
        if advisory == "weak_right":
            return (n + heading_own_var - 1) % n
        if advisory == "strong_right":
            return (n + heading_own_var - 2) % n
        return heading_own_var  # clear

    def compute_distance(self, x_mag: int, y_mag: int) -> int:
        """distance = round(sqrt(x_mag^2 + y_mag^2)) * distance_modifier."""
        return (
            round(math.sqrt(x_mag * x_mag + y_mag * y_mag)) * self.distance_modifier
        )

    def is_safe(self, x_mag: int, y_mag: int) -> bool:
        return self.compute_distance(x_mag, y_mag) >= self.safety_threshold

    def simulate_step(
        self,
        x_mag: int,
        y_mag: int,
        x_sign: int,
        y_sign: int,
        heading_own_var: int,
        advisory: str,
    ) -> tuple[int, int, int, int, int]:
        """
        One environment_update tick.

        Heading is updated first (sequential order in the tree), then position
        uses the new heading via velocity_x_own / velocity_y_own.

        Returns (next_x_var, next_y_var, next_x_sign, next_y_sign, next_heading).
        """
        new_heading_var = self.apply_advisory(heading_own_var, advisory)
        new_heading = new_heading_var * self.degree_multiplier

        vel_x_own = self._vel_x(new_heading, self.speed_own)
        vel_y_own = self._vel_y(new_heading, self.speed_own)

        x = x_mag * self.distance_modifier
        y = y_mag * self.distance_modifier

        next_x = x * x_sign + self.seconds_per_update * (self.vel_x_int - vel_x_own)
        next_y = y * y_sign + self.seconds_per_update * (self.vel_y_int - vel_y_own)

        next_x_sign = -1 if next_x < 0 else 1
        next_y_sign = -1 if next_y < 0 else 1
        next_x_var = int(min(self.max_dist, abs(next_x)) // self.distance_modifier)
        next_y_var = int(min(self.max_dist, abs(next_y)) // self.distance_modifier)

        return next_x_var, next_y_var, next_x_sign, next_y_sign, new_heading_var

    # ------------------------------------------------------------------
    # Angle / NN inputs (DEFINE chain in the template)
    # ------------------------------------------------------------------

    @staticmethod
    def _arctan_xy(x_mag: int, y_mag: int) -> int:
        return 0 if y_mag == 0 else round(math.degrees(math.atan(x_mag / y_mag)))

    @staticmethod
    def _arctan_yx(x_mag: int, y_mag: int) -> int:
        return 0 if x_mag == 0 else round(math.degrees(math.atan(y_mag / x_mag)))

    def _arctan_val(self, x_mag: int, y_mag: int, x_sign: int, y_sign: int) -> int:
        if y_sign == 1:
            return self._arctan_yx(x_mag, y_mag)
        return self._arctan_xy(x_mag, y_mag)

    @staticmethod
    def _normalize_angle(angle_degrees: int) -> int:
        mod = angle_degrees % 360
        pos = mod if mod >= 0 else mod + 360
        return pos - 360 if pos > 180 else pos

    def compute_relative_angle_adjusted(
        self,
        x_mag: int,
        y_mag: int,
        x_sign: int,
        y_sign: int,
        heading_own_var: int,
    ) -> int:
        heading_own = heading_own_var * self.degree_multiplier
        x = x_mag * self.distance_modifier
        y = y_mag * self.distance_modifier
        av = self._arctan_val(x_mag, y_mag, x_sign, y_sign)

        if x_sign == 1 and y == 0:
            rel = 270 - heading_own
        elif x_sign == -1 and y == 0:
            rel = 90 - heading_own
        elif x == 0 and y_sign == 1:
            rel = 360 - heading_own
        elif x == 0 and y_sign == -1:
            rel = 180 - heading_own
        elif x_sign == 1 and y_sign == 1:
            rel = (270 - heading_own) + av
        elif x_sign == 1 and y_sign == -1:
            rel = (180 - heading_own) + av
        elif x_sign == -1 and y_sign == 1:
            rel = (90 - heading_own) - av
        else:
            rel = (180 - heading_own) - av

        return self._normalize_angle(rel)

    def compute_intersect_angle_adjusted(self, heading_own_var: int) -> int:
        heading_own = heading_own_var * self.degree_multiplier
        return self._normalize_angle(heading_own - self.heading_int_degrees)

    def compute_nn_inputs(
        self,
        x_mag: int,
        y_mag: int,
        x_sign: int,
        y_sign: int,
        heading_own_var: int,
    ) -> list[float]:
        """Five normalized NN inputs matching template network_k_1 expressions."""
        dist = self.compute_distance(x_mag, y_mag)
        rel_adj = self.compute_relative_angle_adjusted(
            x_mag, y_mag, x_sign, y_sign, heading_own_var,
        )
        int_adj = self.compute_intersect_angle_adjusted(heading_own_var)
        return [
            (dist - self.distance_mean) / self.distance_range,
            rel_adj / 360.0,
            int_adj / 360.0,
            self.nn_input_speed_own,
            self.nn_input_speed_int,
        ]
