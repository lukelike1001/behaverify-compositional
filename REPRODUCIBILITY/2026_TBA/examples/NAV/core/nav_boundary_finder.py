"""
nav_boundary_finder.py

State-space bounds for the ARCH-COMP NAV benchmark, used to declare the SMV
variable domains that the monolithic table enumerates.

Owns bounds only -- not the NSBT, the trig tables, the fixed-point scaling, or
contract generation. Reads every constant from nav_domain_config.yaml.

TWO METHODS, TWO DIFFERENT CLAIMS
---------------------------------
The distinction matters more than the numbers, so the two are kept apart and
each NavBounds carries the method that produced it.

  "analytic"  Sound a-priori bounds derived from the control authority alone.
              The network is never queried. Safe to declare as an SMV domain:
              no reachable state can fall outside them.

  "sampled"   Closed-loop envelope measured by simulating from a grid over the
              initial set. NOT SOUND -- it is an under-approximation of the
              true reachable tube, since finitely many initial points cannot
              cover an interval. Declaring SMV domains from it would assume
              what the model check is supposed to establish.

Report the sampled envelope to say how loose the sound bounds are; declare the
analytic ones. NavBounds.is_sound records which is which.

DERIVATION OF THE ANALYTIC BOUNDS
---------------------------------
Dynamics (dynamics.m):  x' = v cos(theta),  y' = v sin(theta),
                        v' = u1,            theta' = u2
The output layer is tanh, so |u1|, |u2| <= U with U = 1 structurally. Over a
horizon T, writing the initial interval of a coordinate as [lo, hi]:

  |v(t) - v(0)|     = |int_0^t u1|  <= U t            <= U T
  |theta(t) - theta(0)|              <= U t            <= U T
  |x'(t)| = |v(t) cos(theta(t))| <= |v(t)| <= |v(0)|_max + U t
  |x(t) - x(0)|     <= int_0^t (|v(0)|_max + U s) ds  <= |v(0)|_max T + U T^2/2

so, with V0 = max(|v_lo|, |v_hi|),

  v     in [v_lo     - U T,                 v_hi     + U T]
  theta in [theta_lo - U T,                 theta_hi + U T]
  x     in [x_lo - V0 T - U T^2/2,          x_hi + V0 T + U T^2/2]
  y     likewise.

For this benchmark (v(0) = 0, U = 1, T = 6) the position term is T^2/2 = 18,
which is why the sound box is far larger than the tube in ARCH-COMP Figure 16.
That gap is the cost the monolithic table pays, and it is a result to report
rather than a number to tune away.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np
import yaml

from core.paths import EXAMPLE_ROOT

DEFAULT_CONFIG_PATH = str(EXAMPLE_ROOT / "nav_domain_config.yaml")

# Index of each coordinate in the 4-vector handed to the network.
# See the state-order note in nav_domain_config.yaml.
STATE_ORDER: tuple[str, ...] = ("x", "y", "v", "theta")
X, Y, V, THETA = range(4)

SOUND_METHODS = frozenset({"analytic"})


def load_config(path: str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load NAV benchmark configuration from a YAML file."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class NavBounds:
    """
    An axis-aligned box in NAV state space, tagged with how it was obtained.

    Bounds are (lower, upper) in benchmark units -- metres, m/s, radians --
    before any fixed-point scaling.
    """

    x: tuple[float, float]
    y: tuple[float, float]
    v: tuple[float, float]
    theta: tuple[float, float]
    method: str

    @property
    def is_sound(self) -> bool:
        """True when these bounds may be declared as SMV variable domains."""
        return self.method in SOUND_METHODS

    @property
    def position(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """The ([x_lower, x_upper], [y_lower, y_upper]) pair."""
        return (self.x, self.y)

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "is_sound": self.is_sound,
            "x": list(self.x),
            "y": list(self.y),
            "v": list(self.v),
            "theta": list(self.theta),
        }

    def contains(self, other: NavBounds, tol: float = 0.0) -> bool:
        """True when `other`'s box sits inside this one, up to `tol`."""
        return all(
            self_lo - tol <= other_lo and other_hi <= self_hi + tol
            for (self_lo, self_hi), (other_lo, other_hi) in zip(
                (self.x, self.y, self.v, self.theta),
                (other.x, other.y, other.v, other.theta),
            )
        )

    def widths(self) -> dict[str, float]:
        return {
            name: hi - lo
            for name, (lo, hi) in zip(
                STATE_ORDER, (self.x, self.y, self.v, self.theta)
            )
        }


class NavBoundaryFinder:
    """
    Computes state-space bounds for one NAV controller.

    The ONNX session is created lazily, so `calculate_boundary("analytic")`
    works without the network present -- the sound bounds do not depend on it.
    """

    def __init__(
        self,
        onnx_path: str | None = None,
        cfg: dict[str, Any] | None = None,
        config_path: str = DEFAULT_CONFIG_PATH,
    ) -> None:
        self.onnx_path = str(onnx_path) if onnx_path is not None else None
        self.cfg = cfg if cfg is not None else load_config(config_path)

        initial = self.cfg["initial_set"]
        self.initial_set: dict[str, tuple[float, float]] = {
            name: (float(initial[name][0]), float(initial[name][1]))
            for name in STATE_ORDER
        }
        self.control_period = float(self.cfg["control_period"])
        self.horizon = float(self.cfg["horizon"])
        self.control_bound = float(self.cfg["control_bound"])

        self._session: Any = None
        self._input_name: str | None = None

    # ---------------------------------------------------------------- config

    @property
    def num_control_steps(self) -> int:
        """Control steps over the horizon: 6.0 / 0.2 = 30."""
        steps = self.horizon / self.control_period
        rounded = int(round(steps))
        if abs(steps - rounded) > 1e-9:
            raise ValueError(
                f"horizon {self.horizon} is not a whole number of control "
                f"periods of {self.control_period}"
            )
        return rounded

    # ------------------------------------------------------------ public API

    def calculate_boundary(
        self,
        method: str = "analytic",
        **kwargs: Any,
    ) -> NavBounds:
        """
        Return the state-space box for this controller.

        method="analytic" (default) gives sound bounds and ignores the network.
        method="sampled" gives the measured closed-loop envelope, which is a
        reference figure only -- see the module docstring.
        """
        if method == "analytic":
            return self._analytic_bounds()
        if method == "sampled":
            return self._sampled_bounds(**kwargs)
        raise ValueError(
            f"unknown method {method!r}; expected 'analytic' or 'sampled'"
        )

    def control(self, state: np.ndarray) -> np.ndarray:
        """One forward pass: state (x, y, v, theta) -> u (u1, u2)."""
        session, input_name = self._ensure_session()
        batch = np.asarray(state, dtype=np.float32).reshape(1, 4)
        return session.run(None, {input_name: batch})[0][0].astype(np.float64)

    def simulate(
        self,
        initial_state: np.ndarray,
        substeps_per_control_period: int | None = None,
    ) -> np.ndarray:
        """
        Roll the closed loop out over the horizon under zero-order hold.

        Returns every integration sample as an array of shape (n + 1, 4),
        including the initial state, so callers can take an envelope over the
        whole trajectory rather than only the control instants.
        """
        substeps = int(
            substeps_per_control_period
            if substeps_per_control_period is not None
            else self.cfg["sampling"]["substeps_per_control_period"]
        )
        if substeps < 1:
            raise ValueError("substeps_per_control_period must be >= 1")

        h = self.control_period / substeps
        state = np.asarray(initial_state, dtype=np.float64).copy()
        trajectory = [state.copy()]

        for _step in range(self.num_control_steps):
            u = self.control(state)  # zero-order hold across the period
            for _sub in range(substeps):
                state = self._rk4(state, u, h)
                trajectory.append(state.copy())

        return np.asarray(trajectory)

    # ------------------------------------------------------------- internals

    @staticmethod
    def _derivative(state: np.ndarray, u: np.ndarray) -> np.ndarray:
        """dynamics.m: [v cos(theta), v sin(theta), u1, u2]."""
        return np.array(
            [
                state[V] * np.cos(state[THETA]),
                state[V] * np.sin(state[THETA]),
                u[0],
                u[1],
            ]
        )

    @classmethod
    def _rk4(cls, state: np.ndarray, u: np.ndarray, h: float) -> np.ndarray:
        k1 = cls._derivative(state, u)
        k2 = cls._derivative(state + 0.5 * h * k1, u)
        k3 = cls._derivative(state + 0.5 * h * k2, u)
        k4 = cls._derivative(state + h * k3, u)
        return state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _ensure_session(self) -> tuple[Any, str]:
        if self._session is None:
            if self.onnx_path is None:
                raise ValueError(
                    "no ONNX path was given; only method='analytic' is "
                    "available without a network"
                )
            import onnxruntime  # imported lazily: analytic bounds never need it

            self._session = onnxruntime.InferenceSession(self.onnx_path)
            self._input_name = self._session.get_inputs()[0].name
        assert self._input_name is not None
        return self._session, self._input_name

    def _analytic_bounds(self) -> NavBounds:
        """Sound bounds from |u| <= control_bound. See module docstring."""
        u_max = self.control_bound
        horizon = self.horizon

        v_lo, v_hi = self.initial_set["v"]
        theta_lo, theta_hi = self.initial_set["theta"]

        rate_growth = u_max * horizon
        # Fastest the robot can ever travel, then integrated over the horizon.
        initial_speed_max = max(abs(v_lo), abs(v_hi))
        position_growth = initial_speed_max * horizon + 0.5 * u_max * horizon**2

        def widen(
            interval: tuple[float, float], margin: float
        ) -> tuple[float, float]:
            return (interval[0] - margin, interval[1] + margin)

        return NavBounds(
            x=widen(self.initial_set["x"], position_growth),
            y=widen(self.initial_set["y"], position_growth),
            v=widen((v_lo, v_hi), rate_growth),
            theta=widen((theta_lo, theta_hi), rate_growth),
            method="analytic",
        )

    def _sampled_bounds(
        self,
        samples_per_axis: int | None = None,
        substeps_per_control_period: int | None = None,
    ) -> NavBounds:
        """
        Envelope over trajectories from a grid across the initial set.

        Unsound by construction: a finite grid cannot cover an interval. Use
        for reporting how loose the analytic bounds are, never for declaring
        SMV domains.
        """
        n = int(
            samples_per_axis
            if samples_per_axis is not None
            else self.cfg["sampling"]["samples_per_axis"]
        )
        if n < 1:
            raise ValueError("samples_per_axis must be >= 1")

        axes = [
            np.linspace(lo, hi, n) if hi > lo else np.array([lo])
            for lo, hi in (self.initial_set[name] for name in STATE_ORDER)
        ]

        lower = np.full(4, np.inf)
        upper = np.full(4, -np.inf)
        for corner in itertools.product(*axes):
            trajectory = self.simulate(
                np.array(corner, dtype=np.float64),
                substeps_per_control_period=substeps_per_control_period,
            )
            lower = np.minimum(lower, trajectory.min(axis=0))
            upper = np.maximum(upper, trajectory.max(axis=0))

        return NavBounds(
            x=(float(lower[X]), float(upper[X])),
            y=(float(lower[Y]), float(upper[Y])),
            v=(float(lower[V]), float(upper[V])),
            theta=(float(lower[THETA]), float(upper[THETA])),
            method="sampled",
        )


def main() -> None:
    """Print both boundaries for each shipped network."""
    import argparse

    parser = argparse.ArgumentParser(
        description="State-space bounds for a NAV controller."
    )
    parser.add_argument(
        "--onnx",
        nargs="+",
        default=[
            str(EXAMPLE_ROOT / "networks" / "nn-nav-set.onnx"),
            str(EXAMPLE_ROOT / "networks" / "nn-nav-point.onnx"),
        ],
        help="ONNX controller(s) to measure.",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--samples-per-axis", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    analytic = NavBoundaryFinder(cfg=cfg).calculate_boundary("analytic")

    print(f"horizon {cfg['horizon']} s, control period {cfg['control_period']} s, "
          f"|u| <= {cfg['control_bound']}")
    print()
    print("analytic (SOUND -- declare these as SMV domains; network not queried)")
    for name, (lo, hi) in zip(
        STATE_ORDER, (analytic.x, analytic.y, analytic.v, analytic.theta)
    ):
        print(f"  {name:<5} [{lo:>8.3f}, {hi:>8.3f}]   width {hi - lo:>7.3f}")

    for onnx_path in args.onnx:
        finder = NavBoundaryFinder(onnx_path=onnx_path, cfg=cfg)
        sampled = finder.calculate_boundary(
            "sampled", samples_per_axis=args.samples_per_axis
        )
        analytic_widths = analytic.widths()
        print()
        print(f"sampled (NOT SOUND -- reference only): {onnx_path}")
        for name, (lo, hi) in zip(
            STATE_ORDER, (sampled.x, sampled.y, sampled.v, sampled.theta)
        ):
            width = hi - lo
            ratio = analytic_widths[name] / width if width > 0 else float("inf")
            print(
                f"  {name:<5} [{lo:>8.3f}, {hi:>8.3f}]   width {width:>7.3f}"
                f"   analytic is {ratio:>6.1f}x wider"
            )
        print(f"  contained in analytic box: {analytic.contains(sampled)}")


if __name__ == "__main__":
    main()
