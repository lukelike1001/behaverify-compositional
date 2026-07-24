"""
Shared ACAS Xu plane visualization (Plotly + model params).

Used by acas_contract_explorer.py and acas_lasso_explorer.py. No Gradio.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import yaml

# AcasXu_closed_loop/ (image_scripts/ -> figures/ -> AcasXu_closed_loop/)
ACAS_ROOT = Path(__file__).resolve().parent.parent.parent


class AcasVisualizerCommon:
    """
    One place for ACAS plane viz: yaml physics params, labels/colors, Plotly helpers.

    Construct once at app startup: ``VIZ = AcasVisualizerCommon()``.
    """

    ADVISORY_LABELS = {
        "clear": "Clear (CoC)",
        "weak_left": "Weak Left",
        "weak_right": "Weak Right",
        "strong_left": "Strong Left",
        "strong_right": "Strong Right",
    }

    ADVISORY_COLORS = {
        "clear": "#27ae60",
        "weak_left": "#3498db",
        "weak_right": "#9b59b6",
        "strong_left": "#e67e22",
        "strong_right": "#e74c3c",
    }

    QUADRANT_LABELS = {
        "(+,+)": (1, 1),
        "(+,−)": (1, -1),
        "(−,+)": (-1, 1),
        "(−,−)": (-1, -1),
    }

    OWN_SHIP_COLOR = "#2471a3"
    DANGER_CIRCLE_COLOR = "#555555"
    GRID_LINE_COLOR = "#aaaaaa"
    STEM_COLOR = "#2980b9"
    CYCLE_COLOR = "#8e44ad"
    CURRENT_COLOR = "#e74c3c"

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else ACAS_ROOT
        params = self._load_params(self.root)
        physics = params["physics"]

        self.distance_modifier: int = physics["distance_modifier"]
        self.max_dist: int = physics["max_dist"]
        self.max_dist_var: int = self.max_dist // self.distance_modifier
        self.safety_threshold: int = physics["safety_threshold"]
        self.degree_multiplier: float = physics["degree_multiplier"]
        # Match discrete danger cells: round(sqrt) * modifier < threshold.
        self.safety_radius: float = (
            self.safety_threshold / self.distance_modifier - 0.5
        )
        self.advisories: list[str] = list(params["advisories"])

    @staticmethod
    def _load_params(root: Path) -> dict[str, Any]:
        with open(root / "core" / "acas_model_params.yaml", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def heading_degrees(self, heading_own_var: int) -> float:
        return heading_own_var * self.degree_multiplier

    @staticmethod
    def signed_xy(
        x_mag: int, y_mag: int, x_sign: int, y_sign: int,
    ) -> tuple[float, float]:
        return (float(x_sign * x_mag), float(y_sign * y_mag))

    def add_legend_marker(
        self,
        fig: go.Figure,
        color: str,
        name: str,
        symbol: str = "square",
        size: int = 12,
    ) -> None:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(symbol=symbol, size=size, color=color),
            name=name, showlegend=True, hoverinfo="skip",
        ))

    def add_coordinate_axes(self, fig: go.Figure) -> None:
        fig.add_hline(y=0, line=dict(color=self.GRID_LINE_COLOR, width=0.8))
        fig.add_vline(x=0, line=dict(color=self.GRID_LINE_COLOR, width=0.8))

    def add_danger_zone_circle(self, fig: go.Figure) -> None:
        r = self.safety_radius
        fig.add_shape(
            type="circle", x0=-r, y0=-r, x1=r, y1=r,
            line=dict(color=self.DANGER_CIRCLE_COLOR, width=1.5, dash="dash"),
        )
        self.add_legend_marker(
            fig, self.DANGER_CIRCLE_COLOR,
            f"Danger zone (ρ < {self.safety_threshold} ft)",
            symbol="circle",
        )

    def add_ownship_and_heading_arrow(
        self,
        fig: go.Figure,
        heading_own_var: int,
        *,
        arrow_length_cells: float | None = None,
    ) -> tuple[float, float]:
        """Ownship at origin; returns arrow tip (adx, ady) in signed cell units."""
        heading_deg = self.heading_degrees(heading_own_var)
        heading_rad = math.radians(heading_deg)
        if arrow_length_cells is None:
            arrow_length_cells = self.max_dist_var * 0.22
        adx = math.cos(heading_rad) * arrow_length_cells
        ady = math.sin(heading_rad) * arrow_length_cells

        fig.add_trace(go.Scatter(
            x=[0], y=[0], mode="markers",
            marker=dict(size=11, color=self.OWN_SHIP_COLOR),
            name="Ownship (origin)",
        ))
        fig.add_annotation(
            x=adx, y=ady, ax=0, ay=0, axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowwidth=4, arrowcolor=self.OWN_SHIP_COLOR,
        )
        fig.add_annotation(
            x=adx * 1.12, y=ady * 1.12, text=f"{heading_deg:.0f}°",
            showarrow=False, font=dict(size=15, color=self.OWN_SHIP_COLOR),
        )
        return adx, ady

    def panel_axis_range_for_quadrant(
        self,
        x_sign: int,
        y_sign: int,
        lim: float,
        arrow_tip: tuple[float, float],
    ) -> tuple[float, float, float, float]:
        qx0, qx1 = (0.0, lim) if x_sign == 1 else (-lim, 0.0)
        qy0, qy1 = (0.0, lim) if y_sign == 1 else (-lim, 0.0)
        adx, ady = arrow_tip
        pad = 0.5
        return (
            min(qx0, adx - pad), max(qx1, adx + pad),
            min(qy0, ady - pad), max(qy1, ady + pad),
        )

    @staticmethod
    def empty_figure(message: str = "") -> go.Figure:
        fig = go.Figure()
        if message:
            fig.add_annotation(
                text=message, xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False, font=dict(size=14),
            )
        fig.update_layout(margin=dict(t=40, r=10, b=40, l=60))
        return fig

    def apply_physical_plane_layout(
        self,
        fig: go.Figure,
        *,
        x_range: tuple[float, float] | None = None,
        y_range: tuple[float, float] | None = None,
        title: str = "",
    ) -> None:
        lim = self.max_dist_var + 0.5
        if x_range is None:
            x_range = (-lim, lim)
        if y_range is None:
            y_range = (-lim, lim)
        fig.update_xaxes(range=list(x_range), title="x (× 100 ft, signed)")
        fig.update_yaxes(
            range=list(y_range), title="y (× 100 ft, signed)",
            scaleanchor="x", scaleratio=1,
        )
        fig.update_layout(
            plot_bgcolor="#eaf4fb",
            title=dict(text=title, font=dict(size=13)) if title else None,
            legend=dict(x=1.02, y=1, xanchor="left", font=dict(size=10)),
            margin=dict(t=70, r=10, b=40, l=60),
        )

    @staticmethod
    def field_value_html_table(rows: list[tuple[str, str]]) -> str:
        """Simple two-column HTML table for state / detail panels."""
        body = "".join(
            f"<tr><td style='padding:4px 10px 4px 0;color:#555;vertical-align:top'>"
            f"<b>{name}</b></td>"
            f"<td style='padding:4px 0'>{value}</td></tr>"
            for name, value in rows
        )
        return (
            "<table style='font-family:system-ui,sans-serif;font-size:13px;"
            f"border-collapse:collapse'>{body}</table>"
        )
