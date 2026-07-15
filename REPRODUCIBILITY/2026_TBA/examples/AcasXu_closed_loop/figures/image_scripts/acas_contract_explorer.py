"""
acas_contract_explorer.py

Interactive Gradio app for exploring ACAS Xu A/G contracts.

Lets you filter and select contracts, toggle between continuous / discrete /
both verification modes, and drag an eps slider to see how the bounding box
grows around the exact dangerous state points.

Usage (from AcasXu_closed_loop/):
    python3 figures/image_scripts/acas_contract_explorer.py

    # Custom specs path or port:
    python3 figures/image_scripts/acas_contract_explorer.py --specs path/to/specs.json
    python3 figures/image_scripts/acas_contract_explorer.py --port 7861
"""

import argparse
import json
import math
import sys
from pathlib import Path

import gradio as gr
import plotly.graph_objects as go
import yaml

# ---------------------------------------------------------------------------
# Reach generate_acas_contracts.py (3 hops up from this script's location)
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).parent.parent.parent   # AcasXu_closed_loop/
sys.path.insert(0, str(_ROOT))
from generate_acas_contracts import ADVISORIES, compute_distance, compute_nn_inputs  # noqa: E402
from acas_viability import SUCC, compute_viability_kernel, is_safe  # noqa: E402
from acas_reachability import compute_reachable_states, reachable_physical_by_aprev  # noqa: E402

# ---------------------------------------------------------------------------
# Model parameters
# ---------------------------------------------------------------------------

def _load_params() -> dict:
    with open(_ROOT / "acas_model_params.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)

_P = _load_params()

DISTANCE_MODIFIER = _P["physics"]["distance_modifier"]
MAX_DIST_VAR      = _P["physics"]["max_dist"] // _P["physics"]["distance_modifier"]
SAFETY_THRESHOLD  = _P["physics"]["safety_threshold"]
DEGREE_MULTIPLIER = _P["physics"]["degree_multiplier"]
# Cell-space danger-zone radius: NOT the naive SAFETY_THRESHOLD/DISTANCE_MODIFIER (2.0).
# compute_distance() rounds before comparing to SAFETY_THRESHOLD, so a state is actually
# dangerous iff round(sqrt(xm^2+ym^2)) * DISTANCE_MODIFIER < SAFETY_THRESHOLD, i.e.
# sqrt(xm^2+ym^2) < SAFETY_THRESHOLD/DISTANCE_MODIFIER - 0.5. Do not "fix" this back to
# SAFETY_THRESHOLD/DISTANCE_MODIFIER -- that would draw a geometrically larger circle that
# no longer matches which grid cells the discrete rule actually classifies as dangerous.
SAFETY_RADIUS     = SAFETY_THRESHOLD / DISTANCE_MODIFIER - 0.5

# ---------------------------------------------------------------------------
# Reachability / viability data (computed once at startup -- see acas_viability.py
# and acas_reachability.py; both are cheap pure-Python fixpoint/BFS computations,
# sub-second, so no JSON caching needed here).
# ---------------------------------------------------------------------------

_KERNEL = compute_viability_kernel()
_R_BY_APREV = reachable_physical_by_aprev(compute_reachable_states())

# Ground-truth trajectory: precomputed by acas_lasso_trajectory.py --dump, since it
# requires onnxruntime + real inference -- a heavier dependency than the rest of this
# app needs. Loaded as plain JSON; empty list (feature quietly unavailable) if missing.
_LASSO_PATH = _ROOT / "acas_lasso_trajectory.json"
if _LASSO_PATH.exists():
    with open(_LASSO_PATH, encoding="utf-8") as f:
        _LASSO_STATES = [(tuple(s), a) for s, a in json.load(f)]
else:
    _LASSO_STATES = []

ADVISORY_LABELS = {
    "clear":        "Clear (CoC)",
    "weak_left":    "Weak Left",
    "weak_right":   "Weak Right",
    "strong_left":  "Strong Left",
    "strong_right": "Strong Right",
}

QUADRANT_LABELS = {
    "(+,+)": (1,  1),
    "(+,−)": (1, -1),
    "(−,+)": (-1,  1),
    "(−,−)": (-1, -1),
}

# Hover tooltip text shown in the contract details HTML table.
_FIELD_TOOLTIPS: dict[str, str] = {
    "Contract id":
        "Unique identifier for this A/G contract in the pre-computed spec file.",
    "heading_own_var":
        "Integer index (0-39) encoding the ownship's current heading. "
        "Multiply by 9° to get degrees. Fixed across all states in this contract.",
    "Quadrant":
        "Sign of the (x, y) relative coordinates, representing which of the four spatial "
        "quadrants the intruder occupies relative to the ownship.",
    "Forbidden advisory":
        "The ACAS Xu advisory the NN must NOT output for any dangerous state "
        "in this contract. CROWN verifies this holds over the entire bounding box.",
    "n_states_covered":
        "Number of dangerous (x_mag, y_mag) integer grid states grouped into "
        "this contract. Continuous mode: 1 CROWN call covers all of them. "
        "Discrete mode: 1 CROWN call per state (short-circuits on first UNSAT).",
    "Bounding box dim 1":
        "Range of NN input 1 (normalized distance) across all dangerous states, "
        "with the eps margin added to each side.",
    "Bounding box dim 2":
        "Range of NN input 2 (normalized relative angle to intruder) across all "
        "dangerous states, with the eps margin added to each side.",
    "NN input 3 (intsc °)":
        "NN input 3: normalized intruder heading angle (intersection angle). "
        "Constant for all states in this contract, which is fixed by heading_own_var and quadrant.",
    "NN input 4 (v_own)":
        "NN input 4: normalized ownship speed. Constant across all contracts "
        "(ownship speed is fixed at 20 raw units).",
    "NN input 5 (v_int)":
        "NN input 5: normalized intruder speed. Constant across all contracts "
        "(intruder speed is fixed at 30 raw units).",
}


def _contract_html_table(rows: list[tuple[str, str]]) -> str:
    """Build an HTML table with hover tooltips on field-name cells."""
    th_style = (
        "padding:6px 10px; text-align:left; background:#f0f0f0; "
        "font-weight:bold; border-bottom:2px solid #ccc;"
    )
    td_field_style = (
        "padding:5px 10px; border-bottom:1px solid #e0e0e0; "
        "font-family:monospace; cursor:help; white-space:nowrap;"
    )
    td_val_style = (
        "padding:5px 10px; border-bottom:1px solid #e0e0e0; "
        "font-family:monospace;"
    )
    rows_html = "".join(
        f'<tr>'
        f'<td style="{td_field_style}" title="{_FIELD_TOOLTIPS.get(field, "")}">'
        f'{field} <span style="color:#999;font-size:0.8em;">ⓘ</span></td>'
        f'<td style="{td_val_style}">{value}</td>'
        f'</tr>'
        for field, value in rows
    )
    return (
        f'<table style="width:100%;border-collapse:collapse;font-size:13px;">'
        f'<thead><tr>'
        f'<th style="{th_style}">Field</th>'
        f'<th style="{th_style}">Value</th>'
        f'</tr></thead>'
        f'<tbody>{rows_html}</tbody>'
        f'</table>'
    )

# ---------------------------------------------------------------------------
# Contract data (loaded once at startup)
# ---------------------------------------------------------------------------

_ALL_CONTRACTS: list[dict] = []
_SPECS_PATH: Path = (
    _ROOT / "contracts/crown/continuous_goals/contract_specs_eps1e4.json"
)


def load_contracts(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)["contracts"]


def _nn_pts(contract: dict) -> list[tuple[float, float]]:
    """Return (nn_input_1, nn_input_2) for each dangerous state."""
    xs, ys = contract["x_sign"], contract["y_sign"]
    hv = contract["heading_own_var"]
    result = []
    for xm, ym in contract["dangerous_xy"]:
        inp = compute_nn_inputs(xm, ym, xs, ys, hv)
        result.append((inp[0], inp[1]))
    return result


# ---------------------------------------------------------------------------
# Filter helpers
# ---------------------------------------------------------------------------

def _heading_choices() -> list[str]:
    headings = sorted({c["heading_own_var"] for c in _ALL_CONTRACTS})
    return ["All"] + [f"{h}  ({h * DEGREE_MULTIPLIER}°)" for h in headings]


def _advisory_choices() -> list[str]:
    advs = sorted({c["forbidden_advisory"] for c in _ALL_CONTRACTS})
    return ["All"] + [ADVISORY_LABELS[a] for a in advs]


def _label_to_advisory(label: str) -> str | None:
    if label == "All":
        return None
    return next(k for k, v in ADVISORY_LABELS.items() if v == label)


def filter_and_list(
    heading_var: int,
    quadrant_label: str,
    advisory_label: str,
    min_states: int,
    aprev_label: str,
) -> tuple[list[str], str]:
    """Return filtered contract dropdown choices and a default selection."""
    q      = QUADRANT_LABELS.get(quadrant_label)
    adv    = _label_to_advisory(advisory_label)
    a_prev = _label_to_advisory(aprev_label)  # same label<->key map as forbidden_advisory

    filtered = [
        c for c in _ALL_CONTRACTS
        if c["heading_own_var"] == int(heading_var)
        and (q  is None or (c["x_sign"] == q[0] and c["y_sign"] == q[1]))
        and (adv is None or c["forbidden_advisory"] == adv)
        and (a_prev is None or c["a_prev"] == a_prev)
        and c["n_states_covered"] >= min_states
    ]

    if not filtered:
        return gr.update(choices=[], value=None), "No contracts match the current filters."

    choices = [
        f"id={c['id']}  head={c['heading_own_var']}  "
        f"quad=({'+' if c['x_sign']==1 else '−'},{'+'  if c['y_sign']==1 else '−'})  "
        f"forbid={c['forbidden_advisory']}  n={c['n_states_covered']}"
        for c in filtered
    ]
    return gr.update(choices=choices, value=choices[0]), ""


def _contract_from_choice(choice: str) -> dict | None:
    if not choice:
        return None
    cid = int(choice.split("id=")[1].split()[0])
    matches = [c for c in _ALL_CONTRACTS if c["id"] == cid]
    return matches[0] if matches else None

# ---------------------------------------------------------------------------
# Panel drawing
# ---------------------------------------------------------------------------

_VIABILITY_COLORS = {"interior": "#a9dfbf", "boundary": "#f1c40f",
                     "safe_but_doomed": "#e67e22", "unsafe": "#c0392b"}
_VIABILITY_ORDER = ["interior", "boundary", "safe_but_doomed", "unsafe"]
_REACHABLE_OUTLINE_COLOR = "#1f6f8b"


def _viability_category(xm: int, ym: int, x_sign: int, y_sign: int, heading_own_var: int) -> str:
    """'interior' | 'boundary' | 'safe_but_doomed' | 'unsafe' for this physical state.
    Single source of truth -- used by the panel-1 heatmap and the Cell Detail builders,
    so both can never disagree on a cell's classification."""
    s = (xm, ym, x_sign, y_sign, heading_own_var)
    if s in _KERNEL.unsafe:
        return "unsafe"
    if s in _KERNEL.safe_but_doomed:
        return "safe_but_doomed"
    if s in _KERNEL.boundary:
        return "boundary"
    return "interior"


def _walk_to_unsafe(
    seed: tuple[int, int, int, int, int], max_steps: int = 200,
) -> list[tuple[int, int, int, int, int]]:
    """From a safe_but_doomed (or any safe) seed state, deterministically walk via SUCC
    until is_safe() is False. At each step, prefer the first advisory (in ADVISORIES
    order) whose successor hasn't been visited yet, else ADVISORIES[0] -- purely to
    avoid a cosmetically-repetitive trace before it terminates.

    Termination is mathematically guaranteed for a safe_but_doomed seed, not just
    likely: compute_viability_kernel()'s fixpoint removes doomed states in rounds, and
    a state removed in round k has every advisory leading either directly to Unsafe or
    to a state removed in an earlier round -- so ANY fixed deterministic choice reaches
    Unsafe within a bounded number of steps. max_steps is defensive-only, not required
    by the math.

    Returns the state sequence ending at the first Unsafe state reached (inclusive)."""
    state = seed
    seen = {state}
    trace = [state]
    for _ in range(max_steps):
        if not is_safe(state):
            return trace
        chosen = next((a for a in ADVISORIES if SUCC[state][a] not in seen), ADVISORIES[0])
        state = SUCC[state][chosen]
        trace.append(state)
        seen.add(state)
    return trace


def _unsafe_detail_html(xm: int, ym: int) -> str:
    rho = compute_distance(xm, ym)
    return _contract_html_table([
        ("Category", "Unsafe"),
        ("ρ (distance)", f"{rho} ft"),
        ("Verdict", f"ρ = {rho} ft &lt; {SAFETY_THRESHOLD} ft -- violates the safety invariant."),
    ])


def _interior_detail_html(xm: int, ym: int) -> str:
    rho = compute_distance(xm, ym)
    return _contract_html_table([
        ("Category", "Interior"),
        ("ρ (distance)", f"{rho} ft"),
        ("Verdict", f"ρ = {rho} ft &gt;= {SAFETY_THRESHOLD} ft. All 5 advisories keep this "
                    "state inside the viability kernel V."),
    ])


def _boundary_detail_html(state: tuple[int, int, int, int, int]) -> str:
    xm, ym = state[0], state[1]
    rho = compute_distance(xm, ym)
    allowed = _KERNEL.allowed[state]
    bad = next(a for a in ADVISORIES if a not in allowed)
    nxt = SUCC[state][bad]
    nxt_rho = compute_distance(nxt[0], nxt[1])
    nxt_cat = _viability_category(nxt[0], nxt[1], nxt[2], nxt[3], nxt[4])
    return _contract_html_table([
        ("Category", "Boundary"),
        ("ρ (distance)", f"{rho} ft -- safe right now"),
        ("A choice that exits V", f"{ADVISORY_LABELS[bad]} &rarr; ({nxt[0]},{nxt[1]}), "
                                   f"ρ={nxt_rho} ft ({nxt_cat})"),
        ("Choices that stay safe forever", ", ".join(ADVISORY_LABELS[a] for a in allowed)),
    ])


def _safe_but_doomed_detail_html(state: tuple[int, int, int, int, int]) -> str:
    trace = _walk_to_unsafe(state)
    rows = [("Category", "Safe-but-doomed"),
            ("Verdict", "Safe right now, but every advisory sequence eventually reaches "
                        "Unsafe. One example trace:")]
    for i, s in enumerate(trace):
        rho = compute_distance(s[0], s[1])
        tag = "Unsafe" if not is_safe(s) else "safe"
        rows.append((f"Step {i}", f"({s[0]},{s[1]}) heading_var={s[4]} -- ρ={rho} ft ({tag})"))
    return _contract_html_table(rows)


def _cell_detail_html(xm: int, ym: int, x_sign: int, y_sign: int, heading_own_var: int) -> str:
    """Dispatches on _viability_category(...) to the four builders above."""
    category = _viability_category(xm, ym, x_sign, y_sign, heading_own_var)
    state = (xm, ym, x_sign, y_sign, heading_own_var)
    if category == "unsafe":
        return _unsafe_detail_html(xm, ym)
    if category == "interior":
        return _interior_detail_html(xm, ym)
    if category == "boundary":
        return _boundary_detail_html(state)
    return _safe_but_doomed_detail_html(state)


def _legend_marker(fig: go.Figure, color: str, name: str, symbol: str = "square", size: int = 12) -> None:
    """Add an invisible marker purely to create a legend entry -- neither go.Heatmap
    nor fig.add_shape() produce one on their own in Plotly."""
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                              marker=dict(symbol=symbol, size=size, color=color),
                              name=name, showlegend=True, hoverinfo="skip"))


def _region_boundary_segments(
    cells: set[tuple[int, int]], x_sign: int, y_sign: int,
) -> list[tuple[float, float, float, float]]:
    """4-connected perimeter edges of `cells` -- (xm, ym) grid cells already filtered to
    one heading/quadrant slice -- as (x0, y0, x1, y1) in signed screen coordinates.
    A cell contributes an edge on each of its 4 sides whose neighbor (off-grid counts
    as absent) is not in `cells`. Tracing is done on the whole-cell set directly (no
    field interpolation), so there are no marching-squares-style ambiguous cases; the
    x_sign/y_sign flip is a lattice reflection applied once, per endpoint, at the end."""
    segments = []
    for xm, ym in cells:
        edges = [
            ((xm - 1, ym), (xm - 0.5, ym - 0.5), (xm - 0.5, ym + 0.5)),  # left
            ((xm + 1, ym), (xm + 0.5, ym - 0.5), (xm + 0.5, ym + 0.5)),  # right
            ((xm, ym - 1), (xm - 0.5, ym - 0.5), (xm + 0.5, ym - 0.5)),  # bottom
            ((xm, ym + 1), (xm - 0.5, ym + 0.5), (xm + 0.5, ym + 0.5)),  # top
        ]
        for neighbor, (mx0, my0), (mx1, my1) in edges:
            if neighbor not in cells:
                segments.append((x_sign * mx0, y_sign * my0, x_sign * mx1, y_sign * my1))
    return segments


def _panel1_axis_range(
    x_sign: int, y_sign: int, lim: float, arrow_tip: tuple[float, float],
) -> tuple[float, float, float, float]:
    """(x0, x1, y0, y1): the union of the active quadrant's box and a padded box around
    the heading-arrow tip, so zooming in to make the grid legible never clips the arrow.
    The Viability heatmap is always shown, so this zoom is always applied."""
    qx0, qx1 = (0.0, lim) if x_sign == 1 else (-lim, 0.0)
    qy0, qy1 = (0.0, lim) if y_sign == 1 else (-lim, 0.0)
    adx, ady = arrow_tip
    pad = 0.5
    return (min(qx0, adx - pad), max(qx1, adx + pad),
            min(qy0, ady - pad), max(qy1, ady + pad))


def _draw_physical_original(
    contract: dict | None,
    *,
    heading_own_var: int,
    x_sign: int,
    y_sign: int,
    a_prev: str,
    reachability_choice: str = "Off",
    highlight_br: bool = False,
    show_lasso: bool = False,
) -> go.Figure:
    """
    Panel 1 — Original physical space.

    Shows the ownship at the origin with a heading arrow pointing in its true
    direction. `contract` may be None (overlay-only browsing); x_sign/y_sign/
    heading_own_var are passed in explicitly (not read off the contract) so the
    slice is a single source of truth shared with the overlay controls below.

    The Viability fill (go.Heatmap, one trace for the whole grid) is always shown --
    it is the ground truth for whether a state is currently safe, not an optional
    overlay. Reachability is an independent, non-competing channel: an outline traced
    around the reachable region's perimeter, left un-filled so the viability fill
    underneath stays visible; togglable, and composable with the viability fill.
    B_R and Reach_true are marker overlays on top of both.

    Heading convention (matches the BehaVerify DSL):
      0° = East (+x), angles increase counter-clockwise.
    """
    heading_deg = heading_own_var * DEGREE_MULTIPLIER
    heading_rad = math.radians(heading_deg)
    lim = MAX_DIST_VAR + 0.5

    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color="#aaaaaa", width=0.8))
    fig.add_vline(x=0, line=dict(color="#aaaaaa", width=0.8))

    # --- Viability fill (go.Heatmap, one trace for the whole grid) -- always shown ---
    xs_mag = list(range(MAX_DIST_VAR + 1))
    ys_mag = list(range(MAX_DIST_VAR + 1))
    cat_idx = {name: i for i, name in enumerate(_VIABILITY_ORDER)}

    def _category(xm, ym):
        return _viability_category(xm, ym, x_sign, y_sign, heading_own_var)

    z_mag = [[cat_idx[_category(xm, ym)] for xm in xs_mag] for ym in ys_mag]

    # Screen coords: x_sign/y_sign = -1 reverses ascending order, so sort explicitly
    # rather than assume mag-ascending order is already screen-ascending order.
    x_screen = [x_sign * xm for xm in xs_mag]
    y_screen = [y_sign * ym for ym in ys_mag]
    x_order = sorted(range(len(xs_mag)), key=lambda i: x_screen[i])
    y_order = sorted(range(len(ys_mag)), key=lambda i: y_screen[i])

    n = len(_VIABILITY_ORDER)
    colorscale = []
    for i, name in enumerate(_VIABILITY_ORDER):
        colorscale.append([i / n, _VIABILITY_COLORS[name]])
        colorscale.append([(i + 1) / n, _VIABILITY_COLORS[name]])

    # hoverinfo="skip": the heatmap covers every pixel of every cell, so if it also
    # had per-cell hover text it would always win over marker traces drawn on top at
    # the exact same point (Reach_true, B_R, dangerous states) -- their hover would
    # never fire. Per-cell category info is now available via the click-grid instead.
    fig.add_trace(go.Heatmap(
        x=[x_screen[i] for i in x_order], y=[y_screen[i] for i in y_order],
        z=[[z_mag[iy][ix] for ix in x_order] for iy in y_order],
        zmin=0, zmax=n - 1, colorscale=colorscale, showscale=False,
        xgap=2, ygap=2, opacity=0.65,
        hoverinfo="skip",
    ))
    for name in _VIABILITY_ORDER:
        _legend_marker(fig, _VIABILITY_COLORS[name], name.replace("_", " ").title())

    # --- Reachability outline (perimeter of the reachable region, not a fill) ---
    if reachability_choice != "Off":
        reachable_here = _R_BY_APREV.get(a_prev, frozenset())
        cells = {(xm, ym) for (xm, ym, xs, ys, h) in reachable_here
                 if xs == x_sign and ys == y_sign and h == heading_own_var}
        segments = _region_boundary_segments(cells, x_sign, y_sign)
        xs_line, ys_line = [], []
        for x0, y0, x1, y1 in segments:
            xs_line += [x0, x1, None]
            ys_line += [y0, y1, None]
        fig.add_trace(go.Scatter(
            x=xs_line, y=ys_line, mode="lines",
            line=dict(color=_REACHABLE_OUTLINE_COLOR, width=2.5),
            name=f"Reachable region boundary ({a_prev}, {len(cells)} cells)",
            showlegend=True, hoverinfo="skip",
        ))

    # --- Contract quadrant highlight ---
    qx0 = 0.0 if x_sign == 1 else -lim
    qy0 = 0.0 if y_sign == 1 else -lim
    fig.add_shape(type="rect", x0=qx0, y0=qy0, x1=qx0 + lim, y1=qy0 + lim,
                  fillcolor="#fff9c4", opacity=0.55,
                  line=dict(color="#f39c12", width=1.5), layer="below")
    _legend_marker(fig, "#fff9c4", "Contract quadrant")

    # --- Danger-zone boundary (rho < SAFETY_THRESHOLD) -- outline only, not filled: the
    # invariant is rho >= SAFETY_THRESHOLD (safe); this circle shows where it's VIOLATED,
    # so it must not be labeled/colored like the invariant itself or like a real "region"
    # duplicating the Viability overlay's own (exact, per-cell) Unsafe category.
    radius = SAFETY_RADIUS
    fig.add_shape(type="circle", x0=-radius, y0=-radius, x1=radius, y1=radius,
                  line=dict(color="#555555", width=1.5, dash="dash"))
    _legend_marker(fig, "#555555", f"Danger zone (ρ < {SAFETY_THRESHOLD} ft)", symbol="circle")

    # --- Ownship + heading arrow ---
    fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers",
                              marker=dict(size=11, color="#2471a3"), name="Ownship (origin)"))

    arrow_len = MAX_DIST_VAR * 0.22
    adx = math.cos(heading_rad) * arrow_len
    ady = math.sin(heading_rad) * arrow_len
    fig.add_annotation(x=adx, y=ady, ax=0, ay=0, axref="x", ayref="y",
                        showarrow=True, arrowhead=2, arrowwidth=4, arrowcolor="#2471a3")
    fig.add_annotation(x=adx * 1.12, y=ady * 1.12, text=f"{heading_deg}°",
                        showarrow=False, font=dict(size=15, color="#2471a3"))

    # --- B_R = boundary(V) ∩ R(a_prev) highlight ---
    if highlight_br:
        br_cells = _KERNEL.boundary & _R_BY_APREV.get(a_prev, frozenset())
        br_here = [(xm, ym) for (xm, ym, xs, ys, h) in br_cells
                   if xs == x_sign and ys == y_sign and h == heading_own_var]
        if br_here:
            fig.add_trace(go.Scatter(
                x=[x_sign * xm for xm, ym in br_here], y=[y_sign * ym for xm, ym in br_here],
                mode="markers",
                marker=dict(symbol="star", size=22, color="gold", line=dict(color="black", width=1.2)),
                name=f"B_R = ∂V∩R ({len(br_here)})",
            ))

    # --- True trajectory (Reach_true) markers ---
    if show_lasso:
        lasso_here = [(s[0], s[1]) for s, _ap in _LASSO_STATES
                      if s[2] == x_sign and s[3] == y_sign and s[4] == heading_own_var]
        if lasso_here:
            fig.add_trace(go.Scatter(
                x=[x_sign * xm for xm, ym in lasso_here], y=[y_sign * ym for xm, ym in lasso_here],
                mode="markers",
                marker=dict(symbol="diamond", size=16, color="#8e44ad", line=dict(color="black", width=1.0)),
                name=f"Reach_true ({len(lasso_here)})",
                text=[f"Reach_true: ({xm},{ym})" for xm, ym in lasso_here],
                hoverinfo="text",
            ))

    sign_x = "+" if x_sign == 1 else "−"
    sign_y = "+" if y_sign == 1 else "−"
    x0, x1, y0, y1 = _panel1_axis_range(x_sign, y_sign, lim, (adx, ady))
    title_forbid = f"<br>forbidden: {ADVISORY_LABELS[contract['forbidden_advisory']]}" if contract is not None else ""
    fig.update_xaxes(range=[x0, x1], title="x (× 100 ft, signed)")
    fig.update_yaxes(range=[y0, y1], title="y (× 100 ft, signed)", scaleanchor="x", scaleratio=1)
    fig.update_layout(
        plot_bgcolor="#eaf4fb",
        title=dict(text=(f"Original physical space<br>"
                          f"ownship heading={heading_deg}°  quad=({sign_x},{sign_y}){title_forbid}"), font=dict(size=13)),
        legend=dict(x=1.02, y=1, xanchor="left", font=dict(size=10)),
        margin=dict(t=70, r=10, b=40, l=60),
    )
    return fig


def _axis_limits(contract: dict, eps: float) -> tuple[float, float, float, float]:
    lower = contract["nn_input_lower"]
    upper = contract["nn_input_upper"]
    # Expand by eps on each side (mirrors how bounding box is built)
    bx0, bx1 = lower[0] - eps, upper[0] + eps
    by0, by1 = lower[1] - eps, upper[1] + eps
    pad_x = max((bx1 - bx0) * 0.5, 0.02)
    pad_y = max((by1 - by0) * 0.5, 0.02)
    return bx0 - pad_x, bx1 + pad_x, by0 - pad_y, by1 + pad_y


def _draw_nn_space(
    contract: dict,
    pts: list[tuple[float, float]],
    mode: str,          # "Continuous", "Discrete", "Both"
    eps: float,
    show_labels: bool,
) -> go.Figure:
    lower = contract["nn_input_lower"]
    upper = contract["nn_input_upper"]

    # Bounding box with live eps applied
    bx0 = lower[0] - eps
    bx1 = upper[0] + eps
    by0 = lower[1] - eps
    by1 = upper[1] + eps

    xlim0, xlim1, ylim0, ylim1 = _axis_limits(contract, eps)

    fig = go.Figure()

    if mode in ("Continuous", "Both"):
        fig.add_shape(type="rect", x0=bx0, y0=by0, x1=bx1, y1=by1,
                      fillcolor="#d6eaf8", opacity=0.55 if mode == "Both" else 0.65,
                      line=dict(color="#2e86c1", width=2.0))
        _legend_marker(fig, "#d6eaf8", f"CROWN bounding box  (eps={eps:.0e})")
        fig.add_trace(go.Scatter(x=[bx0, bx1], y=[by0, by1], mode="markers",
                                  marker=dict(symbol="x", size=10, color="#2e86c1", line=dict(width=1.5)),
                                  showlegend=False, hoverinfo="skip"))

    elif mode == "Discrete":
        fig.add_shape(type="rect", x0=bx0, y0=by0, x1=bx1, y1=by1,
                      fillcolor="#d6eaf8", opacity=0.10,
                      line=dict(color="#2e86c1", width=1.5, dash="dash"))
        _legend_marker(fig, "#d6eaf8", "Continuous box (reference)")

    if mode in ("Discrete", "Both"):
        n = contract["n_states_covered"]
        fig.add_trace(go.Scatter(
            x=[p[0] for p in pts], y=[p[1] for p in pts],
            mode="markers+text" if show_labels else "markers",
            marker=dict(size=13, color="#c0392b"),
            text=[str(k) for k in range(1, len(pts) + 1)] if show_labels else None,
            textposition="top right", textfont=dict(size=9, color="#7b241c"),
            name=f"Exact state queries ({n})",
        ))

    if mode == "Continuous":
        fig.add_trace(go.Scatter(
            x=[p[0] for p in pts], y=[p[1] for p in pts], mode="markers",
            marker=dict(size=10, color="#c0392b"),
            name=f"Dangerous states ({contract['n_states_covered']})",
        ))
        info_text = f"1 CROWN call<br>eps = {eps:.2e}"
    elif mode == "Discrete":
        info_text = f"{contract['n_states_covered']} CROWN calls<br>one per exact integer state"
    else:  # Both
        info_text = f"Continuous: 1 call  (eps={eps:.2e})<br>Discrete: {contract['n_states_covered']} calls"

    fig.add_annotation(xref="paper", yref="paper", x=0.97, y=0.03, text=info_text,
                        showarrow=False, align="right", font=dict(size=10),
                        bgcolor="white", opacity=0.85, bordercolor="#aaaaaa", borderpad=4)

    fig.update_xaxes(range=[xlim0, xlim1], title="NN input 1: normalized distance",
                      showgrid=True, gridcolor="#dddddd", griddash="dash")
    fig.update_yaxes(range=[ylim0, ylim1], title="NN input 2: normalized relative angle",
                      showgrid=True, gridcolor="#dddddd", griddash="dash")
    fig.update_layout(
        title=dict(text=f"NN input space  [{mode} mode]", font=dict(size=13)),
        legend=dict(x=1.02, y=1, xanchor="left", font=dict(size=10)),
        margin=dict(t=50, r=10, b=40, l=60),
    )
    return fig

# ---------------------------------------------------------------------------
# Main render function (called by Gradio)
# ---------------------------------------------------------------------------

def _empty_fig(msg: str = "") -> go.Figure:
    fig = go.Figure()
    if msg:
        fig.add_annotation(xref="paper", yref="paper", x=0.5, y=0.5, text=msg,
                            showarrow=False, font=dict(size=13))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def _slice_membership(region: frozenset, x_sign: int, y_sign: int, heading_own_var: int) -> int:
    """Count of (x_mag, y_mag) grid cells in `region` (a frozenset of physical 5-tuples)
    matching this slice. Shared by the panel-1 heatmap and the region-info readout."""
    return sum(
        1 for xm in range(MAX_DIST_VAR + 1) for ym in range(MAX_DIST_VAR + 1)
        if (xm, ym, x_sign, y_sign, heading_own_var) in region
    )


def _current_slice(contract_choice: str, heading_var, quadrant_label: str) -> tuple[int, int, int]:
    """(heading_own_var, x_sign, y_sign) -- single source of truth for "what slice is
    currently active," used by both render() and on_xy_select() so the Cell Detail
    lookup can never disagree with what Panel 1 is actually showing."""
    contract = _contract_from_choice(contract_choice)
    if contract is not None:
        return contract["heading_own_var"], contract["x_sign"], contract["y_sign"]
    x_sign, y_sign = QUADRANT_LABELS.get(quadrant_label, (1, 1))  # "All" -> (+,+)
    return int(heading_var), x_sign, y_sign


_CELL_DETAIL_PLACEHOLDER = f"Enter x_mag/y_mag (0-{MAX_DIST_VAR}) above and click Inspect."


def render(
    contract_choice: str,
    mode: str,
    eps: float,
    show_labels: bool,
    heading_var: int,
    quadrant_label: str,
    aprev_label: str,
    reachability_choice: str,
    highlight_br: bool,
    show_lasso: bool,
) -> tuple:
    """Return (fig_orig, fig_nn, html, region_html, cell_detail) for the five Gradio
    panels. cell_detail resets to the placeholder whenever the slice changes, so an
    inspection made under a previous heading/quadrant/a_prev never leaves stale detail
    on screen."""
    contract = _contract_from_choice(contract_choice)
    a_prev = _label_to_advisory(aprev_label) or "clear"
    slice_heading, slice_x_sign, slice_y_sign = _current_slice(contract_choice, heading_var, quadrant_label)

    fig_orig = _draw_physical_original(
        contract,
        heading_own_var=slice_heading, x_sign=slice_x_sign, y_sign=slice_y_sign,
        a_prev=a_prev,
        reachability_choice=reachability_choice, highlight_br=highlight_br, show_lasso=show_lasso,
    )

    reachable_here = _R_BY_APREV.get(a_prev, frozenset())
    br_cells = _KERNEL.boundary & reachable_here
    region_rows = [
        ("Current slice", f"h={slice_heading}, quad=({'+' if slice_x_sign==1 else '−'},"
                           f"{'+' if slice_y_sign==1 else '−'}), a_prev={a_prev}"),
        ("Interior(V) cells", str(_slice_membership(_KERNEL.interior, slice_x_sign, slice_y_sign, slice_heading))),
        ("Boundary(V) cells", str(_slice_membership(_KERNEL.boundary, slice_x_sign, slice_y_sign, slice_heading))),
        ("Safe-but-doomed cells", str(_slice_membership(_KERNEL.safe_but_doomed, slice_x_sign, slice_y_sign, slice_heading))),
        ("Unsafe cells", str(_slice_membership(_KERNEL.unsafe, slice_x_sign, slice_y_sign, slice_heading))),
        ("Reachable cells (this a_prev)", str(_slice_membership(reachable_here, slice_x_sign, slice_y_sign, slice_heading))),
        ("B_R cells in this slice", str(_slice_membership(br_cells, slice_x_sign, slice_y_sign, slice_heading))),
    ]
    region_html = _contract_html_table(region_rows)

    if contract is None:
        empty = _empty_fig("No contract selected.")
        return fig_orig, empty, "", region_html, _CELL_DETAIL_PLACEHOLDER

    pts = _nn_pts(contract)
    heading_deg = contract["heading_own_var"] * DEGREE_MULTIPLIER
    sign_x = "+" if contract["x_sign"] == 1 else "−"
    sign_y = "+" if contract["y_sign"] == 1 else "−"

    fig_nn = _draw_nn_space(contract, pts, mode, eps, show_labels)

    # Panel 4 — contract metadata as an HTML table with hover tooltips
    lower = contract["nn_input_lower"]
    upper = contract["nn_input_upper"]
    table_rows = [
        ("Contract id",          str(contract["id"])),
        ("heading_own_var",      f"{contract['heading_own_var']} ({heading_deg}°)"),
        ("Quadrant",             f"({sign_x}, {sign_y})"),
        ("Forbidden advisory",   ADVISORY_LABELS[contract["forbidden_advisory"]]),
        ("n_states_covered",     str(contract["n_states_covered"])),
        ("Bounding box dim 1",   f"[{lower[0]:.4f}, {upper[0]:.4f}]"),
        ("Bounding box dim 2",   f"[{lower[1]:.4f}, {upper[1]:.4f}]"),
        ("NN input 3 (intsc °)", f"[{lower[2]:.4f}, {upper[2]:.4f}]"),
        ("NN input 4 (v_own)",   f"{lower[3]:.4f}"),
        ("NN input 5 (v_int)",   f"{lower[4]:.4f}"),
    ]

    return fig_orig, fig_nn, _contract_html_table(table_rows), region_html, _CELL_DETAIL_PLACEHOLDER


def on_xy_select(x_mag, y_mag, heading_var, quadrant_label, aprev_label, contract_choice) -> str:
    """Cell Detail lookup for manually-entered x_mag/y_mag. Guard checks: gr.Number can
    hand back None (empty field) or an out-of-range/non-integer value if the user types
    something odd (spinner arrows still clamp, but typed values bypass that), so validate
    explicitly rather than trusting the widget's min/max/step alone."""
    if x_mag is None or y_mag is None:
        return "Enter both x_mag and y_mag."
    xm, ym = int(round(x_mag)), int(round(y_mag))
    if not (0 <= xm <= MAX_DIST_VAR) or not (0 <= ym <= MAX_DIST_VAR):
        return f"x_mag and y_mag must both be in [0, {MAX_DIST_VAR}]."
    slice_heading, x_sign, y_sign = _current_slice(contract_choice, heading_var, quadrant_label)
    return _cell_detail_html(xm, ym, x_sign, y_sign, slice_heading)

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    advisory_choices = _advisory_choices()
    quadrant_choices = ["All"] + list(QUADRANT_LABELS.keys())

    # Pick a default heading_var that has at least one contract with >= 5 states
    default_heading = next(
        (c["heading_own_var"] for c in _ALL_CONTRACTS if c["n_states_covered"] >= 5),
        _ALL_CONTRACTS[0]["heading_own_var"] if _ALL_CONTRACTS else 0,
    )
    max_heading = max(c["heading_own_var"] for c in _ALL_CONTRACTS)

    # Build initial contract list for the default heading
    _init_choices, _ = filter_and_list(default_heading, "All", "All", 1, "All")
    init_choices = _init_choices["choices"]
    init_value   = _init_choices["value"]

    with gr.Blocks(title="ACAS Xu Contract Explorer") as demo:
        gr.Markdown(
            "## ACAS Xu Assume-Guarantee Contract Explorer\n"
            "Filter contracts by heading, quadrant, and forbidden advisory. "
            "Select a contract to visualize its input region under continuous "
            "and discrete verification modes. Drag the **eps** slider to see "
            "how the bounding box grows around the exact dangerous state points."
        )

        with gr.Row():
            # ── Left column: filters + contract picker ──────────────────────
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("### Filters")
                heading_sl = gr.Slider(
                    minimum=0, maximum=max_heading, step=1,
                    value=default_heading,
                    label="Heading (var)",
                    info=f"heading_own_var × {DEGREE_MULTIPLIER}° = actual heading")
                quadrant_dd = gr.Dropdown(
                    quadrant_choices, value="All", label="Quadrant")
                advisory_dd = gr.Dropdown(
                    advisory_choices, value="All", label="Forbidden advisory")
                aprev_dd = gr.Dropdown(
                    ["All"] + [ADVISORY_LABELS[a] for a in ADVISORIES],
                    value=ADVISORY_LABELS["clear"], label="a_prev (active network)",
                    info="Also drives the Reachability overlay and B_R highlight below")
                min_states_sl = gr.Slider(
                    1, 20, value=1, step=1, label="Min states covered")

                gr.Markdown("### Contract")
                contract_dd = gr.Dropdown(
                    init_choices, value=init_value,
                    label="Select contract", interactive=True)
                filter_status = gr.Markdown("")

                gr.Markdown("### Display")
                mode_radio = gr.Radio(
                    ["Continuous", "Discrete", "Both"],
                    value="Both", label="Verification mode")
                eps_sl = gr.Slider(
                    minimum=0.0, maximum=0.05, value=1e-4, step=1e-5,
                    label="eps (bounding box margin)",
                    info="0 = tight hull of exact points; 1e-4 = contract default")
                labels_cb = gr.Checkbox(
                    value=True, label="Show state index labels (discrete)")

                gr.Markdown("### Reachability\n")
                reachability_radio = gr.Radio(
                    ["Off", "Reachability (for selected a_prev)"],
                    value="Off", label="Reachability overlay",
                    info="Depends on a_prev above")
                br_cb = gr.Checkbox(
                    value=True, label="Highlight reachable boundary (B_R = ∂V ∩ R)")
                lasso_cb = gr.Checkbox(
                    value=False, label="Show true trajectory markers (Reach_true)")
                region_info_html = gr.HTML()

            # ── Right column: panel 1 (flagship view) + click-grid side by side, then 3+4 ──
            with gr.Column(scale=3):
                with gr.Row():
                    plot_orig = gr.Plot(label="1 — Original physical space")
                    with gr.Column(min_width=260):
                        gr.Markdown("#### Inspect a cell")
                        gr.Markdown(
                            f"*Enter magnitudes in [0, {MAX_DIST_VAR}]; sign/heading "
                            "come from the filters at left.*",
                            elem_classes=["hint-text"],
                        )
                        with gr.Row():
                            x_mag_in = gr.Number(
                                value=0, minimum=0, maximum=MAX_DIST_VAR, step=1,
                                precision=0, label="x_mag")
                            y_mag_in = gr.Number(
                                value=0, minimum=0, maximum=MAX_DIST_VAR, step=1,
                                precision=0, label="y_mag")
                        inspect_btn = gr.Button("Inspect", size="sm")
                        gr.Markdown("#### Cell Detail")
                        cell_detail_html = gr.HTML()
                with gr.Row():
                    plot_nn = gr.Plot(label="2: NN input space")
                    with gr.Column(min_width=260):
                        gr.Markdown("#### Contract Details")
                        gr.Markdown(
                            "*Hover over a field name for a description.*",
                            elem_classes=["hint-text"],
                        )
                        info_html = gr.HTML()

        render_outputs = [plot_orig, plot_nn, info_html, region_info_html, cell_detail_html]

        # ── Wire up filters → contract list ─────────────────────────────────
        filter_inputs = [heading_sl, quadrant_dd, advisory_dd, min_states_sl, aprev_dd]
        for ctrl in filter_inputs:
            ctrl.change(
                fn=filter_and_list,
                inputs=filter_inputs,
                outputs=[contract_dd, filter_status],
            )

        # ── Wire up contract/mode/eps/labels/reachability → panels ──────────
        render_inputs = [
            contract_dd, mode_radio, eps_sl, labels_cb,
            heading_sl, quadrant_dd, aprev_dd,
            reachability_radio, br_cb, lasso_cb,
        ]
        for ctrl in render_inputs:
            ctrl.change(
                fn=render,
                inputs=render_inputs,
                outputs=render_outputs,
            )

        # ── x_mag/y_mag entry → Cell Detail panel ───────────────────────────
        xy_inputs = [x_mag_in, y_mag_in, heading_sl, quadrant_dd, aprev_dd, contract_dd]
        for trigger in (inspect_btn.click, x_mag_in.submit, y_mag_in.submit):
            trigger(fn=on_xy_select, inputs=xy_inputs, outputs=[cell_detail_html])

        # ── Initial render ───────────────────────────────────────────────────
        demo.load(
            fn=render,
            inputs=render_inputs,
            outputs=render_outputs,
        )

    return demo

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    global _ALL_CONTRACTS, _SPECS_PATH

    parser = argparse.ArgumentParser(
        description="Interactive ACAS Xu contract explorer (Gradio)."
    )
    parser.add_argument(
        "--specs", type=Path, default=_SPECS_PATH,
        help="Path to contract specs JSON (default: contracts/crown/continuous_goals/contract_specs_eps1e4.json)",
    )
    parser.add_argument(
        "--port", type=int, default=7860,
        help="Port to serve the Gradio app on (default: 7860)",
    )
    parser.add_argument(
        "--share", action="store_true",
        help="Create a public shareable Gradio link",
    )
    args = parser.parse_args()

    _SPECS_PATH    = Path(args.specs).resolve()
    _ALL_CONTRACTS = load_contracts(_SPECS_PATH)

    print(f"Loaded {len(_ALL_CONTRACTS)} contracts from {_SPECS_PATH}")
    print(f"Serving on http://localhost:{args.port}")

    demo = build_ui()
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
