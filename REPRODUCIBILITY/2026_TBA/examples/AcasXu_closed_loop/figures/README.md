# ACAS Xu Closed Loop Figures

Generated figures and interactive demos for the ACAS Xu compositional
verification example.

All scripts live in `image_scripts/` and write output directly to this
directory. Run all commands from `AcasXu_closed_loop/` (one level up).

---

## Quick start — interactive demo

If you want to understand this example before reading anything else, run
the Gradio contract explorer app:

```bash
cd REPRODUCIBILITY/2026_TBA/examples/AcasXu_closed_loop
pip install gradio          # one-time, if not already installed
python3 figures/image_scripts/acas_contract_explorer.py
# → open http://localhost:7860 in your browser
```

The app opens an interactive dashboard — no CROWN or nuXmv needed.
See [the app section below](#acas_contract_explorerpyinteractive-app) for full details.

---

## Layout

```
figures/
└── image_scripts/
    └── acas_contract_explorer.py     # Interactive Gradio demo app  ← start here
```

The interactive app below is the successor to (and replaces) three earlier static-figure
scripts (`acas_discrete_vs_continuous.py`, `acas_input_region.py`, `acas_output_property.py`)
that have been removed.

---

## `acas_contract_explorer.py` — Interactive app

An interactive Gradio dashboard that lets you explore every A/G contract in
the pre-computed spec file without running any verification. Designed as the
fastest way to build intuition for how the compositional pipeline works.

### What it shows

The UI has one flagship panel (with a click-to-inspect grid alongside it), then
two more panels side by side:

**1 — Original physical space:** The full signed `(x, y)` plane as
the intruder sees it. The active quadrant is highlighted in yellow; a dashed
gray circle marks the danger zone (ρ < 200 ft); a heading arrow from the
origin shows the ownship's current heading. Interactive (zoom/pan/hover) via
Plotly.
- **Viability** (always on) — a 4-category heatmap fill (Interior / Boundary /
  Safe-but-doomed / Unsafe) over the physical state space, computed once from
  the physics alone (no networks). This is the ground truth for whether a state
  is currently safe (ρ ≥ 200 ft), so it isn't an optional overlay -- there's no
  reading of this panel that makes sense without it.
- **Reachability** (togglable) — a solid outline traced around the region
  reachable from the closed loop's fixed initial condition, for the
  currently-selected network. Composable with the `B_R` and `Reach_true`
  marker toggles.

(An earlier version of this panel also drew each contract's own safe/dangerous
states as green/red dots. That was removed: "dangerous" there meant "choosing
this contract's specific forbidden advisory from here causes ρ < 200 next
tick," a one-step-lookahead notion tied to one specific contract that excluded
already-unsafe states outright — a different, narrower question than the
Viability overlay's "Unsafe" category, and showing both invited exactly that
confusion.)

**Inspect a cell (next to Panel 1):** Type an `x_mag`/`y_mag` pair (each in
`[0, 10]`, guard-checked -- out-of-range or empty input reports an error
instead of silently clamping) and click **Inspect** (or press Enter in either
field) to populate the **Cell Detail** panel below with category-specific
justification for that cell, under the current heading/quadrant/a_prev/
selected-contract slice:
- **Unsafe** — the ρ < 200 ft inequality for that cell.
- **Interior** — the trivial ρ ≥ 200 ft fact.
- **Boundary** — the safe advisories (`Allowed_V`) vs. the one forbidden
  choice, showing exactly where that choice leads and why it's unsafe.
- **Safe-but-doomed** — a concrete example trace, walked deterministically
  via the physics alone, showing the state is unsafe no matter which
  advisories are chosen from here on.

(An earlier version used a clickable 11×11 grid instead of typed coordinates.
It was retired because the grid's screen-space rows/columns didn't line up
with Panel 1's signed, zoomed axes, making clicks land on the wrong cell.)

**2 — NN input space:** The CROWN verification region drawn over
NN inputs 1 (normalized distance) and 2 (normalized relative angle):
- **Continuous mode** — filled blue bounding box; one CROWN call covers the
  entire shaded region including non-integer states between grid points.
- **Discrete mode** — faint ghost box for reference; individual labeled red
  dots, one per exact integer dangerous state (each requires its own CROWN call).
- **Both** — overlay of continuous box and discrete points simultaneously.

Dragging the **eps** slider grows or shrinks the bounding box margin live, so
you can see exactly how eps affects the over-approximation.

**3 — Contract details:** A table listing the contract id, heading, quadrant,
forbidden advisory, state count, and all five NN input bounds for the selected
contract, plus a region-info readout (interior/boundary/doomed/unsafe/reachable/
`B_R` cell counts for the currently-selected slice).

### Controls

| Control | Description |
|---|---|
| **Heading (var)** slider | Filter by `heading_own_var` (0–39; multiply by 9° for degrees) |
| **Quadrant** dropdown | Filter by `(x_sign, y_sign)` quadrant |
| **Forbidden advisory** dropdown | Filter by which advisory is forbidden |
| **Min states covered** slider | Show only contracts covering ≥ N dangerous states |
| **Select contract** dropdown | Choose among matching contracts |
| **Verification mode** radio | Continuous / Discrete / Both |
| **eps** slider | Bounding box margin (0 = exact hull, 1e-4 = contract default) |
| **Show state index labels** checkbox | Number each discrete point (CROWN call order) |

### Launch

```bash
cd REPRODUCIBILITY/2026_TBA/examples/AcasXu_closed_loop
python3 figures/image_scripts/acas_contract_explorer.py
```

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--specs` | `contracts/crown/discrete/safety/safety_full_contracts.json` | Contract spec JSON |
| `--port` | `7860` | Gradio server port |
| `--share` | *(flag)* | Create a public shareable Gradio link |

**Requirements:** `gradio>=6.0.0`, `plotly`, `yaml` (no CROWN or nuXmv needed).

