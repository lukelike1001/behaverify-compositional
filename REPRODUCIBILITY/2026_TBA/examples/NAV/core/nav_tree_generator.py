"""
nav_tree_generator.py

Writes the NAV NSBT as a BehaVerify `.tree` file.

The tree cannot be hand-written: the trig tables are one line per heading cell
and every constant changes with resolution. This module owns that text and
nothing else -- the lattice and physics live in nav_domain, the state box in
nav_boundary_finder, and running the toolchain in monolithic/nav_table_pipeline.

ONE GENERATOR, TWO PIPELINES
----------------------------
Monolithic and compositional NAV differ in exactly one declaration:

  monolithic     the control is a NEURAL regression variable with a lookup
                 table, so every cell stores one control value.
  compositional  the control is a free VAR assigned nondeterministically, to be
                 constrained afterwards by INVAR contracts.

Everything else -- physics, clock, bounds checking, specification -- is shared,
which is what makes the comparison isolate the neural encoding rather than
incidental modelling differences. Emitting the compositional tree with the
control already free also sidesteps `dsl_with_contracts_to_nuxmv`'s
build-the-table-then-strip-it approach, which cannot run on a continuous domain.

THE TREE
--------
One action node, which is both root and only leaf: read the control, write it to
the blackboard, return success. The environment then moves the robot. There are
no check nodes, because the controller has no sensors -- the obstacle and goal
are baked into its weights, not supplied as inputs -- so a branch on obstacle
membership would model a robot that does not exist. Detection belongs in the
specification. See reports/NAV/2026_09_22_nsbt_design.md section 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from core.nav_domain import (
    CONTROL_SCALE,
    NavDiscreteDynamics,
    NavGrid,
)

MONOLITHIC = "monolithic"
COMPOSITIONAL = "compositional"

# Blackboard / environment variable names used in the emitted tree.
TICK_COUNT = "tick_count"
ROBOT_X, ROBOT_Y, SPEED, HEADING = "robot_x", "robot_y", "speed", "heading"
OVERFLOW = "overflow"
ACCEL, TURN = "u_accel", "u_turn"
NETWORK = "network"

AXIS_TO_VARIABLE = {"x": ROBOT_X, "y": ROBOT_Y, "v": SPEED, "theta": HEADING}


@dataclass(frozen=True)
class NavTreeSpec:
    """Everything the generator needs that is not already in grid/dynamics."""

    horizon_steps: int
    initial_indices: tuple[int, int, int, int]
    obstacle_indices: dict[str, tuple[int, int]]
    goal_indices: dict[str, tuple[int, int]]
    onnx_path: str


def _indent(level: int) -> str:
    return "    " * level


class NavTreeGenerator:
    """Renders a NAV `.tree` file for one grid, one network, and one mode."""

    def __init__(
        self,
        grid: NavGrid,
        dynamics: NavDiscreteDynamics,
        spec: NavTreeSpec,
        mode: str = MONOLITHIC,
    ) -> None:
        if mode not in {MONOLITHIC, COMPOSITIONAL}:
            raise ValueError(f"mode must be {MONOLITHIC!r} or {COMPOSITIONAL!r}")
        self.grid = grid
        self.dynamics = dynamics
        self.spec = spec
        self.mode = mode
        self._validate()

    # ------------------------------------------------------------ validation

    def _validate(self) -> None:
        """
        Refuse to emit a tree that cannot answer the question.

        Both checks come from spike findings: an initial state outside the box
        makes nuXmv abandon its command script and wait at a prompt (looking
        like a fast clean run with no verdict), and a resolution below the
        quantization floor freezes the robot so that every verdict is about a
        model that never moves.
        """
        if not self.grid.contains(self.spec.initial_indices):
            raise ValueError(
                "initial state "
                f"{self.spec.initial_indices} lies outside the declared box "
                f"{[(a.lower_index, a.upper_index) for a in self.grid.axes]}; "
                "nuXmv would reject the assignment and abandon its script"
            )
        quantization = self.dynamics.quantization()
        if not quantization.is_representable:
            frozen = ", ".join(quantization.frozen_axes)
            raise ValueError(
                f"cell size too coarse on: {frozen}. A maximal one-step change "
                f"truncates to zero there, so the robot cannot move. "
                f"cells per step = "
                f"{ {k: round(v, 3) for k, v in quantization.cells_per_step.items()} }"
            )

    # ---------------------------------------------------------- tree sections

    def _constants(self) -> str:
        axis_bounds = ", ".join(
            f"{AXIS_TO_VARIABLE[a.name]}_lo := {a.lower_index}, "
            f"{AXIS_TO_VARIABLE[a.name]}_hi := {a.upper_index}"
            for a in self.grid.axes
        )
        return (
            f"constants {{\n"
            f"{_indent(1)}horizon := {self.spec.horizon_steps},\n"
            f"{_indent(1)}{axis_bounds},\n"
            f"{_indent(1)}heading_offset := {self.dynamics.theta_table_offset},\n"
            f"{_indent(1)}heading_cells := {self.grid.axis('theta').cell_count},\n"
            f"{_indent(1)}accel_divisor := {self.dynamics.velocity_divisor},\n"
            f"{_indent(1)}turn_divisor := {self.dynamics.heading_divisor},\n"
            f"{_indent(1)}x_divisor := {self.dynamics.x_divisor},\n"
            f"{_indent(1)}y_divisor := {self.dynamics.y_divisor}\n"
            f"}} end_constants\n"
        )

    def _trig_array(self, name: str, values: Sequence[int]) -> str:
        rows = "\n".join(
            f"{_indent(2)}condition {{(eq, cell, {index})}} "
            f"assign{{result{{{value}}}}}"
            for index, value in enumerate(values)
        )
        return (
            f"{_indent(1)}variable {{ env {name} DEFINE INT static "
            f"array heading_cells iterative_assign, cell\n"
            f"{rows}\n"
            f"{_indent(2)}assign{{result{{0}}}}\n"
            f"{_indent(1)}}}\n"
        )

    def _control_declaration(self) -> str:
        """The one declaration that differs between the two pipelines."""
        if self.mode == MONOLITHIC:
            return (
                f"{_indent(1)}variable {{ bl {NETWORK} NEURAL regression INT\n"
                f"{_indent(2)}inputs {{ {ROBOT_X}, {ROBOT_Y}, {SPEED}, {HEADING} }}\n"
                f"{_indent(2)}num_outputs {{ 2 }}\n"
                f"{_indent(2)}config {{ table }}\n"
                f"{_indent(2)}source {{ '{self.spec.onnx_path}' }}\n"
                f"{_indent(1)}}}\n"
            )
        return (
            f"{_indent(1)}#{{ compositional: the control is unconstrained here "
            f"and restricted by injected INVAR contracts }}#\n"
        )

    def _variables(self) -> str:
        bound = CONTROL_SCALE
        raw = self._raw_successor_expressions()
        return (
            "variables {\n"
            f"{_indent(1)}variable {{ env {TICK_COUNT} VAR [0, horizon] "
            f"assign{{result{{0}}}}}}\n"
            + "".join(
                f"{_indent(1)}variable {{ env {AXIS_TO_VARIABLE[a.name]} VAR "
                f"[{AXIS_TO_VARIABLE[a.name]}_lo, {AXIS_TO_VARIABLE[a.name]}_hi] "
                f"assign{{result{{{initial}}}}}}}\n"
                for a, initial in zip(self.grid.axes, self.spec.initial_indices)
            )
            + f"{_indent(1)}variable {{ env {OVERFLOW} VAR BOOLEAN "
              f"assign{{result{{False}}}}}}\n"
            + self._trig_array("cos_table", self.dynamics.cosine_table)
            + self._trig_array("sin_table", self.dynamics.sine_table)
            + self._control_declaration()
            + f"{_indent(1)}variable {{ bl {ACCEL} VAR [{-bound}, {bound}] "
              f"assign{{result{{0}}}}}}\n"
            + f"{_indent(1)}variable {{ bl {TURN} VAR [{-bound}, {bound}] "
              f"assign{{result{{0}}}}}}\n"
            + "".join(
                f"{_indent(1)}variable {{ env {name}_raw DEFINE INT "
                f"assign{{result{{{expression}}}}}}}\n"
                for name, expression in raw.items()
            )
            + "} end_variables\n"
        )

    def _raw_successor_expressions(self) -> dict[str, str]:
        """
        Unclamped successors, as DEFINEs.

        Explicit Euler: position reads the pre-update speed and heading, so
        every successor depends only on stage-0 state. That is what lets the
        overflow check run before anything is clamped.
        """
        table_index = f"(add, {HEADING}, heading_offset)"
        return {
            ROBOT_X: f"(add, {ROBOT_X}, (idiv, (mult, {SPEED}, "
                     f"(index, cos_table, {table_index})), x_divisor))",
            ROBOT_Y: f"(add, {ROBOT_Y}, (idiv, (mult, {SPEED}, "
                     f"(index, sin_table, {table_index})), y_divisor))",
            SPEED: f"(add, {SPEED}, (idiv, {ACCEL}, accel_divisor))",
            HEADING: f"(add, {HEADING}, (idiv, {TURN}, turn_divisor))",
        }

    def _out_of_box_test(self) -> str:
        return ", ".join(
            f"(lt, {AXIS_TO_VARIABLE[a.name]}_raw, {AXIS_TO_VARIABLE[a.name]}_lo), "
            f"(gt, {AXIS_TO_VARIABLE[a.name]}_raw, {AXIS_TO_VARIABLE[a.name]}_hi)"
            for a in self.grid.axes
        )

    def _environment_update(self) -> str:
        """
        Order matters. `overflow` is computed first, from the unclamped raws,
        so a breach is recorded before clamping hides it. The four state
        updates are order-independent because every raw reads stage-0 state.
        """
        clamps = "".join(
            f"{_indent(1)}variable_statement {{ {AXIS_TO_VARIABLE[a.name]} assign "
            f"{{ result {{ (max, {AXIS_TO_VARIABLE[a.name]}_lo, "
            f"(min, {AXIS_TO_VARIABLE[a.name]}_hi, "
            f"{AXIS_TO_VARIABLE[a.name]}_raw)) }} }} }}\n"
            for a in self.grid.axes
        )
        return (
            "environment_update {\n"
            f"{_indent(1)}variable_statement {{ {OVERFLOW} assign "
            f"{{ result {{ (or, {OVERFLOW}, {self._out_of_box_test()}) }} }} }}\n"
            f"{clamps}"
            f"{_indent(1)}variable_statement {{ {TICK_COUNT} assign "
            f"{{ result {{ (add, {TICK_COUNT}, 1) }} }} }}\n"
            "} end_environment_update\n"
        )

    def _actions(self) -> str:
        if self.mode == MONOLITHIC:
            accel = f"(index, {NETWORK}, 0)"
            turn = f"(index, {NETWORK}, 1)"
            reads = NETWORK
        else:
            free = f"(loop, control, [{-CONTROL_SCALE}, {CONTROL_SCALE}] " \
                   f"such_that True, control)"
            accel = turn = free
            reads = ""
        return (
            "actions {\n"
            f"{_indent(1)}action {{\n"
            f"{_indent(2)}Act\n"
            f"{_indent(2)}arguments{{}}\n"
            f"{_indent(2)}local_variables {{}} end_local_variables\n"
            f"{_indent(2)}read_variables {{{reads}}} end_read_variables\n"
            f"{_indent(2)}write_variables {{{ACCEL}, {TURN}}} end_write_variables\n"
            f"{_indent(2)}initial_values {{}} end_initial_values\n"
            f"{_indent(2)}update {{\n"
            f"{_indent(3)}variable_statement {{ {ACCEL} assign{{result{{{accel}}}}} }}\n"
            f"{_indent(3)}variable_statement {{ {TURN} assign{{result{{{turn}}}}} }}\n"
            f"{_indent(3)}return_statement {{ result {{ success }} end_result }} "
            f"end_return_statement\n"
            f"{_indent(2)}}} end_update\n"
            f"{_indent(1)}}} end_action\n"
            "} end_actions\n"
        )

    def _region_test(self, region: dict[str, tuple[int, int]]) -> str:
        parts = []
        for axis_name in ("x", "y"):
            low, high = region[axis_name]
            variable = AXIS_TO_VARIABLE[axis_name]
            parts.append(f"(gte, {variable}, {low}), (lte, {variable}, {high})")
        return f"(and, {', '.join(parts)})"

    def _specifications(self) -> str:
        """
        One invariant, three conjuncts.

        Obstacle avoidance holds at every reachable state (the model freezes at
        the horizon, so every reachable state has tick_count <= horizon), and
        the goal is required only at the final step. Both are safety properties
        -- each is violated by a finite prefix -- so no CTL is needed.
        """
        return (
            "specifications {\n"
            f"{_indent(1)}INVARSPEC {{ (and,\n"
            f"{_indent(2)}(not, {OVERFLOW}),\n"
            f"{_indent(2)}(not, {self._region_test(self.spec.obstacle_indices)}),\n"
            f"{_indent(2)}(implies, (eq, {TICK_COUNT}, horizon), "
            f"{self._region_test(self.spec.goal_indices)})\n"
            f"{_indent(1)}) }}\n"
            "} end_specifications\n"
        )

    # -------------------------------------------------------------- assembly

    def render(self) -> str:
        """The complete `.tree` file."""
        configuration = "configuration {\n    neural\n}\n" \
            if self.mode == MONOLITHIC else "configuration {\n}\n"
        return (
            f"#{{ Generated by nav_tree_generator.py -- do not edit by hand.\n"
            f"   mode: {self.mode}\n"
            f"   cells: {self.grid.cell_count}\n"
            f"   cell sizes: "
            f"{ {a.name: a.cell_size for a in self.grid.axes} }\n"
            f"   integration: explicit Euler, position from pre-update state }}#\n"
            + configuration
            + "enumerations {\n}\n"
            + self._constants()
            + self._variables()
            + self._environment_update()
            + "checks {} end_checks\n"
            + "environment_checks {} end_environment_checks\n"
            + self._actions()
            + "sub_trees {} end_sub_trees\n"
            + f"tree {{ Act {{}} }} end_tree\n"
            + f"tick_prerequisite {{ (lt, {TICK_COUNT}, horizon) }} "
              f"end_tick_prerequisite\n"
            + self._specifications()
        )

    def write(self, path: str) -> str:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(self.render())
        return path


def build_spec(
    grid: NavGrid,
    cfg: dict[str, Any],
    onnx_path: str,
) -> NavTreeSpec:
    """Translate the benchmark's physical regions into lattice indices."""
    initial = cfg["initial_set"]
    midpoint = {
        name: (float(initial[name][0]) + float(initial[name][1])) / 2.0
        for name in ("x", "y", "v", "theta")
    }
    horizon = int(round(float(cfg["horizon"]) / float(cfg["control_period"])))
    return NavTreeSpec(
        horizon_steps=horizon,
        initial_indices=grid.to_indices(
            [midpoint["x"], midpoint["y"], midpoint["v"], midpoint["theta"]]
        ),
        obstacle_indices=grid.region_to_indices(
            {"x": cfg["obstacle"]["x"], "y": cfg["obstacle"]["y"]},
            conservative="outward",
        ),
        goal_indices=grid.region_to_indices(
            {"x": cfg["goal"]["x"], "y": cfg["goal"]["y"]},
            conservative="inward",
        ),
        onnx_path=onnx_path,
    )
