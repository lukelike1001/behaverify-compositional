"""
Tests for the emitted NAV NSBT.

The generator's two refusals are the important part. Both encode spike findings
where the failure is silent rather than loud:

  - an initial state outside the declared box makes nuXmv abandon its command
    script and wait at a prompt, which a pipeline records as a fast, clean run
    with no verdict;
  - a cell size above the quantization floor freezes the robot, so every
    verdict describes a model that never moves.

The rest pin that the emitted constants agree with the reference dynamics, so
the tree and `NavDiscreteDynamics` cannot drift apart silently.
"""

from __future__ import annotations

import re

import pytest

from example_imports import activate_example

EXAMPLE_ROOT = activate_example("NAV")

from core.nav_domain import build_domain, load_config, uniform_cell_sizes  # noqa: E402
from core.nav_tree_generator import (  # noqa: E402
    COMPOSITIONAL,
    MONOLITHIC,
    NavTreeGenerator,
    NavTreeSpec,
    build_spec,
)

REPRESENTABLE_CELL = 0.15    # passes the quantization floor
FROZEN_CELL = 0.5            # does not


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config()


def _generator(cfg: dict, cell: float = REPRESENTABLE_CELL, mode: str = MONOLITHIC):
    grid, dynamics, _ = build_domain(uniform_cell_sizes(cell), cfg=cfg)
    return NavTreeGenerator(
        grid, dynamics, build_spec(grid, cfg, "./wrapped.onnx"), mode=mode)


# ---------------------------------------------------------------- refusals


def test_refuses_a_cell_size_that_freezes_the_robot(cfg: dict) -> None:
    with pytest.raises(ValueError, match="cell size too coarse"):
        _generator(cfg, cell=FROZEN_CELL)


def test_refuses_an_initial_state_outside_the_box(cfg: dict) -> None:
    grid, dynamics, _ = build_domain(uniform_cell_sizes(REPRESENTABLE_CELL), cfg=cfg)
    spec = build_spec(grid, cfg, "./wrapped.onnx")
    outside = NavTreeSpec(
        horizon_steps=spec.horizon_steps,
        initial_indices=(10 ** 6, 0, 0, 0),
        obstacle_indices=spec.obstacle_indices,
        goal_indices=spec.goal_indices,
        onnx_path=spec.onnx_path,
    )
    with pytest.raises(ValueError, match="outside the declared box"):
        NavTreeGenerator(grid, dynamics, outside)


def test_rejects_an_unknown_mode(cfg: dict) -> None:
    with pytest.raises(ValueError, match="mode must be"):
        _generator(cfg, mode="hybrid")


# ------------------------------------------------------- the two pipelines


def test_monolithic_declares_a_neural_table(cfg: dict) -> None:
    tree = _generator(cfg, mode=MONOLITHIC).render()
    assert "NEURAL regression INT" in tree
    assert "config { table }" in tree
    assert "inputs { robot_x, robot_y, speed, heading }" in tree


def test_regression_inputs_are_bare_variables(cfg: dict) -> None:
    """
    BehaVerify's regression path nests meta-function results (dsl_to_nuxmv.py
    line 1101 uses `append` where line 1129 uses `+=`), so any arithmetic in
    `inputs {}` produces a rank-3 tensor. Unit conversion lives in the ONNX
    wrapper instead.
    """
    tree = _generator(cfg, mode=MONOLITHIC).render()
    inputs = re.search(r"inputs \{([^}]*)\}", tree).group(1)
    for token in ("idiv", "rdiv", "mult", "add", "sub"):
        assert token not in inputs


def test_compositional_leaves_the_control_free(cfg: dict) -> None:
    tree = _generator(cfg, mode=COMPOSITIONAL).render()
    assert "NEURAL" not in tree
    assert "loop, control" in tree     # nondeterministic assignment


def test_only_the_control_declaration_differs(cfg: dict) -> None:
    """
    The comparison isolates the neural encoding, so physics, clock, bounds
    checking and specification must be byte-identical between the two modes.
    """
    monolithic = _generator(cfg, mode=MONOLITHIC).render()
    compositional = _generator(cfg, mode=COMPOSITIONAL).render()
    for section in ("environment_update", "specifications"):
        pattern = rf"{section} \{{.*?\}} end_{section}"
        left = re.search(pattern, monolithic, re.S)
        right = re.search(pattern, compositional, re.S)
        assert left and right and left.group(0) == right.group(0)


# ------------------------------------------ constants agree with the model


def test_emitted_divisors_match_the_dynamics(cfg: dict) -> None:
    grid, dynamics, _ = build_domain(uniform_cell_sizes(REPRESENTABLE_CELL), cfg=cfg)
    tree = NavTreeGenerator(
        grid, dynamics, build_spec(grid, cfg, "./w.onnx")).render()
    for constant, expected in (
        ("accel_divisor", dynamics.velocity_divisor),
        ("turn_divisor", dynamics.heading_divisor),
        ("x_divisor", dynamics.x_divisor),
        ("y_divisor", dynamics.y_divisor),
        ("heading_offset", dynamics.theta_table_offset),
    ):
        assert f"{constant} := {expected}" in tree


def test_emitted_trig_table_matches_the_dynamics(cfg: dict) -> None:
    grid, dynamics, _ = build_domain(uniform_cell_sizes(REPRESENTABLE_CELL), cfg=cfg)
    tree = NavTreeGenerator(
        grid, dynamics, build_spec(grid, cfg, "./w.onnx")).render()
    block = tree.split("cos_table")[1].split("sin_table")[0]
    for index, value in enumerate(dynamics.cosine_table):
        assert f"condition {{(eq, cell, {index})}} assign{{result{{{value}}}}}" in block


def test_horizon_is_thirty_control_steps(cfg: dict) -> None:
    tree = _generator(cfg).render()
    assert "horizon := 30" in tree
    assert "tick_prerequisite { (lt, tick_count, horizon) }" in tree


# -------------------------------------------------------- the specification


def test_specification_has_all_three_conjuncts(cfg: dict) -> None:
    tree = _generator(cfg).render()
    specification = tree.split("specifications {")[1]
    assert "(not, overflow)" in specification              # bound is proven
    assert "(not, (and, (gte, robot_x" in specification     # obstacle, always
    assert "(implies, (eq, tick_count, horizon)" in specification   # goal, at t=6


def test_no_check_nodes_and_a_single_leaf(cfg: dict) -> None:
    """The controller has no sensors; detection belongs in the specification."""
    tree = _generator(cfg).render()
    assert "checks {} end_checks" in tree
    assert "environment_checks {} end_environment_checks" in tree
    assert "tree { Act {} } end_tree" in tree
    assert tree.count("action {") == 1
