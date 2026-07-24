"""
acas_tree_generator.py

Expand the closed-loop ACAS Xu BehaVerify template by filling REPLACE_*
placeholders with pre-computed integer lookup tables (velocity, distance,
arctan). Physics numbers come from AcasDomain / acas_model_params.yaml.

    from core.acas_tree_generator import AcasTreeGenerator

    AcasTreeGenerator().generate()

CLI (from AcasXu_closed_loop/):

    python3 -m core.acas_tree_generator
    python3 -m core.acas_tree_generator --template tree/acas_closed_loop_template.tree \\
        --output tree/acas_closed_loop.tree
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

# Allow `python3 core/acas_tree_generator.py` as well as -m.
_EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
if str(_EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_ROOT))

from core.acas_domain import AcasDomain
from core.paths import EXAMPLE_ROOT

DEFAULT_TEMPLATE = EXAMPLE_ROOT / "tree" / "acas_closed_loop_template.tree"
DEFAULT_OUTPUT = EXAMPLE_ROOT / "tree" / "acas_closed_loop.tree"

_PLACEHOLDERS = (
    "REPLACE_VELOCITY_X_OWN",
    "REPLACE_VELOCITY_Y_OWN",
    "REPLACE_VELOCITY_X_INT",
    "REPLACE_VELOCITY_Y_INT",
    "REPLACE_DISTANCE",
    "REPLACE_ARCTAN_XY",
    "REPLACE_ARCTAN_YX",
)


# ---------------------------------------------------------------------------
# Pure DSL formatting helpers (nested (if, cond, then, else) chains)
# ---------------------------------------------------------------------------


def _indent(level: int) -> str:
    return " " * 4 * level


def _format_equality_condition(
    values: tuple[int, ...],
    var_names: tuple[str, ...],
) -> str:
    if len(values) != len(var_names):
        raise ValueError(
            f"condition arity mismatch: {len(values)} values vs {len(var_names)} vars"
        )
    if len(var_names) == 1:
        return f"(eq, {var_names[0]}, {values[0]})"
    clauses = [
        f"(eq, {var_names[index]}, {values[index]})"
        for index in range(len(var_names))
    ]
    return "(and, " + ", ".join(clauses) + ")"


def _format_if_chain(
    indent_level: int,
    cond_values: list[tuple[tuple[int, ...], int]],
    var_names: tuple[str, ...],
) -> str:
    """
    Build a right-nested (if, cond, value, ...) expression.

    cond_values is mutated (pop from the end); pass a fresh list.
    """
    if not cond_values:
        raise ValueError("cond_values must be non-empty")

    pre_parts: list[str] = []
    closing = _indent(indent_level)
    while len(cond_values) > 1:
        condition, value = cond_values.pop()
        pre_parts.append(
            f"{_indent(indent_level)}(if, "
            f"{_format_equality_condition(condition, var_names)},\n"
            f"{_indent(indent_level + 1)}{value},\n"
        )
        closing += ")"

    _, default_value = cond_values.pop()
    return (
        "".join(pre_parts)
        + f"{_indent(indent_level)}{default_value}\n"
        + f"{closing}\n"
    )


def _format_result_block(
    cond_values: list[tuple[tuple[int, ...], int]],
    var_names: tuple[str, ...],
) -> str:
    """`result { ... }` body at template indent (no leading indent)."""
    return (
        "result {\n"
        + _format_if_chain(4, list(cond_values), var_names)
        + f"{_indent(3)}}}\n"
    )


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AcasTreeGenerator:
    """
    Expand acas_closed_loop_template.tree → acas_closed_loop.tree.

    Lookup tables cover:
      - velocity (cos/sin × speed) for ownship and intruder, 0..359°
      - Euclidean distance on the (x_mag, y_mag) lattice
      - arctan(x/y) and arctan(y/x) on the same lattice
    """

    template_path: Path = DEFAULT_TEMPLATE
    output_path: Path = DEFAULT_OUTPUT
    domain: AcasDomain | None = None

    def resolve_domain(self) -> AcasDomain:
        return self.domain if self.domain is not None else AcasDomain.from_yaml()

    def velocity_table(self, *, ownship: bool, x_axis: bool) -> str:
        domain = self.resolve_domain()
        speed = domain.speed_own if ownship else domain.speed_int
        var_names = (
            ("heading_own", "speed_own")
            if ownship
            else ("heading_int", "speed_int")
        )
        cond_values: list[tuple[tuple[int, ...], int]] = []
        for heading_degrees in range(360):
            radians = math.radians(heading_degrees)
            component = math.cos(radians) if x_axis else math.sin(radians)
            cond_values.append(
                ((heading_degrees, speed), round(component * speed))
            )
        return _format_result_block(cond_values, var_names)

    def distance_table(self) -> str:
        domain = self.resolve_domain()
        grid_max = domain.max_dist // domain.distance_modifier
        cond_values: list[tuple[tuple[int, ...], int]] = [
            (
                (x_mag, y_mag),
                round(math.sqrt(x_mag * x_mag + y_mag * y_mag))
                * domain.distance_modifier,
            )
            for x_mag in range(0, grid_max + 1)
            for y_mag in range(0, grid_max + 1)
        ]
        return _format_result_block(cond_values, ("x_mag", "y_mag"))

    def arctan_table(self, *, x_over_y: bool) -> str:
        """
        Rounded arctan in degrees on the (x_mag, y_mag) lattice.

        x_over_y=True  → arctan(x_mag / y_mag); 0 when y_mag == 0
        x_over_y=False → arctan(y_mag / x_mag); 0 when x_mag == 0
        """
        domain = self.resolve_domain()
        grid_max = domain.max_dist // domain.distance_modifier
        cond_values: list[tuple[tuple[int, ...], int]] = []
        for x_mag in range(0, grid_max + 1):
            for y_mag in range(0, grid_max + 1):
                if x_over_y:
                    value = (
                        0
                        if y_mag == 0
                        else round(math.degrees(math.atan(x_mag / y_mag)))
                    )
                else:
                    value = (
                        0
                        if x_mag == 0
                        else round(math.degrees(math.atan(y_mag / x_mag)))
                    )
                cond_values.append(((x_mag, y_mag), value))
        return _format_result_block(cond_values, ("x_mag", "y_mag"))

    def expand_template(self, template_text: str) -> str:
        replacements = {
            "REPLACE_VELOCITY_X_OWN": self.velocity_table(
                ownship=True, x_axis=True
            ),
            "REPLACE_VELOCITY_Y_OWN": self.velocity_table(
                ownship=True, x_axis=False
            ),
            "REPLACE_VELOCITY_X_INT": self.velocity_table(
                ownship=False, x_axis=True
            ),
            "REPLACE_VELOCITY_Y_INT": self.velocity_table(
                ownship=False, x_axis=False
            ),
            "REPLACE_DISTANCE": self.distance_table(),
            "REPLACE_ARCTAN_XY": self.arctan_table(x_over_y=True),
            "REPLACE_ARCTAN_YX": self.arctan_table(x_over_y=False),
        }
        expanded = template_text
        for placeholder, body in replacements.items():
            if placeholder not in expanded:
                raise ValueError(
                    f"placeholder {placeholder!r} missing from "
                    f"{self.template_path}"
                )
            expanded = expanded.replace(placeholder, body)
        leftover = [name for name in _PLACEHOLDERS if name in expanded]
        if leftover:
            raise RuntimeError(f"unexpanded placeholders remain: {leftover}")
        return expanded

    def generate(self) -> Path:
        """
        Read template, expand REPLACE_* tables, write output_path.

        Returns the written path.
        """
        if not self.template_path.is_file():
            raise FileNotFoundError(f"template not found: {self.template_path}")
        template_text = self.template_path.read_text(encoding="utf-8")
        expanded = self.expand_template(template_text)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(expanded, encoding="utf-8")
        return self.output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Expand acas_closed_loop_template.tree into "
            "acas_closed_loop.tree with geometric lookup tables."
        )
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=DEFAULT_TEMPLATE,
        help=f"template path (default: {DEFAULT_TEMPLATE})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"output path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args(argv)

    output_path = AcasTreeGenerator(
        template_path=args.template,
        output_path=args.output,
    ).generate()
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
