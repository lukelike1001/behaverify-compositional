"""
nav_wrapper_builder.py

Wrap a NAV controller so BehaVerify's table method can consume it.

The wrapper reads integer lattice coordinates and returns the integer state
delta for (speed, heading). It exists because `dsl_to_nuxmv` stores regression
outputs as `int(output)`, and NAV's tanh outputs lie in (-1, 1), so every
control would truncate to zero and the lookup table would be inert.

Around the untouched original graph:

    Div  by [Sp, Sp, Sv, Sh]     lattice coordinates -> real state
    <original network>
    Mul  by [Sv*dt, Sh*dt]       control -> integer delta on each lattice
    Round

The two output factors differ because speed and heading now live on separate
lattices, so a single scalar cannot serve both. `Round` needs opset 11+, so the
opset is raised from 8. No weights are modified.

The Mul factors also set the control resolution: the delta takes values in
{-round(S*dt), ..., +round(S*dt)}, so `Sv*dt = 1` yields a three-level
controller and larger products yield finer ones.

Usage (from examples/NAV/):

    python3 -m core.nav_wrapper_builder                    # uses the config
    python3 -m core.nav_wrapper_builder --speed-scale 25   # override one axis
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from core.paths import EXAMPLE_ROOT

TARGET_OPSET = 13
NETWORKS = ("point", "set")


def wrapper_filename(
    network: str, position: int, speed: int, heading: int,
) -> str:
    """Wrapper names carry their scales so configurations cannot be confused."""
    return f"nn-nav-{network}-p{position}v{speed}h{heading}.onnx"


def build_wrapper(
    network: str,
    position_scale: int,
    speed_scale: int,
    heading_scale: int,
    dt: float,
    output_dir: Path,
) -> Path:
    """Write one lattice wrapper and return its path."""
    source = EXAMPLE_ROOT / "networks" / f"nn-nav-{network}.onnx"
    model = onnx.load(str(source))
    graph = model.graph

    del model.opset_import[:]
    model.opset_import.append(helper.make_opsetid("", TARGET_OPSET))

    input_name = graph.input[0].name
    output_name = graph.output[0].name

    input_scales = numpy_helper.from_array(
        np.array(
            [position_scale, position_scale, speed_scale, heading_scale],
            dtype=np.float32,
        ),
        name=f"lattice_in_{network}",
    )
    output_scales = numpy_helper.from_array(
        np.array([speed_scale * dt, heading_scale * dt], dtype=np.float32),
        name=f"lattice_out_{network}",
    )
    graph.initializer.extend([input_scales, output_scales])

    real_state = f"{input_name}_real"
    graph.node.insert(
        0, helper.make_node("Div", [input_name, input_scales.name], [real_state]),
    )
    for node in graph.node[1:]:
        for index, operand in enumerate(node.input):
            if operand == input_name:
                node.input[index] = real_state

    scaled = f"{output_name}_scaled"
    delta = f"{output_name}_delta"
    graph.node.append(
        helper.make_node("Mul", [output_name, output_scales.name], [scaled]),
    )
    graph.node.append(helper.make_node("Round", [scaled], [delta]))

    graph.output.remove(graph.output[0])
    graph.output.append(
        helper.make_tensor_value_info(delta, TensorProto.FLOAT, ["BatchSize", 2]),
    )

    onnx.checker.check_model(model)
    destination = output_dir / wrapper_filename(
        network, position_scale, speed_scale, heading_scale,
    )
    onnx.save(model, str(destination))
    return destination


def verify_wrapper(
    network: str,
    path: Path,
    position_scale: int,
    speed_scale: int,
    heading_scale: int,
    dt: float,
    samples: int = 1000,
    seed: int = 0,
) -> int:
    """Compare the wrapper against the original; return the mismatch count."""
    import onnxruntime as ort  # noqa: PLC0415

    original = ort.InferenceSession(
        str(EXAMPLE_ROOT / "networks" / f"nn-nav-{network}.onnx"),
        providers=["CPUExecutionProvider"],
    )
    wrapped = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    original_input = original.get_inputs()[0].name
    wrapped_input = wrapped.get_inputs()[0].name

    divisor = np.array(
        [position_scale, position_scale, speed_scale, heading_scale],
        dtype=np.float32,
    )
    factor = np.array([speed_scale * dt, heading_scale * dt], dtype=np.float32)

    rng = np.random.default_rng(seed)
    mismatches = 0
    for point in rng.integers(-40, 61, size=(samples, 4)).astype(np.float32):
        control = original.run(
            None, {original_input: (point / divisor).reshape(1, 4)},
        )[0].reshape(-1)
        expected = np.round(control * factor)
        actual = wrapped.run(
            None, {wrapped_input: point.reshape(1, 4)},
        )[0].reshape(-1)
        mismatches += int(not np.array_equal(expected, actual))
    return mismatches


def main() -> None:
    from core.nav_tree_generator import NavDomain  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--position-scale", type=int, default=None)
    parser.add_argument("--speed-scale", type=int, default=None)
    parser.add_argument("--heading-scale", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    domain = NavDomain.from_yaml()
    position = args.position_scale or domain.position_scale
    speed = args.speed_scale or domain.speed_scale
    heading = args.heading_scale or domain.heading_scale
    dt = domain.dt

    output_dir = (
        Path(args.output_dir) if args.output_dir else EXAMPLE_ROOT / "networks"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"scales: position={position} speed={speed} heading={heading} dt={dt}")
    print(
        f"control levels: speed +/-{round(speed * dt)}, "
        f"heading +/-{round(heading * dt)}"
    )
    for network in NETWORKS:
        path = build_wrapper(network, position, speed, heading, dt, output_dir)
        note = ""
        if not args.skip_verify:
            bad = verify_wrapper(network, path, position, speed, heading, dt)
            note = f"  ({bad} mismatches / 1000)"
        print(f"  wrote {path.name}{note}")


if __name__ == "__main__":
    main()
