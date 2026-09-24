"""
nav_table_pipeline.py

Monolithic NAV: wrap the network, generate the NSBT, build the SMV table, run
nuXmv, record what happened.

This is the 2025 NEUS table approach applied to a continuous state space. The
encoding is not a choice -- BehaVerify permits `table` and nothing else for
regression networks (`dsl_to_nuxmv.py` raises 'Cannot have a regression network
in nuXmv without using table' at four sites) -- and it was also the best
performer in that paper.

WHY THE NETWORK IS WRAPPED
--------------------------
Two defects in BehaVerify's regression path make the shipped ONNX unusable as
is. Both are worked around in the graph rather than by patching `src/`, so the
monolithic baseline stays stock.

  1. `dsl_to_nuxmv.py:1107` stores `int(network_output)`. NAV's control lies in
     [-1, 1] and `int()` truncates toward zero, so every table entry would be 0
     -- a robot that never accelerates and never turns. Fixed by scaling the
     output by CONTROL_SCALE inside the graph.

  2. The regression input builder uses `current_input.append(...)` where the
     classification one uses `+=` (lines 1101 and 1129). Meta functions return
     lists, so regression networks accept only bare variable references; any
     arithmetic in `inputs {}` produces a rank-3 tensor and onnxruntime rejects
     it. Fixed by scaling the input by the per-axis cell sizes inside the graph,
     so the tree can pass raw lattice indices.

The wrapper is exactly an affine change of units, and `verify_equivalence`
checks that against the original network before anything else runs.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from core.nav_domain import (
    CONTROL_SCALE,
    NavDiscreteDynamics,
    NavGrid,
    build_domain,
    load_config,
    uniform_cell_sizes,
)
from core.nav_tree_generator import MONOLITHIC, NavTreeGenerator, build_spec
from core.paths import EXAMPLE_ROOT

REPRODUCIBILITY_ROOT = EXAMPLE_ROOT.parent.parent
DSL_TO_NUXMV = REPRODUCIBILITY_ROOT / "src" / "dsl_to_nuxmv.py"
METAMODEL = REPRODUCIBILITY_ROOT / "metamodel" / "behaverify.tx"
NUXMV = REPRODUCIBILITY_ROOT / "nuXmv_DL" / "bin" / "nuXmv"

WRAPPED_INPUT = "lattice_index_input"
WRAPPED_OUTPUT = "scaled_control_output"


class NavOnnxWrapper:
    """
    An affine re-unit of the shipped controller: lattice indices in, milli-units
    out. See the module docstring for why this is necessary.
    """

    def __init__(self, source_path: str | Path) -> None:
        self.source_path = Path(source_path)

    def build(self, cell_sizes: Sequence[float], destination: str | Path) -> Path:
        import onnx
        from onnx import TensorProto, helper, numpy_helper

        model = onnx.load(str(self.source_path))
        graph = model.graph
        original_input, original_output = graph.input[0], graph.output[0]

        graph.initializer.append(numpy_helper.from_array(
            np.asarray(cell_sizes, dtype=np.float32), name="cell_size_scale"))
        graph.initializer.append(numpy_helper.from_array(
            np.asarray([float(CONTROL_SCALE)], dtype=np.float32),
            name="control_scale"))

        graph.node.insert(0, helper.make_node(
            "Mul", [WRAPPED_INPUT, "cell_size_scale"], [original_input.name]))
        graph.input.remove(original_input)
        graph.input.append(helper.make_tensor_value_info(
            WRAPPED_INPUT, TensorProto.FLOAT, [None, len(cell_sizes)]))

        graph.node.append(helper.make_node(
            "Mul", [original_output.name, "control_scale"], [WRAPPED_OUTPUT]))
        graph.output.remove(original_output)
        graph.output.append(helper.make_tensor_value_info(
            WRAPPED_OUTPUT, TensorProto.FLOAT, [None, 2]))

        onnx.checker.check_model(model)
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(model, str(destination))
        return destination

    def verify_equivalence(
        self,
        wrapped_path: str | Path,
        cell_sizes: Sequence[float],
        sample_indices: Sequence[Sequence[int]],
        tolerance: float = 1e-2,
    ) -> bool:
        """`wrapped(idx) == CONTROL_SCALE * original(idx * cell_size)`."""
        import onnxruntime as ort

        original = ort.InferenceSession(str(self.source_path))
        wrapped = ort.InferenceSession(str(wrapped_path))
        scales = np.asarray(cell_sizes, dtype=np.float32)
        for indices in sample_indices:
            lattice = np.asarray([indices], dtype=np.float32)
            expected = CONTROL_SCALE * original.run(
                None, {original.get_inputs()[0].name: lattice * scales})[0][0]
            actual = wrapped.run(
                None, {wrapped.get_inputs()[0].name: lattice})[0][0]
            if not np.allclose(expected, actual, atol=tolerance):
                return False
        return True


@dataclass
class NavRunReport:
    """One monolithic run, start to finish."""

    network: str
    cell_sizes: dict[str, float]
    cells: int
    horizon_steps: int
    quantization_cells_per_step: dict[str, float]
    tree_path: str
    smv_path: str
    smv_lines: int
    smv_megabytes: float
    generation_seconds: float
    nuxmv_seconds: float
    nuxmv_return_code: int
    verdict: str
    notes: str = ""

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path


class NavTablePipeline:
    """Wrap -> generate -> translate -> model check, with every step recorded."""

    def __init__(
        self,
        network_path: str | Path,
        cell_sizes: dict[str, float],
        output_directory: str | Path,
        cfg: dict[str, Any] | None = None,
        bounds: Any = None,
    ) -> None:
        self.network_path = Path(network_path)
        self.cell_sizes = cell_sizes
        self.output_directory = Path(output_directory)
        self.grid, self.dynamics, self.cfg = build_domain(
            cell_sizes, cfg=cfg, bounds=bounds)

    # ------------------------------------------------------------ the stages

    def prepare_network(self) -> Path:
        """Build the wrapped ONNX next to the tree and check it is faithful."""
        ordered = [self.cell_sizes[a.name] for a in self.grid.axes]
        destination = self.output_directory / f"{self.network_path.stem}_wrapped.onnx"
        wrapper = NavOnnxWrapper(self.network_path)
        wrapper.build(ordered, destination)
        samples = [
            tuple(a.lower_index for a in self.grid.axes),
            tuple(a.upper_index for a in self.grid.axes),
            tuple(0 for _ in self.grid.axes),
        ]
        if not wrapper.verify_equivalence(destination, ordered, samples):
            raise RuntimeError(
                f"wrapped network at {destination} does not reproduce "
                f"{CONTROL_SCALE} * original(index * cell_size)"
            )
        return destination

    def check_the_robot_moves(self) -> None:
        """
        Reject a resolution whose discrete trajectory never leaves its cell.

        The generator's analytic guard uses the declared control bound (1.0,
        the tanh limit). The network never reaches it, so a configuration can
        pass that check and still be frozen -- h_v = 0.2 is exactly such a
        case. This is the empirical check, and it needs the network the
        generator does not have.
        """
        import onnxruntime as ort

        session = ort.InferenceSession(str(self.network_path))
        input_name = session.get_inputs()[0].name

        def control(indices: Sequence[int]) -> tuple[int, int]:
            physical = np.asarray(self.grid.to_values(indices), dtype=np.float32)
            output = session.run(None, {input_name: physical.reshape(1, 4)})[0][0]
            return (int(CONTROL_SCALE * output[0]), int(CONTROL_SCALE * output[1]))

        initial = self.grid.to_indices([
            (float(self.cfg["initial_set"][name][0])
             + float(self.cfg["initial_set"][name][1])) / 2.0
            for name in ("x", "y", "v", "theta")
        ])
        steps = int(round(
            float(self.cfg["horizon"]) / float(self.cfg["control_period"])))
        if not self.dynamics.trajectory_moves(initial, control, steps):
            raise ValueError(
                f"cell sizes {self.cell_sizes} pass the analytic quantization "
                f"guard but the discrete trajectory never leaves its starting "
                f"cell -- the network's largest output is below one cell per "
                f"step. Any verdict here would describe a stationary robot."
            )

    def generate_tree(self, wrapped_network: Path) -> Path:
        """Emit the NSBT. The ONNX path must be relative to the tree file."""
        spec = build_spec(self.grid, self.cfg, f"./{wrapped_network.name}")
        generator = NavTreeGenerator(
            self.grid, self.dynamics, spec, mode=MONOLITHIC)
        tree_path = self.output_directory / f"{self.network_path.stem}.tree"
        generator.write(str(tree_path))
        return tree_path

    def translate(self, tree_path: Path) -> tuple[Path, float]:
        """Run BehaVerify. This is where the ONNX table is enumerated."""
        smv_path = tree_path.with_suffix(".smv")
        started = time.perf_counter()
        result = subprocess.run(
            [sys.executable, str(DSL_TO_NUXMV), str(METAMODEL),
             str(tree_path), str(smv_path),
             "--no_checks", "--recursion_limit", "20000"],
            capture_output=True, text=True, check=False)
        elapsed = time.perf_counter() - started
        if result.returncode != 0:
            tail = result.stderr.strip().splitlines()[-1] if result.stderr else ""
            raise RuntimeError(f"dsl_to_nuxmv failed: {tail}")
        return smv_path, elapsed

    def model_check(
        self, smv_path: Path, timeout_seconds: int = 3600
    ) -> tuple[str, int, float]:
        """
        Run nuXmv and insist on an explicit verdict.

        `-dynamic` is not optional: without dynamic BDD reordering a 2,835-cell
        model ran past 900 s, and with it the same model finished in under 180.

        A missing verdict is an error, never a data point. nuXmv answers an
        out-of-range initial assignment by abandoning its command script and
        waiting at a prompt, which otherwise looks like a fast, clean run.
        """
        command_file = smv_path.with_suffix(".nuxmv_cmd")
        command_file.write_text("go\ncheck_invar\nquit\n", encoding="utf-8")
        started = time.perf_counter()
        try:
            result = subprocess.run(
                [str(NUXMV), "-dynamic", "-source", str(command_file), str(smv_path)],
                capture_output=True, text=True, timeout=timeout_seconds, check=False)
        except subprocess.TimeoutExpired:
            return "timeout", -1, float(timeout_seconds)
        elapsed = time.perf_counter() - started
        output = result.stdout
        if "cannot assign" in output:
            raise RuntimeError(
                "nuXmv rejected an initial assignment and abandoned its command "
                "script; the declared box does not contain the initial state")
        if result.returncode != 0:
            return f"crash (rc={result.returncode})", result.returncode, elapsed
        if " is true" in output:
            return "true", 0, elapsed
        if " is false" in output:
            return "false", 0, elapsed
        raise RuntimeError(
            f"nuXmv produced no verdict for {smv_path}; this is a pipeline "
            f"failure, not a result")

    # ------------------------------------------------------------------ run

    def run(self, timeout_seconds: int = 3600) -> NavRunReport:
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.check_the_robot_moves()
        wrapped = self.prepare_network()
        tree_path = self.generate_tree(wrapped)
        smv_path, generation_seconds = self.translate(tree_path)
        verdict, return_code, nuxmv_seconds = self.model_check(
            smv_path, timeout_seconds)
        report = NavRunReport(
            network=self.network_path.name,
            cell_sizes=dict(self.cell_sizes),
            cells=self.grid.cell_count,
            horizon_steps=int(round(
                float(self.cfg["horizon"]) / float(self.cfg["control_period"]))),
            quantization_cells_per_step={
                name: round(value, 3)
                for name, value in self.dynamics.quantization().cells_per_step.items()
            },
            tree_path=str(tree_path),
            smv_path=str(smv_path),
            smv_lines=sum(1 for _ in smv_path.open(encoding="utf-8")),
            smv_megabytes=round(smv_path.stat().st_size / 1e6, 3),
            generation_seconds=round(generation_seconds, 3),
            nuxmv_seconds=round(nuxmv_seconds, 3),
            nuxmv_return_code=return_code,
            verdict=verdict,
        )
        report.write(self.output_directory / "run_report.json")
        return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Monolithic NAV table pipeline.")
    parser.add_argument("--onnx", default=str(EXAMPLE_ROOT / "networks" / "nn-nav-set.onnx"))
    parser.add_argument("--cell-size", type=float, required=True,
                        help="uniform cell size on every axis")
    parser.add_argument("--output", default=str(EXAMPLE_ROOT / "results" / "monolithic"))
    parser.add_argument("--bounds", choices=["analytic", "sampled"], default="analytic",
                        help="'sampled' is NOT sound; reference only")
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()

    cfg = load_config()
    bounds = None
    if args.bounds == "sampled":
        from core.nav_boundary_finder import NavBoundaryFinder
        bounds = NavBoundaryFinder(
            onnx_path=args.onnx, cfg=cfg).calculate_boundary("sampled")

    pipeline = NavTablePipeline(
        network_path=args.onnx,
        cell_sizes=uniform_cell_sizes(args.cell_size),
        output_directory=Path(args.output) / f"h{args.cell_size}_{args.bounds}",
        cfg=cfg, bounds=bounds)

    print(f"cells                 {pipeline.grid.cell_count:,}")
    print(f"cells moved per step  "
          f"{ {k: round(v, 2) for k, v in pipeline.dynamics.quantization().cells_per_step.items()} }")
    report = pipeline.run(timeout_seconds=args.timeout)
    print(f"SMV                   {report.smv_lines:,} lines, {report.smv_megabytes} MB "
          f"({report.generation_seconds}s)")
    print(f"nuXmv                 {report.verdict}  ({report.nuxmv_seconds}s)")


if __name__ == "__main__":
    main()
