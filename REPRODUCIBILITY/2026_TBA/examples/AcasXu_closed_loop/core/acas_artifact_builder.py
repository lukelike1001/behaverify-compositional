"""
acas_artifact_builder.py

Builds the two derived artifacts every ACAS Xu pipeline needs before it can
verify anything: the expanded .tree (via AcasTreeGenerator) and the base .smv
(via dsl_to_nuxmv, networks still inlined as lookup tables).

Safety, liveness, and monolithic drivers all open with these same two stages,
so they live here rather than being re-typed in each scripts/ driver. Each
stage is a no-op when its artifact already exists and reuse is requested.

Typical use (from a driver under scripts/):

    builder = AcasArtifactBuilder()
    tree_metrics = builder.ensure_tree(reuse_existing=args.skip_tree)
    smv_metrics = builder.ensure_smv(reuse_existing=args.skip_smv)
"""

from __future__ import annotations

import os
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

_TBA_ROOT = (Path(__file__).resolve().parent.parent / "../..").resolve()
if str(_TBA_ROOT) not in sys.path:
    sys.path.insert(0, str(_TBA_ROOT))

from pipeline.process_memory import ProcessMemory  # noqa: E402

from core.paths import EXAMPLE_ROOT  # noqa: E402

DEFAULT_TREE_PATH = EXAMPLE_ROOT / "tree/acas_closed_loop.tree"
DEFAULT_SMV_PATH = EXAMPLE_ROOT / "symbolic/smv/acas_closed_loop.smv"
DEFAULT_METAMODEL_PATH = _TBA_ROOT / "metamodel/behaverify.tx"
DEFAULT_BEHAVERIFY_SRC = _TBA_ROOT / "src"

# dsl_to_nuxmv's positional flags, named here so the call site reads.
_RECURSION_LIMIT = 10000
_KEEP_STAGE_0 = False
_SKIP_GRAMMAR_CHECK = True


class AcasArtifactBuilder:
    """Generates (or reuses) the tree and base SMV a pipeline runs against."""

    def __init__(
        self,
        *,
        tree_path: Path = DEFAULT_TREE_PATH,
        smv_path: Path = DEFAULT_SMV_PATH,
        metamodel_path: Path = DEFAULT_METAMODEL_PATH,
        behaverify_src: Path = DEFAULT_BEHAVERIFY_SRC,
    ) -> None:
        self.tree_path = tree_path
        self.smv_path = smv_path
        self.metamodel_path = metamodel_path
        self.behaverify_src = behaverify_src

    # -- stages ---------------------------------------------------------

    def ensure_tree(
        self,
        *,
        reuse_existing: bool,
        stage_label: str = "[TREE] generation",
    ) -> dict[str, Any]:
        """Expand the .tree template unless it already exists and may be reused."""
        self._print_banner(stage_label)
        if reuse_existing and self.tree_path.exists():
            return self._skipped(self.tree_path)

        from core.acas_tree_generator import AcasTreeGenerator  # noqa: PLC0415

        start = time.perf_counter()
        AcasTreeGenerator(output_path=self.tree_path).generate()
        wall_sec = time.perf_counter() - start

        print(f"  Generated {self._display(self.tree_path)}  ({wall_sec:.1f}s)")
        return {"wall_sec": round(wall_sec, 3), "skipped": False}

    def ensure_smv(
        self,
        *,
        reuse_existing: bool,
        stage_label: str = "[SMV] base model generation",
    ) -> dict[str, Any]:
        """Compile the tree to SMV unless it already exists and may be reused."""
        self._print_banner(stage_label)
        if reuse_existing and self.smv_path.exists():
            return self._skipped(self.smv_path)
        if not self.tree_path.is_file():
            raise FileNotFoundError(
                f"tree missing at {self.tree_path}; cannot generate SMV"
            )

        self.smv_path.parent.mkdir(parents=True, exist_ok=True)
        tracemalloc.start()
        start = time.perf_counter()
        self._run_dsl_to_nuxmv()
        wall_sec = time.perf_counter() - start
        _, peak_traced_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        smv_lines = self.smv_path.read_text(encoding="utf-8").count("\n")
        print(f"  Generated {self._display(self.smv_path)}  ({wall_sec:.1f}s)")
        return {
            "wall_sec": round(wall_sec, 3),
            "smv_lines": smv_lines,
            "peak_traced_bytes": peak_traced_bytes,
            "peak_rss_kb": ProcessMemory.peak_self_rss_kilobytes(),
            "skipped": False,
        }

    # -- internals ------------------------------------------------------

    def _run_dsl_to_nuxmv(self) -> None:
        """Invoke the BehaVerify compiler from the example root.

        The .tree references its ONNX networks by example-relative path, so the
        compiler must run with EXAMPLE_ROOT as cwd. Tree and SMV arguments are
        passed relative to that same root (os.path.relpath, not
        Path.relative_to, so a path outside the example still resolves).
        """
        src_dir = str(self.behaverify_src)
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        import dsl_to_nuxmv as dsl  # noqa: PLC0415

        previous_cwd = os.getcwd()
        os.chdir(str(EXAMPLE_ROOT))
        try:
            dsl.dsl_to_nuxmv(
                str(self.metamodel_path),
                os.path.relpath(self.tree_path, EXAMPLE_ROOT),
                os.path.relpath(self.smv_path, EXAMPLE_ROOT),
                False,
                False,
                False,
                False,
                _RECURSION_LIMIT,
                _KEEP_STAGE_0,
                _SKIP_GRAMMAR_CHECK,
                None,  # record_times
            )
        finally:
            os.chdir(previous_cwd)

    def _skipped(self, path: Path) -> dict[str, Any]:
        print(f"  Skipped — reusing {self._display(path)}")
        return {"wall_sec": 0.0, "skipped": True}

    @staticmethod
    def _print_banner(stage_label: str) -> None:
        print("\n" + "=" * 60)
        print(stage_label)
        print("=" * 60)

    @staticmethod
    def _display(path: Path) -> str:
        """Path relative to the example root when possible, else absolute."""
        try:
            return str(path.relative_to(EXAMPLE_ROOT))
        except ValueError:
            return str(path)
