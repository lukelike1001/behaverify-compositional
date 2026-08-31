"""
Import switching for the per-example ``core`` packages.

Both examples ship their own top-level ``core`` package
(examples/grid_world/core/ and examples/AcasXu_closed_loop/core/). Python
caches the first one imported under the name ``core``, so a pytest session
touching both examples would resolve every later ``core.*`` import against the
wrong example and fail with ModuleNotFoundError.

Call ``activate_example("grid_world")`` before importing ``core.*`` in a test
module. It drops the cached package and points sys.path at the example asked
for, so import order between test files stops mattering.
"""

from __future__ import annotations

import sys
from pathlib import Path

EXAMPLES_ROOT = Path(__file__).resolve().parents[1] / "examples"


def activate_example(example_name: str) -> Path:
    """Make ``example_name``'s core package the one ``core.*`` resolves to."""
    example_root = EXAMPLES_ROOT / example_name
    if not example_root.is_dir():
        raise ValueError(f"no such example: {example_root}")

    for module_name in [
        name for name in sys.modules
        if name == "core" or name.startswith("core.")
    ]:
        del sys.modules[module_name]

    for other in EXAMPLES_ROOT.iterdir():
        if other != example_root and str(other) in sys.path:
            sys.path.remove(str(other))

    sys.path.insert(0, str(example_root))
    return example_root
