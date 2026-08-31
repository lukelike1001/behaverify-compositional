"""Put the tests directory on sys.path so tests can import example_imports."""

from __future__ import annotations

import sys
from pathlib import Path

_TESTS_ROOT = Path(__file__).resolve().parent
if str(_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TESTS_ROOT))
