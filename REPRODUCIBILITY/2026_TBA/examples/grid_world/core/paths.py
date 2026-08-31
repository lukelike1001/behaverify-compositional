"""
Example-root path anchor for modules under core/.

EXAMPLE_ROOT is grid_world/ (parent of core/), regardless of whether the
importer lives in core/, core/safety/, or core/liveness/.
"""

from pathlib import Path

EXAMPLE_ROOT = Path(__file__).resolve().parent.parent
