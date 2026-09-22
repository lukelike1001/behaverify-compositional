"""
Example-root path anchor for modules under core/.

EXAMPLE_ROOT is NAV/ (parent of core/), regardless of whether the importer
lives in core/ or a subpackage of it.
"""

from pathlib import Path

EXAMPLE_ROOT = Path(__file__).resolve().parent.parent
