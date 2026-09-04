"""
Example-root path anchor for modules under core/.

EXAMPLE_ROOT is NAV/ (parent of core/), regardless of where the importer lives.
"""

from pathlib import Path

EXAMPLE_ROOT = Path(__file__).resolve().parent.parent
