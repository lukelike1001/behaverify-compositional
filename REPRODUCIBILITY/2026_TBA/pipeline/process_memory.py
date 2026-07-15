"""
pipeline.process_memory — peak RSS helpers for pipeline stages.

Used by SMV builders, CROWN batches, and NuxmvVerifier when recording memory.
"""

from __future__ import annotations

import resource


class ProcessMemory:
    """
    Thin wrapper around resource.getrusage peak-RSS counters (Linux, kilobytes).

    Methods are classmethods because there is no per-instance state — only the
    OS accounting for this process and its waited children.
    """

    @classmethod
    def peak_self_rss_kilobytes(cls) -> int:
        """Peak RSS of this process so far (KB). Monotonically nondecreasing on Linux."""
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    @classmethod
    def peak_children_rss_kilobytes(cls) -> int:
        """Peak RSS of all waited child processes (KB)."""
        return resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
