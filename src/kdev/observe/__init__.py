"""Runtime observation backends: CUPTI (torch.profiler), ncu, nsys, NVBit; dispatch probe. Owner: P6.

Backends report availability through ``probe()``; tools pick the best available and name it in the Result
(REA: missing backend = capability_unavailable, never a crash).
"""
from __future__ import annotations

from kdev.models import Check


def probe(ctx: object, problem: str | None = None) -> list[Check]:
    """Doctor checks: ncu present + permitted (ERR_NVGPUCTRPERM / RmProfilingAdminOnly), nsys present,
    compute-sanitizer present, NVBit present (NVBIT_PATH), torch.profiler CUPTI importable. Never raises."""
    raise NotImplementedError("observe.probe (P6)")
