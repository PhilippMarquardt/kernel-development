"""Dispatch probe (M12): which kernels actually launch per scored shape. Owner: P6.

Runs each scored shape once under torch.profiler (CUPTI) eagerly and once as a graph replay; records kernel
name x count x grid per shape; warns when shapes the adapter declares distinct run the same route, and when
a file defines kernels never launched on any scored shape (static kernel list from build/sass).
``summarize_dispatch`` output goes into every `time`/`check` Result (Result.dispatch).
"""
from __future__ import annotations

from typing import Any

from kdev.contract import Result
from kdev.models import JobSpec


def probe_shape(adapter: Any, module: Any, state: Any, shape: Any, seed: int) -> list[dict[str, Any]]:
    """GPU: [{kernel, count, grid, block}] for one shape."""
    raise NotImplementedError("dispatch.probe_shape (P6)")


def summarize_dispatch(per_shape: dict[str, list[dict[str, Any]]], static_kernels: list[str] | None = None
                       ) -> tuple[dict[str, Any], list[str]]:
    """Pure: (compact dispatch dict, warnings)."""
    raise NotImplementedError("dispatch.summarize_dispatch (P6)")


def run_dispatch(spec: JobSpec, run_dir: Any) -> Result:
    raise NotImplementedError("dispatch.run_dispatch (P6)")
