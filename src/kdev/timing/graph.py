"""CUDA-graph capture exactly as the judge does it (recipe from ah.py / calib.py). Owner: P3.

capture(fn): fn() once + synchronize; side-stream warm-up (s.wait_stream(current); with stream(s): fn();
current.wait_stream(s)); torch.cuda.CUDAGraph() capture of fn(); synchronize. PDL launch attributes inside
fn are preserved (no stream juggling inside the capture).
"""
from __future__ import annotations

from typing import Any, Callable


def capture(fn: Callable[[], Any]) -> Any:
    """Return a captured torch.cuda.CUDAGraph of fn (lazy torch import)."""
    raise NotImplementedError("graph.capture (P3)")


def time_replay(graph: Any, flush: Callable[[], None]) -> float:
    """flush(); event record; graph.replay(); event record; synchronize; return elapsed us (raw)."""
    raise NotImplementedError("graph.time_replay (P3)")


def time_eager(fn: Callable[[], Any], flush: Callable[[], None]) -> float:
    """Eager timing (opt-out `--eager`)."""
    raise NotImplementedError("graph.time_eager (P3)")
