"""CUDA-graph capture exactly as the judge does it (recipe from ah.py / calib.py). Owner: P3.

capture(fn): fn() once + synchronize; side-stream warm-up (s.wait_stream(current); with stream(s): fn();
current.wait_stream(s)); torch.cuda.CUDAGraph() capture of fn(); synchronize. PDL launch attributes inside
fn are preserved (no stream juggling inside the capture).
"""
from __future__ import annotations

from typing import Any, Callable


def capture_with_output(fn: Callable[[], Any]) -> tuple[Any, Any, Any]:
    """(graph, eager_output, graph_output): the eager output is the first (warm-up) call's result, the graph
    output is the tensor(s) the captured replay writes into."""
    import torch

    eager_out = fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        graph_out = fn()
    torch.cuda.synchronize()
    return g, eager_out, graph_out


def capture(fn: Callable[[], Any]) -> Any:
    """Return a captured torch.cuda.CUDAGraph of fn (lazy torch import)."""
    return capture_with_output(fn)[0]


def _events() -> tuple[Any, Any]:
    import torch

    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def time_replay(graph: Any, flush: Callable[[], None]) -> float:
    """flush(); event record; graph.replay(); event record; synchronize; return elapsed us (raw)."""
    import torch

    flush()
    e0, e1 = _events()
    e0.record()
    graph.replay()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1000.0


def time_eager(fn: Callable[[], Any], flush: Callable[[], None]) -> float:
    """Eager timing (opt-out `--eager`): includes host launch cost of every kernel in fn."""
    import torch

    flush()
    e0, e1 = _events()
    e0.record()
    fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1000.0
