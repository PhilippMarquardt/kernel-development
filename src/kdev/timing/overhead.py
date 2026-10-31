"""Replay overhead: judge constant (4.0 us) + locally measured empty-graph, 1-kernel and 2-kernel graph
costs. Owner: P3.

The judge measures, per shape, a 1-kernel graph replay (``one_ns``, ~4.9-5.3 us on H200) and a 2-kernel
graph replay (``two_ns``) and subtracts ``overhead = 2*one - two`` (the fixed graph cost without any kernel
body, 3.4-4.4 us, median ~3.9 us) from both sides (judge JSON ``records[].replay_overhead``).

Modes: 'judge' subtracts the constant 4.0 us (default; what calib.py did, reproduces all_calib.md),
'local' subtracts the locally measured 2*one-two (the judge's method; falls back to the empty graph),
'none' nothing (opt-out). All measurements are always reported."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

JUDGE_CONST_US = 4.0
SENSITIVE_BELOW_US = 10.0
MODES = ("judge", "local", "none")


@dataclass
class Overhead:
    judge_const: float = JUDGE_CONST_US
    empty_graph: float | None = None
    one_kernel_graph: float | None = None
    mode: str = "judge"
    two_kernel_graph: float | None = None

    @property
    def local_estimate(self) -> float | None:
        """Judge method on local measurements: 2*one - two (else the empty-graph cost)."""
        if self.one_kernel_graph is not None and self.two_kernel_graph is not None:
            return 2 * self.one_kernel_graph - self.two_kernel_graph
        return self.empty_graph

    @property
    def subtracted(self) -> float:
        """The value actually subtracted per replay under ``mode``."""
        if self.mode == "judge":
            return self.judge_const
        if self.mode == "local":
            est = self.local_estimate
            return self.judge_const if est is None else est
        if self.mode == "none":
            return 0.0
        raise ValueError(f"unknown overhead mode {self.mode!r} (expected one of {MODES})")

    def to_dict(self) -> dict[str, float | None]:
        return {"judge_const": self.judge_const, "empty_graph": self.empty_graph,
                "one_kernel_graph": self.one_kernel_graph, "two_kernel_graph": self.two_kernel_graph,
                "local_estimate": self.local_estimate, "subtracted": self.subtracted}


def from_judge_record(rec: dict[str, Any]) -> Overhead:
    """Overhead block of a judge benchmark record (ns -> us)."""
    ro = rec.get("replay_overhead") or {}
    one = ro.get("one_ns")
    two = ro.get("two_ns")
    return Overhead(mode="local", one_kernel_graph=None if one is None else one / 1000.0,
                    two_kernel_graph=None if two is None else two / 1000.0)


def measure(flush: Any, reps: int = 50) -> Overhead:
    """GPU: median replay time (after ``flush()``) of an empty graph, a 1-kernel (x.add_(1)) graph and a
    2-kernel graph, interleaved. Mode is left at 'judge'; callers set it."""
    import torch

    from kdev.timing import graph as G
    from kdev.timing.stats import median

    x = torch.zeros(1, dtype=torch.float32, device="cuda")

    def one() -> None:
        x.add_(1)

    def two() -> None:
        x.add_(1)
        x.add_(1)

    graphs: dict[str, Any] = {"one": G.capture(one), "two": G.capture(two)}
    try:
        import warnings

        g0 = torch.cuda.CUDAGraph()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")      # "The CUDA Graph is empty" is intended here
            with torch.cuda.graph(g0):
                pass
        torch.cuda.synchronize()
        graphs["empty"] = g0
        G.time_replay(g0, flush)
    except Exception:  # noqa: BLE001 - empty graphs are rejected by some torch/driver versions
        graphs.pop("empty", None)
    ts: dict[str, list[float]] = {k: [] for k in graphs}
    keys = list(graphs)
    for k in keys:                     # warm
        G.time_replay(graphs[k], flush)
    for i in range(reps):
        for k in (keys if i % 2 == 0 else keys[::-1]):
            ts[k].append(G.time_replay(graphs[k], flush))
    return Overhead(empty_graph=median(ts["empty"]) if "empty" in ts else None,
                    one_kernel_graph=median(ts["one"]), two_kernel_graph=median(ts["two"]))
