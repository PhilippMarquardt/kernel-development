"""Timing engine: `time` and `ab` jobs under protocol kdev-tp/1 (docs/PLAN.md 3.7). Owner: P3.

The scheduling/statistics core is written against the :class:`Timer` protocol so it is unit-tested with
:class:`FakeTimer` (no GPU). :class:`TorchTimer` is the real implementation (graph capture + read flush).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from kdev.contract import Result
from kdev.models import JobSpec, Sample, TimingParams


class Timer(Protocol):
    """Prepared timing targets for one shape+seed."""

    def labels(self) -> list[str]: ...
    def replay_us(self, label: str) -> float:
        """Flush, replay the label's graph once, return raw elapsed us."""
        ...
    def overhead_us(self) -> float: ...


@dataclass
class FakeTimer:
    """Deterministic timer for tests: ``means[label]`` us + optional per-call noise sequence."""

    means: dict[str, float]
    noise: list[float] = field(default_factory=list)
    overhead: float = 4.0
    calls: list[str] = field(default_factory=list)

    def labels(self) -> list[str]:
        return list(self.means)

    def replay_us(self, label: str) -> float:
        raise NotImplementedError("FakeTimer.replay_us (P3)")

    def overhead_us(self) -> float:
        return self.overhead


class TorchTimer:
    """Real timer: adapter.load/prepare per file (once), make_inputs(shape, seed), graph.capture of
    adapter.call, Flusher before each replay, CUDA events. Verifies output materialisation (checksum after
    replay vs eager)."""

    def __init__(self, adapter: Any, modules: dict[str, Any], states: dict[str, Any], shape: Any, seed: int,
                 params: TimingParams) -> None:
        raise NotImplementedError("TorchTimer (P3)")

    def labels(self) -> list[str]:
        raise NotImplementedError

    def replay_us(self, label: str) -> float:
        raise NotImplementedError

    def overhead_us(self) -> float:
        raise NotImplementedError


def measure_shape(timer: Timer, shape_id: str, seed: int, params: TimingParams) -> list[Sample]:
    """Run the interleaved schedule for one shape/seed and return samples (overhead subtracted)."""
    raise NotImplementedError("engine.measure_shape (P3)")


def summarize(samples: list[Sample], shapes: list[Any], params: TimingParams, ref_label: str,
              score_fn: Any = None) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Per-shape FileShapeTiming + paired ratio CI + verdicts, and the score block (Result.shapes/.score)."""
    raise NotImplementedError("engine.summarize (P3)")


def run_time(spec: JobSpec, run_dir: Any) -> Result:
    """Worker entry for kind='time' (one file vs the baseline unless params.base == 'none')."""
    raise NotImplementedError("engine.run_time (P3)")


def run_ab(spec: JobSpec, run_dir: Any) -> Result:
    """Worker entry for kind='ab' (N files interleaved; decisions on absolute times, M6)."""
    raise NotImplementedError("engine.run_ab (P3)")
