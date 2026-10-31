"""CUPTI activity via torch.profiler: kernel names, durations, grid/block (works without counter
permission). Owner: P6. Default `profile` backend when ncu is blocked; ``--nopdl`` disables PDL attributes
(env knob provided by the adapter) so per-kernel times are separable."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kdev.contract import Result
from kdev.models import JobSpec


@dataclass
class KernelRecord:
    name: str
    count: int
    total_us: float
    mean_us: float
    grid: list[int] = field(default_factory=list)
    block: list[int] = field(default_factory=list)


def kernel_table(profiler_events: Any) -> list[KernelRecord]:
    """Aggregate torch.profiler key_averages()/events into KernelRecords (accepts fakes in tests)."""
    raise NotImplementedError("cupti.kernel_table (P6)")


def run_profile(spec: JobSpec, run_dir: Any) -> Result:
    """Worker entry for kind='profile' (dispatches to ncu/nsys when params.extra['backend'] asks and the
    backend is permitted)."""
    raise NotImplementedError("cupti.run_profile (P6)")
