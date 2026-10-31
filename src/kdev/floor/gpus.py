"""GPU profile loader (`kdev.gpu/1`, PLAN 3.8). Owner: P6.

Search: $KDEV_HOME/gpus/<name>.toml (measured overrides) then package data src/kdev/data/gpus/<name>.toml.
``detect_name()`` maps nvidia-smi names ('NVIDIA H200') to profile names ('h200').
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Measured:
    value: float
    unit: str = "us"
    source: str = ""
    run: str = ""
    how: str = ""


@dataclass
class GpuProfile:
    name: str
    sm_count: int
    l2_bytes: int
    flush_bytes: int
    cc: str = ""
    smem_per_sm: int = 0
    regs_per_sm: int = 65536
    max_warps_per_sm: int = 64
    max_ctas_per_sm: int = 32
    globaltimer_ns: list[int] = field(default_factory=list)
    peak: dict[str, float] = field(default_factory=dict)
    measured: dict[str, Measured] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


def load_profile(name: str, home: str | None = None) -> GpuProfile:
    raise NotImplementedError("gpus.load_profile (P6)")


def detect_name(smi_name: str) -> str:
    raise NotImplementedError("gpus.detect_name (P6)")


def write_measured(name: str, values: dict[str, Measured], home: str) -> str:
    """Write/merge $KDEV_HOME/gpus/<name>.toml [measured] (floor calibrate --write)."""
    raise NotImplementedError("gpus.write_measured (P6)")
