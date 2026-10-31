"""nvidia-smi queries (clocks, power, temperature, utilisation, GPU identity). Owner: P2.

All functions take ``runner`` so tests feed canned CSV (tests/fixtures/smi/).
"""
from __future__ import annotations

from kdev.models import GpuState
from kdev.util import Runner, run


def query_state(gpu: int = 0, runner: Runner = run) -> GpuState:
    """nvidia-smi --query-gpu=name,uuid,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu
    --format=csv,noheader,nounits. Missing nvidia-smi -> GpuState() with empty fields (never raises)."""
    raise NotImplementedError("smi.query_state (P2)")


def parse_state_csv(text: str) -> GpuState:
    """Pure parser for the CSV line above."""
    raise NotImplementedError("smi.parse_state_csv (P2)")


def clocks_moved(before: GpuState, after: GpuState, tol: float = 0.03) -> bool:
    """True if SM clock moved more than tol (fidelity warning)."""
    raise NotImplementedError("smi.clocks_moved (P2)")


def ncu_permitted() -> bool | None:
    """Parse /proc/driver/nvidia/params RmProfilingAdminOnly (None if unknown)."""
    raise NotImplementedError("smi.ncu_permitted (P2)")
