"""Nsight Compute wrapper (only when permitted). Owner: P6."""
from __future__ import annotations

from kdev.util import Runner, run


def available(runner: Runner = run) -> tuple[bool, str]:
    """(ok, reason): binary present and counters permitted; reason e.g. 'ERR_NVGPUCTRPERM'."""
    raise NotImplementedError("ncu.available (P6)")


def ncu_argv(target_argv: list[str], metrics: str = "SpeedOfLight,Occupancy,WarpStateStats",
             kernel_regex: str | None = None, out: str = "ncu.ncu-rep") -> list[str]:
    raise NotImplementedError("ncu.ncu_argv (P6)")


def parse_csv(text: str) -> list[dict]:
    """`ncu --csv --page details` -> rows."""
    raise NotImplementedError("ncu.parse_csv (P6)")
