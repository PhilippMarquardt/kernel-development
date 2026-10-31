"""Sweep grid + orchestration. Owner: P5.

Flow: expand grid -> one build job per point (build lane, parallel) -> drop points failing build / spills /
C751x (reason recorded) -> quick check (check job, 1 seed) -> ab jobs in batches of <= 4 points + base,
interleaved -> top K by score with CIs; full table in sweep.csv. Grid sources: "K=a,b K2=c" literal, or
"cublaslt:SHAPE" (algorithm enumeration provided by the adapter, later).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kdev.contract import Result


@dataclass
class SweepPoint:
    knobs: dict[str, Any]
    label: str = ""
    build_status: str = ""      # ok | build_failed | spills | c751x
    check_status: str = ""
    score: float | None = None
    ci: tuple[float, float] | None = None
    run_ids: list[str] = field(default_factory=list)


def parse_grid(text: str) -> dict[str, list[Any]]:
    """'TILE=64,128 STAGES=2,3' -> {'TILE': [64, 128], 'STAGES': [2, 3]} (ints/floats coerced)."""
    raise NotImplementedError("grid.parse_grid (P5)")


def expand(grid: dict[str, list[Any]]) -> list[SweepPoint]:
    raise NotImplementedError("grid.expand (P5)")


def run_sweep(file: str, grid: dict[str, list[Any]], *, base: str | None, top: int, build_parallel: int,
              quick_check: bool, ctx: Any) -> Result:
    """Client-side orchestration via ctx.submit (build lane, then GPU lane)."""
    raise NotImplementedError("grid.run_sweep (P5)")
