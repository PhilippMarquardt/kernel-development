"""Hot-loop extraction and per-element histograms. Owner: P5.

Hotness ranking (static): innermost depth first, then instr count x weight of contained mma/ldg/lds/fp
work; line-info weight when available. ``--per N`` divides the loop's histogram by N elements per
iteration (the '~15 ops / 8 weights' analysis of LESSONS 1).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from kdev.sass.cfg import Loop
from kdev.sass.parse import Instr, Kernel


@dataclass
class HotLoop:
    kernel: str
    loop: Loop
    instrs: list[Instr] = field(default_factory=list)
    histogram: dict[str, float] = field(default_factory=dict)
    per_element: dict[str, float] | None = None
    score: float = 0.0


def hot_loops(kernel: Kernel, top: int = 3, per: float | None = None) -> list[HotLoop]:
    raise NotImplementedError("loops.hot_loops (P5)")
