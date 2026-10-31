"""Decode a mark buffer into per-CTA phases. Owner: P6. Pure (bytes in, tables out)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MarkRec:
    t_ns: int
    clk_lo: int
    cta: int
    wg: int
    mark: int


@dataclass
class PhaseStats:
    name: str                 # 'markA->markB'
    p10_ns: float
    p50_ns: float
    p90_ns: float
    n: int


@dataclass
class TimelineSummary:
    phases: list[PhaseStats] = field(default_factory=list)
    cta_start_spread_ns: float = 0.0
    cta_end_spread_ns: float = 0.0
    critical_cta: int | None = None
    resolution_ns: int | None = None     # detected globaltimer granularity (32 or 256 on H200)
    dropped: int = 0
    mark_overhead_us: float | None = None


def parse_buffer(data: bytes) -> tuple[list[MarkRec], int]:
    """(records, dropped) from the raw buffer layout in marks.cuh."""
    raise NotImplementedError("decode.parse_buffer (P6)")


def detect_resolution(recs: list[MarkRec]) -> int | None:
    raise NotImplementedError("decode.detect_resolution (P6)")


def summarize(recs: list[MarkRec], names: dict[int, str]) -> TimelineSummary:
    raise NotImplementedError("decode.summarize (P6)")
