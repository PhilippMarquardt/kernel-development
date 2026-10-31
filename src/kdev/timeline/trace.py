"""Exports: Perfetto/Chrome trace JSON and timeline.csv chart data. Owner: P6."""
from __future__ import annotations

from kdev.timeline.decode import MarkRec


def to_perfetto(recs: list[MarkRec], names: dict[int, str]) -> dict:
    """Chrome trace events: one track per CTA (pid=SM-or-0, tid=cta), 'X' events per phase."""
    raise NotImplementedError("trace.to_perfetto (P6)")


def to_csv_rows(recs: list[MarkRec], names: dict[int, str]) -> list[dict]:
    """Rows {cta, wg, phase, start_ns, end_ns} relative to the earliest mark (dashboard chart data)."""
    raise NotImplementedError("trace.to_csv_rows (P6)")
