"""Exports: Perfetto/Chrome trace JSON and timeline.csv chart data. Owner: P6.

- ``to_perfetto``: Chrome trace-event JSON (open in ui.perfetto.dev / chrome://tracing). One process per SM
  (pid = SM id when KD_MARK_SM records exist, else 0), one thread per (CTA, warpgroup); an 'X' slice per
  phase and an instant event per mark. Timestamps are us relative to the earliest mark.
- ``to_csv_rows`` / ``write_csv``: rows {cta, wg, phase, start_ns, end_ns} (dashboard Gantt data).
- ``chart_json``: compact, chart-ready dict (phase bars p10/p50/p90, mark offsets, a sampled Gantt) for the
  dashboard and for agents (``kdev show RUN --section chart``).
"""
from __future__ import annotations

import csv
from typing import Any

from kdev.timeline.decode import SM_MARK, MarkRec, TimelineSummary, _name

WG_STRIDE = 16   # tid = cta * WG_STRIDE + wg


def _tracks(recs: list[MarkRec]) -> tuple[dict[tuple[int, int], list[MarkRec]], dict[int, int], int]:
    sm_of: dict[int, int] = {}
    tracks: dict[tuple[int, int], list[MarkRec]] = {}
    for r in recs:
        if r.mark == SM_MARK:
            sm_of[r.cta] = r.clk_lo
        else:
            tracks.setdefault((r.cta, r.wg), []).append(r)
    for tr in tracks.values():
        tr.sort(key=lambda r: r.t_ns)
    t0 = min((r.t_ns for tr in tracks.values() for r in tr), default=0)
    return tracks, sm_of, t0


def to_perfetto(recs: list[MarkRec], names: dict[int, str]) -> dict:
    """Chrome trace events: one track per CTA (pid=SM-or-0, tid=cta), 'X' events per phase."""
    tracks, sm_of, t0 = _tracks(recs)
    ev: list[dict[str, Any]] = []
    seen_pid: set[int] = set()
    for (cta, wg), tr in sorted(tracks.items()):
        pid = sm_of.get(cta, 0)
        tid = cta * WG_STRIDE + wg
        if pid not in seen_pid:
            seen_pid.add(pid)
            ev.append({"ph": "M", "name": "process_name", "pid": pid, "tid": 0,
                       "args": {"name": f"SM {pid}" if sm_of else "GPU"}})
        ev.append({"ph": "M", "name": "thread_name", "pid": pid, "tid": tid, "args": {"name": f"cta {cta} wg {wg}"}})
        ev.append({"ph": "M", "name": "thread_sort_index", "pid": pid, "tid": tid, "args": {"sort_index": tid}})
        for a, b in zip(tr, tr[1:]):
            ev.append({"ph": "X", "name": f"{_name(names, a.mark)}->{_name(names, b.mark)}", "pid": pid, "tid": tid,
                       "ts": (a.t_ns - t0) / 1000.0, "dur": (b.t_ns - a.t_ns) / 1000.0, "cat": "phase",
                       "args": {"cta": cta, "wg": wg, "clk": (b.clk_lo - a.clk_lo) & 0xFFFFFFFF}})
        for r in tr:
            ev.append({"ph": "i", "s": "t", "name": _name(names, r.mark), "pid": pid, "tid": tid,
                       "ts": (r.t_ns - t0) / 1000.0, "cat": "mark"})
    return {"traceEvents": ev, "displayTimeUnit": "ns",
            "otherData": {"generator": "kdev timeline", "t0_globaltimer_ns": t0}}


def to_csv_rows(recs: list[MarkRec], names: dict[int, str]) -> list[dict]:
    """Rows {cta, wg, phase, start_ns, end_ns} relative to the earliest mark (dashboard chart data)."""
    tracks, sm_of, t0 = _tracks(recs)
    rows = []
    for (cta, wg), tr in sorted(tracks.items()):
        for a, b in zip(tr, tr[1:]):
            rows.append({"cta": cta, "wg": wg, "sm": sm_of.get(cta, ""),
                         "phase": f"{_name(names, a.mark)}->{_name(names, b.mark)}",
                         "start_ns": a.t_ns - t0, "end_ns": b.t_ns - t0})
    return rows


def write_csv(path: str, rows: list[dict]) -> str:
    cols = ["cta", "wg", "sm", "phase", "start_ns", "end_ns"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


def chart_json(summary: TimelineSummary, rows: list[dict], max_gantt_ctas: int = 64) -> dict:
    """Chart-ready data: phase bars (us), mark offsets (us) and a Gantt of at most ``max_gantt_ctas`` CTAs."""
    ctas = sorted({r["cta"] for r in rows})
    if len(ctas) > max_gantt_ctas:
        step = (len(ctas) - 1) / (max_gantt_ctas - 1)
        keep = {ctas[round(i * step)] for i in range(max_gantt_ctas)}
        if summary.critical_cta is not None:
            keep.add(summary.critical_cta)
    else:
        keep = set(ctas)
    return {
        "schema": "kdev.timeline_chart/1",
        "unit": "us",
        "resolution_ns": summary.resolution_ns,
        "duration_source": summary.duration_source,
        "span_us": round(summary.span_ns / 1000.0, 4),
        "critical_cta": summary.critical_cta,
        "phases": [{"name": p.name, "p10": round(p.p10_ns / 1000, 4), "p50": round(p.p50_ns / 1000, 4),
                    "p90": round(p.p90_ns / 1000, 4), "n": p.n} for p in summary.phases],
        "marks": [{"name": m.name, "p10": round(m.p10_ns / 1000, 4), "p50": round(m.p50_ns / 1000, 4),
                   "p90": round(m.p90_ns / 1000, 4), "min": round(m.min_ns / 1000, 4),
                   "max": round(m.max_ns / 1000, 4), "n": m.n} for m in summary.marks],
        "gantt": [{"cta": r["cta"], "wg": r["wg"], "phase": r["phase"], "start": round(r["start_ns"] / 1000, 4),
                   "end": round(r["end_ns"] / 1000, 4)} for r in rows if r["cta"] in keep],
        "gantt_ctas": len(keep), "total_ctas": len(ctas),
    }
