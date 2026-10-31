"""Render DashboardData to a self-contained static HTML page (inline CSS/JS, no network at view time,
light/dark via prefers-color-scheme). Owner: P8. Template: templates/index.html.

The page is rendered server-side (it reads fine without JavaScript); the full data is also embedded as
``<script type="application/json" id="kdev-data">`` and written to ``dashboard.json`` for agents. The small
inline script only adds table sorting/filtering and relative ages.
"""
from __future__ import annotations

import html
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

from kdev.dashboard.collect import DashboardData

TEMPLATE = Path(__file__).with_name("templates") / "index.html"
MAX_ROWS = 200


def _e(x: Any) -> str:
    return html.escape("" if x is None else str(x), quote=True)


def _num(x: Any, nd: int = 3, suffix: str = "") -> str:
    if not isinstance(x, (int, float)):
        return '<span class="muted">-</span>'
    return f"{x:.{nd}f}{suffix}"


def _us(x: Any) -> str:
    if not isinstance(x, (int, float)):
        return '<span class="muted">-</span>'
    return f"{x:.1f}" if abs(x) >= 10 else f"{x:.2f}"


def _x(x: Any) -> str:
    return f"x{x:.3f}" if isinstance(x, (int, float)) else '<span class="muted">-</span>'


def _pill(text: str, kind: str) -> str:
    """Status pill: always a text label (never colour alone). kind: good|warn|bad|info|muted."""
    return f'<span class="pill {kind}">{_e(text)}</span>'


def _status_pill(status: Any) -> str:
    s = str(status or "").lower()
    if s in ("ok", "done", "pass", "confirmed", "faster", "win"):
        return _pill(s, "good")
    if s in ("fail", "failed", "error", "timeout", "refuted", "incorrect", "disagree", "aborted"):
        return _pill(s, "bad")
    if s in ("running", "claimed", "testing", "queued", "suspect", "waited", "out_of_band"):
        return _pill(s, "warn")
    return _pill(s or "-", "muted")


def _table(headers: Sequence[str], rows: Iterable[Sequence[str]], *, tid: str, empty: str = "none",
           numeric: Sequence[int] = ()) -> str:
    rows = list(rows)
    if not rows:
        return f'<p class="muted">{_e(empty)}</p>'
    more = ""
    if len(rows) > MAX_ROWS:
        more = f'<p class="muted">... {len(rows) - MAX_ROWS} more rows in dashboard.json</p>'
        rows = rows[:MAX_ROWS]
    th = "".join(f'<th{" class=num" if i in numeric else ""}>{_e(h)}</th>' for i, h in enumerate(headers))
    body = "".join("<tr>" + "".join(f'<td{" class=num" if i in numeric else ""}>{c}</td>'
                                    for i, c in enumerate(r)) + "</tr>" for r in rows)
    return (f'<div class="tw"><table class="sortable" id="{_e(tid)}"><thead><tr>{th}</tr></thead>'
            f"<tbody>{body}</tbody></table></div>{more}")


def _gap_bar(pct: Any) -> str:
    """Single-series meter of the gap to board #1 (0..50 % scale), with the number as text."""
    if not isinstance(pct, (int, float)):
        return '<span class="muted">-</span>'
    if pct <= 0:
        return _pill("#1 or better", "good")
    w = min(100.0, pct * 2.0)
    return (f'<span class="meter" title="{pct:.2f}% faster needed to match #1">'
            f'<span style="width:{w:.0f}%"></span></span> {pct:.1f}%')


def _section(title: str, body: str, sid: str, note: str = "") -> str:
    n = f' <span class="muted small">{_e(note)}</span>' if note else ""
    return f'<section id="{_e(sid)}"><h2>{_e(title)}{n}</h2>{body}</section>'


def _problems(d: DashboardData) -> str:
    rows = []
    for p in d.problems:
        bj, bl, g = p.get("best_judged") or {}, p.get("best_local") or {}, p.get("gap") or {}
        rank = "-"
        if g.get("our_rank"):
            rank = f"{g['our_rank']} / {g.get('board_n') or '?'}" + (" (would-be)" if g.get("rank_kind") ==
                                                                     "would-be" else "")
        elif g.get("rank_kind"):
            rank = f"{g['rank_kind']} / {g.get('board_n') or '?'}"
        flags = []
        if p.get("unverified_limits"):
            flags.append(_pill(f"{p['unverified_limits']} unverified limit", "bad"))
        if p.get("disagree"):
            flags.append(_pill(f"{p['disagree']} DISAGREE", "bad"))
        rows.append([
            f"<b>{_e(p['problem'])}</b>",
            _x(bj.get("speedup")) + (f' <span class="muted small">{_e(bj.get("judge_id"))}</span>' if bj else ""),
            f'<span class="age" data-utc="{_e(bj.get("finished_utc") or bj.get("created_utc"))}"></span>' if bj
            else "-",
            _x(bl.get("score")) + (f' <span class="muted small">{_e(bl.get("run_id"))}</span>' if bl else ""),
            _x(g.get("board1")) + (f' <span class="muted small">{_e(g.get("board1_team"))}</span>'
                                   if g.get("board1_team") else ""),
            _x(g.get("board10")), _e(rank), _gap_bar(g.get("gap_to_1_pct")),
            str(p.get("open_ideas", 0)), " ".join(flags) or '<span class="muted">-</span>'])
    note = "board: not scraped (kdev dashboard build --scrape)" if d.board is None else \
        f"board fetched {d.board.get('fetched_utc')} from {d.board.get('url', 'api')}"
    return _section("Problems", _table(
        ["problem", "best judged", "age", "best local (comparable)", "board #1", "board #10", "our rank",
         "gap to #1", "open ideas", "flags"], rows, tid="t-problems",
        empty="no problems in the ledger yet", numeric=(8,)), "problems", note)


def _shape_gaps(d: DashboardData) -> str:
    parts = []
    for p in d.problems:
        g = p.get("gap") or {}
        shapes = g.get("shapes") or [{"shape_id": s.get("shape_id"), "weight": s.get("weight"),
                                      "sub_us": s.get("sub_us"), "base_us": s.get("base_us"),
                                      "speedup": s.get("speedup"), "time_share": None, "cut_to_1_us": None}
                                     for s in p.get("judge_shapes") or []]
        local = {s.get("shape_id"): s for s in (p.get("best_local") or {}).get("shapes") or []}
        if not shapes and not local:
            continue
        rows = []
        for s in shapes:
            loc = local.get(s.get("shape_id")) or {}
            share = s.get("time_share")
            rows.append([_e(s.get("shape_id")), _num(s.get("weight"), 0), _us(s.get("sub_us")),
                         _us(s.get("base_us")), _x(s.get("speedup")),
                         _num(share * 100 if isinstance(share, (int, float)) else None, 1, "%"),
                         _us(s.get("cut_to_1_us")), _us(loc.get("median_us")), _x(loc.get("ratio_median"))])
        if not shapes:
            rows = [[_e(k), _num(v.get("weight"), 0), "-", "-", "-", "-", "-", _us(v.get("median_us")),
                     _x(v.get("ratio_median"))] for k, v in local.items()]
        parts.append(f"<details open><summary><b>{_e(p['problem'])}</b> "
                     f'<span class="muted small">judge {_e((p.get("best_judged") or {}).get("judge_id"))}'
                     f"</span></summary>" + _table(
                         ["shape", "weight", "judge sub us", "judge base us", "speedup", "time share",
                          "cut to #1 us", "local median us", "local ratio"], rows,
                         tid=f"t-shape-{p['problem']}", numeric=(1, 2, 3, 4, 5, 6, 7, 8)) + "</details>")
    body = "".join(parts) or '<p class="muted">no judged per-shape data yet (kdev judge record)</p>'
    return _section("Per-shape gap", body, "shapes",
                    "board publishes totals only: 'cut to #1' = per-shape cut if the gap closed uniformly")


def _job_rows(jobs: list[dict[str, Any]]) -> list[list[str]]:
    return [[_e(j.get("job_id")), _status_pill(j.get("state")), _e(j.get("kind")), _e(j.get("agent")),
             _e(j.get("position") or ""), _e(j.get("run_id") or ""),
             f'<span class="age" data-utc="{_e(j.get("started_utc") or j.get("submitted_utc"))}"></span>',
             _e(j.get("error_code") or "")] for j in jobs]


def _queue(d: DashboardData) -> str:
    q = d.queue or {}
    dm = q.get("daemon") or {}
    if dm:
        alive = dm.get("alive", True)
        head = (_pill("daemon up" if alive else "daemon stale", "good" if alive else "bad") +
                f' <span class="muted small">pid {_e(dm.get("pid"))} on {_e(dm.get("host"))}, heartbeat '
                f'{_e(dm.get("ts_utc"))}, gpu lane {_e(dm.get("gpu_lane_job") or "idle")}</span>')
    else:
        head = _pill("no daemon", "warn") + ' <span class="muted small">direct (flock) mode</span>'
    jobs = list(q.get("running") or []) + list(q.get("queued") or [])
    hdr = ["job", "state", "kind", "agent", "pos", "run", "since", "error"]
    body = (f"<p>{head} <span class='muted small'>source: {_e(q.get('source', '-'))}</span></p>"
            + "<h3>Running / queued</h3>" + _table(hdr, _job_rows(jobs), tid="t-queue", empty="GPU idle, queue empty")
            + "<h3>Recent jobs</h3>" + _table(hdr, _job_rows(list(q.get("recent") or [])), tid="t-jobs"))
    off = [[f'<span class="age" data-utc="{_e(o.get("ts_utc"))}"></span>', _e(o.get("pid")),
            f"<code>{_e(str(o.get('cmdline') or '')[:120])}</code>", _e(o.get("gpu_mem_mib")),
            _e(o.get("during_job")), _status_pill(o.get("action"))] for o in d.offenders]
    body += "<h3>Offenders (last 3 h)</h3>" + _table(["when", "pid", "cmdline", "MiB", "during job", "action"],
                                                     off, tid="t-off", empty="none")
    return _section("GPU queue", body, "queue")


def _runs(d: DashboardData) -> str:
    rows = [[f"<code>{_e(r.get('run_id'))}</code>", _e(r.get("tool")), _status_pill(r.get("status")),
             _status_pill(r.get("verdict")), _e(r.get("problem")), _e(r.get("agent")),
             "" if r.get("comparable") in (1, True, None) else _pill("not comparable", "warn"),
             _e(r.get("exclusive")), _e(r.get("error_code") or ""),
             f'<span class="age" data-utc="{_e(r.get("created_utc"))}"></span>'] for r in d.recent_runs]
    return _section("Recent runs", _table(["run", "tool", "status", "verdict", "problem", "agent", "comparable",
                                           "excl", "error", "age"], rows, tid="t-runs",
                                          empty="no runs recorded"), "runs", "kdev show <run> for details")


def _ideas(d: DashboardData) -> str:
    limits = {c["idea_id"]: c for c in d.unverified_limit_claims}
    rows = []
    for i in d.ideas:
        flag = ""
        if i["idea_id"] in limits:
            flag = _pill("needs " + "+".join(limits[i["idea_id"]]["missing"]), "bad")
        gain = i.get("expected_gain_pct")
        rows.append([str(i["idea_id"]), _e(i.get("problem")), _e(i.get("kind")), _status_pill(i.get("status")),
                     _e(i.get("owner") or ""), f"{_e(i.get('title'))}" + (
                         f'<div class="muted small">{_e(i.get("variant"))}</div>' if i.get("variant") else ""),
                     _num(gain, 1, "%") if gain is not None else "",
                     f'{i.get("evidence_n", 0)} <span class="muted small">(+{i.get("evidence_for", 0)}/'
                     f'-{i.get("evidence_against", 0)})</span>', flag,
                     f'<span class="age" data-utc="{_e(i.get("updated_utc"))}"></span>'])
    open_n = sum(1 for i in d.ideas if i.get("status") in ("open", "claimed", "testing"))
    return _section("Idea board", _table(["id", "problem", "kind", "status", "owner", "title / variant", "gain",
                                          "evidence", "flags", "updated"], rows, tid="t-ideas",
                                         empty="no ideas yet (kdev idea add)", numeric=(0,)),
                    "ideas", f"{open_n} open/claimed/testing; {len(limits)} unverified limit claims")


def _calib(d: DashboardData) -> str:
    rows = [[_e(c.get("problem")), _e(c.get("shape_id")), _e(c.get("label")), _x(c.get("local_speedup")),
             _x(c.get("judge_speedup")), _num(c.get("local_over_judge"), 3), _status_pill(c.get("flag")),
             f"<code>{_e(c.get('judge_id'))}</code>"] for c in d.calibration]
    return _section("Calibration (local vs judge)", _table(
        ["problem", "shape", "label", "local", "judge", "local/judge", "flag", "judge id"], rows, tid="t-calib",
        empty="no calibration runs (kdev calib FILE:JUDGE.json)", numeric=(3, 4, 5)), "calib")


def render_body(data: DashboardData) -> str:
    warn = "".join(f"<li>{_e(w)}</li>" for w in data.warnings)
    head = f'<ul class="warnings">{warn}</ul>' if warn else ""
    nav = " ".join(f'<a href="#{s}">{t}</a>' for s, t in (("problems", "problems"), ("shapes", "shapes"),
                                                           ("queue", "queue"), ("runs", "runs"),
                                                           ("ideas", "ideas"), ("calib", "calibration")))
    return (f'<nav>{nav} <input id="filter" placeholder="filter rows" aria-label="filter rows"></nav>{head}'
            + _problems(data) + _shape_gaps(data) + _queue(data) + _runs(data) + _ideas(data) + _calib(data))


def render_html(data: DashboardData) -> str:
    tpl = TEMPLATE.read_text(encoding="utf-8")
    blob = json.dumps(data.to_dict(), separators=(",", ":"), default=str).replace("</", "<\\/")
    values = {"TITLE": "kdev dashboard", "GENERATED": _e(data.generated_utc), "HOME": _e(data.home),
              "BODY": render_body(data), "DATA_JSON": blob}
    # one pass, so placeholder-like text inside the data is never substituted
    return re.sub(r"\{\{([A-Z_]+)\}\}", lambda m: values.get(m.group(1), m.group(0)), tpl)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write(data: DashboardData, out_dir: str) -> dict[str, str]:
    """Write index.html + dashboard.json atomically; return {'html': path, 'json': path}."""
    out = Path(out_dir)
    h, j = out / "index.html", out / "dashboard.json"
    _atomic_write(j, json.dumps(data.to_dict(), indent=1, default=str))
    _atomic_write(h, render_html(data))
    return {"html": str(h), "json": str(j)}
