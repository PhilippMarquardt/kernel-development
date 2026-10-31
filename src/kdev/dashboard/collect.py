"""Gather dashboard data (pure reads). Owner: P8.

Sources: the ledger (``$KDEV_HOME/ledger.db``) and the queue (``$KDEV_HOME/queue.db``), both opened
**read-only** (``mode=ro``; nothing is created or migrated) and queried against the frozen schemas
(PLAN 3.4/3.5), so the dashboard works while the owning packages evolve; ``kdev.service.queue.snapshot`` and
``kdev.ledger.query.best`` are used when implemented (their output is attached as-is), and
``kdev.dashboard.board.scrape`` is optional. Output is one JSON-serialisable dict (also written as
dashboard.json next to index.html, so agents can read it without HTML).
"""
from __future__ import annotations

import calendar
import dataclasses
import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from kdev.config import Config
from kdev.util import utc_now

_BAD_STATUS = ("fail", "error", "reject", "timeout", "cancel", "duplicate")
_OPEN_IDEA = ("open", "claimed", "testing")


@dataclass
class DashboardData:
    generated_utc: str = ""
    problems: list[dict[str, Any]] = field(default_factory=list)   # best judged/local, same-hour base, gap
    recent_runs: list[dict[str, Any]] = field(default_factory=list)
    queue: dict[str, Any] = field(default_factory=dict)
    offenders: list[dict[str, Any]] = field(default_factory=list)
    ideas: list[dict[str, Any]] = field(default_factory=list)
    calibration: list[dict[str, Any]] = field(default_factory=list)
    board: dict[str, Any] | None = None
    unverified_limit_claims: list[dict[str, Any]] = field(default_factory=list)
    home: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(dataclasses.asdict(self), default=str))


def connect_ro(path: Path) -> sqlite3.Connection | None:
    """Open an SQLite db read-only (None if it does not exist)."""
    if not Path(path).exists():
        return None
    conn = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def _rows(conn: sqlite3.Connection, sql: str, params: tuple = (), warn: list[str] | None = None) -> list[dict]:
    try:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    except sqlite3.Error as e:
        if warn is not None:
            warn.append(f"ledger/queue query failed ({e}): {sql.split()[0:6]}")
        return []


def _jload(s: Any) -> Any:
    if not isinstance(s, str):
        return s
    try:
        return json.loads(s)
    except ValueError:
        return s


def _ok_judge(status: Any) -> bool:
    st = str(status or "").lower()
    return not any(b in st for b in _BAD_STATUS)


# ---------------------------------------------------------------------------------------------- ledger


def _problem_names(conn: sqlite3.Connection, warn: list[str]) -> list[str]:
    rows = _rows(conn, "SELECT problem FROM runs WHERE problem IS NOT NULL UNION "
                       "SELECT problem FROM judge_results WHERE problem IS NOT NULL UNION "
                       "SELECT problem FROM ideas WHERE problem IS NOT NULL", warn=warn)
    return sorted({r["problem"] for r in rows if r["problem"]})


def _best_judged(conn: sqlite3.Connection, problem: str, warn: list[str]) -> dict[str, Any] | None:
    rows = _rows(conn, "SELECT judge_id, file_md5, mode, status, speedup, speedup_ci_json, cluster, node, "
                       "toolchain, created_utc, finished_utc, duplicate_of FROM judge_results "
                       "WHERE problem=? AND speedup IS NOT NULL ORDER BY speedup DESC", (problem,), warn)
    for r in rows:
        if _ok_judge(r["status"]) and not r["duplicate_of"]:
            r["speedup_ci"] = _jload(r.pop("speedup_ci_json"))
            return r
    return None


def _judge_shapes(conn: sqlite3.Connection, judge_id: str, warn: list[str]) -> list[dict[str, Any]]:
    rows = _rows(conn, "SELECT shape_id, case_json, sub_us, base_us, speedup FROM judge_shapes WHERE judge_id=? "
                       "ORDER BY shape_id", (judge_id,), warn)
    for r in rows:
        case = _jload(r["case_json"])
        r["weight"] = case.get("weight") if isinstance(case, dict) else None
    return rows


def _best_local(conn: sqlite3.Connection, problem: str, warn: list[str]) -> dict[str, Any] | None:
    rows = _rows(conn, "SELECT s.run_id, s.label, s.score, s.ci_lo, s.ci_hi, s.vs_label, r.tool, r.agent, "
                       "r.created_utc, a.path, a.md5 FROM scores s JOIN runs r ON r.run_id=s.run_id "
                       "LEFT JOIN artifacts a ON a.run_id=s.run_id AND a.label=s.label "
                       "WHERE r.problem=? AND r.comparable=1 AND r.status='ok' AND s.score IS NOT NULL "
                       "ORDER BY s.score DESC LIMIT 1", (problem,), warn)
    if not rows:
        return None
    best = rows[0]
    best["shapes"] = _rows(conn, "SELECT shape_id, weight, median_us, ci_lo_us, ci_hi_us, ratio_median "
                                 "FROM shape_results WHERE run_id=? AND label=? AND seed=-1 ORDER BY shape_id",
                           (best["run_id"], best["label"]), warn)
    return best


def _p4_best(conn: sqlite3.Connection, problem: str) -> Any:
    """kdev.ledger.query.best when implemented (same-hour baseline etc.); None otherwise."""
    try:
        from kdev.ledger import query
        return query.best(conn, problem)
    except Exception:  # noqa: BLE001 - optional enrichment (stub or schema drift)
        return None


def _ideas(conn: sqlite3.Connection, warn: list[str]) -> list[dict[str, Any]]:
    ideas = _rows(conn, "SELECT idea_id, problem, title, hypothesis, kind, status, owner, expected_gain_pct, "
                        "tags, variant, created_utc, updated_utc, resolution FROM ideas "
                        "ORDER BY updated_utc DESC LIMIT 300", warn=warn)
    ev = _rows(conn, "SELECT e.idea_id, e.ref_kind, e.ref, e.supports, r.tool AS run_tool FROM idea_evidence e "
                     "LEFT JOIN runs r ON e.ref_kind='run' AND r.run_id=e.ref", warn=warn)
    by: dict[int, list[dict]] = {}
    for e in ev:
        by.setdefault(e["idea_id"], []).append(e)
    for i in ideas:
        evs = by.get(i["idea_id"], [])
        tools = {str(e.get("run_tool") or "") for e in evs}
        i["evidence_n"] = len(evs)
        i["evidence_for"] = sum(1 for e in evs if e.get("supports") == "for")
        i["evidence_against"] = sum(1 for e in evs if e.get("supports") == "against")
        i["has_sass"] = any(t.startswith("sass") for t in tools)
        i["has_sweep"] = "sweep" in tools
    return ideas


def _unverified_limits(ideas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """LESSONS 1: an 'at the limit' claim needs a sass run AND a sweep run as evidence."""
    out = []
    for i in ideas:
        if i.get("kind") != "limit" or i.get("status") in ("refuted", "abandoned"):
            continue
        missing = [n for n, ok in (("sass", i.get("has_sass")), ("sweep", i.get("has_sweep"))) if not ok]
        if missing:
            out.append({"idea_id": i["idea_id"], "problem": i.get("problem"), "title": i.get("title"),
                        "owner": i.get("owner"), "status": i.get("status"), "missing": missing})
    return out


def _calibration(conn: sqlite3.Connection, warn: list[str]) -> list[dict[str, Any]]:
    rows = _rows(conn, "SELECT c.run_id, c.judge_id, c.label, c.shape_id, c.local_speedup, c.judge_speedup, "
                       "c.local_over_judge, c.flag, r.problem, r.created_utc FROM calib c "
                       "LEFT JOIN runs r ON r.run_id=c.run_id ORDER BY r.created_utc DESC LIMIT 500", warn=warn)
    latest: dict[tuple, dict] = {}
    for r in rows:                               # newest first: keep the latest per (problem, shape)
        latest.setdefault((r.get("problem"), r["shape_id"]), r)
    return sorted(latest.values(), key=lambda r: (str(r.get("problem")), r["shape_id"]))


def _recent_runs(conn: sqlite3.Connection, n: int, warn: list[str]) -> list[dict[str, Any]]:
    return _rows(conn, "SELECT run_id, tool, status, verdict, problem, agent, created_utc, comparable, "
                       "exclusive, error_code FROM runs ORDER BY created_utc DESC LIMIT ?", (n,), warn)


# ---------------------------------------------------------------------------------------------- queue


def _queue(cfg: Config, warn: list[str], recent: int = 20, offenders_since_s: float = 3 * 3600) -> dict[str, Any]:
    try:
        from kdev.service import queue as q
        snap = q.snapshot(cfg, recent=recent, offenders_since_s=offenders_since_s)
        d = json.loads(json.dumps(dataclasses.asdict(snap), default=str))
        d["source"] = "service.queue.snapshot"
        return d
    except Exception:  # noqa: BLE001 - not implemented yet / daemon db busy: direct read-only fallback
        pass
    conn = connect_ro(cfg.queue_path)
    out: dict[str, Any] = {"daemon": {}, "running": [], "queued": [], "recent": [], "offenders": [], "gpu": {},
                           "source": "queue.db (read-only)"}
    if conn is None:
        out["source"] = "none (no queue.db)"
        return out
    try:
        cols = "job_id, lane, kind, state, priority, agent, run_id, submitted_utc, started_utc, finished_utc, " \
               "exit_code, error_code, waited_s"
        out["running"] = _rows(conn, f"SELECT {cols} FROM jobs WHERE state='running' ORDER BY started_utc",
                               warn=warn)
        queued = _rows(conn, f"SELECT {cols} FROM jobs WHERE state='queued' "
                             "ORDER BY priority DESC, submitted_utc", warn=warn)
        for i, j in enumerate(queued):
            j["position"] = i + 1
        out["queued"] = queued
        out["recent"] = _rows(conn, f"SELECT {cols} FROM jobs WHERE state NOT IN ('queued','running') "
                                    "ORDER BY finished_utc DESC LIMIT ?", (recent,), warn)
        hb = _rows(conn, "SELECT pid, host, ts_utc, version, gpu_lane_job, build_workers FROM heartbeat", warn=warn)
        if hb:
            d = hb[0]
            age = _age_s(d.get("ts_utc"))
            d["age_s"] = age
            d["alive"] = age is not None and age < 15
            out["daemon"] = d
        since = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - offenders_since_s))
        out["offenders"] = _rows(conn, "SELECT ts_utc, pid, cmdline, gpu_mem_mib, during_job, action FROM offenders "
                                       "WHERE ts_utc>=? ORDER BY ts_utc DESC LIMIT 50", (since,), warn)
    finally:
        conn.close()
    return out


def _age_s(ts: Any) -> float | None:
    try:
        t = calendar.timegm(time.strptime(str(ts)[:19], "%Y-%m-%dT%H:%M:%S"))
        return round(time.time() - t, 1)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------------------------- main


def collect(cfg: Config, *, scrape: bool = False, recent: int = 30, team: str | None = None,
            extra_problems: list[str] | None = None,
            board_fetch: Callable[[str, float], Any] | None = None) -> DashboardData:
    """Build the dashboard data. Never writes to the ledger or queue. Missing dbs give empty sections plus a
    warning; ``scrape=True`` adds the public board (GET only) and per-problem/per-shape gaps."""
    data = DashboardData(generated_utc=utc_now(), home=str(cfg.home))
    warn = data.warnings
    conn = connect_ro(cfg.ledger_path)
    names: list[str] = list(extra_problems or [])
    if conn is None:
        warn.append(f"no ledger at {cfg.ledger_path} (nothing recorded yet)")
    else:
        try:
            names = sorted(set(names) | set(_problem_names(conn, warn)))
            data.ideas = _ideas(conn, warn)
            data.unverified_limit_claims = _unverified_limits(data.ideas)
            data.calibration = _calibration(conn, warn)
            data.recent_runs = _recent_runs(conn, recent, warn)
            for p in names:
                bj = _best_judged(conn, p, warn)
                bl = _best_local(conn, p, warn)
                entry: dict[str, Any] = {
                    "problem": p, "best_judged": bj, "best_local": bl,
                    "judge_shapes": _judge_shapes(conn, bj["judge_id"], warn) if bj else [],
                    "open_ideas": sum(1 for i in data.ideas if i.get("problem") == p
                                      and i.get("status") in _OPEN_IDEA),
                    "unverified_limits": sum(1 for c in data.unverified_limit_claims if c.get("problem") == p),
                    "disagree": sum(1 for c in data.calibration if c.get("problem") == p
                                    and str(c.get("flag")).upper() == "DISAGREE"),
                }
                p4 = _p4_best(conn, p)
                if p4 is not None:
                    entry["ledger_best"] = json.loads(json.dumps(p4, default=str))
                data.problems.append(entry)
        finally:
            conn.close()
    if conn is None:
        data.problems = [{"problem": p, "best_judged": None, "best_local": None, "judge_shapes": [],
                          "open_ideas": 0, "unverified_limits": 0, "disagree": 0} for p in names]
    data.queue = _queue(cfg, warn)
    data.offenders = list(data.queue.get("offenders") or [])
    if scrape:
        from kdev.dashboard import board as b
        data.board = b.scrape(problems=names, team=team, fetch=board_fetch)
        gap_rows = {g["problem"]: g for g in b.gaps(data.board, data.problems)}
        for p in data.problems:
            p["gap"] = gap_rows.get(p["problem"])
        for p, msg in data.board.get("errors", {}).items():
            warn.append(f"board {p}: {msg}")
    return data
