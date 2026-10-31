"""Ledger queries: runs, "has this been tried", bests, per-shape history. Owner: P4."""
from __future__ import annotations

import csv
import io
import json
import os
import sqlite3
from dataclasses import asdict, dataclass, field
from typing import Any

from kdev.contract import ErrorCode, KdevError
from kdev.ledger._util import jload, md5_file, norm_md5

EXPORT_TABLES = ("runs", "artifacts", "shape_results", "scores", "builds", "kernels", "judge_results",
                 "judge_shapes", "calib", "ideas", "idea_evidence", "board_rows")


@dataclass
class TriedHit:
    match: str                 # md5 | norm_md5 | knobs | idea_text
    run_id: str | None = None
    judge_id: str | None = None
    idea_id: int | None = None
    tool: str = ""
    verdict: str = ""
    created_utc: str = ""
    comparable: bool = True
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TriedReport:
    query: dict[str, Any] = field(default_factory=dict)
    hits: list[TriedHit] = field(default_factory=list)

    @property
    def tried(self) -> bool:
        return bool(self.hits)


def runs(conn: sqlite3.Connection, *, problem: str | None = None, md5: str | None = None,
         tool: str | None = None, since_utc: str | None = None, limit: int = 30) -> list[dict[str, Any]]:
    """Recorded runs, newest first. ``md5`` matches any artifact's md5 or norm_md5."""
    where, params = [], []
    if problem:
        where.append("r.problem=?")
        params.append(problem)
    if tool:
        where.append("r.tool=?")
        params.append(tool)
    if since_utc:
        where.append("r.created_utc>=?")
        params.append(since_utc)
    if md5:
        where.append("r.run_id IN (SELECT run_id FROM artifacts WHERE md5=? OR norm_md5=?)")
        params += [md5, md5]
    sql = ("SELECT r.run_id, r.tool, r.status, r.verdict, r.problem, r.agent, r.created_utc, r.comparable, "
           "r.nvcc_version, r.flush_method, r.exclusive, r.optouts_json, r.run_dir, r.summary_json, "
           "(SELECT group_concat(label || ':' || substr(md5,1,8), ' ') FROM artifacts a "
           "WHERE a.run_id=r.run_id) AS files FROM runs r")
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY r.created_utc DESC LIMIT ?"
    out = []
    for r in conn.execute(sql, (*params, int(limit))):
        d = dict(r)
        d["comparable"] = bool(d["comparable"])
        d["summary"] = jload(d.pop("summary_json"), {})
        d["optouts"] = jload(d.pop("optouts_json"), [])
        out.append(d)
    return out


def _knobs_match(have: dict[str, Any], want: dict[str, Any]) -> bool:
    return all(k in have and str(have[k]) == str(v) for k, v in want.items())


def tried(conn: sqlite3.Connection, *, file: str | None = None, text: str | None = None,
          knobs: dict[str, Any] | None = None, problem: str | None = None) -> TriedReport:
    """Exact md5, norm_md5 (comment-insensitive), knob-subset match within the problem, FTS over ideas."""
    from kdev.ledger import ideas as _ideas

    rep = TriedReport(query={"file": file, "text": text, "knobs": knobs or {}, "problem": problem})
    seen: set[tuple[str, str]] = set()

    def add(h: TriedHit) -> None:
        key = (h.match if h.match in ("knobs", "idea_text") else "file",
               str(h.run_id or h.judge_id or h.idea_id))
        if key not in seen:
            seen.add(key)
            rep.hits.append(h)

    if file:
        if not os.path.isfile(file):
            raise KdevError(ErrorCode.NOT_FOUND, f"no such file: {file}", details={"path": file})
        md5 = md5_file(file)
        try:
            nm = norm_md5(file)
        except (OSError, ValueError):
            nm = None
        rep.query.update(md5=md5, norm_md5=nm)
        for col, val in (("md5", md5), ("norm_md5", nm)):
            if not val:
                continue
            for r in conn.execute(
                    f"SELECT a.run_id, a.label, a.path, r.tool, r.verdict, r.created_utc, r.comparable "
                    f"FROM artifacts a JOIN runs r ON r.run_id=a.run_id WHERE a.{col}=? "
                    f"ORDER BY r.created_utc DESC", (val,)):
                add(TriedHit(col, run_id=r["run_id"], tool=r["tool"], verdict=r["verdict"] or "",
                             created_utc=r["created_utc"], comparable=bool(r["comparable"]),
                             detail=f"{r['label']} {r['path'] or ''}".strip()))
            jcol = "file_md5" if col == "md5" else "norm_md5"
            for r in conn.execute(f"SELECT judge_id, status, speedup, created_utc, label, cluster "
                                  f"FROM judge_results WHERE {jcol}=? ORDER BY created_utc DESC", (val,)):
                add(TriedHit(col, judge_id=r["judge_id"], tool="judge", verdict=r["status"] or "",
                             created_utc=r["created_utc"] or "",
                             detail=f"x{r['speedup']:.4f} {r['cluster'] or ''} {r['label'] or ''}".strip()
                             if r["speedup"] else (r["label"] or "")))
        try:
            from kdev.ledger._util import sha256_file
            sha = sha256_file(file)
            for r in conn.execute("SELECT judge_id, status, speedup, created_utc, label FROM judge_results "
                                  "WHERE content_sha256=?", (sha,)):
                add(TriedHit("md5", judge_id=r["judge_id"], tool="judge", verdict=r["status"] or "",
                             created_utc=r["created_utc"] or "", detail="judge content_hash match"))
        except OSError:
            pass
    if knobs:
        sql = ("SELECT a.run_id, a.label, a.knobs_json, a.path, r.tool, r.verdict, r.created_utc, r.comparable "
               "FROM artifacts a JOIN runs r ON r.run_id=a.run_id WHERE a.knobs_json IS NOT NULL"
               + (" AND r.problem=?" if problem else "") + " ORDER BY r.created_utc DESC")
        for r in conn.execute(sql, (problem,) if problem else ()):
            have = jload(r["knobs_json"], {}) or {}
            if isinstance(have, dict) and _knobs_match(have, knobs):
                add(TriedHit("knobs", run_id=r["run_id"], tool=r["tool"], verdict=r["verdict"] or "",
                             created_utc=r["created_utc"], comparable=bool(r["comparable"]),
                             detail=f"{r['label']} {json.dumps(have, sort_keys=True)}"))
    if text:
        for i in _ideas.search(conn, text, problem=problem):
            add(TriedHit("idea_text", idea_id=i.idea_id, tool=f"idea:{i.kind}", verdict=i.status,
                         created_utc=i.created_utc,
                         detail=(i.title + (f" [variant: {i.variant}]" if i.variant else ""))[:160]))
    return rep


def best(conn: sqlite3.Connection, problem: str | None = None) -> list[dict[str, Any]]:
    """Per problem: best judged (judge_results) and best local comparable score, with same-hour baseline."""
    from kdev.judge.record import ledger as judge_ledger

    judged = {d["problem"]: d for d in judge_ledger(conn, problem)}
    sql = ("SELECT r.problem, s.run_id, s.label, s.score, s.ci_lo, s.ci_hi, r.created_utc, r.tool "
           "FROM scores s JOIN runs r ON r.run_id=s.run_id WHERE r.comparable=1 AND s.score IS NOT NULL "
           "AND instr(s.label, '/') = 0" + (" AND r.problem=?" if problem else "")
           + " ORDER BY r.problem, s.score DESC")
    local: dict[str, dict[str, Any]] = {}
    for r in conn.execute(sql, (problem,) if problem else ()):
        if r["problem"] not in local:
            d = dict(r)
            a = conn.execute("SELECT path, md5 FROM artifacts WHERE run_id=? AND label=?",
                             (r["run_id"], r["label"])).fetchone()
            d["path"], d["md5"] = (a["path"], a["md5"]) if a else (None, None)
            local[r["problem"]] = d
    probs = sorted(set(judged) | set(local), key=lambda x: str(x))
    return [{"problem": p, "judged": judged.get(p), "local": local.get(p)} for p in probs]


def shape_history(conn: sqlite3.Connection, problem: str | None = None, shape: str | None = None,
                  limit: int = 200) -> list[dict[str, Any]]:
    """Per-shape history, oldest first: judged rows (judge_shapes) and local pooled rows (shape_results).
    ``shape`` is a substring of the shape id."""
    out: list[dict[str, Any]] = []
    jsql = ("SELECT j.created_utc AS utc, 'judge' AS source, j.judge_id AS ref, j.label, j.cluster, "
            "s.shape_id, s.sub_us, s.base_us, s.speedup FROM judge_shapes s JOIN judge_results j "
            "ON j.judge_id=s.judge_id WHERE j.status IN ('ok','fail')")
    lsql = ("SELECT r.created_utc AS utc, 'local' AS source, r.run_id AS ref, a.path AS label, NULL AS cluster, "
            "s.shape_id, s.median_us AS sub_us, NULL AS base_us, s.ratio_median AS speedup "
            "FROM shape_results s JOIN runs r ON r.run_id=s.run_id LEFT JOIN artifacts a "
            "ON a.run_id=s.run_id AND a.label=s.label WHERE s.seed=-1 AND r.comparable=1")
    for sql, pcol in ((jsql, "j.problem"), (lsql, "r.problem")):
        params: list[Any] = []
        if problem:
            sql += f" AND {pcol}=?"
            params.append(problem)
        if shape:
            sql += " AND instr(s.shape_id, ?) > 0"
            params.append(shape)
        out += [dict(r) for r in conn.execute(sql, params)]
    out.sort(key=lambda d: (d["shape_id"], d["utc"] or ""))
    return out[-limit:] if limit else out


def export(conn: sqlite3.Connection, fmt: str = "jsonl", tables: tuple[str, ...] = ("runs",)) -> str:
    """Tables as JSONL (one object per row, with a ``_table`` key) or CSV (one block per table)."""
    for t in tables:
        if t not in EXPORT_TABLES:
            raise KdevError(ErrorCode.INVALID_REQUEST, f"unknown table {t!r}", details={"field": "table"},
                            recovery="one of " + ", ".join(EXPORT_TABLES))
    buf = io.StringIO()
    for t in tables:
        cur = conn.execute(f"SELECT * FROM {t}")
        cols = [c[0] for c in cur.description]
        if fmt == "jsonl":
            for r in cur:
                buf.write(json.dumps({"_table": t, **dict(zip(cols, tuple(r)))}, default=str) + "\n")
        elif fmt == "csv":
            if len(tables) > 1:
                buf.write(f"# table: {t}\n")
            w = csv.writer(buf, lineterminator="\n")
            w.writerow(cols)
            for r in cur:
                w.writerow(tuple(r))
        else:
            raise KdevError(ErrorCode.INVALID_REQUEST, f"bad format {fmt!r}", details={"field": "format"})
    return buf.getvalue()
