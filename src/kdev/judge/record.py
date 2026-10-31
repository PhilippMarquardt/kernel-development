"""`kdev judge record JOB.json`: ingest an `okc status --json` / `okc check --json` result. Owner: P4.
Fixture: tests/fixtures/judge/judge_moe.json (keys: id, cluster, content_hash, duplicate_of, result{records,
machine{node}, speedup, speedup_ci, evaluator, log_tail}).

Recorded per judge job: problem, mode, status (ok|fail|duplicate|...), speedup + CI, cluster, node, nvcc version
(from log_tail when printed), created/finished UTC, the judge's content hash (``content_sha256``), team,
optional label/role/pair_id, and the md5/norm_md5 of the exact file when ``--file`` is given (LESSONS 6:
record the exact variant). Per shape: case, sub_us, base_us, speedup.
"""
from __future__ import annotations

import sqlite3
from typing import Any

from kdev.judge._parse import JRun, parse_judge
from kdev.ledger._util import epoch_to_utc, jdump, md5_file, norm_md5, utc_now, utc_to_epoch

SAME_HOUR_S = 3600.0
JUDGE_NOISE = 0.02          # judge noise +-1.5-2 % (LESSONS 3)


def record_parsed(conn: sqlite3.Connection, jr: JRun, *, file_path: str | None = None,
                  raw_path: str | None = None, label: str | None = None, role: str | None = None,
                  pair_id: str | None = None, file_md5: str | None = None,
                  file_norm_md5: str | None = None) -> str:
    md5, nm = file_md5, file_norm_md5
    if file_path:
        md5 = md5_file(file_path)
        try:
            nm = norm_md5(file_path)
        except (OSError, ValueError):
            nm = None
    old = conn.execute("SELECT file_md5, norm_md5, label, role, pair_id, raw_path FROM judge_results "
                       "WHERE judge_id=?", (jr.judge_id,)).fetchone()
    if old is not None:      # re-recording keeps what we knew before (e.g. the file md5 from `--file`)
        md5 = md5 or old["file_md5"]
        nm = nm or old["norm_md5"]
        label = label or old["label"]
        role = role or old["role"]
        pair_id = pair_id or old["pair_id"]
        raw_path = raw_path or old["raw_path"]
    row = {
        "judge_id": jr.judge_id, "problem": jr.problem, "file_md5": md5, "norm_md5": nm, "mode": jr.mode,
        "status": jr.status, "speedup": jr.speedup, "speedup_ci_json": jdump(jr.speedup_ci),
        "cluster": jr.cluster, "node": jr.node, "toolchain": jr.toolchain,
        "created_utc": epoch_to_utc(jr.created_at), "finished_utc": epoch_to_utc(jr.finished_at),
        "duplicate_of": jr.duplicate_of, "raw_path": raw_path, "recorded_utc": utc_now(),
        "content_sha256": jr.content_sha256, "team": jr.team, "label": label, "role": role, "pair_id": pair_id,
    }
    cols = list(row)
    with conn:
        conn.execute(f"INSERT OR REPLACE INTO judge_results ({','.join(cols)}) "
                     f"VALUES ({','.join('?' * len(cols))})", [row[c] for c in cols])
        conn.execute("DELETE FROM judge_shapes WHERE judge_id=?", (jr.judge_id,))
        for s in jr.shapes:
            conn.execute("INSERT OR REPLACE INTO judge_shapes(judge_id,shape_id,case_json,sub_us,base_us,speedup) "
                         "VALUES (?,?,?,?,?,?)",
                         (jr.judge_id, s.shape_id, jdump(s.case), s.sub_us, s.base_us, s.speedup))
    return jr.judge_id


def record(conn: sqlite3.Connection, judge_json: dict[str, Any], file_path: str | None = None,
           raw_path: str | None = None, *, label: str | None = None, role: str | None = None,
           pair_id: str | None = None) -> str:
    """Upsert judge_results + judge_shapes; returns judge_id. Uses kdev.timing.calib.parse_judge_json."""
    return record_parsed(conn, parse_judge(judge_json), file_path=file_path, raw_path=raw_path, label=label,
                         role=role, pair_id=pair_id)


def _row(r: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(r) if r is not None else None


def same_hour(conn: sqlite3.Connection, judge_row: dict[str, Any], window_s: float = SAME_HOUR_S
              ) -> list[dict[str, Any]]:
    """Other judged runs of the same problem within +-window of ``judge_row`` (base rows first)."""
    t = utc_to_epoch(judge_row.get("created_utc"))
    if t is None:
        return []
    cur = conn.execute("SELECT judge_id, label, role, pair_id, speedup, cluster, created_utc, status "
                       "FROM judge_results WHERE problem=? AND judge_id != ? AND status='ok' "
                       "AND mode IN ('check','submit')", (judge_row.get("problem"), judge_row["judge_id"]))
    out = []
    for r in cur:
        rt = utc_to_epoch(r["created_utc"])
        if rt is not None and abs(rt - t) <= window_s:
            d = dict(r)
            d["dt_min"] = round((rt - t) / 60.0, 1)
            out.append(d)
    cl = judge_row.get("cluster")
    out.sort(key=lambda d: (d["role"] != "base", d["cluster"] != cl, abs(d["dt_min"])))
    return out


def baseline_for(conn: sqlite3.Connection, judge_row: dict[str, Any]) -> dict[str, Any] | None:
    """The same-hour baseline of a judged run: the role=base row of its pair, else the nearest base-labelled
    run of the same problem within an hour (None if there is none: LESSONS 3 says judge one)."""
    if judge_row.get("pair_id"):
        r = conn.execute("SELECT * FROM judge_results WHERE pair_id=? AND role='base' AND status='ok' "
                         "AND judge_id != ? ORDER BY created_utc",
                         (judge_row["pair_id"], judge_row["judge_id"])).fetchall()
        if r:
            sp = [x["speedup"] for x in r if x["speedup"]]
            d = dict(r[0])
            d["pooled_speedup"] = _gmean(sp)
            d["n"] = len(sp)
            return d
    for d in same_hour(conn, judge_row):
        if d.get("role") == "base":
            return d
    return None


def _gmean(xs: list[float]) -> float | None:
    import math
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def ledger(conn: sqlite3.Connection, problem: str | None = None) -> list[dict[str, Any]]:
    """Judged bests and same-hour baselines per problem."""
    sql = ("SELECT * FROM judge_results WHERE status='ok' AND mode IN ('check','submit') AND speedup IS NOT NULL"
           + (" AND problem=?" if problem else "") + " ORDER BY problem, speedup DESC")
    best: dict[str, dict[str, Any]] = {}
    counts: dict[str, int] = {}
    for r in conn.execute(sql, (problem,) if problem else ()):
        p = r["problem"]
        counts[p] = counts.get(p, 0) + 1
        if p not in best:
            best[p] = dict(r)
    out = []
    for p, b in best.items():
        base = baseline_for(conn, b)
        top = conn.execute("SELECT team, speedup, taken_utc FROM board_rows WHERE problem=? "
                           "AND taken_utc=(SELECT max(taken_utc) FROM board_rows WHERE problem=?) "
                           "ORDER BY speedup DESC LIMIT 1", (p, p)).fetchone()
        out.append({
            "problem": p, "judge_id": b["judge_id"], "speedup": b["speedup"], "label": b["label"],
            "cluster": b["cluster"], "created_utc": b["created_utc"], "file_md5": b["file_md5"],
            "n_judged": counts[p],
            "baseline": ({"judge_id": base["judge_id"], "speedup": base.get("pooled_speedup") or base["speedup"],
                          "label": base.get("label"), "cluster": base.get("cluster"),
                          "created_utc": base.get("created_utc")} if base else None),
            "vs_baseline": ((b["speedup"] / (base.get("pooled_speedup") or base["speedup"]))
                            if base and (base.get("pooled_speedup") or base.get("speedup")) else None),
            "board_top": dict(top) if top else None,
            "cluster_mismatch": bool(base and base.get("cluster") and b["cluster"]
                                     and base.get("cluster") != b["cluster"]),
        })
    return out
