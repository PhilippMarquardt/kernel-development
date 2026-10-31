"""Ingest Results into the ledger. Owner: P4. Called by the daemon / direct mode after every job (P2) and by
`kdev ledger import`. Idempotent per run_id (child rows are replaced).

Mapping (Result envelope ``kdev.result/1`` + ``meta.json`` beside it):
- ``runs``: envelope + ``fidelity`` (+ meta: job spec, host, gpu, toolchain). ``comparable`` = 0 when
  fidelity has opt-outs, exclusivity is not ok/waited, the toolchain mismatched, the protocol is not the current
  one, or a timing tool (time/ab/sweep/calib) has no fidelity at all.
- ``artifacts``: meta job ``files`` (path, label, md5, knobs, defines, role) plus labels seen in
  ``shapes[].per_file`` (md5 only). ``norm_md5`` is computed when the path exists.
- ``shape_results``: one pooled row (seed -1) per (label, shape) with median/CI/n; ``ratio`` goes on the
  numerator label; per-seed medians become rows with ``seed`` set.
- ``scores``: ``score.per_file`` (+ ``score.delta`` as the CI of the numerator vs ``den``).
- optional summary sections written by other tools: ``summary.builds`` [{label, ext_name, cache_key,
  cache_hit, seconds, so_path, so_notes}], ``summary.kernels`` [{label, kernel|name, regs, spill_store,
  spill_load, stack, smem_static, barriers, c751x|warnings, sass_instr}], ``summary.calib`` [{judge_id,
  label, shape_id, local_speedup, judge_speedup, local_over_judge, flag}].
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any

from kdev.contract import PROTOCOL, ErrorCode, Fidelity, KdevError, Result
from kdev.ledger._util import jdump, norm_md5, utc_now

TIMING_TOOLS = ("time", "ab", "sweep", "calib")
CHILD_TABLES = ("artifacts", "shape_results", "scores", "builds", "kernels", "calib")
_RUN_ID_TS = re.compile(r"^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})")


def _created_from_run_id(run_id: str) -> str | None:
    m = _RUN_ID_TS.match(run_id or "")
    if not m:
        return None
    y, mo, d, h, mi, s = m.groups()
    return f"{y}-{mo}-{d}T{h}:{mi}:{s}Z"


def _comparable(result: Result) -> tuple[bool, list[str]]:
    why: list[str] = []
    f = result.fidelity
    if f is None:
        if result.tool in TIMING_TOOLS:
            why.append("no fidelity record")
        return (not why), why
    if f.optouts:
        why.append("optouts=" + ",".join(f.optouts))
    if f.exclusive not in ("ok", "waited"):
        why.append(f"exclusive={f.exclusive}")
    if not f.toolchain.get("match", True):
        why.append("toolchain mismatch")
    if result.protocol != PROTOCOL:
        why.append(f"protocol {result.protocol} != {PROTOCOL}")
    return (not why), why


def _job(meta: dict[str, Any]) -> dict[str, Any]:
    for k in ("job", "spec", "job_spec"):
        if isinstance(meta.get(k), dict):
            return meta[k]
    return meta


def _overhead(f: Fidelity | None) -> float | None:
    if f is None or not f.overhead_us:
        return None
    o = f.overhead_us
    for k in ("subtracted", "judge_const"):
        if isinstance(o.get(k), (int, float)):
            return float(o[k])
    return None


def rows_from_result(result: Result, meta: dict[str, Any] | None = None) -> dict[str, list[dict[str, Any]]]:
    """Pure: Result -> {table: [row dicts]} (unit-tested without sqlite)."""
    meta = meta or {}
    job = _job(meta)
    run_id = result.run_id or meta.get("run_id") or job.get("run_id")
    if not run_id:
        raise KdevError(ErrorCode.INVALID_REQUEST, "result has no run_id", details={"field": "run_id"},
                        recovery="ingest a result.json written by RunDir.write_result")
    f = result.fidelity
    comparable, _why = _comparable(result)
    tc = (f.toolchain if f else {}) or {}
    meta_tc = meta.get("toolchain") if isinstance(meta.get("toolchain"), dict) else {}
    gpu = meta.get("gpu") if isinstance(meta.get("gpu"), dict) else {}
    host = meta.get("host")
    if isinstance(host, dict):
        host = host.get("name") or host.get("hostname")
    params = job.get("params") if isinstance(job.get("params"), dict) else {}
    created = (meta.get("created_utc") or job.get("submitted_utc") or _created_from_run_id(run_id)
               or utc_now())
    run = {
        "run_id": run_id, "job_id": result.job_id or job.get("job_id") or None, "tool": result.tool,
        "status": result.status, "verdict": result.verdict or None,
        "problem": job.get("problem") or meta.get("problem") or result.summary.get("problem"),
        "adapter": job.get("adapter") or meta.get("adapter"),
        "agent": job.get("agent") or meta.get("agent") or None,
        "host": host, "gpu_name": gpu.get("name") or meta.get("gpu_name"),
        "gpu_uuid": gpu.get("uuid") or meta.get("gpu_uuid"),
        "created_utc": created, "finished_utc": meta.get("finished_utc") or result.summary.get("finished_utc"),
        "protocol": result.protocol, "comparable": int(comparable),
        "nvcc_version": tc.get("nvcc") or meta_tc.get("nvcc_version"), "toolchain_pin": tc.get("pin"),
        "toolchain_json": jdump(tc or meta_tc or None),
        "flush_method": (f.flush or {}).get("method") if f else None,
        "flush_bytes": (f.flush or {}).get("bytes") if f else None,
        "graph": (int(bool(f.graph)) if f else None), "exclusive": f.exclusive if f else None,
        "waited_s": f.waited_s if f else None, "seeds_json": jdump(f.seeds if f else None),
        "reps": params.get("reps"), "overhead_us": _overhead(f),
        "optouts_json": jdump(f.optouts if f else None),
        "clocks_before_json": jdump((f.clocks or {}).get("before") if f else None),
        "clocks_after_json": jdump((f.clocks or {}).get("after") if f else None),
        "run_dir": result.files.get("dir") or meta.get("run_dir"),
        "summary_json": jdump(result.summary or None),
        "error_code": result.error.code if result.error else None,
    }
    out: dict[str, list[dict[str, Any]]] = {t: [] for t in ("runs",) + CHILD_TABLES}
    out["runs"].append(run)

    arts: dict[str, dict[str, Any]] = {}
    for fr in job.get("files") or []:
        if not isinstance(fr, dict):
            continue
        label = fr.get("label") or "A"
        path = fr.get("path")
        nm = fr.get("norm_md5")
        if not nm and path and os.path.isfile(path):
            try:
                nm = norm_md5(path)
            except (OSError, ValueError):
                nm = None
        arts[label] = {"run_id": run_id, "label": label, "role": fr.get("role") or "sub", "path": path,
                       "md5": fr.get("md5") or "", "norm_md5": nm, "knobs_json": jdump(fr.get("knobs") or None),
                       "defines": " ".join(fr.get("defines") or []) or None, "so_md5": fr.get("so_md5")}

    for sh in result.shapes or []:
        sid = sh.get("shape_id")
        if not sid:
            continue
        ratio_by_num = {r.get("num"): r for r in (sh.get("ratios") or []) if isinstance(r, dict)}
        if isinstance(sh.get("ratio"), dict):
            ratio_by_num.setdefault(sh["ratio"].get("num"), sh["ratio"])
        for pf in sh.get("per_file") or []:
            label = pf.get("label") or "A"
            ratio = ratio_by_num.get(label) or {}
            if label not in arts:
                arts[label] = {"run_id": run_id, "label": label, "role": "sub", "path": None,
                               "md5": pf.get("md5") or "", "norm_md5": None, "knobs_json": None,
                               "defines": None, "so_md5": None}
            elif not arts[label]["md5"] and pf.get("md5"):
                arts[label]["md5"] = pf["md5"]
            is_num = ratio.get("num") == label
            out["shape_results"].append({
                "run_id": run_id, "label": label, "shape_id": sid, "load": sh.get("load"),
                "weight": sh.get("weight"), "seed": -1, "median_us": pf.get("median_us"),
                "ci_lo_us": pf.get("ci_lo_us"), "ci_hi_us": pf.get("ci_hi_us"), "n": pf.get("n"),
                "ratio_den_label": ratio.get("den") if is_num else None,
                "ratio_median": ratio.get("median") if is_num else None,
                "ratio_ci_lo": ratio.get("ci_lo") if is_num else None,
                "ratio_ci_hi": ratio.get("ci_hi") if is_num else None,
                "check_status": sh.get("check_status") or sh.get("verdict") or None})
            seeds = pf.get("seeds") or []
            meds = pf.get("per_seed_median_us") or []
            if len(seeds) > 1 and len(seeds) == len(meds):
                for s, m in zip(seeds, meds):
                    out["shape_results"].append({
                        "run_id": run_id, "label": label, "shape_id": sid, "load": sh.get("load"),
                        "weight": sh.get("weight"), "seed": int(s), "median_us": m, "ci_lo_us": None,
                        "ci_hi_us": None, "n": None, "ratio_den_label": None, "ratio_median": None,
                        "ratio_ci_lo": None, "ratio_ci_hi": None, "check_status": None})
    out["artifacts"] = list(arts.values())

    sc = result.score or {}
    per = sc.get("per_file") or {}
    delta = sc.get("delta") or {}
    num, den = delta.get("num"), delta.get("den")
    if per and not (num and den) and len(per) == 2 and delta:
        den, num = list(per)[0], list(per)[1]
    for label, val in per.items():
        row = {"run_id": run_id, "label": label, "score": val, "ci_lo": None, "ci_hi": None, "vs_label": None}
        if isinstance(val, dict):
            row.update(score=val.get("median", val.get("score")), ci_lo=val.get("ci_lo"), ci_hi=val.get("ci_hi"))
        if label == num and delta:
            row.update(vs_label=den)
        out["scores"].append(row)
    if delta and num and den:
        out["scores"].append({"run_id": run_id, "label": f"{num}/{den}", "score": delta.get("median"),
                              "ci_lo": delta.get("ci_lo"), "ci_hi": delta.get("ci_hi"), "vs_label": den})

    summ = result.summary or {}
    for b in summ.get("builds") or []:
        if isinstance(b, dict) and b.get("ext_name"):
            out["builds"].append({"run_id": run_id, "label": b.get("label") or "A", "ext_name": b["ext_name"],
                                  "cache_key": b.get("cache_key"),
                                  "cache_hit": None if b.get("cache_hit") is None else int(bool(b["cache_hit"])),
                                  "seconds": b.get("seconds"), "so_path": b.get("so_path"),
                                  "so_notes": (" | ".join(b["so_notes"]) if isinstance(b.get("so_notes"), list)
                                               else b.get("so_notes"))})
    for k in summ.get("kernels") or []:
        if not isinstance(k, dict):
            continue
        name = k.get("kernel") or k.get("name")
        if not name:
            continue
        c751 = k.get("c751x")
        if c751 is None and k.get("warnings"):
            c751 = " | ".join(w for w in k["warnings"] if "C751" in str(w)) or None
        out["kernels"].append({"run_id": run_id, "label": k.get("label") or "A", "kernel": name,
                               "regs": k.get("regs"), "spill_store": k.get("spill_store"),
                               "spill_load": k.get("spill_load"), "stack": k.get("stack"),
                               "smem_static": k.get("smem_static"), "barriers": k.get("barriers"),
                               "c751x": c751, "sass_instr": k.get("sass_instr")})
    for c in (summ.get("calib") or []) + (summ.get("calib_rows") or []):
        if isinstance(c, dict) and c.get("judge_id") and c.get("shape_id"):
            out["calib"].append({"run_id": run_id, "judge_id": c["judge_id"], "label": c.get("label") or "A",
                                 "shape_id": c["shape_id"], "local_speedup": c.get("local_speedup"),
                                 "judge_speedup": c.get("judge_speedup"),
                                 "local_over_judge": c.get("local_over_judge"), "flag": c.get("flag")})
    return out


def _insert(conn: sqlite3.Connection, table: str, row: dict[str, Any]) -> None:
    cols = list(row)
    conn.execute(f"INSERT OR REPLACE INTO {table} ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                 [row[c] for c in cols])


def write_rows(conn: sqlite3.Connection, rows: dict[str, list[dict[str, Any]]]) -> str:
    """Upsert the output of rows_from_result in one transaction (children replaced)."""
    run_id = rows["runs"][0]["run_id"]
    with conn:
        for t in CHILD_TABLES:
            conn.execute(f"DELETE FROM {t} WHERE run_id=?", (run_id,))
        for t in ("runs",) + CHILD_TABLES:
            for r in rows.get(t, []):
                _insert(conn, t, r)
    return run_id


def _load_result_path(result_path: str | Path) -> tuple[Path, dict[str, Any], dict[str, Any] | None]:
    p = Path(result_path)
    if p.is_dir():
        p = p / "result.json"
    if not p.is_file():
        raise KdevError(ErrorCode.NOT_FOUND, f"no result.json at {result_path}", details={"path": str(p)},
                        recovery="pass a run dir or a result.json path")
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except ValueError as e:
        raise KdevError(ErrorCode.INVALID_REQUEST, f"{p}: not JSON ({e})", details={"path": str(p)}) from e
    meta = None
    mp = p.with_name("meta.json")
    if mp.is_file():
        try:
            meta = json.loads(mp.read_text(encoding="utf-8"))
        except ValueError:
            meta = None
    return p, d, meta


def ingest_result(result_path: str | Path, conn: sqlite3.Connection | None = None,
                  meta: dict[str, Any] | None = None) -> str:
    """Read result.json (+ meta.json beside it) and upsert runs/artifacts/shape_results/scores/builds/
    kernels/calib rows; sets comparable=0 when fidelity opt-outs exist, exclusivity failed, toolchain
    mismatched or protocol != current. Returns run_id."""
    p, d, file_meta = _load_result_path(result_path)
    if not str(d.get("schema", "")).startswith("kdev.result/"):
        raise KdevError(ErrorCode.INVALID_REQUEST, f"{p}: not a kdev.result document",
                        details={"path": str(p), "schema": d.get("schema")},
                        recovery="judge JSONs go through `kdev judge record`")
    m = dict(file_meta or {})
    m.update(meta or {})
    m.setdefault("run_dir", str(p.parent))
    result = Result.from_dict(d)
    own = conn is None
    if own:
        from kdev.ledger.db import open_default
        conn = open_default()
    try:
        return write_rows(conn, rows_from_result(result, m))
    finally:
        if own:
            conn.close()


def mark_protocol_incomparable(conn: sqlite3.Connection, current_protocol: str) -> int:
    """Protocol bump: set comparable=0 on rows of older protocols; return count."""
    with conn:
        cur = conn.execute("UPDATE runs SET comparable=0 WHERE protocol != ? AND comparable != 0",
                           (current_protocol,))
    return cur.rowcount
