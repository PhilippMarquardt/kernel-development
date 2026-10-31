"""Per-shape judge/local calibration table (P3). Written by every `kdev calib`, applied by `time`/`ab`.

Why: kdev and okc/tools/calib.py agree with each other (same session: median 0.5-0.8 %), but both differ
from the judge per shape by a median ~2.4 % and up to 17 % on 3-6 us shapes (bf16, 2026-10-31). The
residual is judge-side and shape-specific: the judge subtracts a per-shape replay overhead (2*one - two,
3.8-4.4 us) where the local protocol subtracts a constant 4.0 us, the judge node/cluster differs, and the
local SGLang baseline is slower than the judge's on some shapes (bf16 router 12 %, attention ~5 %). So
the gap is recorded instead of "fixed": ``ratio = judge_us / local_us`` per shape, separately for the
submission and the baseline, from each calib run, and `time`/`ab` report ``~judge us = local us x
median ratio`` next to the measured value.

Uncertainty per shape = max(judge noise 1.5 %, half the spread of the observed ratios) plus, on shapes
under ``SMALL_US`` (overhead-dominated), the judge's per-shape overhead spread (``SMALL_ABS_US``). Such
shapes are flagged ``small``: compare absolute us there, never local speedups.

Storage: ``$KDEV_HOME/calib/<problem>.json`` (``kdev.calibtab/1``), observations keyed by the case
(judge case minus seed/load/weight/repeats/router_*), newest ``KEEP`` per shape and role.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

SCHEMA = "kdev.calibtab/1"
KEEP = 12
JUDGE_NOISE = 0.015
SMALL_US = 6.0
SMALL_ABS_US = 0.3
_DROP = ("seed", "load", "repeats", "weight", "router_load", "router_distinct")


def case_key(case: dict[str, Any]) -> str:
    return json.dumps({k: case[k] for k in sorted(case) if k not in _DROP}, sort_keys=True, default=str)


def _home(home: str | None = None) -> Path:
    return Path(home or os.environ.get("KDEV_HOME") or os.path.expanduser("~/.kdev"))


def path_for(problem: str, home: str | None = None) -> Path:
    return _home(home) / "calib" / f"{os.path.basename(os.path.normpath(problem or 'unknown'))}.json"


def load(problem: str, home: str | None = None) -> dict[str, Any]:
    p = path_for(problem, home)
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("schema") == SCHEMA:
            return d
    except (OSError, ValueError):
        pass
    return {"schema": SCHEMA, "problem": problem, "shapes": {}}


def save(tab: dict[str, Any], problem: str, home: str | None = None) -> Path:
    p = path_for(problem, home)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(tab, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, p)
    return p


def add_observations(tab: dict[str, Any], obs: list[dict[str, Any]]) -> int:
    """obs: {key, shape_id, role (sub|base), local_us, judge_us, run_id, judge_id, node, cluster, md5,
    judge_ovh_us}. Returns the number added."""
    n = 0
    for o in obs:
        if not o.get("local_us") or not o.get("judge_us"):
            continue
        ent = tab["shapes"].setdefault(o["key"], {"shape_id": o.get("shape_id"), "sub": [], "base": []})
        ent["shape_id"] = o.get("shape_id") or ent.get("shape_id")
        lst = ent.setdefault(o["role"], [])
        rec = {k: o.get(k) for k in ("local_us", "judge_us", "run_id", "judge_id", "node", "cluster", "md5",
                                     "judge_ovh_us")}
        rec["ratio"] = o["judge_us"] / o["local_us"]
        rec["t"] = round(time.time(), 1)
        # one observation per (run, judge job): re-ingesting the same calib run is idempotent
        lst[:] = [x for x in lst if not (x.get("run_id") == rec["run_id"] and x.get("judge_id") == rec["judge_id"])]
        lst.append(rec)
        del lst[:-KEEP]
        n += 1
    tab["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return n


def _median(v: list[float]) -> float:
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def stats_for(ent: dict[str, Any], role: str) -> dict[str, Any] | None:
    lst = [x for x in ent.get(role) or [] if x.get("ratio")]
    if not lst:
        return None
    r = [x["ratio"] for x in lst]
    med = _median(r)
    half = (max(r) - min(r)) / 2 if len(r) > 1 else 0.0
    loc = _median([x["local_us"] for x in lst])
    small = loc < SMALL_US
    unc = max(JUDGE_NOISE, half) + (SMALL_ABS_US / loc if small and loc > 0 else 0.0)
    return {"ratio": med, "unc": unc, "n": len(lst), "small": small, "local_us": loc,
            "judges": sorted({str(x.get("judge_id")) for x in lst})}


def estimate(tab: dict[str, Any], case: dict[str, Any], role: str, local_us: float | None) -> dict[str, Any] | None:
    """{judge_est_us, ratio, unc, n, small} for a local time on this case, or None without a calibration."""
    if not local_us or not isinstance(local_us, (int, float)) or math.isnan(local_us):
        return None
    ent = tab.get("shapes", {}).get(case_key(case))
    if not ent:
        return None
    st = stats_for(ent, role) or (stats_for(ent, "sub") if role != "base" else None)
    if st is None:
        return None
    small = st["small"] or local_us < SMALL_US
    unc = st["unc"] if (st["small"] or not small) else st["unc"] + SMALL_ABS_US / local_us
    return {"judge_est_us": local_us * st["ratio"], "ratio": st["ratio"], "unc": unc, "n": st["n"],
            "small": small}


def annotate(rows: list[dict[str, Any]], shapes: list[Any], problem: str | None, base_label: str,
             home: str | None = None) -> dict[str, Any] | None:
    """Adds ``judge_est`` to every per_file entry of ``rows`` that has a calibration (sub ratio for
    submissions, base ratio for the baseline). Returns a summary (or None when the problem has no table)."""
    if not problem:
        return None
    tab = load(problem, home)
    if not tab.get("shapes"):
        return None
    cases = {}
    for sh in shapes:
        sid = getattr(sh, "shape_id", None) or (sh.get("shape_id") if isinstance(sh, dict) else None)
        case = getattr(sh, "case", None) if not isinstance(sh, dict) else sh.get("case")
        if sid is not None and case is not None:
            cases[sid] = case
    n_cal = n_small = 0
    for r in rows:
        case = cases.get(r.get("shape_id"))
        if case is None:
            continue
        hit = False
        for pf in r.get("per_file") or []:
            role = "base" if pf.get("label") == base_label else "sub"
            est = estimate(tab, case, role, pf.get("median_us"))
            if est:
                pf["judge_est"] = est
                hit = True
                n_small += 1 if est["small"] and role == "sub" else 0
        n_cal += 1 if hit else 0
    if not n_cal:
        return None
    return {"table": str(path_for(problem, home)), "shapes_calibrated": n_cal, "shapes": len(rows),
            "small_sub_shapes": n_small, "updated_utc": tab.get("updated_utc")}


def observations_from_calib(crows: list[dict[str, Any]], labels: list[str], shapes: list[Any], judges: dict[str, Any],
                            md5s: dict[str, str], run_id: str) -> list[dict[str, Any]]:
    """calib rows (timing.calib.calib_rows) -> table observations (each label's sub, and the baseline once
    per judge JSON)."""
    by_sid = {getattr(s, "shape_id", None): s for s in shapes}
    out: list[dict[str, Any]] = []
    base_done: set[tuple[str, str]] = set()
    for c in crows:
        sh = by_sid.get(c["shape_id"])
        if sh is None:
            continue
        key = case_key(dict(sh.case))
        for lb in labels:
            f = c["files"][lb]
            jr = judges.get(lb)
            if jr is None or not f.get("judge_us"):
                continue
            js = next((s for s in jr.shapes if s.shape_id == c["shape_id"]), None)
            common = {"key": key, "shape_id": c["shape_id"], "run_id": run_id, "judge_id": jr.judge_id,
                      "node": jr.node, "cluster": jr.cluster, "judge_ovh_us": js.overhead_us if js else None}
            out.append(dict(common, role="sub", local_us=f.get("local_us"), judge_us=f.get("judge_us"),
                            md5=md5s.get(lb)))
            jb = js.base_us if js else None
            if jb and (jr.judge_id, c["shape_id"]) not in base_done:
                base_done.add((jr.judge_id, c["shape_id"]))
                out.append(dict(common, role="base", local_us=c.get("local_base_us"), judge_us=jb, md5=None))
    return out


def table_lines(tab: dict[str, Any], max_rows: int = 30) -> list[str]:
    ents = sorted(tab.get("shapes", {}).values(), key=lambda e: str(e.get("shape_id")))
    lines = ["shape".ljust(30) + f"{'sub J/L':>9}{'+-':>6}{'n':>3}{'base J/L':>10}{'+-':>6}{'n':>3}  flags"]
    for e in ents[:max_rows]:
        s, b = stats_for(e, "sub"), stats_for(e, "base")

        def f(x: dict[str, Any] | None) -> str:
            return (f"{x['ratio']:9.3f}{x['unc'] * 100:5.1f}%{x['n']:3d}" if x else f"{'-':>9}{'':>6}{'':>3}")

        flags = "small" if (s and s["small"]) else ""
        if s and abs(s["ratio"] - 1) > 0.05:
            flags += (" " if flags else "") + "gap>5%"
        lines.append(str(e.get("shape_id"))[:29].ljust(30) + f(s) + " " + f(b) + "  " + flags)
    if len(ents) > max_rows:
        lines.append(f"... {len(ents) - max_rows} more shapes (--json)")
    return lines
