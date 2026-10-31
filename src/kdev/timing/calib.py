"""Judge-vs-local calibration (port of okc/tools/calib.py + calib_seeds.py). Owner: P3.

Parsing and comparison are pure (fixtures: tests/fixtures/judge/judge_moe.json, cal_moe.json); the timing
part reuses the engine session (same graph capture, read flush, interleave, overhead and seeds as `time`).
Rows go to the ledger ``calib`` table via the Result (``shapes[].calib`` and ``summary.calib_rows``).
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any

from kdev.contract import ErrorCode, KdevError, Result
from kdev.models import JobSpec

DROP_CASE_KEYS = ("load", "repeats", "weight", "router_load", "router_distinct")
ID_KEYS = ("layer", "batch", "bs", "m", "q_len", "ctx", "turn", "prefix")
DEFAULT_BAND = (0.95, 1.05)


@dataclass
class JudgeShape:
    shape_id: str
    case: dict[str, Any]
    sub_us: float | None
    base_us: float | None
    speedup: float | None
    seed: int | None = None
    weight: float = 1.0
    load: str = ""
    index: int = 0
    runs: int | None = None
    overhead_us: float | None = None       # judge's measured 2*one - two for this shape
    record: dict[str, Any] = field(default_factory=dict)


@dataclass
class JudgeRun:
    judge_id: str
    cluster: str | None
    node: str | None
    speedup: float | None
    status: str
    shapes: list[JudgeShape]
    raw: dict[str, Any]
    speedup_ci: list[float] | None = None


def shape_id_for(case: dict[str, Any], index: int | None = None) -> str:
    """Readable stable id: '<load>/<layer>/<batch>/<q_len>/...' from the identifying case keys."""
    parts = [str(case.get("load", "")) or "case"]
    for k in ID_KEYS:
        if k in case:
            v = case[k]
            parts.append(str(v) if k in ("layer",) else f"{k}{v}")
    return "/".join(parts)


def parse_judge_json(d: dict[str, Any]) -> JudgeRun:
    """`okc status --json` -> JudgeRun (records with event == 'benchmark'; sub_us = baseline_ns/1000/speedup)."""
    r = d.get("result") or d
    recs = [x for x in (r.get("records") or []) if x.get("event") == "benchmark"]
    shapes: list[JudgeShape] = []
    seen: dict[str, int] = {}
    for i, rec in enumerate(recs):
        full = dict(rec.get("case") or {})
        case = {k: v for k, v in full.items() if k not in DROP_CASE_KEYS}
        sid = shape_id_for(full)
        if sid in seen:
            seen[sid] += 1
            sid = f"{sid}#{seen[sid]}"
        else:
            seen[sid] = 0
        sp = rec.get("speedup", rec.get("paired_ratio"))
        bns = rec.get("baseline_ns")
        base_us = bns / 1000.0 if bns is not None else None
        sub_us = base_us / sp if base_us is not None and sp else None
        ro = rec.get("replay_overhead") or {}
        shapes.append(JudgeShape(shape_id=sid, case=case, sub_us=sub_us, base_us=base_us, speedup=sp,
                                 seed=case.get("seed"), weight=float(full.get("weight", 1.0) or 1.0),
                                 load=str(full.get("load", "")), index=i, runs=rec.get("runs"),
                                 overhead_us=(ro["overhead_ns"] / 1000.0) if "overhead_ns" in ro else None,
                                 record=rec))
    return JudgeRun(judge_id=str(d.get("id") or d.get("job_id") or ""), cluster=d.get("cluster"),
                    node=(r.get("machine") or {}).get("node"), speedup=r.get("speedup"),
                    status=str(d.get("status") or r.get("status") or ""), shapes=shapes, raw=d,
                    speedup_ci=r.get("speedup_ci"))


def judge_score(judge: JudgeRun) -> float:
    """Recompute the judge's problem speedup from its records (load-weighted geomean of base/sub)."""
    loads: dict[str, list[float]] = {}
    for s in judge.shapes:
        if s.base_us is None or s.sub_us is None:
            continue
        a = loads.setdefault(s.load, [0.0, 0.0])
        a[0] += s.weight * s.base_us
        a[1] += s.weight * s.sub_us
    vals = [b / a for b, a in loads.values() if a > 0]
    return math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else float("nan")


def compare(local_speedups: dict[str, float], judge: JudgeRun, band: tuple[float, float] = DEFAULT_BAND
            ) -> list[dict[str, Any]]:
    """Per shape: local, judge, local/judge, flag (ok | out_of_band)."""
    rows = []
    for s in judge.shapes:
        loc = local_speedups.get(s.shape_id)
        if loc is None:
            continue
        lj = loc / s.speedup if s.speedup else None
        flag = "ok" if lj is not None and band[0] <= lj <= band[1] else ("out_of_band" if lj is not None else "-")
        rows.append({"shape_id": s.shape_id, "local": loc, "judge": s.speedup, "local_over_judge": lj,
                     "flag": flag})
    return rows


def ab_disagree(local_ab: float, judge_ab: float, min_effect: float = 0.02, max_gap: float = 0.03) -> bool:
    """calib.py rule: direction flip with effect > 2 %, or |local/judge - 1| > 3 %."""
    flip = (local_ab - 1) * (judge_ab - 1) < 0 and max(abs(local_ab - 1), abs(judge_ab - 1)) > min_effect
    return flip or abs(local_ab / judge_ab - 1) > max_gap


def _match(judge: JudgeRun, ref_shape: JudgeShape) -> JudgeShape | None:
    """Same shape in another judge run: by shape_id, else by index when the case matches (minus seed)."""
    for s in judge.shapes:
        if s.shape_id == ref_shape.shape_id:
            return s
    if ref_shape.index < len(judge.shapes):
        s = judge.shapes[ref_shape.index]
        strip = lambda c: {k: v for k, v in c.items() if k != "seed"}  # noqa: E731
        if strip(s.case) == strip(ref_shape.case):
            return s
    return None


def calib_rows(rows: list[dict[str, Any]], judges: dict[str, JudgeRun | None], labels: list[str],
               base_label: str, ref_judge: JudgeRun, band: tuple[float, float] = DEFAULT_BAND
               ) -> list[dict[str, Any]]:
    """Per shape and label: judge sub/base us, local sub/base us, speedups, local/judge, flag; for 2+
    labels also local vs judge A/B (vs the first label) with the DISAGREE rule."""
    by_id = {r["shape_id"]: r for r in rows}
    out = []
    for js in ref_judge.shapes:
        r = by_id.get(js.shape_id)
        if r is None:
            continue
        us = {p["label"]: p["median_us"] for p in r["per_file"]}
        sp = {rt["num"]: rt["speedup"] for rt in (r.get("ratios") or []) if rt["den"] == base_label}
        row: dict[str, Any] = {"shape_id": js.shape_id, "load": js.load, "weight": js.weight,
                               "local_base_us": us.get(base_label), "judge_base_us": js.base_us, "files": {}}
        for lb in labels:
            jr = judges.get(lb)
            jm = _match(jr, js) if jr else None
            loc = sp.get(lb)
            lj = loc / jm.speedup if loc and jm and jm.speedup else None
            flag = None if lj is None else ("ok" if band[0] <= lj <= band[1] else "out_of_band")
            row["files"][lb] = {"local_us": us.get(lb), "local_sp": loc,
                                "judge_us": jm.sub_us if jm else None, "judge_sp": jm.speedup if jm else None,
                                "local_over_judge": lj, "flag": flag,
                                "judge_id": jr.judge_id if jr else None}
        if len(labels) >= 2:
            a = row["files"][labels[0]]
            for lb in labels[1:]:
                f = row["files"][lb]
                if a["local_sp"] and f["local_sp"]:
                    la = f["local_sp"] / a["local_sp"]
                    f["local_ab"] = la
                    if a["judge_sp"] and f["judge_sp"]:
                        ja = f["judge_sp"] / a["judge_sp"]
                        f["judge_ab"] = ja
                        f["ab_flag"] = "DISAGREE" if ab_disagree(la, ja) else "ok"
                        if f["ab_flag"] == "DISAGREE":
                            f["flag"] = "DISAGREE"
        out.append(row)
    return out


def render_calib_md(crows: list[dict[str, Any]], labels: list[str], header: str = "") -> str:
    lines = [header] if header else []
    for lb in labels:
        lines += [f"\n### {lb}",
                  "| shape | judge sub us | local sub us | judge base us | local base us | judge x | local x |"
                  " local/judge |", "|---|---|---|---|---|---|---|---|"]
        for r in crows:
            f = r["files"][lb]

            def g(x: Any, p: int = 1) -> str:
                return "-" if x is None else f"{x:.{p}f}"

            star = " **" if f.get("flag") in ("out_of_band", "DISAGREE") else ""
            lines.append(f"| {r['shape_id']} | {g(f['judge_us'])} | {g(f['local_us'])} | {g(r['judge_base_us'])} | "
                         f"{g(r['local_base_us'])} | {g(f['judge_sp'], 3)} | {g(f['local_sp'], 3)} | "
                         f"{g(f['local_over_judge'], 3)}{star} |")
    return "\n".join(lines) + "\n"


def _parse_files(spec: JobSpec) -> tuple[dict[str, JudgeRun | None], JudgeRun]:
    judges: dict[str, JudgeRun | None] = {}
    for i, f in enumerate(spec.files):
        lb = f.label or chr(ord("A") + i)
        if f.judge_json and f.judge_json != "-":
            with open(f.judge_json, encoding="utf-8") as fh:
                judges[lb] = parse_judge_json(json.load(fh))
        else:
            judges[lb] = None
    ref = next((j for j in judges.values() if j is not None), None)
    if ref is None:
        raise KdevError(ErrorCode.INVALID_REQUEST, "calib needs at least one judge JSON for the shape list",
                        recovery="kdev calib FILE.py:JUDGE.json (okc status <job> --json)")
    return judges, ref


def run_calib(spec: JobSpec, run_dir: Any) -> Result:
    """Worker entry for kind='calib' (files carry judge_json)."""
    import time

    from kdev.adapters.base import Shape
    from kdev.timing import engine as E
    from kdev.timing._io import run_id as _rid
    from kdev.timing._io import write_json, write_text

    if not spec.files:
        raise KdevError(ErrorCode.INVALID_REQUEST, "no files", recovery="kdev calib FILE.py:JUDGE.json")
    judges, ref = _parse_files(spec)
    params = dict(spec.params or {})
    extra = dict(params.get("extra") or {})
    # Shapes = the judge's timed cases (with the judge's seeds); the adapter only supplies the GPU side.
    extra["cases"] = [{"shape_id": s.shape_id, "case": s.case, "weight": s.weight, "load": s.load,
                       "seed": s.seed} for s in ref.shapes]
    extra.setdefault("adapter", extra.get("adapter", "direct"))
    params["extra"] = extra
    params.setdefault("reps", 12)
    params["base"] = params.get("base") if params.get("base") not in (None, "none") else "baseline"
    spec2 = JobSpec.from_dict(dict(spec.to_dict(), params=params))
    ad, _, _ = E.resolve_adapter(spec2)
    files = E._files_for(spec2, "time", lambda: ad.baseline_path())
    labels = [f.label for f in files if f.label != E.BASE_LABEL]
    ss = E.open_session(spec2, run_dir, files)
    # always the judge's shapes, in judge order (the adapter's own scored list is not used here)
    ss.shapes = [Shape(c["shape_id"], dict(c["case"]), weight=c["weight"], load=c["load"], seed=c["seed"])
                 for c in extra["cases"] if not ss.params.shapes or ss.params.shapes in c["shape_id"]
                 or ss.params.shapes in str(c["case"])]
    if hasattr(ss.adapter, "set_shapes"):
        ss.adapter.set_shapes(ss.shapes)
    try:
        E.time_shapes(ss)
    finally:
        E.close_session(ss)
    rows, score = E.summarize(ss.samples, ss.shapes, ss.params, E.BASE_LABEL, E._score_fn(ss),
                              md5s={f.label: f.md5 for f in files}, checks=ss.checks)
    crows = calib_rows(rows, judges, labels, E.BASE_LABEL, ref)
    by_id = {c["shape_id"]: c for c in crows}
    for r in rows:
        if r["shape_id"] in by_id:
            r["calib"] = by_id[r["shape_id"]]
    fid = E.fidelity(ss)
    ljs = {lb: [c["files"][lb]["local_over_judge"] for c in crows if c["files"][lb]["local_over_judge"]]
           for lb in labels}
    flagged = {lb: [c["shape_id"] for c in crows if c["files"][lb].get("flag") in ("out_of_band", "DISAGREE")]
               for lb in labels}
    disagree = [c["shape_id"] for c in crows for lb in labels[1:] if c["files"][lb].get("ab_flag") == "DISAGREE"]
    from kdev.timing.stats import median

    per_label = {}
    for lb in labels:
        v = ljs[lb]
        jr = judges.get(lb)
        per_label[lb] = {"median_local_over_judge": median(v) if v else None,
                         "range": [min(v), max(v)] if v else None, "flagged": flagged[lb],
                         "judge_id": jr.judge_id if jr else None, "judge_node": jr.node if jr else None,
                         "judge_speedup": jr.speedup if jr else None,
                         "local_score": (score or {}).get("per_file", {}).get(lb),
                         "local_score_ci": (score or {}).get("ci", {}).get(lb)}
    verdict = "DISAGREE" if disagree else ("out_of_band" if any(flagged.values()) else "calibrated")
    hdr = (f"# kdev calib {_rid(run_dir)}  ({time.strftime('%Y-%m-%d %H:%M')})\n\n" +
           E.fidelity_line(fid) + "\n")
    md = render_calib_md(crows, labels, hdr)
    mdp = write_text(run_dir, "calib.md", md)
    cjp = write_json(run_dir, "calib.json", {"rows": crows, "per_label": per_label,
                                             "fidelity": fid.__dict__, "summary_score": score})
    write_json(run_dir, "shapes.json", rows)
    rd = str(getattr(run_dir, "path", run_dir))
    res = Result(tool="calib", status="ok", verdict=verdict, run_id=_rid(run_dir), job_id=spec.job_id or None,
                 shapes=rows, score=score, fidelity=fid,
                 files={"calib": str(mdp), "calib_json": str(cjp), "raw": os.path.join(rd, "raw.jsonl"),
                        "log": os.path.join(rd, "log.txt")})
    res.summary = {"labels": labels, "per_label": per_label, "disagree": disagree, "n_shapes": len(crows),
                   "reps": ss.params.reps, "adapter": ss.adapter_source,
                   "calib_rows": [{"shape_id": c["shape_id"], "label": lb,
                                   "local_speedup": c["files"][lb]["local_sp"],
                                   "judge_speedup": c["files"][lb]["judge_sp"],
                                   "local_over_judge": c["files"][lb]["local_over_judge"],
                                   "flag": c["files"][lb]["flag"], "judge_id": c["files"][lb]["judge_id"]}
                                  for c in crows for lb in labels]}
    if spec.problem or ss.problem_dir:
        res.summary["problem"] = spec.problem or os.path.basename(os.path.normpath(ss.problem_dir))
    lines = [f"{verdict.upper()}  calib  " + "  ".join(
        f"{lb}: local/judge median {pl['median_local_over_judge']:.3f} range "
        f"[{pl['range'][0]:.3f}, {pl['range'][1]:.3f}] flagged {len(pl['flagged'])}"
        for lb, pl in per_label.items() if pl["median_local_over_judge"])]
    lines.append("shape".ljust(34) + f"{'base us L/J':>16}" + "".join(f"{lb + ' us L/J':>18}{'L/J x':>8}" for lb in labels))
    for c in crows[:15]:
        s = c["shape_id"][:33].ljust(34) + f"{_f(c['local_base_us'])}/{_f(c['judge_base_us'])}".rjust(16)
        for lb in labels:
            f = c["files"][lb]
            s += f"{_f(f['local_us'])}/{_f(f['judge_us'])}".rjust(18)
            s += (f"{f['local_over_judge']:.3f}" if f["local_over_judge"] else "-").rjust(8)
            s += "*" if f.get("flag") in ("out_of_band", "DISAGREE") else " "
        lines.append(s)
    if len(crows) > 15:
        lines.append(f"... {len(crows) - 15} more rows in calib.md")
    # the fidelity line is added once by kdev.output.render
    res.lines = lines
    if ss.error is not None:
        res.status, res.error = "error", ss.error
    if any(v is False for v in ss.checks.values()):
        res.warnings.append("correctness check failed on timed inputs: " + str(ss.check_msgs[:3]))
    res.warnings += ss.warnings
    return res


def _f(x: float | None) -> str:
    if x is None:
        return "-"
    return f"{x:.1f}"
