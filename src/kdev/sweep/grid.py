"""Sweep grid + orchestration. Owner: P5.

Flow: expand grid -> one build per point (parallel, outside the GPU lock; same hash-keyed cache the GPU lane
checks) -> prepare-time discovery per built point (the `kdev build` loop: GPU discover job, rebuild the
recorded calls, until complete; ``--no-discover`` skips it) -> drop points failing build / spills / C751x /
still-pending discovery (reason recorded) -> quick check (check job, 1 seed) ->
ab jobs in batches of <= 4 points + base, interleaved -> top K by score with CIs; full table in sweep.csv.
Grid sources: "K=a,b K2=c" literal, or "cublaslt:SHAPE" (algorithm enumeration provided by the adapter, later).

Knobs reach the code as ``-DK=V`` on every extension compile (and as JSON in ``$KDEV_KNOBS`` for modules that
read them at import), so a knob must be a preprocessor macro (``#ifndef TILE / #define TILE 128``) or read
from ``KDEV_KNOBS``.

GPU-lane contract used (P3 Result envelope): ``score.per_file[label]`` (higher = better) per batch, with the
base in every batch, so points are ranked by ``score[label] / score[base]``; a CI is taken from
``score.ci_per_file[label] = [lo, hi]`` or ``score.delta`` (2-file jobs) when present.
"""
from __future__ import annotations

import csv
import io
import itertools
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from kdev.contract import ErrorCode, KdevError, Result
from kdev.models import FileRef, JobSpec

BATCH = 4


@dataclass
class SweepPoint:
    knobs: dict[str, Any]
    label: str = ""
    build_status: str = ""      # ok | build_failed | spills | c751x
    check_status: str = ""
    score: float | None = None
    ci: tuple[float, float] | None = None
    run_ids: list[str] = field(default_factory=list)
    reason: str = ""
    rel: float | None = None    # score / base score of the same batch
    regs_max: int | None = None
    discover_rounds: int = 0    # prepare-time discovery rounds run for this point (kdev build loop)


def _coerce(v: str) -> Any:
    s = v.strip()
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    for t in (int, float):
        try:
            return t(s)
        except ValueError:
            pass
    return s


def parse_grid(text: str) -> dict[str, list[Any]]:
    """'TILE=64,128 STAGES=2,3' -> {'TILE': [64, 128], 'STAGES': [2, 3]} (ints/floats coerced)."""
    if not text or not text.strip():
        raise KdevError(ErrorCode.INVALID_REQUEST, "empty --grid", recovery="use --grid 'K=a,b K2=c,d'")
    if text.strip().startswith("cublaslt:"):
        raise KdevError(ErrorCode.NOT_IMPLEMENTED, "cublaslt grid source is not implemented yet",
                        recovery="enumerate algorithm ids as a literal grid, e.g. --grid 'ALGO=0,1,2,3'")
    grid: dict[str, list[Any]] = {}
    for tok in re.split(r"[\s;]+", text.strip()):
        if not tok:
            continue
        k, sep, vals = tok.partition("=")
        if not sep or not k or not vals:
            raise KdevError(ErrorCode.INVALID_REQUEST, f"bad grid term {tok!r}", recovery="use K=a,b")
        if k in grid:
            raise KdevError(ErrorCode.INVALID_REQUEST, f"knob {k} given twice in --grid")
        grid[k] = [_coerce(v) for v in vals.split(",") if v != ""]
        if not grid[k]:
            raise KdevError(ErrorCode.INVALID_REQUEST, f"knob {k} has no values")
    return grid


def expand(grid: dict[str, list[Any]]) -> list[SweepPoint]:
    keys = list(grid)
    pts = []
    for n, combo in enumerate(itertools.product(*(grid[k] for k in keys)), 1):
        knobs = dict(zip(keys, combo))
        pts.append(SweepPoint(knobs=knobs, label=f"p{n}"))
    return pts


def knob_str(knobs: dict[str, Any]) -> str:
    return " ".join(f"{k}={v}" for k, v in knobs.items())


# ------------------------------------------------------------------------------------------- build stage
def build_points(file: str, points: list[SweepPoint], *, parallel: int, problem: str | None, home: str | None,
                 builder: Callable[..., Any] | None = None) -> dict[str, Any]:
    """Builds every point (thread pool; each build is a subprocess). Fills build_status/reason/regs_max.
    Returns {label: BuildReport}."""
    if builder is None:
        from kdev.build.compile import build_file as builder  # noqa: PLC0415

    def one(p: SweepPoint):
        try:
            return p.label, builder(file, knobs=p.knobs, problem=problem, home=home)
        except KdevError as e:
            return p.label, e

    with ThreadPoolExecutor(max_workers=max(1, parallel)) as ex:
        reps = dict(ex.map(one, points))
    for p in points:
        r = reps[p.label]
        if isinstance(r, KdevError):
            p.build_status, p.reason = "build_failed", f"{r.code.value}: {r.message[:160]}"
            continue
        fails = [e for e in r.errors if e.startswith(("build_failed", "toolchain_mismatch"))]
        ks = [k for e in r.exts for k in e.kernels]
        p.regs_max = max((k.regs or 0 for k in ks), default=None)
        if fails:
            p.build_status, p.reason = "build_failed", fails[0][:200]
        elif any(k.spill_store or k.spill_load for k in ks):
            sp = [k for k in ks if k.spill_store or k.spill_load]
            p.build_status = "spills"
            p.reason = f"{len(sp)} kernel(s) spill ({sp[0].spill_store}/{sp[0].spill_load} B st/ld)"
        elif any("C751" in w for k in ks for w in k.warnings):
            p.build_status, p.reason = "c751x", "ptxas C751x: wgmma serialised"
        else:
            p.build_status = "ok"
    return reps


def discover_points(file: str, points: list[SweepPoint], reports: dict[str, Any], *, ctx: Any,
                    problem: str | None, home: str | None, builder: Callable[..., Any] | None = None) -> list[str]:
    """The prepare-time discovery of ``kdev build`` for every built point (its knobs reach prepare() as -D /
    $KDEV_KNOBS, so each variant can build different extensions): GPU discover job -> rebuild the recorded
    calls, until a pass is complete. A point still pending is dropped (``needs_build`` otherwise wastes its
    check/ab job). Returns warnings; stops discovering when GPU jobs are unavailable."""
    from kdev.build.discover import discover_loop  # noqa: PLC0415

    if builder is None:
        from kdev.build.compile import build_file as builder  # noqa: PLC0415
    warns: list[str] = []
    for p in points:
        def rebuild(path: str, p: SweepPoint = p) -> Any:
            return builder(path, knobs=p.knobs, problem=problem, home=home)

        try:
            rep, rounds, w = discover_loop(file, reports.get(p.label), rebuild, dict(p.knobs), [], ctx)
        except Exception as e:  # noqa: BLE001 - no GPU route (fake/remote ctx): keep the plain builds
            warns.append(f"discovery skipped: {type(e).__name__}: {str(e)[:120]}")
            break
        p.discover_rounds = len(rounds)
        p.run_ids += [r["run_id"] for r in rounds if r.get("run_id")]
        if w:
            warns += [f"{p.label}: {x}" for x in w]
            if any("skipped" in x for x in w):
                break
        if rounds and rounds[-1]["verdict"] != "complete":
            p.build_status, p.reason = "needs_build", (f"prepare-time extensions still pending after "
                                                       f"{len(rounds)} discovery round(s)")
        elif rep is not None and not getattr(rep, "ok", True):
            p.build_status, p.reason = "build_failed", "rebuild after discovery failed"
    return warns


# --------------------------------------------------------------------------------------------- GPU stages
def _submit(ctx: Any, spec: JobSpec) -> Result:
    try:
        r = ctx.submit(spec)
    except NotImplementedError as e:
        raise KdevError(ErrorCode.CAPABILITY_UNAVAILABLE, f"GPU job routing unavailable: {e}") from e
    if isinstance(r, Result) and r.error is not None and r.error.code in ("capability_unavailable",
                                                                           "not_implemented", "gpu_busy_timeout"):
        raise KdevError(ErrorCode(r.error.code), r.error.message)
    return r


def _ref(file: str, p: SweepPoint, role: str = "variant") -> FileRef:
    return FileRef(path=os.path.abspath(file), label=p.label, knobs=dict(p.knobs), role=role)


def check_specs(file: str, points: list[SweepPoint], *, problem: str | None, agent: str, parent: str = ""
                ) -> list[JobSpec]:
    return [JobSpec(kind="check", files=[_ref(file, p)], params={"seeds": 1, "mode": "graph", "repeat": 1},
                    problem=problem, agent=agent, run_name=f"sweep-{p.label}", parent_job=parent or None)
            for p in points]


def ab_specs(file: str, base: str, points: list[SweepPoint], *, problem: str | None, agent: str,
             params: dict[str, Any] | None = None, parent: str = "") -> list[JobSpec]:
    specs = []
    for b in range(0, len(points), BATCH):
        chunk = points[b:b + BATCH]
        files = [FileRef(path=os.path.abspath(base), label="base", role="base")] + [_ref(file, p) for p in chunk]
        specs.append(JobSpec(kind="ab", files=files, params={"base": "none", **(params or {})}, problem=problem,
                             agent=agent, run_name=f"sweep-ab{b // BATCH + 1}", parent_job=parent or None))
    return specs


def _score_of(res: Result, label: str) -> tuple[float | None, tuple[float, float] | None]:
    sc = res.score or {}
    per = sc.get("per_file") or {}
    val = per.get(label)
    ci = None
    cip = (sc.get("ci_per_file") or {}).get(label)
    if isinstance(cip, (list, tuple)) and len(cip) == 2:
        ci = (float(cip[0]), float(cip[1]))
    return (float(val) if isinstance(val, (int, float)) else None), ci


def to_csv(points: list[SweepPoint]) -> str:
    buf = io.StringIO()
    keys = list(dict.fromkeys(k for p in points for k in p.knobs))
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["label", *keys, "build_status", "check_status", "score", "rel_base", "ci_lo", "ci_hi",
                "regs_max", "reason", "run_ids"])
    for p in points:
        w.writerow([p.label, *[p.knobs.get(k, "") for k in keys], p.build_status, p.check_status,
                    "" if p.score is None else f"{p.score:.6g}", "" if p.rel is None else f"{p.rel:.6g}",
                    p.ci[0] if p.ci else "", p.ci[1] if p.ci else "", p.regs_max if p.regs_max is not None else "",
                    p.reason, " ".join(p.run_ids)])
    return buf.getvalue()


def run_sweep(file: str, grid: dict[str, list[Any]], *, base: str | None, top: int, build_parallel: int,
              quick_check: bool, ctx: Any, dry_run: bool = False, builder: Callable[..., Any] | None = None,
              drop: tuple[str, ...] = ("build_failed", "spills", "c751x"), params: dict[str, Any] | None = None,
              discover: bool = True) -> Result:
    """Client-side orchestration: parallel builds here (compile-only, no GPU), then check/ab jobs via
    ``ctx.submit`` (service GPU lane or direct flock mode). Without a GPU route the result stops after the
    build stage (verdict 'built_only'). Returns the sweep Result; ``summary['points']`` holds every point."""
    from kdev.build import _runs  # noqa: PLC0415

    if not os.path.isfile(file):
        raise KdevError(ErrorCode.NOT_FOUND, f"no such file: {file}")
    base = base or file
    points = expand(grid)
    if len(points) > 512:
        raise KdevError(ErrorCode.INVALID_REQUEST, f"grid has {len(points)} points (> 512)",
                        recovery="split the grid or fix knobs you already know")
    agent = getattr(ctx, "agent", "") or ""
    problem = getattr(ctx, "problem", None)
    res = Result(tool="sweep")
    if dry_run:
        specs = ab_specs(file, base, points, problem=problem, agent=agent, params=params)
        res.verdict = "dry_run"
        res.summary = {"points": [asdict(p) for p in points[:64]], "n_points": len(points),
                       "ab_jobs": len(specs)}
        res.lines = [f"{len(points)} points, {len(specs)} ab batch(es) of <= {BATCH} + base"] + \
            [f"  {p.label}: {knob_str(p.knobs)}" for p in points[:15]]
        if len(points) > 15:
            res.lines.append(f"  ... {len(points) - 15} more points")
        return res
    run = _runs.new_run(ctx, "sweep", os.path.splitext(os.path.basename(file))[0])
    home = str(_runs.home(ctx))
    reps = build_points(file, points, parallel=build_parallel, problem=problem, home=home, builder=builder)
    live = [p for p in points if p.build_status == "ok" or p.build_status not in drop]
    gpu_note = ""
    disc_warn: list[str] = []
    if discover and file.endswith(".py"):
        disc_warn = discover_points(file, [p for p in live if p.build_status == "ok"], reps, ctx=ctx,
                                    problem=problem, home=home, builder=builder)
        live = [p for p in live if p.build_status not in ("needs_build", "build_failed")]
    try:
        if quick_check and live:
            for spec, p in zip(check_specs(file, live, problem=problem, agent=agent), live):
                r = _submit(ctx, spec)
                p.check_status = "pass" if r.status == "ok" else (r.verdict or r.status)
                if r.run_id:
                    p.run_ids.append(r.run_id)
                if r.status != "ok":
                    p.reason = p.reason or f"check {p.check_status}"
            live = [p for p in live if p.check_status == "pass"]
        for spec in ab_specs(file, base, live, problem=problem, agent=agent, params=params):
            r = _submit(ctx, spec)
            base_score, _ = _score_of(r, "base")
            for f in spec.files[1:]:
                p = next(x for x in live if x.label == f.label)
                if r.run_id:
                    p.run_ids.append(r.run_id)
                p.score, p.ci = _score_of(r, p.label)
                if p.score is not None and base_score:
                    p.rel = p.score / base_score
                if r.status != "ok" and not p.reason:
                    p.reason = f"ab {r.verdict or r.status}"
    except KdevError as e:
        gpu_note = f"{e.code.value}: {e.message[:160]}"
    ranked = sorted([p for p in live if p.rel is not None or p.score is not None],
                    key=lambda p: -(p.rel if p.rel is not None else p.score or 0.0))
    csv_p = _runs.write_text(run, "sweep.csv", to_csv(points))
    js_p = _runs.write_json(run, "sweep.json", {"schema": "kdev.sweep/1", "file": file, "base": base, "grid": grid,
                                                "points": [asdict(p) for p in points]})
    res.files.update({"csv": str(csv_p), "sweep": str(js_p)})
    n_fail = {s: sum(1 for p in points if p.build_status == s)
              for s in ("build_failed", "spills", "c751x", "needs_build")}
    if not n_fail["needs_build"]:
        del n_fail["needs_build"]
    res.verdict = "built_only" if gpu_note else ("ranked" if ranked else "no_results")
    res.status = "ok" if any(p.build_status == "ok" for p in points) else "fail"
    res.summary = {"n_points": len(points), "built_ok": sum(p.build_status == "ok" for p in points),
                   "dropped": n_fail, "timed": len(ranked),
                   "top": [{"label": p.label, "knobs": p.knobs, "rel": p.rel, "score": p.score, "ci": p.ci}
                           for p in ranked[:top]]}
    dropped = ", ".join(f"{k} {v}" for k, v in n_fail.items() if v) or "none"
    lines = [f"{len(points)} points: built {res.summary['built_ok']}, dropped {dropped}"
             + (f"; timed {len(ranked)}" if ranked else "")]
    if ranked:
        lines.append(f"  top {min(top, len(ranked))} by score vs base:")
        for p in ranked[:top]:
            ci = f" [{p.ci[0]:.3f}, {p.ci[1]:.3f}]" if p.ci else ""
            rel = f"x{p.rel:.3f}" if p.rel is not None else "-"
            lines.append(f"    {p.label:5s} {rel:8s} score {p.score if p.score is not None else '-'}{ci}  "
                         f"{knob_str(p.knobs)}")
    else:
        for p in points[:12]:
            lines.append(f"  {p.label:5s} {p.build_status:12s} regs {p.regs_max or '-':>4}  {knob_str(p.knobs)}"
                         + (f"  ({p.reason[:60]})" if p.reason else ""))
        if len(points) > 12:
            lines.append(f"  ... {len(points) - 12} more in sweep.csv")
    bad = [p for p in points if p.build_status not in ("ok", "")][:3]
    for p in bad:
        lines.append(f"  dropped {p.label} ({knob_str(p.knobs)}): {p.build_status}: {p.reason[:90]}")
    if gpu_note:
        res.warnings.append(f"GPU stage skipped: {gpu_note}")
    res.warnings += disc_warn
    n_disc = sum(1 for p in points if p.discover_rounds)
    if n_disc:
        res.summary["discovered"] = n_disc
    res.lines = lines
    res.next.append(f"kdev show {run.run_id} --section csv")
    return _runs.finish(run, res)
