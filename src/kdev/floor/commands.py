"""CLI: floor, floor calibrate. Owner: P6.

``kdev floor`` (static, no GPU): roofline (bytes / flops / MUFU ops vs the GPU profile's peak and measured
constants) + latency skeleton (a plan of dependent trips, kernel boundaries, merges, ...) per shape, the floor
= max(roofline, skeleton), and - when the current per-shape us are given - the gap and the OKC score
sensitivity (% of score per us saved, % gain if the shape hit its floor).

Plan TOML (``--plan``): either one shape at top level, or one table per shape:

    [shapes."d1/sliding"]
    bytes = 5.2e6          # DRAM bytes moved
    flops = 1.1e8          # tensor flops (dtype below)
    dtype = "fp8"
    mufu = 2.0e5           # ex2/rcp/rsqrt ops
    us = 3.46              # current time (judge units) -> gap and sensitivity
    weight = 46            # optional: judge weight / load (else from the problem adapter)
    load = "d1"
    dram_trips = 3         # skeleton terms (kdev.floor.skeleton.TERM_TO_CONSTANT), stream_bytes, x_us = ...
    merges = 1
"""
from __future__ import annotations

import os
import subprocess
import sys
from argparse import Namespace
from typing import Any

from kdev.contract import Effects, ErrorCode, KdevError, Result
from kdev.registry import Arg, Context, Registry

P = "P6"

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib


def _profile(args: Namespace, ctx: Context) -> Any:
    from kdev.floor.gpus import load_profile
    name = getattr(args, "gpu", None) or os.environ.get("KDEV_GPU") or "h200"
    return load_profile(name, home=ctx.home)


def _parse_terms(s: str | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for part in (s or "").replace(",", " ").split():
        k, eq, v = part.partition("=")
        if not eq:
            raise KdevError(ErrorCode.INVALID_REQUEST, f"bad --terms item {part!r} (want K=V)",
                            recovery="e.g. --terms 'dram_trips=3 merges=1 kernel_boundaries=0'")
        out[k] = float(v)
    return out


def _adapter_shapes(problem: str | None) -> dict[str, Any]:
    if not problem:
        return {}
    try:
        from kdev.adapters.registry import get_adapter
        return {s.shape_id: s for s in get_adapter(problem=problem).shapes()}
    except Exception:  # noqa: BLE001 - weights are optional
        return {}


def floor_rows(profile: Any, entries: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], float | None]:
    """Pure core of `kdev floor`: per entry roofline + skeleton + floor (+ sensitivity when ``us`` given)."""
    from kdev.floor.roofline import geomean_sensitivity, roofline
    from kdev.floor.skeleton import skeleton
    rows = []
    for e in entries:
        r: dict[str, Any] = {"shape_id": e.get("shape_id", "-"), "load": e.get("load", ""),
                             "weight": float(e.get("weight", 1.0)), "us": e.get("us")}
        roof = None
        if e.get("bytes") or e.get("flops") or e.get("mufu"):
            roof = roofline(profile, float(e.get("bytes", 0)), float(e.get("flops", 0)), str(e.get("dtype", "bf16")),
                            float(e.get("mufu", 0)))
            r.update(mem_us_peak=roof.mem_us_peak, mem_us_achievable=roof.mem_us_achievable,
                     compute_us=roof.compute_us, mufu_us=roof.mufu_us, roofline_us=roof.floor_us,
                     roofline_us_achievable=roof.floor_us_achievable, bound=roof.bound)
        sk = skeleton(profile, e)
        has_terms = bool(set(sk.terms) - {"launch"})
        if has_terms or roof is None:
            r.update(skeleton_us=sk.total_us, skeleton_terms=sk.terms, dominant=sk.dominant,
                     missing_constants=sk.missing_constants)
        cands = [x for x in (r.get("roofline_us"), r.get("skeleton_us")) if x is not None]
        r["floor_us"] = max(cands) if cands else None
        cands_a = [x for x in (r.get("roofline_us_achievable"), r.get("skeleton_us")) if x is not None]
        r["floor_us_achievable"] = max(cands_a) if cands_a else None
        if r["floor_us"] is not None and r.get("skeleton_us") is not None and r.get("roofline_us") is not None:
            r["binding"] = "skeleton" if r["skeleton_us"] >= r["roofline_us"] else "roofline"
        if r["us"] is not None and r["floor_us"] is not None:
            r["gap_us"] = float(r["us"]) - r["floor_us"]
            r["pct_of_floor"] = 100.0 * r["floor_us"] / float(r["us"]) if float(r["us"]) > 0 else None
        rows.append(r)
    total = None
    timed = [r for r in rows if r["us"] is not None]
    if timed:
        sens, total = geomean_sensitivity([{"shape_id": r["shape_id"], "load": r["load"], "weight": r["weight"],
                                            "us": float(r["us"]), "floor_us": r.get("floor_us_achievable")}
                                           for r in timed])
        for r, s in zip(timed, sens):
            r["pct_per_us"] = s.pct_per_us
            r["gain_pct_at_floor"] = s.gain_pct_at_floor
    return rows, total


def _entries(args: Namespace, ctx: Context) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if getattr(args, "plan", None):
        try:
            with open(args.plan, "rb") as f:
                d = tomllib.load(f)
        except OSError as e:
            raise KdevError(ErrorCode.NOT_FOUND, f"plan {args.plan}: {e}") from e
        except tomllib.TOMLDecodeError as e:
            raise KdevError(ErrorCode.INVALID_REQUEST, f"plan {args.plan}: {e}") from e
        if isinstance(d.get("shapes"), dict):
            defaults = {k: v for k, v in d.items() if k != "shapes" and not isinstance(v, dict)}
            for sid, e in d["shapes"].items():
                entries.append({**defaults, **e, "shape_id": sid})
        else:
            entries.append({**d, "shape_id": d.get("shape_id", "plan")})
    inline = _parse_terms(getattr(args, "terms", None))
    if args.bytes is not None or args.flops is not None or getattr(args, "mufu", None) or inline:
        e = {"shape_id": args.shape or "cli", "bytes": args.bytes or 0, "flops": args.flops or 0,
             "dtype": args.dtype or "bf16", "mufu": getattr(args, "mufu", None) or 0, **inline}
        if getattr(args, "us", None) is not None:
            e["us"] = args.us
        entries.append(e)
    if args.shape and getattr(args, "plan", None):
        entries = [e for e in entries if e["shape_id"] == args.shape or args.shape in e["shape_id"]] or entries
    if not entries:
        raise KdevError(ErrorCode.INVALID_REQUEST, "nothing to compute",
                        recovery="give --bytes/--flops/--mufu, --terms 'dram_trips=3 ...' or --plan PLAN.toml")
    shapes = _adapter_shapes(ctx.problem or getattr(args, "problem", None))
    for e in entries:
        s = shapes.get(e["shape_id"])
        if s is not None:
            e.setdefault("weight", s.weight)
            e.setdefault("load", s.load)
    return entries


def cmd_floor(args: Namespace, ctx: Context) -> Result:
    from kdev.floor._text import table, us
    prof = _profile(args, ctx)
    rows, total = floor_rows(prof, _entries(args, ctx))
    tab = []
    for r in rows:
        tab.append([r["shape_id"], us(r.get("mem_us_peak")), us(r.get("mem_us_achievable")), us(r.get("compute_us")),
                    us(r.get("mufu_us")), us(r.get("skeleton_us")), us(r.get("floor_us")), us(r.get("floor_us_achievable")),
                    us(r["us"]) if r["us"] is not None else "-", us(r.get("gap_us")),
                    f"{r['pct_per_us']:.2f}" if r.get("pct_per_us") is not None else "-",
                    f"{r['gain_pct_at_floor']:+.2f}" if r.get("gain_pct_at_floor") is not None else "-"])
    lines = [f"gpu {prof.name}: dram {prof.peak.get('dram_tbps')} TB/s peak (judge), "
             f"{prof.const('stream_tbps')} TB/s measured; units: us (judge = replay - 4.0)"]
    lines += table(["shape", "mem_pk", "mem_ach", "compute", "mufu", "skeleton", "floor", "floor_ach", "now", "gap",
                    "%/us", "gain%"], tab)
    for r in rows[:3]:
        if r.get("skeleton_terms"):
            terms = " + ".join(f"{k} {v:.2f}" for k, v in sorted(r["skeleton_terms"].items(), key=lambda x: -x[1]))
            lines.append(f"{r['shape_id']} skeleton: {terms}" + (f"  MISSING {', '.join(r['missing_constants'])}"
                                                                 if r.get("missing_constants") else ""))
    if total is not None:
        lines.append(f"all shapes at floor_ach: score {total:+.2f}%")
    worst = max((r for r in rows if r.get("gap_us") is not None), key=lambda r: r.get("pct_per_us", 0) * r["gap_us"],
                default=None)
    verdict = "ok"
    warnings = []
    for r in rows:
        if r.get("us") is not None and r.get("floor_us") is not None and float(r["us"]) < r["floor_us"] * 0.98:
            warnings.append(f"{r['shape_id']}: measured {r['us']} us is below the floor {r['floor_us']:.2f} us "
                            "(suspect: check the plan, or a skipped computation)")
            verdict = "suspect"
    nxt = [f"kdev timeline FILE --shape {worst['shape_id']}  (largest weighted gap)"] if worst else []
    return Result(tool="floor", verdict=verdict, summary={"gpu": prof.name, "rows": rows, "gain_pct_all_at_floor": total},
                  lines=lines, warnings=warnings, next=nxt)


def cmd_floor_calibrate(args: Namespace, ctx: Context) -> Result:
    from kdev.config import no_gpu
    from kdev.floor import microbench
    from kdev.models import JobSpec
    from kdev.util import new_id, utc_now
    warnings: list[str] = []
    benches = [b for b in (getattr(args, "benches", None) or "").replace(",", " ").split() if b]
    unknown = [b for b in benches if b not in microbench.BENCHES]
    if unknown:
        raise KdevError(ErrorCode.INVALID_REQUEST, f"unknown benches: {', '.join(unknown)}",
                        recovery=f"choose from: {', '.join(microbench.BENCHES)}")
    if not no_gpu() and not getattr(args, "no_prebuild", False):
        py = getattr(getattr(ctx, "config", None), "python", "") or sys.executable
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        try:
            cp = subprocess.run(microbench.build_argv(py), capture_output=True, text=True, timeout=1800, env=env,
                                check=False)
            if cp.returncode != 0:
                warnings.append("microbench prebuild failed (compiles inside the job): "
                                + (cp.stderr or cp.stdout).strip().splitlines()[-1][:160] if (cp.stderr or cp.stdout)
                                else "microbench prebuild failed")
        except (OSError, subprocess.TimeoutExpired) as e:
            warnings.append(f"microbench prebuild failed: {e}")
    spec = JobSpec(kind="floor_calibrate", files=[], lane="gpu", job_id=new_id("j"),
                   params={"reps": args.reps, "flush": "read",
                           "extra": {"gpu": args.gpu, "write": bool(args.write), "reps": args.reps,
                                     "benches": benches or None}},
                   problem=ctx.problem, agent=ctx.agent or os.environ.get("KDEV_AGENT", ""),
                   priority=int(getattr(args, "priority", 0) or 0), timeout_s=int(getattr(args, "timeout", 1800) or 1800),
                   run_name=getattr(args, "run_name", "") or "calibrate", submitted_utc=utc_now(),
                   requires_prebuilt=False)
    res = ctx.submit(spec, detach=bool(getattr(args, "detach", False)), direct=bool(getattr(args, "direct", False)),
                     timeout_s=getattr(args, "timeout", None))
    res.warnings = warnings + list(res.warnings)
    return res


def register(reg: Registry) -> None:
    reg.command("floor", "roofline + latency-skeleton floor (judge-style and achievable)",
                cmd_floor,
                args=[Arg(("--shape",), "shape id (with --problem / --plan)", default=None),
                      Arg(("--bytes",), "DRAM bytes moved", type="float", default=None),
                      Arg(("--flops",), "tensor flops", type="float", default=None),
                      Arg(("--mufu",), "MUFU (SFU) ops: ex2/rcp/rsqrt/...", type="float", default=None),
                      Arg(("--dtype",), "bf16|fp16|fp8|fp4|int8|tf32|fp32", default="bf16"),
                      Arg(("--terms",), "skeleton terms 'dram_trips=3 merges=1 kernel_boundaries=0 ...'", default=None),
                      Arg(("--us",), "current time of the shape (judge units) -> gap and %%/us", type="float",
                          default=None),
                      Arg(("--plan",), "skeleton plan TOML (one shape or [shapes.ID] tables)", type="path",
                          default=None),
                      Arg(("--gpu",), "GPU profile name (default $KDEV_GPU or h200)", default=None)],
                effects=Effects(gpu="none"), package=P)
    reg.command("floor calibrate", "run the microbench suite and fill the GPU profile's measured constants",
                cmd_floor_calibrate,
                args=[Arg(("--gpu",), "profile name (default: detected from the device name)", default=None),
                      Arg(("--write",), "write $KDEV_HOME/gpus/<gpu>.toml", action="store_true"),
                      Arg(("--reps",), "interleaved replays per variant", type="int", default=30),
                      Arg(("--benches",), "subset, comma separated (default: all)", default=None),
                      Arg(("--no-prebuild",), "skip the compile-only prebuild of the microbench extension",
                          action="store_true")],
                effects=Effects(gpu="queued", writes=("run_dir", "files")), gpu=True, long_running=True,
                package=P)
