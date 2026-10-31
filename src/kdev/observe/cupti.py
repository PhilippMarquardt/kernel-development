"""CUPTI activity via torch.profiler: kernel names, durations, grid/block (works without counter
permission). Owner: P6. Default `profile` backend when ncu is blocked; ``--nopdl`` disables PDL attributes
(env knob provided by the adapter) so per-kernel times are separable.

Backends for ``run_profile`` (``params.extra['backend']``): ``auto`` = ncu when ``ncu.available()`` else
CUPTI; ``ncu`` / ``nsys`` re-run this module as a target process (``python -m kdev.observe.cupti --target
JOB.json``) under the tool; an ncu run that reports ERR_NVGPUCTRPERM falls back to CUPTI and says so.
CUPTI timing: graph replay after the read flush (judge recipe) unless ``params.graph`` is false; the flush's
own kernels are profiled once alone and removed from the table.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from kdev.contract import ErrorCode, KdevError, Result
from kdev.models import JobSpec

NOPDL_ENV = ("KDEV_NOPDL", "OKC_NOPDL")


@dataclass
class KernelRecord:
    name: str
    count: int
    total_us: float
    mean_us: float
    grid: list[int] = field(default_factory=list)
    block: list[int] = field(default_factory=list)


def _as_dims(v: Any) -> list[int]:
    if isinstance(v, (list, tuple)):
        try:
            return [int(x) for x in v]
        except (TypeError, ValueError):
            return []
    return []


def _iter_kernel_events(ev: Any) -> list[tuple[str, float, list[int], list[int], int]]:
    """Normalise profiler inputs to (name, dur_us, grid, block, count)."""
    out = []
    if isinstance(ev, dict) and "traceEvents" in ev:
        ev = ev["traceEvents"]
    for e in ev or []:
        if isinstance(e, dict):
            if e.get("ph") not in (None, "X"):
                continue
            cat = str(e.get("cat", "kernel")).lower()
            if cat != "kernel":
                continue
            args = e.get("args") or {}
            dur = e.get("dur_us", e.get("dur", 0.0))
            out.append((str(e.get("name", "")), float(dur or 0.0), _as_dims(e.get("grid", args.get("grid"))),
                        _as_dims(e.get("block", args.get("block"))), int(e.get("count", 1))))
            continue
        # torch FunctionEvent / FunctionEventAvg (key_averages)
        dev = str(getattr(e, "device_type", "")).upper()
        name = str(getattr(e, "key", None) or getattr(e, "name", ""))
        if hasattr(e, "count") and not hasattr(e, "time_range"):          # averaged row
            tot = float(getattr(e, "self_device_time_total", 0.0) or getattr(e, "self_cuda_time_total", 0.0)
                        or getattr(e, "device_time_total", 0.0) or getattr(e, "cuda_time_total", 0.0))
            if tot <= 0:
                continue
            out.append((name, tot, [], [], int(e.count)))
            continue
        if "CUDA" not in dev:
            continue
        tr = getattr(e, "time_range", None)
        dur = float(getattr(tr, "elapsed_us", lambda: 0.0)()) if tr is not None else 0.0
        out.append((name, dur, [], [], 1))
    return out


def kernel_table(profiler_events: Any) -> list[KernelRecord]:
    """Aggregate torch.profiler key_averages()/events into KernelRecords (accepts fakes in tests).

    Accepts a Chrome-trace dict (``prof.export_chrome_trace`` JSON, kernels have ``cat: kernel`` and
    ``args.grid/block``), a list of such event dicts, or torch FunctionEvent / key_averages rows. Rows are
    aggregated per (name, grid, block) and sorted by total time."""
    agg: dict[tuple[str, tuple[int, ...], tuple[int, ...]], list[float]] = {}
    for name, dur, grid, block, cnt in _iter_kernel_events(profiler_events):
        k = (name, tuple(grid), tuple(block))
        a = agg.setdefault(k, [0, 0.0])
        a[0] += cnt
        a[1] += dur
    rows = [KernelRecord(n, int(c), float(t), float(t) / c if c else 0.0, list(g), list(b))
            for (n, g, b), (c, t) in agg.items()]
    rows.sort(key=lambda r: (-r.total_us, r.name))
    return rows


def profile_events(fn: Callable[[], Any]) -> list[dict]:
    """GPU: run ``fn`` under torch.profiler (CUDA activity) and return Chrome-trace kernel events."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    from kdev.observe.guard import mark_profiled
    torch.cuda.synchronize()
    mark_profiled("torch.profiler (CUPTI)")      # CUPTI stays on: later timings in this process are tainted
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        fn()
        torch.cuda.synchronize()
    fd, path = tempfile.mkstemp(suffix=".json", prefix="kdev_prof_")
    os.close(fd)
    try:
        prof.export_chrome_trace(path)
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    return [e for e in d.get("traceEvents", []) if str(e.get("cat", "")).lower() == "kernel"]


def _set_nopdl(on: bool) -> None:
    for k in NOPDL_ENV:
        if on:
            os.environ[k] = "1"
        else:
            os.environ.pop(k, None)


def _cupti_profile(spec: JobSpec, reps: int, warnings: list[str]) -> list[dict[str, Any]]:
    from kdev.observe import _rt
    params = spec.params or {}
    extra = params.get("extra") or {}
    adapter = _rt.get_adapter_for(spec)
    f = spec.files[0]
    _rt.use_prebuilt(spec, f)
    module = adapter.load(f.path, "prof")
    state = adapter.prepare(module)
    flush = _rt.make_flush(params.get("flush_bytes"), params.get("flush", "read"))
    flush_names = {r.name for r in kernel_table(profile_events(flush))}
    use_graph = params.get("graph", True)
    shapes = _rt.select_shapes(adapter, params, extra.get("shape"))
    out = []
    for sh in shapes:
        inputs = adapter.make_inputs(sh, _rt.seed_for(adapter, sh))
        fn = lambda inputs=inputs: adapter.call(module, state, inputs)  # noqa: E731
        fn()
        if use_graph:
            g = _rt.capture(fn)
            body = lambda g=g: [(flush(), g.replay()) for _ in range(reps)]  # noqa: E731
        else:
            body = lambda fn=fn: [(flush(), fn()) for _ in range(reps)]  # noqa: E731
        ev = profile_events(body)
        table = [r for r in kernel_table(ev) if r.name not in flush_names]
        if use_graph and not table:
            warnings.append(f"{sh.shape_id}: no kernels seen in graph replay; re-profiled eagerly")
            table = [r for r in kernel_table(profile_events(lambda: [(flush(), fn()) for _ in range(reps)]))
                     if r.name not in flush_names]
        out.append({"shape_id": sh.shape_id, "reps": reps,
                    "kernels": [dict(asdict(r), per_call=r.count / reps, per_call_us=r.total_us / reps)
                                for r in table],
                    "total_per_call_us": sum(r.total_us for r in table) / reps})
    return out


def _external(tool: str, spec: JobSpec, rd: Path, warnings: list[str]) -> list[dict[str, Any]] | None:
    """Run the target driver under ncu/nsys; None when the tool is unusable (caller falls back)."""
    from kdev.observe import ncu, nsys
    from kdev.util import run
    job = rd / "target_job.json"
    job.write_text(json.dumps(spec.to_dict()), encoding="utf-8")
    target = [sys.executable, "-m", "kdev.observe.cupti", "--target", str(job)]
    extra = (spec.params or {}).get("extra") or {}
    if tool == "ncu":
        exe = ncu.find_ncu() or "ncu"
        rep = rd / "ncu.ncu-rep"
        argv = ncu.ncu_argv(target, metrics=extra.get("metrics") or "SpeedOfLight,Occupancy,WarpStateStats",
                            kernel_regex=extra.get("kernel"), out=str(rep), ncu=exe)
        p = run(argv, timeout=spec.timeout_s)
        (rd / "ncu.log").write_text(p.stdout + "\n" + p.stderr, encoding="utf-8")
        if ncu.is_perm_error(p.stdout + p.stderr):
            warnings.append("capability_unavailable: ncu (ERR_NVGPUCTRPERM) -> cupti")
            return None
        if p.returncode != 0 or not rep.exists():
            warnings.append(f"ncu failed rc={p.returncode} (see ncu.log) -> cupti")
            return None
        c = run(ncu.ncu_csv_argv(str(rep), exe), timeout=600)
        (rd / "ncu.csv").write_text(c.stdout, encoding="utf-8")
        rows = ncu.summarize_rows(ncu.parse_csv(c.stdout))
        return [{"shape_id": "all", "backend": "ncu", "kernels": rows}]
    exe = nsys.find_nsys() or "nsys"
    out = rd / "nsys"
    argv = nsys.nsys_argv(target, out=str(out))
    argv[0] = exe
    p = run(argv, timeout=spec.timeout_s)
    (rd / "nsys.log").write_text(p.stdout + "\n" + p.stderr, encoding="utf-8")
    rep = next((x for x in (rd / "nsys.nsys-rep", rd / "nsys.qdrep") if x.exists()), None)
    if p.returncode != 0 or rep is None:
        warnings.append(f"nsys failed rc={p.returncode} (see nsys.log) -> cupti")
        return None
    sa = nsys.stats_argv(str(rep))
    sa[0] = exe
    s = run(sa, timeout=600)
    (rd / "nsys_kern_sum.csv").write_text(s.stdout, encoding="utf-8")
    return [{"shape_id": "all", "backend": "nsys", "kernels": nsys.parse_kern_sum(s.stdout)}]


def run_profile(spec: JobSpec, run_dir: Any) -> Result:
    """Worker entry for kind='profile' (dispatches to ncu/nsys when params.extra['backend'] asks and the
    backend is permitted)."""
    from kdev.floor._text import table, us
    from kdev.observe import _rt, ncu, nsys
    rd = _rt.run_path(run_dir)
    params = spec.params or {}
    extra = params.get("extra") or {}
    backend = extra.get("backend", "auto")
    reps = int(extra.get("reps", 20))
    warnings: list[str] = []
    _set_nopdl(bool(extra.get("nopdl")))
    if not spec.files:
        raise KdevError(ErrorCode.INVALID_REQUEST, "profile needs one file")
    used = "cupti"
    data = None
    if backend in ("auto", "ncu"):
        ok, why = ncu.available()
        if ok:
            data = _external("ncu", spec, rd, warnings)
            used = "ncu" if data is not None else "cupti"
        elif backend == "ncu" or "not found" not in why:
            warnings.append(f"capability_unavailable: ncu ({why}) -> cupti")
    elif backend == "nsys":
        if nsys.available():
            data = _external("nsys", spec, rd, warnings)
            used = "nsys" if data is not None else "cupti"
        else:
            warnings.append("capability_unavailable: nsys (not found) -> cupti")
    if data is None:
        data = _cupti_profile(spec, reps, warnings)
    files = {"profile": _rt.write_json(rd / "profile.json", {"backend": used, "shapes": data})}
    rows = []
    for sh in data:
        for k in sh["kernels"][:5]:
            if used == "cupti":
                rows.append([sh["shape_id"], k["name"][:60], f"{k['per_call']:.2g}", us(k["mean_us"]),
                             us(k["per_call_us"]), "x".join(map(str, k["grid"])) or "-"])
            elif used == "nsys":
                rows.append([sh["shape_id"], k["name"][:60], str(k["count"]), us(k["avg_us"]), us(k["total_us"]), "-"])
            else:
                dur = next((v for m, v in k["metrics"].items() if m.endswith("/Duration")), "-")
                rows.append([sh["shape_id"], k["kernel"][:60], "1", dur, "-", k["grid"]])
    lines = [f"backend: {used}" + ("  (PDL disabled via KDEV_NOPDL)" if extra.get("nopdl") else "")]
    lines += table(["shape", "kernel", "n/call", "mean_us", "us/call", "grid"], rows, more_hint="profile.json")
    top = data[0]["kernels"][0] if data and data[0]["kernels"] else None
    summ = {"backend": used, "shapes": len(data),
            "top": (top.get("name") or top.get("kernel")) if top else None}
    return Result(tool="profile", verdict="ok", summary=summ, files=files, warnings=warnings, lines=lines,
                  next=["kdev dispatch FILE  (which kernels each scored shape launches)"])


def _target_main(job_path: str) -> int:
    """Target process for ncu/nsys: run every selected shape a few times eagerly (no profiling here)."""
    import torch
    from kdev.observe import _rt
    spec = JobSpec.from_dict(json.loads(Path(job_path).read_text(encoding="utf-8")))
    params = spec.params or {}
    extra = params.get("extra") or {}
    _set_nopdl(bool(extra.get("nopdl")))
    adapter = _rt.get_adapter_for(spec)
    f = spec.files[0]
    _rt.use_prebuilt(spec, f)
    module = adapter.load(f.path, "prof")
    state = adapter.prepare(module)
    for sh in _rt.select_shapes(adapter, params, extra.get("shape")):
        inputs = adapter.make_inputs(sh, _rt.seed_for(adapter, sh))
        for _ in range(int(extra.get("target_reps", 3))):
            adapter.call(module, state, inputs)
        torch.cuda.synchronize()
    return 0


if __name__ == "__main__":  # pragma: no cover
    if len(sys.argv) == 3 and sys.argv[1] == "--target":
        sys.exit(_target_main(sys.argv[2]))
    print("usage: python -m kdev.observe.cupti --target JOB.json", file=sys.stderr)
    sys.exit(2)
