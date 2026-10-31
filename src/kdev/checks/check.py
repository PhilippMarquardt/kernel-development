"""`kdev check` (worker kind='check'). Owner: P7.

Per case (tests + scored + extra held-out shapes) and per fresh seed: eager and/or graph-replay output,
repeated R times; adapter.check against the reference with the judge tolerances. Classification:
- wrong: fails on every repeat;
- intermittent: fails k/R repeats (race; e.g. PDL hoist) -> hint `kdev sass --audit pdl`;
- nondeterministic: all pass but outputs differ bitwise between repeats (report max ulp; fp32 atomics);
- pass.
Graph mode is judge-like: capture once (warm-up + side-stream warm-up first, like the harnesses), replay R
times on the captured inputs (each replay compared with the eager output: <= 1 ulp is the fp32-atomic-order
allowance), then copy a FRESH seed's inputs into the captured buffers and replay once more (``graph_fresh``):
the judge replays one graph on fresh inputs, so anything cached from ``data`` or branching on its values
fails here.
``--cold`` measures prepare + compile time from an empty extension cache against protocol.budget_s.
Always attaches the dispatch summary (kdev.observe.dispatch) when that backend is available.

Job params (``JobSpec.params``, TimingParams): seeds, seed_list, shapes (filter), repeat, mode; ``extra``:
extra_shapes, tests_only, cold, dispatch (default True).
"""
from __future__ import annotations

import gc
import json
import os
import secrets
import struct
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from kdev.contract import ErrorCode, KdevError, Result
from kdev.models import JobSpec

CLASSES = ("pass", "nondeterministic", "intermittent", "wrong", "error")
SEVERITY = {k: i for i, k in enumerate(CLASSES)}
ATOMIC_ULP = 1                     # eager-vs-graph allowance (fp32 atomic accumulation order)
COLD_ENV = ("TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR", "SGLANG_JIT_CACHE_DIR", "CUDA_CACHE_PATH",
            "TVM_FFI_CACHE_DIR", "FLASHINFER_WORKSPACE_BASE")
FATAL_MARKERS = ("illegal memory access", "CUDA error", "unspecified launch failure", "device-side assert")


@dataclass
class CaseOutcome:
    shape_id: str
    seed: int
    mode: str                     # eager | graph | graph_fresh
    passes: int
    repeats: int
    klass: str
    max_rel_err: float | None = None
    max_ulp: int | None = None
    message: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------------------------------- pure helpers

def classify(passes: list[bool], bitwise_equal: bool) -> str:
    """Pure classification used above. ``passes``: one entry per repeat."""
    if not passes:
        return "error"
    n_ok = sum(1 for p in passes if p)
    if n_ok == 0:
        return "wrong"
    if n_ok < len(passes):
        return "intermittent"
    return "pass" if bitwise_equal else "nondeterministic"


def worst_class(classes: list[str]) -> str:
    return max(classes, key=lambda k: SEVERITY.get(k, len(CLASSES))) if classes else "error"


def ordered_bits(bits: int, width: int) -> int:
    """Sign-magnitude float bit pattern (as an unsigned int of ``width`` bits) -> monotonic integer."""
    bits &= (1 << width) - 1
    sign = 1 << (width - 1)
    return -(bits & (sign - 1)) if bits & sign else bits


def ulp_distance_f32(a: float, b: float) -> int:
    """Distance in float32 ulps between two Python floats rounded to float32 (pure; for tests/reports)."""
    ia = struct.unpack("<I", struct.pack("<f", a))[0]
    ib = struct.unpack("<I", struct.pack("<f", b))[0]
    return abs(ordered_bits(ia, 32) - ordered_bits(ib, 32))


def pick_seeds(n: int, seed_list: list[int] | None = None) -> list[int]:
    """Fresh, unpredictable seeds (reward-hack guard) unless an explicit list is given (reproduction)."""
    if seed_list:
        return [int(s) for s in seed_list]
    return [secrets.randbelow(2**31 - 2) + 1 for _ in range(max(1, int(n)))]


def shape_matches(shape: Any, filt: str | None) -> bool:
    if not filt:
        return True
    return any(f and (f in shape.shape_id or f in str(shape.case) or f == shape.load) for f in filt.split("|"))


def select_cases(adapter: Any, *, tests_only: bool = False, extra: bool = False, filt: str | None = None) -> list:
    cases = list(adapter.tests())
    if not tests_only:
        cases += adapter.shapes()
        if extra:
            cases += adapter.extra_shapes()
    return [c for c in cases if shape_matches(c, filt)]


def summarize(outcomes: list[CaseOutcome]) -> dict[str, Any]:
    counts = {k: 0 for k in CLASSES}
    for o in outcomes:
        counts[o.klass] = counts.get(o.klass, 0) + 1
    bad = [o for o in outcomes if o.klass in ("wrong", "intermittent", "error")]
    ulps = [o.max_ulp for o in outcomes if o.max_ulp is not None]
    vs_eager = [o.extra.get("vs_eager_rel_l2") for o in outcomes if o.extra.get("vs_eager_rel_l2") is not None]
    return {
        "counts": counts,
        "n": len(outcomes),
        "failures": [{"shape_id": o.shape_id, "seed": o.seed, "mode": o.mode, "klass": o.klass,
                      "passes": f"{o.passes}/{o.repeats}", "message": o.message[:200]} for o in bad[:10]],
        "n_failures": len(bad),
        "max_ulp": max(ulps) if ulps else None,
        "graph_vs_eager_rel_l2": max(vs_eager) if vs_eager else None,
    }


def per_shape(outcomes: list[CaseOutcome], shapes: list[Any]) -> list[dict[str, Any]]:
    meta = {s.shape_id: s for s in shapes}
    rows: dict[str, dict[str, Any]] = {}
    for o in outcomes:
        r = rows.setdefault(o.shape_id, {"shape_id": o.shape_id, "weight": getattr(meta.get(o.shape_id), "weight", 1.0),
                                         "load": getattr(meta.get(o.shape_id), "load", ""),
                                         "scored": getattr(meta.get(o.shape_id), "scored", False),
                                         "classes": [], "max_rel_err": None, "tol": None, "max_ulp": None})
        r["classes"].append(o.klass)
        if o.max_rel_err is not None:
            r["max_rel_err"] = max(r["max_rel_err"] or 0.0, o.max_rel_err)
        if o.extra.get("tol") is not None:
            r["tol"] = o.extra["tol"]
        if o.max_ulp is not None:
            r["max_ulp"] = max(r["max_ulp"] or 0, o.max_ulp)
    out = []
    for r in rows.values():
        r["check_status"] = worst_class(r.pop("classes"))
        out.append(r)
    return out


def _fmt(x: float | None) -> str:
    return "-" if x is None else f"{x:.2e}"


def render_lines(verdict: str, summary: dict[str, Any], rows: list[dict[str, Any]]) -> list[str]:
    c = summary["counts"]
    head = (f"{verdict.upper()}  check  {summary.get('problem', '')}  cases={summary['n']} pass={c['pass']} "
            f"nondet={c['nondeterministic']} intermittent={c['intermittent']} wrong={c['wrong']} error={c['error']}")
    lines = [head]
    if summary.get("max_ulp") is not None or summary.get("graph_vs_eager_rel_l2") is not None:
        gv = summary.get("graph_vs_eager_rel_l2")
        lines.append(f"max ulp between repeats {summary.get('max_ulp')} (<= {ATOMIC_ULP}: fp32 atomic order)  "
                     f"graph-vs-eager rel L2 {'-' if gv is None else f'{gv:.1e}'}")
    lines.append(f"{'shape':22s} {'status':16s} {'rel_err':>9s} {'tol':>9s} {'ulp':>5s}")
    order = sorted(rows, key=lambda r: (-SEVERITY.get(r["check_status"], 9), r["shape_id"]))
    for r in order[:15]:
        lines.append(f"{r['shape_id'][:22]:22s} {r['check_status']:16s} {_fmt(r['max_rel_err']):>9s} "
                     f"{_fmt(r['tol']):>9s} {str(r['max_ulp'] if r['max_ulp'] is not None else '-'):>5s}")
    if len(order) > 15:
        lines.append(f"... {len(order) - 15} more shapes in cases.jsonl")
    for f in summary["failures"][:5]:
        lines.append(f"  {f['klass']}: {f['shape_id']} seed={f['seed']} {f['mode']} {f['passes']} :: {f['message'][:110]}")
    cold = summary.get("cold")
    if cold:
        lines.append(f"cold: load {cold['load_s']:.1f}s prepare {cold['prepare_s']:.1f}s first calls "
                     f"{cold['first_calls_s']:.1f}s total {cold['total_s']:.1f}s / budget {cold['budget_s']}s "
                     f"-> {cold['status']}")
    return lines


def next_hints(summary: dict[str, Any], file: str) -> list[str]:
    c = summary["counts"]
    hints = []
    if c["intermittent"]:
        hints += [f"kdev sass {file} --audit pdl", f"kdev sanitize {file} --tool racecheck"]
    elif c["wrong"] or c["error"]:
        hints.append(f"kdev sanitize {file} --tool memcheck")
    elif c["nondeterministic"]:
        hints.append("nondeterministic within tolerance: fine for the judge if max ulp is small (fp32 atomics)")
    return hints[:2]


def artifact_path(run_dir: Any, name: str) -> Path | None:
    """Path of an artifact inside a run dir (RunDir, str or Path); None when there is no run dir."""
    if run_dir is None:
        return None
    base = Path(getattr(run_dir, "path", run_dir))
    base.mkdir(parents=True, exist_ok=True)
    return base / name


def write_jsonl(run_dir: Any, name: str, rows: list[dict[str, Any]]) -> str | None:
    p = artifact_path(run_dir, name)
    if p is None:
        return None
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    return str(p)


def write_json(run_dir: Any, name: str, obj: Any) -> str | None:
    p = artifact_path(run_dir, name)
    if p is None:
        return None
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, p)
    return str(p)


def cold_verdict(load_s: float, prepare_s: float, first_calls_s: float, budget_s: float,
                 prepare_budget_s: float) -> dict[str, Any]:
    """PASS / RISK (> 60% of the run budget: little time left for tests + timing) / FAIL."""
    total = load_s + prepare_s + first_calls_s
    status = "PASS"
    if prepare_s > prepare_budget_s or total > budget_s:
        status = "FAIL"
    elif total > 0.6 * budget_s or prepare_s > 0.8 * prepare_budget_s:
        status = "RISK"
    return {"load_s": load_s, "prepare_s": prepare_s, "first_calls_s": first_calls_s, "total_s": total,
            "budget_s": budget_s, "prepare_budget_s": prepare_budget_s, "status": status}


# ------------------------------------------------------------------------------------- torch helpers

def tensors_of(obj: Any) -> list[Any]:
    """Tensors in an output/inputs structure, in a stable order."""
    import torch
    out: list[Any] = []
    if isinstance(obj, torch.Tensor):
        out.append(obj)
    elif isinstance(obj, dict):
        for k in sorted(obj, key=str):
            out += tensors_of(obj[k])
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            out += tensors_of(x)
    return out


def clone_output(obj: Any) -> Any:
    import torch
    if isinstance(obj, torch.Tensor):
        return obj.detach().clone()
    if isinstance(obj, tuple):
        return tuple(clone_output(x) for x in obj)
    if isinstance(obj, list):
        return [clone_output(x) for x in obj]
    if isinstance(obj, dict):
        return {k: clone_output(v) for k, v in obj.items()}
    return obj


_INT_VIEW = {"torch.float32": ("int32", 32), "torch.float64": ("int64", 64), "torch.bfloat16": ("int16", 16),
             "torch.float16": ("int16", 16), "torch.float8_e4m3fn": ("int8", 8), "torch.float8_e5m2": ("int8", 8)}


ULP_FLOOR = 2.0 ** -12   # elements below this fraction of the tensor's max |x| are ignored by the ulp metric


def tensor_max_ulp(a: Any, b: Any) -> int:
    """Max ulp distance between two same-dtype float tensors over elements with |x| >= ULP_FLOOR * max|x| (near-zero
    elements make ulp distances meaningless); NaN pairs ignored."""
    import torch
    if not a.numel():
        return 0
    view, width = _INT_VIEW[str(a.dtype)]
    ia = a.contiguous().view(getattr(torch, view)).to(torch.int64)
    ib = b.contiguous().view(getattr(torch, view)).to(torch.int64)
    mag = (1 << (width - 1)) - 1
    oa = torch.where(ia < 0, -(ia & mag), ia)
    ob = torch.where(ib < 0, -(ib & mag), ib)
    d = (oa - ob).abs()
    af, bf = a.float(), b.float()
    big = torch.nan_to_num(af.abs(), nan=0.0, posinf=0.0).max() * ULP_FLOOR
    keep = ((af.abs() >= big) | (bf.abs() >= big)) & ~(torch.isnan(af) & torch.isnan(bf))
    d = torch.where(keep, d, torch.zeros_like(d))
    return int(d.max())


def compare_outputs(a: Any, b: Any) -> dict[str, Any]:
    """{'bitwise', 'max_ulp' (float tensors, see tensor_max_ulp), 'int_diffs' (integer tensors), 'max_abs_diff',
    'structure'}."""
    import torch
    ta, tb = tensors_of(a), tensors_of(b)
    if len(ta) != len(tb) or any(x.shape != y.shape or x.dtype != y.dtype for x, y in zip(ta, tb)):
        return {"bitwise": False, "max_ulp": None, "int_diffs": 0, "max_abs_diff": None, "rel_l2": None,
                "structure": False}
    bitwise, max_ulp, int_diffs, max_abs, rel = True, None, 0, 0.0, 0.0
    for x, y in zip(ta, tb):
        if str(x.dtype) in _INT_VIEW:
            u = tensor_max_ulp(x, y)
            max_ulp = u if max_ulp is None else max(max_ulp, u)
            same = bool(torch.equal(x, y)) or bool(((x == y) | (torch.isnan(x.float()) & torch.isnan(y.float()))).all())
            bitwise &= same
            if not same:
                dxy = torch.nan_to_num(x.float() - y.float(), nan=0.0)
                max_abs = max(max_abs, float(dxy.abs().max()))
                rel = max(rel, float(dxy.norm() / torch.nan_to_num(x.float(), nan=0.0).norm().clamp(min=1e-12)))
        else:
            n = int((x != y).sum())
            int_diffs += n
            bitwise &= n == 0
    return {"bitwise": bitwise, "max_ulp": max_ulp, "int_diffs": int_diffs, "max_abs_diff": max_abs, "rel_l2": rel,
            "structure": True}


def copy_into(dst: Any, src: Any) -> bool:
    """Copy src's tensors into dst's (same structure, shapes, dtypes) in place; False (dst untouched) on any
    mismatch. Non-tensor leaves must be equal."""
    import torch

    def same(d: Any, s: Any) -> bool:
        if isinstance(d, torch.Tensor):
            return isinstance(s, torch.Tensor) and d.shape == s.shape and d.dtype == s.dtype
        if isinstance(d, dict):
            return isinstance(s, dict) and set(d) == set(s) and all(same(d[k], s[k]) for k in d)
        if isinstance(d, (list, tuple)):
            return type(d) is type(s) and len(d) == len(s) and all(same(x, y) for x, y in zip(d, s))
        return d == s

    if not same(dst, src):
        return False
    for d, s in zip(tensors_of(dst), tensors_of(src)):
        d.copy_(s)
    return True


def capture_graph(adapter: Any, module: Any, state: Any, data: Any) -> tuple[Any, Any]:
    """Warm-up, side-stream warm-up, capture (the harness/judge sequence). Returns (graph, static output)."""
    import torch
    adapter.call(module, state, data)
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        adapter.call(module, state, data)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = adapter.call(module, state, data)
    torch.cuda.synchronize()
    return g, out


def _is_fatal(e: BaseException) -> bool:
    return any(m in str(e) for m in FATAL_MARKERS)


def run_case(adapter: Any, module: Any, state: Any, shape: Any, seed: int, *, modes: tuple[str, ...],
             repeats: int, fresh_seed: int | None = None) -> list[CaseOutcome]:
    """All outcomes of one (shape, seed): eager, graph, graph_fresh as requested. Raises only on fatal CUDA errors
    (after recording the outcome)."""
    import torch

    outs: list[CaseOutcome] = []
    data = adapter.make_inputs(shape, seed)
    eager0 = None

    def record(mode: str, results: list[Any], ref_out: Any, msgs: list[str], extra: dict[str, Any]) -> None:
        passes = [r.ok for r in results]
        cmp_all = [compare_outputs(ref_out, o) for o in extra.pop("_outs", [])]
        bitwise = all(c["bitwise"] for c in cmp_all)
        ulps = [c["max_ulp"] for c in cmp_all if c["max_ulp"] is not None]
        errs = [r.max_rel_err for r in results if r.max_rel_err is not None]
        tol = next((r.tol for r in results if r.tol is not None), None)
        bad = next((r.message for r in results if not r.ok), None)
        extra = dict(extra, tol=tol, int_diffs=sum(c["int_diffs"] for c in cmp_all),
                     max_abs_diff=max((c["max_abs_diff"] or 0.0) for c in cmp_all) if cmp_all else None)
        klass = classify(passes, bitwise)
        if klass == "pass" and extra.get("vs_eager_bitwise") is False:
            klass = "nondeterministic"
        outs.append(CaseOutcome(shape.shape_id, seed, mode, sum(passes), len(passes), klass,
                                max(errs) if errs else None, max(ulps) if ulps else (0 if cmp_all else None),
                                bad or (msgs[0] if msgs else ""), extra))

    try:
        if "eager" in modes or "graph" in modes:
            results, cl = [], []
            for _ in range(repeats if "eager" in modes else 1):
                o = clone_output(adapter.call(module, state, data))
                torch.cuda.synchronize()
                results.append(adapter.check(data, o))
                cl.append(o)
            eager0 = cl[0]
            if "eager" in modes:
                record("eager", results, eager0, [r.message for r in results], {"_outs": cl[1:]})
            del cl
        if "graph" in modes:
            g, gout = capture_graph(adapter, module, state, data)
            results, cl, vs, vs_int, vs_rel, vs_bit = [], [], 0, 0, 0.0, True
            for _ in range(repeats):
                g.replay()
                torch.cuda.synchronize()
                o = clone_output(gout)
                results.append(adapter.check(data, o))
                cl.append(o)
                if eager0 is not None:
                    c = compare_outputs(eager0, o)
                    vs = max(vs, c["max_ulp"] or 0)
                    vs_int += c["int_diffs"]
                    vs_rel = max(vs_rel, c["rel_l2"] or 0.0)
                    vs_bit &= c["bitwise"]
            record("graph", results, cl[0], [r.message for r in results],
                   {"_outs": cl[1:], "vs_eager_max_ulp": vs if eager0 is not None else None,
                    "vs_eager_int_diffs": vs_int, "vs_eager_rel_l2": vs_rel if eager0 is not None else None,
                    "vs_eager_bitwise": vs_bit if eager0 is not None else None})
            del cl
            if fresh_seed is not None:
                fresh = adapter.make_inputs(shape, fresh_seed)
                if copy_into(data, fresh):
                    del fresh
                    g.replay()
                    torch.cuda.synchronize()
                    r = adapter.check(data, clone_output(gout))
                    outs.append(CaseOutcome(shape.shape_id, fresh_seed, "graph_fresh", int(r.ok), 1,
                                            "pass" if r.ok else "wrong", r.max_rel_err, None, r.message,
                                            {"tol": r.tol, "captured_seed": seed}))
                else:
                    outs.append(CaseOutcome(shape.shape_id, fresh_seed, "graph_fresh", 0, 0, "pass", None, None,
                                            "skipped: a second seed's inputs differ in structure/shape",
                                            {"skipped": True}))
            del g, gout
    except Exception as e:  # noqa: BLE001 - recorded as an 'error' outcome
        outs.append(CaseOutcome(shape.shape_id, seed, "+".join(modes), 0, 0, "error",
                                message=f"{type(e).__name__}: {str(e)[:300]}"))
        if _is_fatal(e):
            raise
    finally:
        del data
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
    return outs


def _modes(mode: str) -> tuple[str, ...]:
    return {"eager": ("eager",), "graph": ("graph",), "both": ("eager", "graph")}.get(mode, ("eager", "graph"))


def _set_cold_env(run_dir: Any) -> dict[str, str]:
    import tempfile
    base = artifact_path(run_dir, "cold_cache") or Path(tempfile.mkdtemp(prefix="kdev_cold_"))
    env = {}
    for var in COLD_ENV:
        d = base / var.lower()
        d.mkdir(parents=True, exist_ok=True)
        os.environ[var] = env[var] = str(d)
    return env


def _dispatch(adapter: Any, module: Any, state: Any, seeds: list[int]) -> tuple[dict[str, Any] | None, list[str]]:
    try:
        from kdev.observe.dispatch import probe_shape, summarize_dispatch
        per = {s.shape_id: probe_shape(adapter, module, state, s, seeds[0]) for s in adapter.shapes()}
        return summarize_dispatch(per)
    except NotImplementedError:
        return None, ["dispatch probe not available yet (kdev.observe.dispatch)"]
    except Exception as e:  # noqa: BLE001
        return None, [f"dispatch probe failed: {type(e).__name__}: {str(e)[:120]}"]


def _get_adapter(spec: JobSpec) -> Any:
    from kdev.adapters.registry import get_adapter
    f = spec.files[0].path if spec.files else None
    return get_adapter(problem=spec.problem, file=f, adapter=spec.adapter)


def run_check(spec: JobSpec, run_dir: Any) -> Result:
    """Worker entry point for kind='check'. Never raises (errors become Results)."""
    try:
        return _run_check(spec, run_dir)
    except KdevError as e:
        return Result.from_error("check", e)


def _run_check(spec: JobSpec, run_dir: Any) -> Result:
    if not spec.files:
        raise KdevError(ErrorCode.INVALID_REQUEST, "check needs a submission file")
    p = dict(spec.params)
    ex = dict(p.get("extra") or {})
    adapter = _get_adapter(spec)
    fref = spec.files[0]
    seeds = pick_seeds(p.get("seeds") or 3, p.get("seed_list"))
    repeats = max(1, int(p.get("repeat") or 3))
    modes = _modes(str(p.get("mode") or "both"))
    cases = select_cases(adapter, tests_only=bool(ex.get("tests_only")), extra=bool(ex.get("extra_shapes")),
                         filt=p.get("shapes"))
    if not cases:
        raise KdevError(ErrorCode.INVALID_REQUEST, f"no cases match --shapes {p.get('shapes')!r}")
    warnings: list[str] = []
    cold = None
    t0 = time.monotonic()
    if ex.get("cold"):
        _set_cold_env(run_dir)
    module = adapter.load(fref.path, fref.label or "A")
    for k, v in (fref.knobs or {}).items():
        setattr(module, k, v)
    load_s = time.monotonic() - t0
    state = adapter.prepare(module)
    prepare_s = getattr(adapter, "last_prepare_s", None) or 0.0
    if ex.get("cold"):
        import torch
        t1 = time.monotonic()
        for s in adapter.shapes():
            d = adapter.make_inputs(s, seeds[0])
            adapter.call(module, state, d)
            torch.cuda.synchronize()
            del d
        cold = cold_verdict(load_s, prepare_s, time.monotonic() - t1, adapter.protocol.budget_s,
                            adapter.protocol.prepare_budget_s)

    outcomes: list[CaseOutcome] = []
    fatal = None
    for shape in cases:
        for seed in seeds:
            fresh = seed ^ 0x5DEECE66 if "graph" in modes else None
            try:
                outcomes += run_case(adapter, module, state, shape, seed, modes=modes, repeats=repeats,
                                     fresh_seed=fresh)
            except Exception as e:  # noqa: BLE001 - fatal CUDA error: the context is gone
                outcomes.append(CaseOutcome(shape.shape_id, seed, "+".join(modes), 0, 0, "error",
                                            message=f"{type(e).__name__}: {str(e)[:300]}"))
                fatal = f"fatal CUDA error at {shape.shape_id} seed {seed}: {str(e)[:160]}; remaining cases skipped"
                break
        if fatal:
            break
    if fatal:
        warnings.append(fatal)

    dispatch = None
    if ex.get("dispatch", True) and not fatal:
        dispatch, dw = _dispatch(adapter, module, state, seeds)
        warnings += dw if isinstance(dw, list) else []

    summary = summarize(outcomes)
    summary.update(problem=adapter.problem, file=fref.path, md5=fref.md5, seeds=seeds, repeats=repeats,
                   modes=list(modes), n_cases=len(cases), prepare_s=round(prepare_s, 2))
    if cold:
        summary["cold"] = cold
    failed = summary["n_failures"] > 0 or (cold is not None and cold["status"] == "FAIL")
    verdict = "fail" if failed else "pass"
    rows = per_shape(outcomes, cases)
    res = Result(tool="check", status="fail" if failed else "ok", verdict=verdict, job_id=spec.job_id or None,
                 summary=summary, shapes=rows, dispatch=dispatch, warnings=warnings,
                 next=next_hints(summary, fref.path))
    path = write_jsonl(run_dir, "cases.jsonl", [asdict(o) for o in outcomes])
    if path:
        res.files["cases"] = path
    res.lines = render_lines(verdict, summary, rows)
    return res
