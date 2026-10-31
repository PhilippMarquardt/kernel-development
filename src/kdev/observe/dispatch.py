"""Dispatch probe (M12): which kernels actually launch per scored shape. Owner: P6.

Runs each scored shape once under torch.profiler (CUPTI) eagerly and once as a graph replay; records kernel
name x count x grid per shape; warns when shapes the adapter declares distinct run the same route, and when
a file defines kernels never launched on any scored shape (static kernel list from build/sass).
``summarize_dispatch`` output goes into every `time`/`check` Result (Result.dispatch).

The lesson behind it: hours were spent optimising code paths that never ran (an entry point overwritten later
in the file, an embedded alternative source, a shape routed to a different kernel). Every timing should name
the kernels that actually ran.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any

from kdev.contract import ErrorCode, KdevError, Result
from kdev.models import JobSpec

LIBRARY_PATTERNS = (r"\bat::", r"\bat_cuda_detail\b", r"elementwise_kernel", r"reduce_kernel", r"vectorized_",
                    r"\bcublas", r"\bcutlass", r"sm\d+_xmma", r"\bnvjet", r"\bcudnn", r"triton_", r"\bflash_fwd",
                    r"\bmemcpy", r"\bmemset", r"\bfill_kernel", r"\bindex_elementwise", r"CatArrayBatched")


def _mangled_base(s: str) -> str | None:
    """Last name component of an Itanium-mangled symbol (_Z3foo.., _ZN2ns3fooE.., _Z3fooILi1EE..)."""
    if not s.startswith("_Z"):
        return None
    i = 3 if s.startswith("_ZL") else 2
    nested = s.startswith("N", i)
    if nested:
        i += 1
        while i < len(s) and s[i] in "rVK":
            i += 1
    last = None
    while i < len(s):
        if s[i].isdigit():
            j = i
            while j < len(s) and s[j].isdigit():
                j += 1
            n = int(s[i:j])
            last = s[j:j + n]
            i = j + n
            if not nested:
                break
        elif s.startswith("St", i):          # std::
            i += 2
        else:                                # 'I' template args, 'E' end of the nested name, ...
            break
    return last


def base_name(kernel: str) -> str:
    """Short identifier for matching static (mangled/SASS) and runtime (demangled CUPTI) kernel names:
    'void ns::foo<1, 2>(Params)' -> 'foo', '_ZN2ns3fooILi1EEEv7Params' -> 'foo'."""
    k = kernel.strip()
    m = _mangled_base(k)
    if m:
        return m
    k = re.sub(r"^(void|static|__global__)\s+", "", k)
    k = k.replace("(anonymous namespace)::", "")     # CUPTI's demangling of kernels in an unnamed namespace
    depth = 0
    cut = len(k)
    for i, ch in enumerate(k):           # strip args at top level
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif ch == "(" and depth == 0:
            cut = i
            break
    k = k[:cut]
    k = re.sub(r"<.*>$", "", k)          # template args
    return k.split("::")[-1].strip() or kernel


def _route_sig(kernels: list[dict[str, Any]]) -> tuple:
    return tuple(sorted((k.get("kernel", ""), tuple(k.get("grid") or []), int(k.get("count", 1))) for k in kernels))


def summarize_dispatch(per_shape: dict[str, list[dict[str, Any]]], static_kernels: list[str] | None = None
                       ) -> tuple[dict[str, Any], list[str]]:
    """Pure: (compact dispatch dict, warnings)."""
    compact: dict[str, Any] = {}
    warnings: list[str] = []
    for sid, ks in per_shape.items():
        compact[sid] = [{"kernel": k.get("kernel", ""), "count": int(k.get("count", 1)),
                         "grid": list(k.get("grid") or [])} for k in ks]
        if not ks:
            warnings.append(f"{sid}: no kernel launched (cached output? host-side fallback?)")
    sigs: dict[tuple, list[str]] = {}
    for sid, ks in per_shape.items():
        if ks:
            sigs.setdefault(_route_sig(ks), []).append(sid)
    for sids in sigs.values():
        if len(sids) > 1:
            warnings.append(f"same route (identical kernels x grid x count) for shapes: {', '.join(sids[:6])}"
                            + (f" (+{len(sids) - 6})" if len(sids) > 6 else ""))
    launched = {base_name(k.get("kernel", "")) for ks in per_shape.values() for k in ks}
    if static_kernels:
        never = sorted({base_name(s) for s in static_kernels} - launched)
        if never:
            warnings.append(f"kernels defined but never launched on any scored shape: {', '.join(never[:8])}"
                            + (f" (+{len(never) - 8})" if len(never) > 8 else "") + " (M12: dead path?)")
    lib = Counter()
    for ks in per_shape.values():
        for k in ks:
            if any(re.search(p, k.get("kernel", "")) for p in LIBRARY_PATTERNS):
                lib[base_name(k.get("kernel", ""))] += 1
    if lib:
        warnings.append("library/torch kernels in the timed call: " + ", ".join(f"{n} ({c} shapes)"
                                                                               for n, c in lib.most_common(5)))
    return compact, warnings


def _events_to_kernels(events: list[dict]) -> list[dict[str, Any]]:
    from kdev.observe.cupti import kernel_table
    return [{"kernel": r.name, "count": r.count, "grid": r.grid, "block": r.block, "total_us": r.total_us}
            for r in kernel_table(events)]


def probe_shape(adapter: Any, module: Any, state: Any, shape: Any, seed: int) -> list[dict[str, Any]]:
    """GPU: [{kernel, count, grid, block}] for one shape (one eager call after a warm-up call)."""
    from kdev.observe.cupti import profile_events
    inputs = adapter.make_inputs(shape, seed)
    adapter.call(module, state, inputs)
    return _events_to_kernels(profile_events(lambda: adapter.call(module, state, inputs)))


def _probe_graph(adapter: Any, module: Any, state: Any, shape: Any, seed: int) -> list[dict[str, Any]]:
    from kdev.observe import _rt
    from kdev.observe.cupti import profile_events
    inputs = adapter.make_inputs(shape, seed)
    g = _rt.capture(lambda: adapter.call(module, state, inputs))
    return _events_to_kernels(profile_events(lambda: g.replay()))


def _static_kernels() -> list[str]:
    """Kernel names of every loaded extension .so (best effort via kdev.sass, P5)."""
    try:
        from kdev.observe import _rt
        from kdev.sass.disasm import sass_text
        from kdev.sass.parse import parse_sass
    except Exception:  # noqa: BLE001
        return []
    names: list[str] = []
    for p in _rt.loaded_libraries():
        if "torch_extensions" not in p and "/cache/ext/" not in p and "kdev" not in p:
            continue
        try:
            names += [k.name for k in parse_sass(sass_text(p))]
        except Exception:  # noqa: BLE001
            continue
    return names


def run_dispatch(spec: JobSpec, run_dir: Any) -> Result:
    from kdev.floor._text import table
    from kdev.observe import _rt
    if not spec.files:
        raise KdevError(ErrorCode.INVALID_REQUEST, "dispatch needs one file")
    rd = _rt.run_path(run_dir)
    params = spec.params or {}
    adapter = _rt.get_adapter_for(spec)
    f = spec.files[0]
    _rt.use_prebuilt(spec, f)
    module = adapter.load(f.path, "dispatch")
    state = adapter.prepare(module)
    eager: dict[str, list[dict[str, Any]]] = {}
    graph: dict[str, list[dict[str, Any]]] = {}
    warnings: list[str] = []
    for sh in _rt.select_shapes(adapter, params):
        seed = _rt.seed_for(adapter, sh)
        eager[sh.shape_id] = probe_shape(adapter, module, state, sh, seed)
        if params.get("graph", True):
            try:
                graph[sh.shape_id] = _probe_graph(adapter, module, state, sh, seed)
            except Exception as e:  # noqa: BLE001
                warnings.append(f"{sh.shape_id}: graph capture failed: {str(e)[:100]}")
    static = _static_kernels()
    compact, w2 = summarize_dispatch(eager, static)
    warnings += w2
    for sid, gk in graph.items():
        en = {k["kernel"] for k in eager.get(sid, [])}
        gn = {k["kernel"] for k in gk}
        if gk and en != gn:
            warnings.append(f"{sid}: graph replay runs different kernels than eager "
                            f"(+{len(gn - en)} / -{len(en - gn)})")
    files = {"dispatch": _rt.write_json(rd / "dispatch.json", {"eager": eager, "graph": graph,
                                                                  "static_kernels": static})}
    rows = [[sid, base_name(k["kernel"])[:48], k["count"], "x".join(map(str, k["grid"])) or "-"]
            for sid, ks in compact.items() for k in ks]
    lines = table(["shape", "kernel", "count", "grid"], rows, more_hint="dispatch.json")
    return Result(tool="dispatch", verdict="warn" if warnings else "ok", dispatch=compact, files=files,
                  warnings=warnings, lines=lines,
                  summary={"shapes": len(compact), "kernels": len({k["kernel"] for ks in eager.values() for k in ks})})


def probe_isolated(spec: JobSpec, out_dir: str, *, python: str | None = None, timeout_s: float = 900,
                   runner: Any = None) -> tuple[dict[str, Any], list[str]]:
    """Dispatch probe in a separate process (torch.profiler leaves CUPTI enabled, adding 2-5 us to every
    later timing in the same process). For `time`/`check` workers that want Result.dispatch: call this, never
    probe_shape, in the timing process. Returns (compact dispatch dict, warnings); never raises on child
    failure (the warning says why)."""
    import json as _json
    import os
    import sys
    from pathlib import Path

    from kdev.util import run
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    job = d / "dispatch_job.json"
    job.write_text(_json.dumps(spec.to_dict()), encoding="utf-8")
    argv = [python or sys.executable, "-m", "kdev.observe.dispatch", "--child", str(job), str(d)]
    try:
        p = (runner or run)(argv, timeout=timeout_s, env=dict(os.environ))
    except Exception as e:  # noqa: BLE001
        return {}, [f"dispatch probe failed to start: {e}"]
    res = d / "dispatch_result.json"
    if p.returncode != 0 or not res.is_file():
        tail = (p.stderr or p.stdout or "").strip().splitlines()[-1:] or ["?"]
        return {}, [f"dispatch probe failed rc={p.returncode}: {tail[0][:160]}"]
    r = _json.loads(res.read_text(encoding="utf-8"))
    return r.get("dispatch") or {}, list(r.get("warnings") or [])


def _child_main(job_path: str, out_dir: str) -> int:
    import json as _json
    from pathlib import Path
    spec = JobSpec.from_dict(_json.loads(Path(job_path).read_text(encoding="utf-8")))
    try:
        res = run_dispatch(spec, out_dir)
    except KdevError as e:
        res = Result.from_error("dispatch", e)
    Path(out_dir, "dispatch_result.json").write_text(res.to_json(), encoding="utf-8")
    return 0 if res.status == "ok" else 1


if __name__ == "__main__":  # pragma: no cover
    import sys as _sys
    if len(_sys.argv) == 4 and _sys.argv[1] == "--child":
        _sys.exit(_child_main(_sys.argv[2], _sys.argv[3]))
    print("usage: python -m kdev.observe.dispatch --child JOB.json OUT_DIR", file=_sys.stderr)
    _sys.exit(2)
