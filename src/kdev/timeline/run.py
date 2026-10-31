"""Worker entry for kind='timeline' (GPU). Owner: P6.

Builds (via the prebuilt instrumented copy), allocates the mark buffer, runs one shape (graph replay after
read flush, like timing), decodes, writes trace.json/timeline.csv, and measures mark overhead as marked vs
unmarked replay delta. ``params.extra['ab']`` = two define sets -> interleaved timelines.

Job layout (built by ``kdev timeline``): files with role ``sub``/``variant`` are instrumented copies (one per
define set, label A/B for --ab), an optional file with role ``base`` is the unmarked original (overhead).
``params.extra``: ``marks`` {id: name}, ``shape``, ``ctas`` (all|sample:N), ``reps`` (default 15),
``capacity`` (records, default 4096 CTAs x 8 WG x 33 slots), ``seed``.
After the first (warm-up) call of a marked module every loaded extension exporting ``kd_marks_init`` gets
the buffer; then the call is captured as a graph and replayed ``reps`` times interleaved with the unmarked
graph (ABBA), each replay after the read flush, buffer zeroed before the flush.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from kdev.contract import ErrorCode, KdevError, Result
from kdev.models import JobSpec

DEFAULT_CAPACITY = 4096 * 8 * 33
_CLAIMED: set[str] = set()


def _names_for(spec: JobSpec, f: Any) -> dict[int, str]:
    extra = (spec.params or {}).get("extra") or {}
    m = extra.get("marks")
    if not m:
        mj = Path(f.path).parent / "marks.json"
        if mj.is_file():
            m = json.loads(mj.read_text(encoding="utf-8")).get("marks", {})
    return {int(k): v for k, v in (m or {}).items()}


class _Marked:
    """One instrumented module prepared for one shape."""

    def __init__(self, spec: JobSpec, adapter: Any, f: Any, shape: Any, seed: int, capacity: int, tag: str) -> None:
        import torch
        from kdev.observe import _rt
        _rt.use_prebuilt(spec, f)
        self.label = f.label
        self.notes: list[str] = []
        before = set(_rt.loaded_libraries())
        self.module = adapter.load(f.path, tag)
        self.state = adapter.prepare(self.module)
        self.inputs = adapter.make_inputs(shape, seed)
        fn = lambda: adapter.call(self.module, self.state, self.inputs)  # noqa: E731
        fn()
        torch.cuda.synchronize()
        self.capacity = capacity
        self.buf = torch.zeros(16 + 16 * capacity, dtype=torch.uint8, device="cuda")
        all_libs = _rt.marks_libraries()
        owned = set(_rt.extension_paths(self.module))                     # M = load_inline(...) in the module
        mine = [(p, h) for p, h in all_libs if os.path.realpath(p) in owned]
        new = [(p, h) for p, h in all_libs if p not in before]             # loaded by this module (fresh process)
        self.libs = mine or new or [(p, h) for p, h in all_libs if p not in _CLAIMED]   # --ab: own .so each
        _CLAIMED.update(p for p, _ in self.libs)
        if not self.libs and all_libs:
            raise KdevError(ErrorCode.INVALID_REQUEST, f"{f.path} [{f.label}]: no new instrumented extension was "
                            "loaded (both --ab variants resolve to the same build?)")
        if not self.libs:
            raise KdevError(ErrorCode.CAPABILITY_UNAVAILABLE,
                            f"{f.path}: no loaded extension exports kd_marks_init",
                            recovery="build the instrumented copy (kdev build COPY --defines -DKDEV_MARKS) and make "
                                     "sure the adapter loads that copy, not a cached unmarked build")
        flags = None
        for p, init in self.libs:
            rc = init(self.buf.data_ptr(), capacity)
            if rc < 0:
                raise KdevError(ErrorCode.INTERNAL, f"kd_marks_init failed rc={rc} in {p}")
            if flags is None:
                flags = rc
            elif rc != flags:                 # another layout / KD_MAX_* cannot share this buffer: disable it
                init(None, 0)
                self.notes.append(f"{os.path.basename(p)}: different mark layout (flags {rc:#x} vs {flags:#x}); "
                                  "its marks are disabled")
        self.flags = int(flags or 0) | 0x80000000
        hdr = [0, capacity, 0, self.flags]
        self.hdr = torch.tensor([x - (1 << 32) if x >= 1 << 31 else x for x in hdr], dtype=torch.int32,
                                device="cuda").view(torch.uint8)
        self.graph = _rt.capture(fn)

    def reset(self) -> None:
        self.buf.zero_()
        self.buf[:16].copy_(self.hdr)

    def read(self) -> bytes:
        return self.buf.cpu().numpy().tobytes()


def _plain(spec: JobSpec, adapter: Any, f: Any, shape: Any, seed: int) -> Any:
    from kdev.observe import _rt
    _rt.use_prebuilt(spec, f)
    module = adapter.load(f.path, "tl_plain")
    state = adapter.prepare(module)
    inputs = adapter.make_inputs(shape, seed)
    return _rt.capture(lambda: adapter.call(module, state, inputs))


def run_timeline(spec: JobSpec, run_dir: Any) -> Result:
    from kdev.floor._text import table, us
    from kdev.observe import _rt
    from kdev.timeline import decode, trace
    from kdev.observe.guard import check_clean
    check_clean("timeline")              # CUPTI left on by a profiler would add 2-5 us per replay
    rd = _rt.run_path(run_dir)
    params = spec.params or {}
    extra = params.get("extra") or {}
    marked_files = [f for f in spec.files if f.role != "base"]
    base_files = [f for f in spec.files if f.role == "base"]
    if not marked_files:
        raise KdevError(ErrorCode.INVALID_REQUEST, "timeline job has no instrumented file")
    adapter = _rt.get_adapter_for(spec)
    shape = _rt.select_shapes(adapter, params, extra.get("shape"))[0]
    seed = int(extra.get("seed", _rt.seed_for(adapter, shape)))
    reps = int(extra.get("reps", 15))
    cap = int(extra.get("capacity", DEFAULT_CAPACITY))
    flush = _rt.make_flush(params.get("flush_bytes"), params.get("flush", "read"))
    names = _names_for(spec, marked_files[0])
    variants = [_Marked(spec, adapter, f, shape, seed, cap, f"tl{i}") for i, f in enumerate(marked_files)]
    plain = _plain(spec, adapter, base_files[0], shape, seed) if base_files else None
    times: dict[str, list[float]] = {v.label: [] for v in variants}
    plain_t: list[float] = []
    sums: dict[str, list[Any]] = {v.label: [] for v in variants}
    recs_by: dict[str, list[tuple[float, list[Any]]]] = {v.label: [] for v in variants}
    dropped: dict[str, int] = {v.label: 0 for v in variants}
    order: list[Any] = list(variants) + ([plain] if plain is not None else [])
    for r in range(reps + 1):
        seq = order if r % 2 == 0 else list(reversed(order))
        for item in seq:
            if item is plain:
                t = _rt.time_replay_us(plain, flush)
                if r:
                    plain_t.append(t)
                continue
            item.reset()
            t = _rt.time_replay_us(item.graph, flush)
            if not r:
                continue                      # first round = warm-up
            times[item.label].append(t)
            recs, drop = decode.parse_buffer(item.read())
            recs = decode.sample_ctas(recs, extra.get("ctas", "all"))
            s = decode.summarize(recs, names)
            s.dropped = drop
            dropped[item.label] += drop
            sums[item.label].append(s)
            recs_by[item.label].append((s.span_ns, recs))
    out_files: dict[str, str] = {}
    per_label: dict[str, Any] = {}
    lines: list[str] = []
    warnings: list[str] = []
    for v in variants:
        lab = v.label
        merged = decode.merge_summaries(sums[lab])
        if plain_t:
            merged.mark_overhead_us = _rt.median(times[lab]) - _rt.median(plain_t)
        rs = sorted(recs_by[lab], key=lambda x: x[0])
        rep_recs = rs[len(rs) // 2][1] if rs else []
        sfx = "" if len(variants) == 1 else f"_{lab}"
        out_files[f"trace{sfx}"] = _rt.write_json(rd / f"trace{sfx}.json", trace.to_perfetto(rep_recs, names))
        rows = trace.to_csv_rows(rep_recs, names)
        out_files[f"csv{sfx}"] = trace.write_csv(str(rd / f"timeline{sfx}.csv"), rows)
        out_files[f"chart{sfx}"] = _rt.write_json(rd / f"chart{sfx}.json", trace.chart_json(merged, rows))
        out_files[f"timeline{sfx}"] = _rt.write_json(rd / f"timeline{sfx}.json", merged.to_dict())
        per_label[lab] = {"replay_us_median": _rt.median(times[lab]), "span_us": merged.span_ns / 1000,
                          "resolution_ns": merged.resolution_ns, "duration_source": merged.duration_source,
                          "mark_overhead_us": merged.mark_overhead_us, "critical_cta": merged.critical_cta,
                          "dropped": dropped[lab], "ctas": merged.n_ctas}
        warnings.extend(f"{lab}: {n}" for n in v.notes)
        if dropped[lab]:
            warnings.append(f"{lab}: {dropped[lab]} marks dropped (raise params.extra.capacity / KD_MAX_WG)")
        if merged.resolution_ns and merged.resolution_ns >= decode.COARSE_NS and merged.duration_source != "clock64":
            warnings.append(f"{lab}: globaltimer tick {merged.resolution_ns} ns; mark times +-{merged.resolution_ns} ns")
        hdr = f"[{lab}] " if len(variants) > 1 else ""
        lines.append(f"{hdr}shape {shape.shape_id} seed {seed}: replay {us(_rt.median(times[lab]) - 4.0)} us (-4.0), "
                     f"span {us(merged.span_ns / 1000)} us, {merged.n_ctas} CTAs, tick {merged.resolution_ns} ns, "
                     f"durations={merged.duration_source}"
                     + (f", mark overhead {merged.mark_overhead_us:+.2f} us" if merged.mark_overhead_us is not None else ""))
        lines.append(f"{hdr}CTA start spread {us(merged.cta_start_spread_ns / 1000)} us, end spread "
                     f"{us(merged.cta_end_spread_ns / 1000)} us, critical CTA {merged.critical_cta}")
        lines += table(["mark", "p10_us", "p50_us", "p90_us", "max_us", "n"],
                       [[m.name, us(m.p10_ns / 1000), us(m.p50_ns / 1000), us(m.p90_ns / 1000), us(m.max_ns / 1000),
                         m.n] for m in merged.marks], max_rows=8 if len(variants) > 1 else 12,
                       more_hint=out_files[f"timeline{sfx}"])
        lines += table(["phase", "p10_us", "p50_us", "p90_us", "n"],
                       [[p.name, us(p.p10_ns / 1000), us(p.p50_ns / 1000), us(p.p90_ns / 1000), p.n]
                        for p in merged.phases], max_rows=6 if len(variants) > 1 else 10,
                       more_hint=out_files[f"timeline{sfx}"])
    summary = {"shape": shape.shape_id, "seed": seed, "reps": reps, "per_label": per_label,
               "plain_replay_us_median": _rt.median(plain_t) if plain_t else None}
    return Result(tool="timeline", verdict="ok", summary=summary, files=out_files, warnings=warnings, lines=lines,
                  next=["open trace.json in ui.perfetto.dev", "kdev show RUN --section chart"])
