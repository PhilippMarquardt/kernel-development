"""Worker entry for kind='timeline' (GPU). Owner: P6.

Judge-like conditions: every module (instrumented copies and their unmarked originals) is loaded and prepared
once; per shape and seed each gets fresh inputs (``seeds_for_shape``: the judge seed first, then the
calib_seeds rotation, so routing/lengths change per seed), a warm call and a CUDA graph; then the graphs are
replayed ``replays`` times (+1 warm-up round), modules interleaved ABBA, each replay after the read flush, the
mark buffer zeroed before the flush. Phased lock (PROTOCOL 10): load/prepare/capture under the SHARED flock,
the replays EXCLUSIVE. Mark overhead = marked replay median - plain replay median of the same file and shape.

Job layout (built by ``kdev timeline``): files with role ``sub``/``variant`` are instrumented copies (label A,
and B for --diff / --ab), role ``base`` files are the unmarked originals (labels ``A.plain`` / ``B.plain``).
``params.extra``: ``marks`` {id: name} (or ``marks_by`` {label: {id: name}}), ``shape`` (one shape id) or
``params.shapes`` (comma filter), ``seeds`` (default 3), ``reps`` (replays, default 5), ``roles`` {wg: role},
``t0`` (mark name), ``seed_list`` (explicit seeds for every shape), ``ctas`` (all|sample:N), ``capacity_ctas`` (default 1024), ``names`` overrides.
After the warm call of a marked module every extension it owns that exports ``kd_marks_init`` gets the buffer
(re-checked after each new shape: lazily built extensions are attached too).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from kdev.contract import ErrorCode, KdevError, Result
from kdev.models import JobSpec

DEFAULT_CAPACITY_CTAS = 1024
DEFAULT_SEEDS = 3
DEFAULT_REPS = 5
_CLAIMED: set[str] = set()
_OWNER: dict[str, int] = {}          # .so path -> id(_Mod) whose buffer it currently writes to


def _names_for(spec: JobSpec, f: Any) -> dict[int, str]:
    extra = (spec.params or {}).get("extra") or {}
    m = (extra.get("marks_by") or {}).get(f.label) or extra.get("marks")
    if not m:
        mj = Path(f.path).parent / "marks.json"
        if mj.is_file():
            m = json.loads(mj.read_text(encoding="utf-8")).get("marks", {})
    names = {int(k): v for k, v in (m or {}).items()}
    over = extra.get("names") or {}
    by_name = {v: k for k, v in names.items()}
    for k, v in over.items():
        if str(k) in by_name:
            names[by_name[str(k)]] = v
        elif str(k).isdigit():
            names[int(k)] = v
    return names


def _flags_info(flags: int) -> dict[str, int]:
    lay = flags & 3
    return {"layout": lay, "max_wg": (flags >> 16) & 0xFF, "max_marks": (flags >> 8) & 0xFF,
            "cta_recs": (1 << ((flags >> 2) & 0x3F)) if lay == 2 else 0}


class _Mod:
    """One loaded + prepared module (marked or plain) with one graph per (shape, seed)."""

    def __init__(self, spec: JobSpec, adapter: Any, f: Any, tag: str, marked: bool, cap_ctas: int) -> None:
        from kdev.observe import _rt
        self.spec, self.adapter, self.f, self.marked = spec, adapter, f, marked
        self.label = f.label
        self.notes: list[str] = []
        self.cap_ctas = cap_ctas
        _rt.use_prebuilt(spec, f)
        self.module = adapter.load(f.path, tag)
        self.state = adapter.prepare(self.module)
        self.graphs: dict[tuple[str, int], Any] = {}
        self.inputs: dict[tuple[str, int], Any] = {}
        self.inited: set[str] = set()
        self.buf: Any = None
        self.hdr: Any = None
        self.flags = 0
        self.capacity = 0
        self.read_n = 16
        self.libs: list[tuple[str, Any]] = []

    def _attach(self) -> None:
        import torch
        from kdev.observe import _rt
        all_libs = _rt.marks_libraries()
        owned = {os.path.realpath(p) for p in _rt.extension_paths(self.module)}
        mine = [(p, h) for p, h in all_libs if os.path.realpath(p) in owned and p not in self.inited]
        shared = [p for p, _ in mine if p in _CLAIMED]
        if shared:                       # identical source in two files (e.g. the same helper TU): one .so
            self.notes.append("shares " + ", ".join(os.path.basename(p) for p in shared) + " with another file "
                              "(identical CUDA source): its buffer is re-pointed before each replay")
        if not mine and not self.inited:
            mine = [(p, h) for p, h in all_libs if p not in _CLAIMED]
        for p, init in mine:
            flags = init(None, 0)
            if flags < 0:
                raise KdevError(ErrorCode.INTERNAL, f"kd_marks_init(NULL) failed rc={flags} in {p}")
            if self.buf is None:
                fi = _flags_info(flags)
                per_cta = max(1, fi["max_wg"]) * (fi["cta_recs"] if fi["layout"] == 2 else fi["max_marks"] + 1)
                self.capacity = self.cap_ctas * per_cta if fi["layout"] != 1 else self.cap_ctas * 256
                self.buf = torch.zeros(16 + 16 * self.capacity, dtype=torch.uint8, device="cuda")
                self.flags = flags | 0x80000000
                hdr = [0, self.capacity, fi["cta_recs"], self.flags]
                self.hdr = torch.tensor([x - (1 << 32) if x >= 1 << 31 else x for x in hdr], dtype=torch.int32,
                                        device="cuda").view(torch.uint8)
            if (flags | 0x80000000) != self.flags:
                self.notes.append(f"{os.path.basename(p)}: different mark layout (flags {flags:#x} vs "
                                  f"{self.flags:#x}); its marks are disabled")
                self.inited.add(p)
                continue
            rc = init(self.buf.data_ptr(), self.capacity)
            if rc < 0:
                raise KdevError(ErrorCode.INTERNAL, f"kd_marks_init failed rc={rc} in {p}")
            self.inited.add(p)
            self.libs.append((p, init))
            _CLAIMED.add(p)
            _OWNER[p] = id(self)

    def add(self, shape: Any, seed: int) -> None:
        import torch
        from kdev.observe import _rt
        from kdev.timeline import decode
        _rt.use_prebuilt(self.spec, self.f)
        inputs = self.adapter.make_inputs(shape, seed)
        fn = lambda: self.adapter.call(self.module, self.state, inputs)  # noqa: E731
        fn()
        torch.cuda.synchronize()
        if self.marked:
            self._attach()
            if self.buf is None:
                raise KdevError(ErrorCode.CAPABILITY_UNAVAILABLE,
                                f"{self.f.path} [{self.label}]: no loaded extension exports kd_marks_init",
                                recovery="build the instrumented copy (kdev timeline builds it unless --no-build) "
                                         "and make sure the file loads its CUDA through load_inline")
            self.reset()
        g = _rt.capture(fn)
        if self.marked:
            torch.cuda.synchronize()
            data = self.read(full=True)
            self.read_n = max(self.read_n, decode.used_bytes(data, margin_ctas=8))
        key = (shape.shape_id, seed)
        self.graphs[key] = g
        self.inputs[key] = inputs

    def reset(self) -> None:
        for p, init in self.libs:          # a .so shared with another module points at the last one's buffer
            if _OWNER.get(p) != id(self):
                init(self.buf.data_ptr(), self.capacity)
                _OWNER[p] = id(self)
        if self.buf is not None:
            self.buf.zero_()
            self.buf[:16].copy_(self.hdr)

    def read(self, full: bool = False) -> bytes:
        n = self.buf.numel() if full else min(self.buf.numel(), self.read_n)
        return self.buf[:n].cpu().numpy().tobytes()


def _phased(run_dir: Any) -> Any:
    if os.environ.get("KDEV_PHASE_MODE") != "phased":
        return None
    from kdev.service.phase import PhasedLock
    from kdev.timing._io import log
    return PhasedLock.from_env(log=lambda m: log(run_dir, m))


def run_timeline(spec: JobSpec, run_dir: Any) -> Result:
    from kdev.observe.guard import check_clean
    check_clean("timeline")              # CUPTI left on by a profiler would add 2-5 us per replay
    pl = _phased(run_dir)
    if pl is not None:
        pl.prepare()                     # SHARED flock before any CUDA context exists
    import torch

    from kdev.observe import _rt
    from kdev.timeline import analyze as A
    from kdev.timeline import decode, trace
    from kdev.timing.seeds import seeds_for_shape
    rd = _rt.run_path(run_dir)
    params = spec.params or {}
    extra = params.get("extra") or {}
    marked_files = [f for f in spec.files if f.role != "base"]
    base_files = [f for f in spec.files if f.role == "base"]
    if not marked_files:
        raise KdevError(ErrorCode.INVALID_REQUEST, "timeline job has no instrumented file")
    adapter = _rt.get_adapter_for(spec)
    shapes = _rt.select_shapes(adapter, params, extra.get("shape"))
    if not extra.get("shape") and not params.get("shapes"):
        shapes = shapes[:1]
    n_seeds = len(extra.get("seed_list") or []) or max(1, int(extra.get("seeds") or DEFAULT_SEEDS))
    reps = max(1, int(extra.get("reps") or DEFAULT_REPS))
    roles = {int(k): v for k, v in (extra.get("roles") or {}).items()}
    t0 = extra.get("t0") or None
    cap_ctas = int(extra.get("capacity_ctas") or DEFAULT_CAPACITY_CTAS)
    flush = _rt.make_flush(params.get("flush_bytes"), params.get("flush", "read"))
    names = {f.label: _names_for(spec, f) for f in marked_files}
    mods = [_Mod(spec, adapter, f, f"tl{i}", True, cap_ctas) for i, f in enumerate(marked_files)]
    mods += [_Mod(spec, adapter, f, f"tlp{i}", False, cap_ctas) for i, f in enumerate(base_files)]
    plain_of = {m.label: (m.label[:-6] if m.label.endswith(".plain") else marked_files[0].label)
                for m in mods if not m.marked}
    groups: list[tuple[Any, int]] = []
    for sh in shapes:
        sl = [int(x) for x in (extra.get("seed_list") or [])]
        for sd in (sl or seeds_for_shape(_rt.seed_for(adapter, sh), n_seeds)):
            for m in mods:
                m.add(sh, int(sd))
            groups.append((sh, int(sd)))
    torch.cuda.synchronize()
    if pl is not None:
        pl.park()
        pl.exclusive()
    times: dict[tuple[str, str], list[float]] = {}
    docs: dict[tuple[str, str], list[dict[str, Any]]] = {}
    reps_recs: dict[tuple[str, str], list[tuple[float, list[Any]]]] = {}
    dropped: dict[str, int] = {}
    for sh, sd in groups:
        key = (sh.shape_id, sd)
        for r in range(reps + 1):
            seq = mods if r % 2 == 0 else list(reversed(mods))
            for m in seq:
                if m.marked:
                    m.reset()
                t = _rt.time_replay_us(m.graphs[key], flush)
                if not r:
                    continue                      # first round = warm-up
                times.setdefault((m.label, sh.shape_id), []).append(t)
                if not m.marked:
                    continue
                recs, drop = decode.parse_buffer(m.read())
                recs = decode.sample_ctas(recs, extra.get("ctas", "all"))
                dropped[m.label] = dropped.get(m.label, 0) + drop
                doc = A.analyze(recs, names[m.label], roles=roles, t0=t0)
                docs.setdefault((m.label, sh.shape_id), []).append(doc)
                reps_recs.setdefault((m.label, sh.shape_id), []).append((float(doc.get("span_us") or 0.0), recs))
    exclusive = pl.facts.to_dict() if pl is not None else None
    # ---- aggregate
    report: dict[str, Any] = {"schema": "kdev.timeline/2", "shapes": [s.shape_id for s in shapes], "seeds": n_seeds,
                              "replays": reps, "t0": t0, "roles": roles, "files": {}}
    marked_labels = [m.label for m in mods if m.marked]
    for m in mods:
        if m.marked:
            report["files"][m.label] = {"source": m.f.path, "shapes": {}, "notes": m.notes,
                                        "dropped": dropped.get(m.label, 0)}
    for lab in marked_labels:
        plain_lab = next((p for p, ml in plain_of.items() if ml == lab), None)
        for sh in shapes:
            merged = A.merge(docs.get((lab, sh.shape_id), []))
            mt = _rt.median(times.get((lab, sh.shape_id), []))
            pt = _rt.median(times.get((plain_lab, sh.shape_id), [])) if plain_lab else None
            ov = (mt - pt) if pt is not None else None
            mpc = float(merged.get("marks_per_cta") or 0.0)
            timing = {"marked_us": mt, "plain_us": pt, "overhead_us": ov,
                      "ns_per_mark": (ov * 1000.0 / mpc) if (ov is not None and mpc) else None,
                      "marked_reps": len(times.get((lab, sh.shape_id), [])),
                      "plain_reps": len(times.get((plain_lab, sh.shape_id), [])) if plain_lab else 0}
            report["files"][lab]["shapes"][sh.shape_id] = {"timing": timing, "timeline": merged}
    diffs: dict[str, Any] = {}
    if len(marked_labels) >= 2:
        la, lb = marked_labels[0], marked_labels[1]
        for sh in shapes:
            a = report["files"][la]["shapes"][sh.shape_id]
            b = report["files"][lb]["shapes"][sh.shape_id]
            d = A.diff(a["timeline"], b["timeline"])
            ta, tb = a["timing"], b["timing"]
            d["timing"] = {"d_marked_us": tb["marked_us"] - ta["marked_us"],
                           "d_plain_us": (tb["plain_us"] - ta["plain_us"]) if (ta["plain_us"] is not None and
                                                                             tb["plain_us"] is not None) else None}
            diffs[sh.shape_id] = d
        report["diff"] = {"a": la, "b": lb, "shapes": diffs}
    # ---- files
    out_files: dict[str, str] = {}
    out_files["timeline"] = _rt.write_json(rd / "timeline.json", report)
    for lab in marked_labels:
        sfx = "" if len(marked_labels) == 1 else f"_{lab}"
        rs = sorted(reps_recs.get((lab, shapes[0].shape_id), []), key=lambda x: x[0])
        rep_recs = rs[len(rs) // 2][1] if rs else []
        nm = names[lab]
        out_files[f"trace{sfx}"] = _rt.write_json(rd / f"trace{sfx}.json", trace.to_perfetto(rep_recs, nm))
        rows = trace.to_csv_rows([r for r in rep_recs if r.kind != decode.KIND_VALUE], nm)
        out_files[f"csv{sfx}"] = trace.write_csv(str(rd / f"timeline{sfx}.csv"), rows)
        legacy = decode.summarize(rep_recs, nm)
        out_files[f"chart{sfx}"] = _rt.write_json(rd / f"chart{sfx}.json", trace.chart_json(legacy, rows))
    # ---- text
    full: list[str] = []
    lines: list[str] = []
    warnings: list[str] = []
    n_sh = len(shapes)
    per = max(7, 34 // max(1, n_sh * (1 if diffs else len(marked_labels))))
    for sh in shapes:
        for lab in marked_labels:
            e = report["files"][lab]["shapes"][sh.shape_id]
            head = f"[{lab}] {sh.shape_id}" if len(marked_labels) > 1 else sh.shape_id
            full += A.render(e["timeline"], head, max_marks=200, max_phases=200, timing=e["timing"])
            if not diffs:
                lines += A.render(e["timeline"], head, max_marks=max(3, (per - 4) // 2 + 1),
                                  max_phases=max(2, (per - 4) // 2 - 1), timing=e["timing"])
        if diffs:
            d = diffs[sh.shape_id]
            full += A.render_diff(d, sh.shape_id, max_rows=200, timing=d["timing"])
            lines += A.render_diff(d, sh.shape_id, max_rows=max(3, (per - 3) // 2), timing=d["timing"])
    (rd / "timeline.txt").write_text("\n".join(full) + "\n", encoding="utf-8")
    out_files["text"] = str(rd / "timeline.txt")
    for m in mods:
        warnings.extend(f"{m.label}: {n}" for n in m.notes)
    for lab, n in dropped.items():
        if n:
            warnings.append(f"{lab}: {n} marks dropped over all replays (raise --capacity-ctas, or the per-region "
                            "log is full: -DKD_CTA_RECS_LOG2=9)")
    for lab in marked_labels:
        for sh in shapes:
            e = report["files"][lab]["shapes"][sh.shape_id]
            tl, tm = e["timeline"], e["timing"]
            if t0 and tl.get("t0") == "first":
                warnings.append(f"{lab} {sh.shape_id}: t0 mark {t0!r} not recorded; offsets from the first mark")
            res = tl.get("resolution_ns")
            if res and res >= decode.COARSE_NS and tl.get("duration_source") != "clock64":
                warnings.append(f"{lab} {sh.shape_id}: globaltimer tick {res} ns; mark times +-{res} ns")
            if tm.get("overhead_us") is not None and tm.get("plain_us") and \
                    tm["overhead_us"] > max(0.5, 0.02 * tm["plain_us"]):
                warnings.append(f"{lab} {sh.shape_id}: marks cost {tm['overhead_us']:+.2f} us "
                                f"({100 * tm['overhead_us'] / tm['plain_us']:.1f}%): gaps/tails next to marks are "
                                "inflated; drop marks off the hot path or compare like with like (--diff)")
    summary = {"shapes": report["shapes"], "seeds": n_seeds, "reps": reps, "labels": marked_labels,
               "per_label": {lab: {sid: {**e["timing"], "span_us": e["timeline"].get("span_us"),
                                         "end_spread_us": e["timeline"].get("end_spread_us"),
                                         "n_ctas": e["timeline"].get("n_ctas"),
                                         "resolution_ns": e["timeline"].get("resolution_ns")}
                                   for sid, e in report["files"][lab]["shapes"].items()} for lab in marked_labels}}
    if exclusive is not None:
        summary["phase"] = dict(exclusive, mode="phased")
    return Result(tool="timeline", verdict="ok", summary=summary, files=out_files, warnings=warnings, lines=lines,
                  next=["kdev show RUN --section text  (all marks/phases)", "open trace.json in ui.perfetto.dev"])
