"""Decode a mark buffer into per-CTA phases. Owner: P6. Pure (bytes in, tables out).

Buffer layout: see marks.cuh (16-byte header {count, capacity, aux, flags} + 16-byte records; layouts
0 slots / 1 append log / 2 per-(CTA, warpgroup) log; record kinds time / info / value).
A *track* is one (CTA, warpgroup); a *phase* ``A->B`` is the time between consecutive marks of a track.
Mark times (``marks``) are offsets from the earliest mark of the replay (the kernel start as seen by the
marks), so "mark 'reds' p50 2.1 us" reads like the ad hoc fctl/gt3 timelines.

Resolution: %globaltimer ticks every 32 or 256 ns on H200 (kernel dependent). ``detect_resolution`` reports
the tick (gcd of distinct timestamp deltas). When the tick is coarse (>= 128 ns) and an SM clock rate can be
estimated from the records themselves (clock_lo delta / globaltimer delta over long tracks), phase durations
use the clock deltas (same SM within a track) and ``duration_source`` says ``clock64``.
"""
from __future__ import annotations

import math
import struct
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

HDR = struct.Struct("<IIII")
REC = struct.Struct("<QIHBB")
SM_MARK = 255
COARSE_NS = 128
KIND_TIME, KIND_INFO, KIND_VALUE = 0, 1, 2


@dataclass
class MarkRec:
    t_ns: int
    clk_lo: int               # clock_lo, or the info word (KIND_INFO) / the value (KIND_VALUE)
    cta: int
    wg: int
    mark: int
    kind: int = KIND_TIME


def _split_wg(b: int) -> tuple[int, int]:
    """wg byte -> (warpgroup, kind): bit 7 value record, bit 6 info record (marks.cuh)."""
    if b & 0x80:
        return b & 0x3F, KIND_VALUE
    if b & 0x40:
        return b & 0x3F, KIND_INFO
    return b & 0x3F, KIND_TIME


def _wg_byte(r: MarkRec) -> int:
    return (r.wg & 0x3F) | (0x80 if r.kind == KIND_VALUE else 0x40 if r.kind == KIND_INFO else 0)


@dataclass
class PhaseStats:
    name: str                 # 'markA->markB'
    p10_ns: float
    p50_ns: float
    p90_ns: float
    n: int
    min_ns: float = 0.0
    max_ns: float = 0.0


@dataclass
class MarkStats:
    name: str
    mark: int
    p10_ns: float
    p50_ns: float
    p90_ns: float
    min_ns: float
    max_ns: float
    n: int


@dataclass
class TimelineSummary:
    phases: list[PhaseStats] = field(default_factory=list)
    cta_start_spread_ns: float = 0.0
    cta_end_spread_ns: float = 0.0
    critical_cta: int | None = None
    resolution_ns: int | None = None     # detected globaltimer granularity (32 or 256 on H200)
    dropped: int = 0
    mark_overhead_us: float | None = None
    marks: list[MarkStats] = field(default_factory=list)
    span_ns: float = 0.0                 # earliest mark .. latest mark
    n_ctas: int = 0
    n_tracks: int = 0
    n_sms: int = 0                       # distinct SMs seen (KD_MARK_SM records)
    sm_mhz: float | None = None          # SM clock estimated from clock_lo / globaltimer
    duration_source: str = "globaltimer"  # globaltimer | clock64
    critical_path: list[dict[str, Any]] = field(default_factory=list)  # phases of the critical CTA's last track
    replays: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_header(data: bytes) -> dict[str, int]:
    if len(data) < HDR.size:
        raise ValueError(f"mark buffer too short ({len(data)} bytes)")
    count, cap, aux, flags = HDR.unpack_from(data, 0)
    layout = flags & 3
    return {"count": count, "capacity": cap, "smid_bits": aux, "aux": aux, "flags": flags,
            "valid": bool(flags & 0x80000000), "layout": layout, "max_wg": (flags >> 16) & 0xFF,
            "max_marks": (flags >> 8) & 0xFF, "cta_recs": (1 << ((flags >> 2) & 0x3F)) if layout == 2 else 0}


def _recs_numpy(body: bytes, idx_cta: Any = None) -> list[MarkRec] | None:
    try:
        import numpy as np
    except ImportError:
        return None
    dt = np.dtype([("t", "<u8"), ("clk", "<u4"), ("cta", "<u2"), ("wg", "u1"), ("id", "u1")])
    a = np.frombuffer(body, dtype=dt, count=len(body) // dt.itemsize)
    keep = np.nonzero(a["t"])[0]
    sub = a[keep]
    ctas = sub["cta"].astype(np.int64)
    if idx_cta is not None:
        full = idx_cta(keep)
        ctas = np.where((full & 0xFFFF) == ctas, full, ctas)
    out = []
    for t, clk, cta, wgb, mid in zip(sub["t"].tolist(), sub["clk"].tolist(), ctas.tolist(), sub["wg"].tolist(),
                                     sub["id"].tolist()):
        wg, kind = _split_wg(wgb)
        out.append(MarkRec(t, clk, cta, wg, mid, kind))
    return out


def _recs_struct(body: bytes) -> list[MarkRec]:
    out = []
    for t, clk, cta16, wgb, mark in REC.iter_unpack(body):
        if t:
            wg, kind = _split_wg(wgb)
            out.append(MarkRec(t, clk, cta16, wg, mark, kind))
    return out


def parse_buffer(data: bytes) -> tuple[list[MarkRec], int]:
    """(records, dropped) from the raw buffer layout in marks.cuh."""
    h = parse_header(data)
    avail = (len(data) - HDR.size) // REC.size
    cap = min(h["capacity"], avail) if h["capacity"] else avail
    body = bytes(memoryview(data)[HDR.size: HDR.size + cap * REC.size])
    recs: list[MarkRec] = []
    if h["valid"] and h["layout"] == 2:
        R = max(2, h["cta_recs"])
        mw = max(1, h["max_wg"])
        for reg in range(cap // R):
            off = reg * R * REC.size
            n = struct.unpack_from("<I", body, off)[0]
            if n == 0:
                continue
            n = min(n, R - 1)
            seg = body[off + REC.size: off + (n + 1) * REC.size]
            got = _recs_struct(seg) if n < 64 else (_recs_numpy(seg) or _recs_struct(seg))
            for r in got:
                r.cta = reg // mw
            recs.extend(got)
        return recs, h["count"]
    if h["valid"] and h["layout"] == 0:
        per_cta = max(1, h["max_wg"]) * (h["max_marks"] + 1)
        got = _recs_numpy(body, lambda idx: idx // per_cta)
        if got is not None:
            return got, h["count"]
        for idx, (t, clk, cta16, wgb, mark) in enumerate(REC.iter_unpack(body)):
            if t == 0:
                continue
            cta = idx // per_cta
            if (cta & 0xFFFF) != cta16:
                cta = cta16
            wg, kind = _split_wg(wgb)
            recs.append(MarkRec(t, clk, cta, wg, mark, kind))
        return recs, h["count"]
    n = min(h["count"], cap)
    return _recs_struct(body[: n * REC.size]), max(0, h["count"] - h["capacity"])


def build_buffer(recs: Iterable[MarkRec], capacity: int | None = None, layout: int = 1, max_wg: int = 8,
                 max_marks: int = 32, cta_recs_log2: int = 4) -> bytes:
    """Inverse of parse_buffer (tests, fixtures): a log, slot or per-(CTA, warpgroup) log buffer image."""
    recs = list(recs)
    flags = 0x80000000 | (max_wg << 16) | (max_marks << 8) | (cta_recs_log2 << 2) | layout

    def pack(r: MarkRec) -> bytes:
        return REC.pack(r.t_ns, r.clk_lo & 0xFFFFFFFF, r.cta & 0xFFFF, _wg_byte(r), r.mark)

    if layout == 1:
        cap = capacity if capacity is not None else len(recs)
        body = b"".join(pack(r) for r in recs[:cap])
        body += bytes(REC.size * (cap - min(cap, len(recs))))
        return HDR.pack(len(recs), cap, 0, flags) + body
    if layout == 2:
        R = 1 << cta_recs_log2
        n_cta = max((r.cta for r in recs), default=-1) + 1
        cap = capacity if capacity is not None else n_cta * max_wg * R
        slots = bytearray(REC.size * cap)
        counts: dict[int, int] = {}
        dropped = 0
        for r in recs:
            reg = r.cta * max_wg + r.wg
            n = counts.get(reg, 0)
            if r.wg >= max_wg or (reg + 1) * R > cap or n + 1 >= R:
                dropped += 1
                continue
            counts[reg] = n + 1
            slots[(reg * R + 1 + n) * REC.size: (reg * R + 2 + n) * REC.size] = pack(r)
        for reg, n in counts.items():
            struct.pack_into("<I", slots, reg * R * REC.size, n)
        return HDR.pack(dropped, cap, R, flags) + bytes(slots)
    per_cta = max_wg * (max_marks + 1)
    n_cta = max((r.cta for r in recs), default=-1) + 1
    cap = capacity if capacity is not None else n_cta * per_cta
    slots = bytearray(REC.size * cap)
    dropped = 0
    for r in recs:
        mid = max_marks if r.mark == SM_MARK else r.mark
        s = (r.cta * max_wg + r.wg) * (max_marks + 1) + mid
        if r.wg >= max_wg or mid > max_marks or s >= cap:
            dropped += 1
            continue
        slots[s * REC.size:(s + 1) * REC.size] = pack(r)
    return HDR.pack(dropped, cap, 0, flags) + bytes(slots)


def used_bytes(data: bytes, margin_ctas: int = 0) -> int:
    """Bytes of the buffer prefix that hold records (header + up to the last CTA region/slot in use, plus
    ``margin_ctas``): the host reads only this prefix per replay once a warm-up call sized it."""
    h = parse_header(data)
    recs, _ = parse_buffer(data)
    if not recs:
        return len(data)
    if h["layout"] == 1:
        return HDR.size + REC.size * min(h["count"], h["capacity"])
    top = max(r.cta for r in recs) + 1 + margin_ctas
    if h["layout"] == 2:
        n = top * max(1, h["max_wg"]) * max(2, h["cta_recs"])
    else:
        n = top * max(1, h["max_wg"]) * (h["max_marks"] + 1)
    return HDR.size + REC.size * min(n, h["capacity"] or n)


def percentile(xs: list[float], q: float) -> float:
    """Linear-interpolated percentile (q in 0..100) of a non-empty list."""
    s = sorted(xs)
    if not s:
        return float("nan")
    if len(s) == 1:
        return float(s[0])
    k = (len(s) - 1) * q / 100.0
    lo = math.floor(k)
    hi = min(lo + 1, len(s) - 1)
    return float(s[lo] + (s[hi] - s[lo]) * (k - lo))


def detect_resolution(recs: list[MarkRec]) -> int | None:
    """globaltimer tick in ns (gcd of deltas between distinct timestamps), None with < 3 distinct stamps."""
    vals = sorted({r.t_ns for r in recs if r.mark != SM_MARK})
    if len(vals) < 3:
        return None
    g = 0
    for a, b in zip(vals, vals[1:]):
        g = math.gcd(g, b - a)
        if g == 1:
            break
    return g or None


def _name(names: dict[Any, str], m: int) -> str:
    """Mark name; ``names`` may be keyed by int or by str (JSON round trip)."""
    if m in names:
        return names[m]
    return names.get(str(m), f"m{m}")


def _sm_mhz(tracks: dict[tuple[int, int], list[MarkRec]], min_span_ns: int = 1500) -> float | None:
    """Pooled clock rate (sum of clock deltas / sum of globaltimer deltas over long tracks): the timer's
    quantisation error averages out across tracks."""
    sc = st = 0
    for tr0 in tracks.values():
        tr = [r for r in tr0 if r.kind == KIND_TIME]
        if len(tr) < 2:
            continue
        dt = tr[-1].t_ns - tr[0].t_ns
        if dt < min_span_ns:
            continue
        sc += (tr[-1].clk_lo - tr[0].clk_lo) & 0xFFFFFFFF
        st += dt
    return sc / st * 1000.0 if st else None


def summarize(recs: list[MarkRec], names: dict[int, str]) -> TimelineSummary:
    """Per-phase / per-mark percentiles across tracks, CTA spreads, critical CTA, resolution (one replay)."""
    s = TimelineSummary()
    sm_of: dict[int, int] = {}
    tracks: dict[tuple[int, int], list[MarkRec]] = {}
    for r in recs:
        if r.mark == SM_MARK:
            sm_of[r.cta] = r.clk_lo
            continue
        if r.kind == KIND_VALUE:
            continue
        tracks.setdefault((r.cta, r.wg), []).append(r)
    if not tracks:
        return s
    for tr in tracks.values():
        c0 = min(tr, key=lambda r: r.t_ns).clk_lo
        tr.sort(key=lambda r, c0=c0: (r.t_ns, (r.clk_lo - c0) & 0xFFFFFFFF))
    allr = [r for tr in tracks.values() for r in tr]
    t0 = min(r.t_ns for r in allr)
    s.resolution_ns = detect_resolution(allr)
    s.sm_mhz = _sm_mhz(tracks)
    use_clk = bool(s.resolution_ns and s.resolution_ns >= COARSE_NS and s.sm_mhz)
    s.duration_source = "clock64" if use_clk else "globaltimer"
    s.n_tracks = len(tracks)
    s.n_sms = len(set(sm_of.values()))
    # marks: offsets from t0
    per_mark: dict[int, list[float]] = {}
    for r in allr:
        per_mark.setdefault(r.mark, []).append(float(r.t_ns - t0))
    order = sorted(per_mark, key=lambda m: percentile(per_mark[m], 50))
    s.marks = [MarkStats(_name(names, m), m, percentile(v, 10), percentile(v, 50), percentile(v, 90), min(v), max(v),
                         len(v)) for m in order for v in [per_mark[m]]]

    def dur(a: MarkRec, b: MarkRec) -> float:
        if use_clk and a.kind == KIND_TIME and b.kind == KIND_TIME:
            return ((b.clk_lo - a.clk_lo) & 0xFFFFFFFF) * 1000.0 / s.sm_mhz  # type: ignore[operator]
        return float(b.t_ns - a.t_ns)

    per_phase: dict[str, list[float]] = {}
    phase_start: dict[str, list[float]] = {}
    for tr in tracks.values():
        for a, b in zip(tr, tr[1:]):
            nm = f"{_name(names, a.mark)}->{_name(names, b.mark)}"
            per_phase.setdefault(nm, []).append(dur(a, b))
            phase_start.setdefault(nm, []).append(float(a.t_ns - t0))
    porder = sorted(per_phase, key=lambda k: percentile(phase_start[k], 50))
    s.phases = [PhaseStats(k, percentile(v, 10), percentile(v, 50), percentile(v, 90), len(v), min(v), max(v))
                for k in porder for v in [per_phase[k]]]
    first: dict[int, int] = {}
    last: dict[int, int] = {}
    last_track: dict[int, tuple[int, int]] = {}
    for (cta, wg), tr in tracks.items():
        first[cta] = min(first.get(cta, tr[0].t_ns), tr[0].t_ns)
        if tr[-1].t_ns >= last.get(cta, -1):
            last[cta] = tr[-1].t_ns
            last_track[cta] = (cta, wg)
    s.n_ctas = len(first)
    s.cta_start_spread_ns = float(max(first.values()) - min(first.values()))
    s.cta_end_spread_ns = float(max(last.values()) - min(last.values()))
    s.span_ns = float(max(last.values()) - t0)
    s.critical_cta = max(last, key=lambda c: (last[c], -c))
    tr = tracks[last_track[s.critical_cta]]
    s.critical_path = [{"phase": f"{_name(names, a.mark)}->{_name(names, b.mark)}", "start_ns": float(a.t_ns - t0),
                        "dur_ns": dur(a, b), "wg": a.wg} for a, b in zip(tr, tr[1:])]
    if tr:
        s.critical_path.insert(0, {"phase": f"start->{_name(names, tr[0].mark)}", "start_ns": 0.0,
                                   "dur_ns": float(tr[0].t_ns - t0), "wg": tr[0].wg})
    return s


def merge_summaries(sums: list[TimelineSummary]) -> TimelineSummary:
    """Median over replays of every per-phase / per-mark statistic (phases matched by name)."""
    sums = [x for x in sums if x.n_tracks]
    if not sums:
        return TimelineSummary(replays=0)
    if len(sums) == 1:
        return sums[0]

    def med(xs: list[float]) -> float:
        return percentile(xs, 50)

    out = TimelineSummary(replays=len(sums))
    names_order: list[str] = []
    by: dict[str, list[PhaseStats]] = {}
    for x in sums:
        for p in x.phases:
            if p.name not in by:
                names_order.append(p.name)
            by.setdefault(p.name, []).append(p)
    out.phases = [PhaseStats(k, med([p.p10_ns for p in v]), med([p.p50_ns for p in v]), med([p.p90_ns for p in v]),
                             int(med([p.n for p in v])), med([p.min_ns for p in v]), med([p.max_ns for p in v]))
                  for k in names_order for v in [by[k]]]
    mo: list[str] = []
    mb: dict[str, list[MarkStats]] = {}
    for x in sums:
        for m in x.marks:
            if m.name not in mb:
                mo.append(m.name)
            mb.setdefault(m.name, []).append(m)
    out.marks = [MarkStats(k, v[0].mark, med([m.p10_ns for m in v]), med([m.p50_ns for m in v]),
                           med([m.p90_ns for m in v]), med([m.min_ns for m in v]), med([m.max_ns for m in v]),
                           int(med([m.n for m in v]))) for k in mo for v in [mb[k]]]
    out.marks.sort(key=lambda m: m.p50_ns)
    out.cta_start_spread_ns = med([x.cta_start_spread_ns for x in sums])
    out.cta_end_spread_ns = med([x.cta_end_spread_ns for x in sums])
    out.span_ns = med([x.span_ns for x in sums])
    crit = Counter(x.critical_cta for x in sums).most_common(1)[0][0]
    out.critical_cta = crit
    pick = next(x for x in sums if x.critical_cta == crit)
    out.critical_path = pick.critical_path
    res = [x.resolution_ns for x in sums if x.resolution_ns]
    out.resolution_ns = min(res) if res else None
    mhz = [x.sm_mhz for x in sums if x.sm_mhz]
    out.sm_mhz = med(mhz) if mhz else None
    out.duration_source = Counter(x.duration_source for x in sums).most_common(1)[0][0]
    out.dropped = sum(x.dropped for x in sums)
    out.n_ctas = max(x.n_ctas for x in sums)
    out.n_tracks = max(x.n_tracks for x in sums)
    out.n_sms = max(x.n_sms for x in sums)
    return out


def sample_ctas(recs: list[MarkRec], spec: str) -> list[MarkRec]:
    """'all' or 'sample:N' (N CTAs evenly spread over the CTA ids, always incl. the critical one)."""
    if not spec or spec == "all":
        return recs
    if not spec.startswith("sample:"):
        raise ValueError(f"--ctas must be all|sample:N, got {spec!r}")
    n = int(spec.split(":", 1)[1])
    ctas = sorted({r.cta for r in recs})
    if n >= len(ctas):
        return recs
    last: dict[int, int] = {}
    for r in recs:
        if r.mark != SM_MARK:
            last[r.cta] = max(last.get(r.cta, 0), r.t_ns)
    keep = {ctas[int(i * (len(ctas) - 1) / max(1, n - 1))] for i in range(n)}
    if last:
        keep.add(max(last, key=lambda c: last[c]))
    return [r for r in recs if r.cta in keep]
