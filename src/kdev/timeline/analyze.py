"""Per-replay timeline analysis, merge over replays/seeds, compact tables and A/B diffs. Owner: P6. Pure.

One replay's records -> ``analyze`` -> a dict of numbers (no per-CTA lists, so replays merge key by key):

- ``marks``: per mark, the CTAs' (last) occurrence in us from ``t0`` (the earliest record of the ``--t0`` mark,
  default the earliest record): ``min`` / ``med`` (the CTA at index n//2, like the agents' scripts) / ``max`` /
  ``mean`` / ``spread`` (max - min) / ``n`` CTAs / ``per_cta`` occurrences per CTA.
- ``phases``: ``A->B`` between consecutive records of one track (CTA, warpgroup). Per CTA the phase is summed over
  its occurrences (segments of a loop) and over the CTA's tracks the largest track sum is taken: ``cta_mean`` /
  ``cta_med`` / ``cta_max`` over CTAs, ``slow`` = the slowest CTA's value, ``occ_med`` = median single occurrence,
  ``occ`` = occurrences per CTA.
- ``slowest``: the CTA whose last record is latest (``end_us``, ``lag_us`` behind the median CTA end) and its phase
  sums; ``start_spread_us`` / ``end_spread_us`` across CTAs; ``span_us``.
- ``roles``: tracks grouped by warpgroup role (``--roles 0-1=consumer 2=producer``; default ``wg<N>``): waits =
  phases whose start mark name contains ``wait`` plus value records (``@kdev:acc`` / ``DBGV``-style) whose name
  contains ``wait``; ``wait_mean`` / ``wait_max`` / ``wait_slow`` per track, per-value ``mean`` / ``max`` / ``slow``.
  Value records are clock cycles, converted with the SM clock estimated from the marks (else ``sm_mhz_default``).

``merge`` takes the median of every number over replays (and seeds); ``render`` prints the compact table;
``diff`` compares two merged docs (B - A).
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from kdev.timeline.decode import COARSE_NS, KIND_TIME, KIND_VALUE, SM_MARK, MarkRec, _name, _sm_mhz, detect_resolution

DEFAULT_SM_MHZ = 1980.0


def _stats(vals: list[float]) -> dict[str, float]:
    s = sorted(vals)
    n = len(s)
    return {"min": s[0], "med": s[n // 2], "max": s[-1], "mean": sum(s) / n, "spread": s[-1] - s[0], "n": n}


def _med(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if not n:
        return float("nan")
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def analyze(recs: list[MarkRec], names: dict[Any, str], *, roles: dict[int, str] | None = None,
            t0: str | None = None, sm_mhz_default: float = DEFAULT_SM_MHZ) -> dict[str, Any]:
    roles = roles or {}
    timed = [r for r in recs if r.kind != KIND_VALUE and r.mark != SM_MARK]
    vals = [r for r in recs if r.kind == KIND_VALUE]
    if not timed and not vals:
        return {"empty": True}
    nm = {}

    def name(m: int) -> str:
        if m not in nm:
            nm[m] = _name(names, m)
        return nm[m]

    tracks: dict[tuple[int, int], list[MarkRec]] = {}
    for r in timed:
        tracks.setdefault((r.cta, r.wg), []).append(r)
    for tr in tracks.values():
        tr.sort(key=lambda r: r.t_ns)
    res = detect_resolution(timed)
    mhz = _sm_mhz(tracks)
    use_clk = bool(res and res >= COARSE_NS and mhz)
    out: dict[str, Any] = {"resolution_ns": res, "sm_mhz": mhz,
                           "duration_source": "clock64" if use_clk else "globaltimer"}
    base = None
    if t0:
        c = [r.t_ns for r in timed if name(r.mark) == t0]
        base = min(c) if c else None
    out["t0"] = t0 if base is not None else "first"
    if base is None:
        base = min((r.t_ns for r in timed), default=min((r.t_ns for r in vals), default=0))

    def us(t: int) -> float:
        return (t - base) / 1000.0

    def dur(a: MarkRec, b: MarkRec) -> float:
        if use_clk and a.kind == KIND_TIME and b.kind == KIND_TIME:
            return ((b.clk_lo - a.clk_lo) & 0xFFFFFFFF) / mhz  # type: ignore[operator]
        return (b.t_ns - a.t_ns) / 1000.0

    # ---- marks
    last: dict[str, dict[int, int]] = {}
    cnt: dict[str, Counter[int]] = {}
    start: dict[int, int] = {}
    end: dict[int, int] = {}
    for r in timed:
        n = name(r.mark)
        d = last.setdefault(n, {})
        if r.t_ns >= d.get(r.cta, -1):
            d[r.cta] = r.t_ns
        cnt.setdefault(n, Counter())[r.cta] += 1
        start[r.cta] = min(start.get(r.cta, r.t_ns), r.t_ns)
        end[r.cta] = max(end.get(r.cta, r.t_ns), r.t_ns)
    marks: dict[str, Any] = {}
    for n, d in last.items():
        st = _stats([us(t) for t in d.values()])
        st["per_cta"] = sum(cnt[n].values()) / max(1, len(cnt[n]))
        marks[n] = st
    out["marks"] = marks
    out["mark_order"] = sorted(marks, key=lambda k: (marks[k]["med"], marks[k]["min"]))
    if end:
        ends = sorted(end.values())
        out["n_ctas"] = len(end)
        out["span_us"] = us(ends[-1])
        out["start_spread_us"] = (max(start.values()) - min(start.values())) / 1000.0
        out["end_spread_us"] = (ends[-1] - ends[0]) / 1000.0
        out["end_med_us"] = us(ends[len(ends) // 2])
    out["n_tracks"] = len(tracks)
    out["marks_per_cta"] = len(timed) / max(1, len(end))

    # ---- phases
    tsum: dict[str, dict[tuple[int, int], float]] = {}
    occ: dict[str, list[float]] = {}
    pstart: dict[str, list[float]] = {}
    for key, tr in tracks.items():
        for a, b in zip(tr, tr[1:]):
            p = f"{name(a.mark)}->{name(b.mark)}"
            d = dur(a, b)
            tsum.setdefault(p, {})
            tsum[p][key] = tsum[p].get(key, 0.0) + d
            occ.setdefault(p, []).append(d)
            pstart.setdefault(p, []).append(us(a.t_ns))
    slow = max(end, key=lambda c: (end[c], -c)) if end else None
    phases: dict[str, Any] = {}
    for p, d in tsum.items():
        per_cta: dict[int, float] = {}
        for (cta, _wg), v in d.items():
            per_cta[cta] = max(per_cta.get(cta, 0.0), v)
        st = _stats(list(per_cta.values()))
        phases[p] = {"cta_mean": st["mean"], "cta_med": st["med"], "cta_max": st["max"], "n": st["n"],
                     "slow": per_cta.get(slow, 0.0) if slow is not None else 0.0, "occ_med": _med(occ[p]),
                     "occ": len(occ[p]) / max(1, len(per_cta)), "start_med": _med(pstart[p])}
    out["phases"] = phases
    out["phase_order"] = sorted(phases, key=lambda k: phases[k]["start_med"])
    if slow is not None:
        out["slowest"] = {"cta": slow, "end_us": us(end[slow]), "start_us": us(start[slow]),
                          "lag_us": us(end[slow]) - out["end_med_us"],
                          "phases": {p: phases[p]["slow"] for p in out["phase_order"] if phases[p]["slow"]}}

    # ---- roles and value records
    clk_mhz = mhz or sm_mhz_default
    rl: dict[str, Any] = {}

    def role(wg: int) -> str:
        return roles.get(wg, f"wg{wg}")

    by_role: dict[str, list[tuple[int, int]]] = {}
    for key in tracks:
        by_role.setdefault(role(key[1]), []).append(key)
    vsum: dict[str, dict[str, dict[tuple[int, int], float]]] = {}
    for r in vals:
        vsum.setdefault(role(r.wg), {}).setdefault(name(r.mark), {})
        d = vsum[role(r.wg)][name(r.mark)]
        d[(r.cta, r.wg)] = d.get((r.cta, r.wg), 0.0) + (r.clk_lo & 0xFFFFFFFF) / clk_mhz
    for ro in sorted(set(by_role) | set(vsum)):
        keys = by_role.get(ro, [])
        waits: dict[tuple[int, int], float] = {k: 0.0 for k in keys}
        ph: dict[str, Any] = {}
        for p, d in tsum.items():
            mine = [d[k] for k in keys if k in d]
            if not mine:
                continue
            ph[p] = {"mean": sum(mine) / len(mine), "max": max(mine)}
            if "wait" in p.split("->", 1)[0].lower():
                for k in keys:
                    waits[k] = waits.get(k, 0.0) + d.get(k, 0.0)
        vv: dict[str, Any] = {}
        for vn, d in vsum.get(ro, {}).items():
            xs = list(d.values())
            slow_v = max((v for (c, _w), v in d.items() if c == slow), default=0.0)
            vv[vn] = {"mean": sum(xs) / len(xs), "max": max(xs), "slow": slow_v, "n": len(xs)}
            if "wait" in vn.lower():
                for k, v in d.items():
                    waits[k] = waits.get(k, 0.0) + v
        spans = [(tracks[k][-1].t_ns - tracks[k][0].t_ns) / 1000.0 for k in keys if len(tracks[k]) > 1]
        w = list(waits.values())
        rl[ro] = {"tracks": len(keys) or len({k for d in vsum.get(ro, {}).values() for k in d}),
                  "span_mean": sum(spans) / len(spans) if spans else 0.0,
                  "wait_mean": sum(w) / len(w) if w else 0.0, "wait_max": max(w) if w else 0.0,
                  "wait_slow": max((v for (c, _w), v in waits.items() if c == slow), default=0.0),
                  "phases": ph, "values": vv}
    out["roles"] = rl
    return out


def _merge(vals: list[Any], key: str = "") -> Any:
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    if all(isinstance(v, dict) for v in vals):
        keys: list[str] = []
        for v in vals:
            for k in v:
                if k not in keys:
                    keys.append(k)
        return {k: _merge([v.get(k) for v in vals], k) for k in keys}
    if key == "cta" or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
        if all(isinstance(v, list) for v in vals):
            return max(vals, key=len)
        return Counter(map(lambda x: x if not isinstance(x, list) else tuple(x), vals)).most_common(1)[0][0]
    return _med([float(v) for v in vals])


def merge(docs: list[dict[str, Any]]) -> dict[str, Any]:
    """Median of every number over replays; orders recomputed from the merged medians."""
    docs = [d for d in docs if d and not d.get("empty")]
    if not docs:
        return {"empty": True, "replays": 0}
    m = _merge(docs)
    m["replays"] = len(docs)
    mk = m.get("marks") or {}
    m["mark_order"] = sorted(mk, key=lambda k: (mk[k]["med"], mk[k]["min"]))
    ph = m.get("phases") or {}
    m["phase_order"] = sorted(ph, key=lambda k: ph[k]["start_med"])
    sl = m.get("slowest") or {}
    if sl:
        sl["ctas"] = dict(Counter(d["slowest"]["cta"] for d in docs if d.get("slowest")).most_common(3))
    return m


def _f(x: Any) -> str:
    if x is None:
        return "-"
    x = float(x)
    return f"{x:.1f}" if abs(x) >= 100 else f"{x:.2f}"


def _d(x: Any) -> str:
    if x is None:
        return "-"
    return f"{float(x):+.2f}"


def render(m: dict[str, Any], head: str, *, max_marks: int = 8, max_phases: int = 6,
           timing: dict[str, Any] | None = None) -> list[str]:
    """Compact text for one (file, shape)."""
    from kdev.floor._text import table
    if m.get("empty"):
        return [f"{head}: no marks recorded"]
    t = timing or {}
    tick = m.get("resolution_ns")
    parts = [f"{head}: {int(m.get('replays', 0))} replays"]
    if t.get("marked_us") is not None:
        parts.append(f"replay {_f(t['marked_us'])} us")
    if t.get("plain_us") is not None:
        ov = t.get("overhead_us")
        parts.append(f"plain {_f(t['plain_us'])}, mark cost {_d(ov)} us"
                     + (f" ({100 * ov / t['plain_us']:.1f}%, {_f(t.get('ns_per_mark'))} ns/mark)"
                        if ov is not None and t.get("plain_us") else ""))
    parts.append(f"span {_f(m.get('span_us'))} us, tick {tick} ns, t0={m.get('t0')}")
    lines = [", ".join(parts)]
    mk = m.get("marks") or {}
    rows = [[n, _f(mk[n]["min"]), _f(mk[n]["med"]), _f(mk[n]["max"]), _f(mk[n]["spread"]),
             f"{mk[n].get('per_cta', 1):.1f}"] for n in m.get("mark_order", [])]
    lines += ["  " + x for x in table(["mark", "min", "med", "max", "spread", "n/CTA"], rows, max_rows=max_marks)]
    ph = m.get("phases") or {}
    order = sorted(m.get("phase_order", []), key=lambda k: -ph[k]["cta_mean"])[:max_phases]
    order = [p for p in m.get("phase_order", []) if p in order]
    rows = [[p, _f(ph[p]["cta_mean"]), _f(ph[p]["cta_max"]), _f(ph[p]["slow"]), _f(ph[p]["occ_med"]),
             f"{ph[p]['occ']:.1f}"] for p in order]
    if rows:
        lines += ["  " + x for x in table(["phase (sum/CTA, top by mean)", "mean", "max", "slowest", "1x med",
                                           "n/CTA"], rows, max_rows=max_phases)]
    sl = m.get("slowest") or {}
    if sl:
        avg = {p: ph[p]["cta_mean"] for p in ph}
        top = sorted(((v - avg.get(p, 0.0), p) for p, v in (sl.get("phases") or {}).items()), reverse=True)[:3]
        lines.append(f"  CTAs {int(m.get('n_ctas', 0))}: start spread {_f(m.get('start_spread_us'))}, end spread "
                     f"{_f(m.get('end_spread_us'))} us; slowest CTA {sl.get('cta')} ends {_f(sl.get('end_us'))} "
                     f"({_d(sl.get('lag_us'))} vs med end)"
                     + ("; vs mean CTA: " + ", ".join(f"{p} {_d(dv)}" for dv, p in top if dv > 0.005) if top else ""))
    rl = m.get("roles") or {}
    if rl and (len(rl) > 1 or any(r.get("values") or r.get("wait_max") for r in rl.values())):
        bits = []
        for ro, r in rl.items():
            b = f"{ro} x{int(r.get('tracks') or 0)} wait {_f(r.get('wait_mean'))}/{_f(r.get('wait_max'))}"
            vv = r.get("values") or {}
            if vv:
                b += " [" + ", ".join(f"{k} {_f(v['mean'])}/{_f(v['max'])}" for k, v in list(vv.items())[:3]) + "]"
            bits.append(b)
        lines.append("  roles (wait us mean/max per track): " + " | ".join(bits))
    return lines


def diff(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """B - A of the merged docs: marks (med, max), phases (cta_mean, cta_max, slow), spreads, role waits."""
    out: dict[str, Any] = {"marks": {}, "phases": {}, "roles": {}}
    for k in ("span_us", "start_spread_us", "end_spread_us", "end_med_us"):
        if a.get(k) is not None and b.get(k) is not None:
            out[k] = b[k] - a[k]
    ma, mb = a.get("marks") or {}, b.get("marks") or {}
    for n in [x for x in (a.get("mark_order") or []) if x in mb] + [x for x in (b.get("mark_order") or []) if x not in ma]:
        if n in ma and n in mb:
            out["marks"][n] = {"a_med": ma[n]["med"], "b_med": mb[n]["med"], "d_med": mb[n]["med"] - ma[n]["med"],
                               "d_max": mb[n]["max"] - ma[n]["max"], "d_spread": mb[n]["spread"] - ma[n]["spread"]}
        else:
            out["marks"][n] = {"only": "A" if n in ma else "B"}
    pa, pb = a.get("phases") or {}, b.get("phases") or {}
    for p in list(dict.fromkeys(list(a.get("phase_order") or []) + list(b.get("phase_order") or []))):
        if p in pa and p in pb:
            out["phases"][p] = {"a_mean": pa[p]["cta_mean"], "b_mean": pb[p]["cta_mean"],
                                "d_mean": pb[p]["cta_mean"] - pa[p]["cta_mean"],
                                "d_max": pb[p]["cta_max"] - pa[p]["cta_max"], "d_slow": pb[p]["slow"] - pa[p]["slow"]}
        else:
            out["phases"][p] = {"only": "A" if p in pa else "B"}
    ra, rb = a.get("roles") or {}, b.get("roles") or {}
    for ro in ra:
        if ro in rb:
            out["roles"][ro] = {"d_wait_mean": rb[ro]["wait_mean"] - ra[ro]["wait_mean"],
                                "d_wait_max": rb[ro]["wait_max"] - ra[ro]["wait_max"]}
    return out


def render_diff(d: dict[str, Any], head: str, *, max_rows: int = 8, timing: dict[str, Any] | None = None) -> list[str]:
    from kdev.floor._text import table
    t = timing or {}
    bits = [f"{head} B-A:"]
    if t.get("d_plain_us") is not None:
        bits.append(f"plain replay {_d(t['d_plain_us'])} us")
    if t.get("d_marked_us") is not None:
        bits.append(f"marked {_d(t['d_marked_us'])} us")
    bits.append(f"span {_d(d.get('span_us'))}, end spread {_d(d.get('end_spread_us'))} us")
    lines = [" ".join(bits[:1]) + " " + ", ".join(bits[1:])]
    mk = d.get("marks") or {}
    rows = [[n, _f(v.get("a_med")), _f(v.get("b_med")), _d(v.get("d_med")), _d(v.get("d_max"))]
            for n, v in mk.items() if "only" not in v]
    lines += ["  " + x for x in table(["mark", "A med", "B med", "dmed", "dmax"], rows, max_rows=max_rows)]
    ph = d.get("phases") or {}
    prow = sorted(((abs(v.get("d_mean", 0.0)), p, v) for p, v in ph.items() if "only" not in v), reverse=True)
    rows = [[p, _f(v["a_mean"]), _f(v["b_mean"]), _d(v["d_mean"]), _d(v["d_slow"])] for _, p, v in prow[:max_rows]]
    if rows:
        lines += ["  " + x for x in table(["phase (sum/CTA, top |d|)", "A", "B", "dmean", "dslowest"], rows,
                                         max_rows=max_rows)]
    only = [f"{p} ({v['only']} only)" for p, v in list(mk.items()) + list(ph.items()) if "only" in v]
    if only:
        lines.append("  only in one file: " + ", ".join(only[:6]) + (" ..." if len(only) > 6 else ""))
    return lines
