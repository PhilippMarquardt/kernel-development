"""Excess budget from an ab Result (pure; no torch). Method of xs_ (muse_bf16_proj xs/EXCESS_BUDGET.md):

- every variant removes one component of the base file (wrong results by design, timing only);
- saved(component) = base RAW us - variant RAW us per shape (+ = the component costs time), judge-mode median
  RAW (fast mode: net us);
- ``stream`` (consumers skip all per-stage work, same bytes) splits the excess into compute = base - stream and
  skeleton = stream - overhead - floor;
- excess = base RAW - overhead - floor (floor from ``--floor SHAPE=US``: the pure-read floor), remainder =
  excess - sum(singles). Single ablations are marginals and overlap on one critical path, so their sum is NOT
  the excess (non-additivity is reported, not hidden);
- a control variant (all hooks compiled in, nothing removed) gives the noise band: |saved| below
  max(NOISE_US, |control - base|) is marked ``~``.
"""
from __future__ import annotations

import math
from typing import Any

NOISE_US = 0.15
DEFAULT_OVERHEAD_US = 4.0
STREAM = "stream"
CONTROL = "control"


def _num(x: Any) -> float | None:
    return float(x) if isinstance(x, (int, float)) and math.isfinite(x) else None


def _us(p: dict[str, Any]) -> float | None:
    v = _num(p.get("median_raw_us"))
    return v if v is not None else _num(p.get("median_us"))


def budget(rows: list[dict[str, Any]], base: str, comps: dict[str, str], floors: dict[str, float] | None = None,
           noise_us: float = NOISE_US) -> dict[str, Any]:
    """rows: ab Result.shapes; base: label of the base file; comps: {label: component}. Returns
    {components, shapes: [{shape_id, weight, base_us, overhead_us, floor_us, excess_us, saved: {comp: us},
    noise: {comp: bool}, sum_singles, remainder_us, compute_us, skeleton_us}], totals: {comp: weighted us}}."""
    floors = floors or {}
    names = [c for c in comps.values() if c != CONTROL]
    singles = [c for c in names if c != STREAM]
    out_rows = []
    totals = {c: 0.0 for c in names}
    for r in rows:
        pf = {p["label"]: p for p in r.get("per_file") or []}
        if base not in pf:
            continue
        b = _us(pf[base])
        if b is None:
            continue
        raw = _num(pf[base].get("median_raw_us")) is not None
        ovh = _num(pf[base].get("overhead_us"))
        ovh = (ovh if ovh is not None else DEFAULT_OVERHEAD_US) if raw else 0.0
        ctl = next((lb for lb, c in comps.items() if c == CONTROL), None)
        band = noise_us
        if ctl and ctl in pf and _us(pf[ctl]) is not None:
            band = max(noise_us, abs(_us(pf[ctl]) - b))
        saved: dict[str, float | None] = {}
        for lb, c in comps.items():
            if c == CONTROL:
                continue
            v = _us(pf[lb]) if lb in pf else None
            saved[c] = None if v is None else b - v
        w = float(r.get("weight", 1.0) or 1.0)
        for c, s in saved.items():
            if s is not None:
                totals[c] += w * s
        fl = _num(floors.get(r["shape_id"]))
        excess = None if fl is None else b - ovh - fl
        ssum = sum(s for c, s in saved.items() if c in singles and s is not None)
        st = saved.get(STREAM)
        out_rows.append({
            "shape_id": r["shape_id"], "weight": w, "base_us": b, "overhead_us": ovh, "floor_us": fl,
            "excess_us": excess, "saved": saved, "noise": {c: s is not None and abs(s) < band for c, s in saved.items()},
            "noise_band_us": band, "sum_singles": ssum,
            "remainder_us": None if excess is None else excess - ssum,
            "compute_us": st, "skeleton_us": None if (st is None or fl is None) else (b - st) - ovh - fl,
            "verdicts": {comps.get(rt.get("num"), rt.get("num")): rt.get("verdict") for rt in r.get("ratios") or []},
        })
    return {"schema": "kdev.ablate_budget/1", "base": base, "components": names, "labels": comps,
            "shapes": out_rows, "totals": totals, "noise_us": noise_us}


def _c(x: float | None, noise: bool = False, w: int = 7) -> str:
    if x is None:
        return "-".rjust(w)
    return (f"{x:+.2f}" + ("~" if noise else "")).rjust(w)


def render(b: dict[str, Any], max_rows: int = 30) -> list[str]:
    names = b["components"]
    singles = [c for c in names if c != STREAM]
    has_ex = any(r["excess_us"] is not None for r in b["shapes"])
    hdr = "shape".ljust(16) + f"{'base':>7}" + (f"{'excess':>8}" if has_ex else "") \
        + "".join(f"{c[:8]:>9}" for c in names) + f"{'sum':>8}" + (f"{'remain':>8}" if has_ex else "")
    if STREAM in names and has_ex:
        hdr += f"{'skel':>7}"
    lines = [hdr]
    for r in b["shapes"][:max_rows]:
        s = r["shape_id"][:15].ljust(16) + f"{r['base_us']:7.2f}" + (_c(r["excess_us"], w=8) if has_ex else "")
        s += "".join(_c(r["saved"].get(c), r["noise"].get(c, False), 9) for c in names)
        s += _c(r["sum_singles"], w=8) + (_c(r["remainder_us"], w=8) if has_ex else "")
        if STREAM in names and has_ex:
            s += _c(r["skeleton_us"])
        lines.append(s)
    if len(b["shapes"]) > max_rows:
        lines.append(f"... {len(b['shapes']) - max_rows} more rows in budget.md")
    tot = b["totals"]
    rank = sorted(singles, key=lambda c: -tot.get(c, 0.0))
    if rank:
        lines.append("ranking (weight x us saved, summed): " + "  ".join(f"{c} {tot[c]:+.2f}" for c in rank))
    lines.append(f"us saved by removing the component (+ = it costs time); ~ = within noise "
                 f"(max({b['noise_us']} us, |control - base|)); singles are marginals, they need not add up")
    return lines


def markdown(b: dict[str, Any], title: str = "") -> str:
    names = b["components"]
    out = [f"# Excess budget{(' - ' + title) if title else ''}", "",
           "us saved by removing the component (+ = the component costs time); `~` = within the noise band. "
           "Values: median RAW us (judge mode). excess = base - overhead - floor; remainder = excess - sum of "
           "singles (singles are marginals on one critical path: non-additive).", "",
           "| shape | weight | base | overhead | floor | excess | " + " | ".join(names)
           + " | sum singles | remainder | skeleton |", "|---" * (9 + len(names)) + "|"]
    f = lambda x, n=False: "-" if x is None else f"{x:+.2f}{'~' if n else ''}"  # noqa: E731
    for r in b["shapes"]:
        out.append(f"| {r['shape_id']} | {r['weight']:g} | {r['base_us']:.2f} | {r['overhead_us']:.2f} | "
                   f"{'-' if r['floor_us'] is None else format(r['floor_us'], '.2f')} | {f(r['excess_us'])} | "
                   + " | ".join(f(r["saved"].get(c), r["noise"].get(c, False)) for c in names)
                   + f" | {f(r['sum_singles'])} | {f(r['remainder_us'])} | {f(r['skeleton_us'])} |")
    out += ["", "## Ranking (weight x us saved, summed over shapes)", "", "| component | weighted us |", "|---|---|"]
    for c in sorted(names, key=lambda c: -b["totals"].get(c, 0.0)):
        out.append(f"| {c} | {b['totals'][c]:+.3f} |")
    out += ["", "labels: " + ", ".join(f"{lb}={c}" for lb, c in b["labels"].items()) + f", base={b['base']}", ""]
    return "\n".join(out)
