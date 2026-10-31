"""Roofline floor (judge-style peak constants and measured achievable bandwidth). Owner: P6. Pure.

Per shape: memory time from bytes (peak ``dram_tbps`` = the judge's floor constant, and the measured
``stream_tbps`` = achievable), compute time from flops at the dtype's tensor peak, and MUFU time from SFU op
counts (``mufu_per_clk_sm`` x SMs x clock). ``floor_us`` (judge-style) = max(peak memory, compute, MUFU);
``floor_us_achievable`` uses the measured stream bandwidth instead of the datasheet number.

``geomean_sensitivity`` answers "what is 1 us worth on this shape": for the OKC score (geomean over loads of
base_load / sub_load, sub_load = sum of weight * us), d ln(score) / d us_s = -w_s / (n_loads * sub_load(s)).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from kdev.contract import ErrorCode, KdevError
from kdev.floor.gpus import GpuProfile

DTYPE_ALIASES = {"bf16": "bf16", "bfloat16": "bf16", "fp16": "fp16", "f16": "fp16", "half": "fp16",
                 "fp8": "fp8", "e4m3": "fp8", "e5m2": "fp8", "fp4": "fp4", "nvfp4": "fp4", "int8": "int8",
                 "i8": "int8", "tf32": "tf32", "fp32": "fp32", "f32": "fp32", "float": "fp32"}


@dataclass
class RooflineFloor:
    bytes: float
    flops: float
    dtype: str
    mem_us_peak: float
    mem_us_achievable: float
    compute_us: float
    floor_us: float
    bound: str          # memory | compute | mufu
    mufu: float = 0.0
    mufu_us: float = 0.0
    floor_us_achievable: float = 0.0
    notes: list[str] = field(default_factory=list)


def peak_tflops(profile: GpuProfile, dtype: str) -> float:
    d = DTYPE_ALIASES.get(dtype.lower(), dtype.lower())
    v = profile.peak.get(f"{d}_tflops")
    if v is None and d == "fp16":
        v = profile.peak.get("bf16_tflops")
    if not v:
        raise KdevError(ErrorCode.INVALID_REQUEST, f"GPU profile {profile.name} has no peak for dtype {dtype!r}",
                        recovery=f"use one of: {', '.join(k[:-7] for k, x in profile.peak.items() if k.endswith('_tflops') and x)}")
    return v


def mufu_rate_per_us(profile: GpuProfile) -> float:
    """SFU ops per microsecond on the whole GPU."""
    per_clk = profile.peak.get("mufu_per_clk_sm", 16.0)
    ghz = profile.peak.get("sm_clock_ghz", 1.98)
    return per_clk * profile.sm_count * ghz * 1e3


def roofline(profile: GpuProfile, bytes_: float, flops: float, dtype: str = "bf16", mufu: float = 0.0
             ) -> RooflineFloor:
    """Roofline floor in us for one shape (bytes moved from DRAM, flops, MUFU op count)."""
    if bytes_ < 0 or flops < 0 or mufu < 0:
        raise KdevError(ErrorCode.INVALID_REQUEST, "bytes, flops and mufu must be >= 0")
    notes: list[str] = []
    dram = profile.peak.get("dram_tbps")
    if not dram:
        raise KdevError(ErrorCode.INVALID_REQUEST, f"GPU profile {profile.name} has no peak.dram_tbps")
    mem_peak = bytes_ / (dram * 1e12) * 1e6
    stream = profile.const("stream_tbps")
    if stream:
        mem_ach = bytes_ / (stream * 1e12) * 1e6
    else:
        mem_ach = mem_peak
        notes.append("no measured stream_tbps: achievable = peak (run kdev floor calibrate)")
    compute = flops / (peak_tflops(profile, dtype) * 1e12) * 1e6 if flops else 0.0
    mufu_us = mufu / mufu_rate_per_us(profile) if mufu else 0.0
    terms = {"memory": mem_peak, "compute": compute, "mufu": mufu_us}
    bound = max(terms, key=lambda k: terms[k])
    floor = terms[bound]
    floor_ach = max(mem_ach, compute, mufu_us)
    return RooflineFloor(bytes=float(bytes_), flops=float(flops), dtype=DTYPE_ALIASES.get(dtype.lower(), dtype),
                         mem_us_peak=mem_peak, mem_us_achievable=mem_ach, compute_us=compute, floor_us=floor,
                         bound=bound, mufu=float(mufu), mufu_us=mufu_us, floor_us_achievable=floor_ach, notes=notes)


@dataclass
class ShapeSensitivity:
    shape_id: str
    load: str
    weight: float
    us: float
    pct_per_us: float             # score change (%) per 1 us saved on this shape
    floor_us: float | None = None
    gain_pct_at_floor: float | None = None   # score change (%) if this shape alone reached floor_us


def geomean_sensitivity(rows: list[dict[str, Any]]) -> tuple[list[ShapeSensitivity], float | None]:
    """rows: [{shape_id, load, weight, us, floor_us?}] (current per-shape us, judge units).

    Returns (per-shape sensitivities, total gain % if every shape reached its floor). Score model: OKC
    load-weighted geomean (geomean over loads of base_load / sub_load), so the base times cancel out of the
    relative change and are not needed."""
    loads: dict[str, float] = {}
    for r in rows:
        loads[r.get("load", "")] = loads.get(r.get("load", ""), 0.0) + float(r.get("weight", 1.0)) * float(r["us"])
    n = len(loads) or 1
    out: list[ShapeSensitivity] = []
    log_all = 0.0
    saved: dict[str, float] = {}
    for r in rows:
        ld, w, us = r.get("load", ""), float(r.get("weight", 1.0)), float(r["us"])
        tot = loads[ld]
        pct = 100.0 * w / (n * tot) if tot > 0 else 0.0
        fl = r.get("floor_us")
        gain = None
        if fl is not None and tot > 0:
            new = tot - w * (us - float(fl))
            gain = 100.0 * (math.exp(math.log(tot / new) / n) - 1.0) if new > 0 else None
            saved[ld] = saved.get(ld, 0.0) + w * (us - float(fl))
        out.append(ShapeSensitivity(r.get("shape_id", ""), ld, w, us, pct, None if fl is None else float(fl), gain))
    total = None
    if saved and len(saved) == len(loads) and all(r.get("floor_us") is not None for r in rows):
        for ld, tot in loads.items():
            new = tot - saved.get(ld, 0.0)
            if new <= 0:
                return out, None
            log_all += math.log(tot / new)
        total = 100.0 * (math.exp(log_all / n) - 1.0)
    return out, total
