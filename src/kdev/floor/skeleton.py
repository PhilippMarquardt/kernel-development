"""Latency skeleton floor from measured constants (DESIGN 3.3). Owner: P6. Pure.

A plan (TOML or dict) lists terms: {dram_trips = 2, kernel_boundaries = 1, graph_nodes = 0, membar = 1,
cluster_barriers = 2, atom_acqrel = 0, stream_bytes = 2.0e8, ...}; each term multiplies a GPU-profile
measured constant. Output: total floor us, per-term breakdown, dominant term.

Conventions (judge units = replay us minus 4.0):
- a ``launch`` term (one null kernel, ``null_kernel_us``) is added unless the plan sets ``launch = false``;
- ``stream_bytes`` are charged at the measured ``stream_tbps``;
- any numeric key ending in ``_us`` is taken as a direct microsecond term (e.g. ``compute_us = 1.2``);
- keys that describe the shape rather than a latency term (bytes, flops, dtype, us, weight, load, mufu,
  shape_id, name, notes, terms) are ignored; unknown numeric keys are reported in ``missing_constants``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kdev.floor.gpus import GpuProfile

TERM_TO_CONSTANT = {
    "dram_trips": "dram_trip_132cta_us",
    "dram_trips_1cta": "dram_trip_1cta_us",
    "l2_hits": "l2_hit_us",
    "kernel_boundaries": "kernel_boundary_us",
    "graph_nodes": "extra_graph_node_us",
    "coop_launch": "coop_launch_us",
    "membar": "membar_gpu_us",
    "atom_acqrel": "atom_acqrel_cold_us",
    "cluster_barriers": "barrier_cluster_us",
    "pdl_boundaries": "pdl_boundary_us",
    "gather_rounds": "gather_round_8k_us",
    "merges": "merge_lastcta_us",
    "merges_256": "merge_lastcta_256_us",
    "fence_cta": "fence_cta_us",
    "icache_cold": "icache_cold_us",
}

NON_TERMS = {"bytes", "flops", "dtype", "us", "weight", "load", "mufu", "shape_id", "name", "notes", "terms",
             "launch", "skeleton", "measured_us", "kernel"}


@dataclass
class SkeletonFloor:
    total_us: float
    terms: dict[str, float] = field(default_factory=dict)
    dominant: str = ""
    missing_constants: list[str] = field(default_factory=list)


def _num(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return None


def skeleton(profile: GpuProfile, plan: dict[str, Any]) -> SkeletonFloor:
    """Sum of count x measured constant per term (us, judge units)."""
    terms: dict[str, float] = {}
    missing: list[str] = []
    body = dict(plan.get("terms", {})) if isinstance(plan.get("terms"), dict) else {}
    body.update({k: v for k, v in plan.items() if k != "terms"})
    if body.get("launch", True) is not False:
        c = profile.const("null_kernel_us")
        if c is None:
            missing.append("null_kernel_us")
        else:
            terms["launch"] = c
    for k, v in body.items():
        if k in NON_TERMS:
            continue
        n = _num(v)
        if n is None or n == 0:
            continue
        if k == "stream_bytes":
            bw = profile.const("stream_tbps") or profile.peak.get("dram_tbps")
            if not bw:
                missing.append("stream_tbps")
                continue
            terms[k] = n / (bw * 1e12) * 1e6
        elif k in TERM_TO_CONSTANT:
            c = profile.const(TERM_TO_CONSTANT[k])
            if c is None:
                missing.append(TERM_TO_CONSTANT[k])
                continue
            terms[k] = n * c
        elif k.endswith("_us"):
            terms[k] = n
        else:
            missing.append(f"{k} (unknown term)")
    total = sum(terms.values())
    dominant = max(terms, key=lambda t: terms[t]) if terms else ""
    return SkeletonFloor(total_us=total, terms=terms, dominant=dominant, missing_constants=missing)
