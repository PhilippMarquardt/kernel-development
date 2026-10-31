"""Latency skeleton floor from measured constants (DESIGN 3.3). Owner: P6. Pure.

A plan (TOML or dict) lists terms: {dram_trips = 2, kernel_boundaries = 1, graph_nodes = 0, membar = 1,
cluster_barriers = 2, atom_acqrel = 0, stream_bytes = 2.0e8, ...}; each term multiplies a GPU-profile
measured constant. Output: total floor us, per-term breakdown, dominant term.
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
}


@dataclass
class SkeletonFloor:
    total_us: float
    terms: dict[str, float] = field(default_factory=dict)
    dominant: str = ""
    missing_constants: list[str] = field(default_factory=list)


def skeleton(profile: GpuProfile, plan: dict[str, Any]) -> SkeletonFloor:
    raise NotImplementedError("skeleton.skeleton (P6)")
