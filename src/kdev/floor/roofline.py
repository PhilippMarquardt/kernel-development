"""Roofline floor (judge-style peak constants and measured achievable bandwidth). Owner: P6. Pure."""
from __future__ import annotations

from dataclasses import dataclass

from kdev.floor.gpus import GpuProfile


@dataclass
class RooflineFloor:
    bytes: float
    flops: float
    dtype: str
    mem_us_peak: float
    mem_us_achievable: float
    compute_us: float
    floor_us: float
    bound: str          # memory | compute


def roofline(profile: GpuProfile, bytes_: float, flops: float, dtype: str = "bf16") -> RooflineFloor:
    raise NotImplementedError("roofline.roofline (P6)")
