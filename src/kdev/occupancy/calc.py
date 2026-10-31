"""Occupancy calculator (pure). Owner: P6.

Inputs from ptxas (kdev.build.ptxas.parse_ptxas) / cuobjdump -res-usage: regs, smem static, dynamic smem,
block size, launch bounds, setmaxnreg groups. Output per kernel: CTAs/SM by each limiter (regs, smem,
warps, CTA slots), the binding limiter, warps/SMSP, and the register rule e.g. "5 warps/SMSP caps regs at
96" (regs per thread <= 65536 / (4 * warps_per_smsp * 32) rounded down to the allocation granule of 8).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from kdev.floor.gpus import GpuProfile


@dataclass
class OccupancyReport:
    kernel: str
    regs: int
    block: int
    smem_bytes: int
    ctas_per_sm: dict[str, int] = field(default_factory=dict)   # {regs, smem, warps, slots}
    limiter: str = ""
    active_ctas: int = 0
    warps_per_smsp: float = 0.0
    reg_cap_for_next_warp: int | None = None
    notes: list[str] = field(default_factory=list)


def occupancy(profile: GpuProfile, kernel: str, regs: int, block: int, smem_bytes: int,
              setmaxnreg: list[tuple[int, int]] | None = None) -> OccupancyReport:
    raise NotImplementedError("calc.occupancy (P6)")


def max_regs_for_warps(profile: GpuProfile, warps_per_smsp: int) -> int:
    raise NotImplementedError("calc.max_regs_for_warps (P6)")
