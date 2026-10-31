"""Seed policy (M7). Owner: P3.

Default: 1 seed for deterministic shapes, 5 when the adapter's input_variance() has routing/lengths. The
judge's per-shape seed is used first when known; extra seeds rotate like calib_seeds.py:
seed + j * 0x2545F4914F6CDD1D (mod 2**63).
"""
from __future__ import annotations

SEED_STRIDE = 0x2545F4914F6CDD1D


def default_seed_count(input_variance: dict[str, bool]) -> int:
    raise NotImplementedError("seeds.default_seed_count (P3)")


def seeds_for_shape(base_seed: int | None, count: int, fallback: int = 11) -> list[int]:
    raise NotImplementedError("seeds.seeds_for_shape (P3)")
