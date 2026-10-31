"""Seed policy (M7). Owner: P3.

Default: 1 seed for deterministic shapes, 5 when the adapter's input_variance() has routing/lengths. The
judge's per-shape seed is used first when known; extra seeds rotate like calib_seeds.py:
seed + j * 0x2545F4914F6CDD1D (mod 2**63 - 1, exactly as calib_seeds.py so its runs reproduce).
"""
from __future__ import annotations

SEED_STRIDE = 0x2545F4914F6CDD1D
SEED_MOD = 2**63 - 1
VARIANCE_SEEDS = 5
DETERMINISTIC_SEEDS = 1


def default_seed_count(input_variance: dict[str, bool]) -> int:
    """5 when any declared input-variance source (routing, lengths, ...) is true, else 1."""
    return VARIANCE_SEEDS if any(bool(v) for v in (input_variance or {}).values()) else DETERMINISTIC_SEEDS


def seeds_for_shape(base_seed: int | None, count: int, fallback: int = 11) -> list[int]:
    """``count`` seeds: the judge's seed (or ``fallback``) first, then the calib_seeds.py rotation."""
    s0 = fallback if base_seed is None else int(base_seed)
    out = [s0]
    for j in range(1, max(1, int(count))):
        out.append((s0 + j * SEED_STRIDE) % SEED_MOD)
    return out
