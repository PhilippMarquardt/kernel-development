"""Interleaving schedules (pure). Owner: P3.

abba: rep i even -> labels in order, odd -> reversed (calib.py / cq_run.py; the judge alternates which side
goes first and pairs reps two at a time, so the order cancels). rotate: rep i starts at label i % n.
The schedule is recorded per sample (order_pos) in raw.jsonl.
"""
from __future__ import annotations

SCHEMES = ("abba", "rotate")


def order_for_rep(n_files: int, rep: int, scheme: str = "abba") -> list[int]:
    """File indices in the order they are replayed in repetition ``rep``."""
    if n_files <= 0:
        return []
    if scheme == "abba":
        fwd = list(range(n_files))
        return fwd if rep % 2 == 0 else fwd[::-1]
    if scheme == "rotate":
        s = rep % n_files
        return [(s + j) % n_files for j in range(n_files)]
    raise ValueError(f"unknown interleave scheme {scheme!r} (expected one of {SCHEMES})")


def full_schedule(n_files: int, reps: int, scheme: str = "abba", warmup: int = 1) -> list[tuple[int, int, int]]:
    """List of (rep, order_pos, file_index); warm-up reps have rep < 0 and are not recorded.

    Warm-up reps replay every file once in forward order."""
    out: list[tuple[int, int, int]] = []
    for w in range(warmup):
        rep = -(warmup - w)
        for pos in range(n_files):
            out.append((rep, pos, pos))
    for rep in range(reps):
        for pos, idx in enumerate(order_for_rep(n_files, rep, scheme)):
            out.append((rep, pos, idx))
    return out


def first_index(n_files: int, rep: int, scheme: str = "abba") -> int:
    """Which file ran first in ``rep`` (the ``order`` argument of stats.paired_ratios)."""
    o = order_for_rep(n_files, rep, scheme)
    return o[0] if o else -1
