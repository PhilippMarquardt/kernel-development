"""Interleaving schedules (pure). Owner: P3.

abba: rep i even -> labels in order, odd -> reversed (calib.py / cq_run.py). rotate: rep i starts at label
i % n. The schedule is recorded per sample (order_pos) in raw.jsonl.
"""
from __future__ import annotations


def order_for_rep(n_files: int, rep: int, scheme: str = "abba") -> list[int]:
    raise NotImplementedError("schedule.order_for_rep (P3)")


def full_schedule(n_files: int, reps: int, scheme: str = "abba", warmup: int = 1) -> list[tuple[int, int, int]]:
    """List of (rep, order_pos, file_index); warm-up reps have rep < 0 and are not recorded."""
    raise NotImplementedError("schedule.full_schedule (P3)")
