"""Judge-vs-local calibration (port of okc/tools/calib.py + calib_seeds.py). Owner: P3.

Parsing and comparison are pure (fixtures: tests/fixtures/judge/judge_moe.json, cal_moe.json); the timing
part reuses engine.measure_shape. Rows go to the ledger ``calib`` table via the Result.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kdev.contract import Result
from kdev.models import JobSpec

DROP_CASE_KEYS = ("load", "repeats", "weight", "router_load", "router_distinct")


@dataclass
class JudgeShape:
    shape_id: str
    case: dict[str, Any]
    sub_us: float | None
    base_us: float | None
    speedup: float | None
    seed: int | None = None


@dataclass
class JudgeRun:
    judge_id: str
    cluster: str | None
    node: str | None
    speedup: float | None
    status: str
    shapes: list[JudgeShape]
    raw: dict[str, Any]


def parse_judge_json(d: dict[str, Any]) -> JudgeRun:
    """`okc status --json` -> JudgeRun (records with event == 'benchmark'; sub_us = baseline_ns/1000/speedup)."""
    raise NotImplementedError("calib.parse_judge_json (P3)")


def compare(local_speedups: dict[str, float], judge: JudgeRun, band: tuple[float, float] = (0.95, 1.05)
            ) -> list[dict[str, Any]]:
    """Per shape: local, judge, local/judge, flag (ok | out_of_band)."""
    raise NotImplementedError("calib.compare (P3)")


def ab_disagree(local_ab: float, judge_ab: float, min_effect: float = 0.02, max_gap: float = 0.03) -> bool:
    """calib.py rule: direction flip with effect > 2 %, or |local/judge - 1| > 3 %."""
    raise NotImplementedError("calib.ab_disagree (P3)")


def run_calib(spec: JobSpec, run_dir: Any) -> Result:
    """Worker entry for kind='calib' (files carry judge_json)."""
    raise NotImplementedError("calib.run_calib (P3)")
