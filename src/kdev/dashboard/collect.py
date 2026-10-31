"""Gather dashboard data (pure reads). Owner: P8.

Sources: kdev.ledger.query.best/runs, kdev.ledger.ideas.list_ideas, ledger calib rows,
kdev.service.queue.snapshot, kdev.dashboard.board.scrape (optional). Output is one JSON-serialisable dict
(also written as dashboard.json next to index.html, so agents can read it without HTML).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kdev.config import Config


@dataclass
class DashboardData:
    generated_utc: str = ""
    problems: list[dict[str, Any]] = field(default_factory=list)   # best judged/local, same-hour base, gap
    recent_runs: list[dict[str, Any]] = field(default_factory=list)
    queue: dict[str, Any] = field(default_factory=dict)
    offenders: list[dict[str, Any]] = field(default_factory=list)
    ideas: list[dict[str, Any]] = field(default_factory=list)
    calibration: list[dict[str, Any]] = field(default_factory=list)
    board: dict[str, Any] | None = None
    unverified_limit_claims: list[dict[str, Any]] = field(default_factory=list)


def collect(cfg: Config, *, scrape: bool = False, recent: int = 30) -> DashboardData:
    raise NotImplementedError("dashboard.collect.collect (P8)")
