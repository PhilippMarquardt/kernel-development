"""Optional read-only public board scrape (via kdev.judge.okc_cli.leaderboard only). Owner: P8.
Diffs the board's per-shape numbers against our bests (LESSONS 4: re-scan every 3 h)."""
from __future__ import annotations

from typing import Any


def scrape(problem: str | None = None) -> dict[str, Any]:
    raise NotImplementedError("dashboard.board.scrape (P8)")


def gaps(board: dict[str, Any], ours: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per problem/shape: our best vs board best, gap %."""
    raise NotImplementedError("dashboard.board.gaps (P8)")
