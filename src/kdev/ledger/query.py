"""Ledger queries: runs, "has this been tried", bests. Owner: P4."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TriedHit:
    match: str                 # md5 | norm_md5 | knobs | idea_text
    run_id: str | None = None
    judge_id: str | None = None
    idea_id: int | None = None
    tool: str = ""
    verdict: str = ""
    created_utc: str = ""
    comparable: bool = True
    detail: str = ""


@dataclass
class TriedReport:
    query: dict[str, Any] = field(default_factory=dict)
    hits: list[TriedHit] = field(default_factory=list)


def runs(conn: sqlite3.Connection, *, problem: str | None = None, md5: str | None = None,
         tool: str | None = None, since_utc: str | None = None, limit: int = 30) -> list[dict[str, Any]]:
    raise NotImplementedError("ledger.query.runs (P4)")


def tried(conn: sqlite3.Connection, *, file: str | None = None, text: str | None = None,
          knobs: dict[str, Any] | None = None, problem: str | None = None) -> TriedReport:
    """Exact md5, norm_md5 (comment-insensitive), knob-subset match within the problem, FTS over ideas."""
    raise NotImplementedError("ledger.query.tried (P4)")


def best(conn: sqlite3.Connection, problem: str | None = None) -> list[dict[str, Any]]:
    """Per problem: best judged (judge_results) and best local comparable score, with same-hour baseline."""
    raise NotImplementedError("ledger.query.best (P4)")


def export(conn: sqlite3.Connection, fmt: str = "jsonl", tables: tuple[str, ...] = ("runs",)) -> str:
    raise NotImplementedError("ledger.query.export (P4)")
