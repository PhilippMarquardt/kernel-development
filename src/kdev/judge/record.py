"""`kdev judge record JOB.json`: ingest an `okc status --json` / `okc check --json` result. Owner: P4.
Fixture: tests/fixtures/judge/judge_moe.json (keys: id, cluster, content_hash, duplicate_of, result{records,
machine{node}, speedup, speedup_ci, evaluator, log_tail})."""
from __future__ import annotations

import sqlite3
from typing import Any


def record(conn: sqlite3.Connection, judge_json: dict[str, Any], file_path: str | None = None,
           raw_path: str | None = None) -> str:
    """Upsert judge_results + judge_shapes; returns judge_id. Uses kdev.timing.calib.parse_judge_json."""
    raise NotImplementedError("judge.record.record (P4)")


def ledger(conn: sqlite3.Connection, problem: str | None = None) -> list[dict[str, Any]]:
    """Judged bests and same-hour baselines per problem."""
    raise NotImplementedError("judge.record.ledger (P4)")
