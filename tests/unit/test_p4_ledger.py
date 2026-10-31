"""P4: ledger, ideas, judge record. Placeholder; P4 replaces with ingest/tried/best/claim tests."""
from __future__ import annotations

import sqlite3

from kdev.ledger.db import SCHEMA_FILE
from kdev.registry import build_registry


def test_ledger_schema_is_valid_sql():
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA_FILE.read_text(encoding="utf-8"))
    tables = {r[0] for r in conn.execute("select name from sqlite_master where type='table'")}
    assert {"runs", "artifacts", "shape_results", "scores", "builds", "kernels", "judge_results",
            "judge_shapes", "calib", "ideas", "idea_evidence"} <= tables


def test_ledger_commands_registered():
    names = set(build_registry().names())
    assert {"ledger query", "ledger tried", "idea add", "idea claim", "idea resolve", "judge record"} <= names
