"""Ledger connection + migrations. Owner: P4."""
from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 1
SCHEMA_FILE = Path(__file__).with_name("schema.sql")


def connect(path: str | Path) -> sqlite3.Connection:
    """Open/create the ledger (WAL, busy_timeout 10 s, row_factory=sqlite3.Row) and migrate to
    SCHEMA_VERSION. Falls back to a LIKE-based search if fts5 is unavailable."""
    raise NotImplementedError("ledger.db.connect (P4)")


def migrate(conn: sqlite3.Connection) -> int:
    """Apply schema.sql and numbered migrations; return the resulting version."""
    raise NotImplementedError("ledger.db.migrate (P4)")


def readonly_query(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list[dict]:
    """`ledger query --sql`: only a single SELECT/WITH statement is accepted (authorizer denies writes)."""
    raise NotImplementedError("ledger.db.readonly_query (P4)")
