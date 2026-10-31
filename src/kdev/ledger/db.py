"""Ledger connection + migrations. Owner: P4.

``schema.sql`` is the frozen version-1 schema (PLAN 3.4). Additive changes live in ``_MIGRATIONS`` below
(numbered, applied in order, recorded in ``meta.schema_version``); each one is listed in docs/CHANGES.md.

Version 2 (P4, 2026-10-31, additive only):
- ``judge_results`` gains ``content_sha256`` (the judge's ``content_hash``: what the judge dedups on),
  ``team``, ``label`` (file name / evidence label), ``role`` (base|cand|other) and ``pair_id`` (rows judged as
  one same-hour pair by ``kdev judge pair`` or grouped by the importer).
- ``board_rows``: public leaderboard snapshots (``okc leaderboard`` / research JSONs).
- triggers keeping the external-content ``ideas_fts`` index in sync with ``ideas``.
"""
from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path

SCHEMA_VERSION = 2
SCHEMA_FILE = Path(__file__).with_name("schema.sql")

_FTS_TRIGGERS = (
    """CREATE TRIGGER IF NOT EXISTS ideas_fts_ai AFTER INSERT ON ideas BEGIN
         INSERT INTO ideas_fts(rowid, title, hypothesis, variant)
         VALUES (new.idea_id, new.title, coalesce(new.hypothesis, ''), coalesce(new.variant, ''));
       END""",
    """CREATE TRIGGER IF NOT EXISTS ideas_fts_ad AFTER DELETE ON ideas BEGIN
         INSERT INTO ideas_fts(ideas_fts, rowid, title, hypothesis, variant)
         VALUES ('delete', old.idea_id, old.title, coalesce(old.hypothesis, ''), coalesce(old.variant, ''));
       END""",
    """CREATE TRIGGER IF NOT EXISTS ideas_fts_au AFTER UPDATE ON ideas BEGIN
         INSERT INTO ideas_fts(ideas_fts, rowid, title, hypothesis, variant)
         VALUES ('delete', old.idea_id, old.title, coalesce(old.hypothesis, ''), coalesce(old.variant, ''));
         INSERT INTO ideas_fts(rowid, title, hypothesis, variant)
         VALUES (new.idea_id, new.title, coalesce(new.hypothesis, ''), coalesce(new.variant, ''));
       END""",
)

# version -> statements. ALTER TABLE ADD COLUMN tolerates "duplicate column" (re-run after a crash).
_MIGRATIONS: dict[int, list[str]] = {
    2: [
        "ALTER TABLE judge_results ADD COLUMN content_sha256 TEXT",
        "ALTER TABLE judge_results ADD COLUMN team TEXT",
        "ALTER TABLE judge_results ADD COLUMN label TEXT",
        "ALTER TABLE judge_results ADD COLUMN role TEXT",
        "ALTER TABLE judge_results ADD COLUMN pair_id TEXT",
        "CREATE INDEX IF NOT EXISTS ix_judge_sha ON judge_results(content_sha256)",
        "CREATE INDEX IF NOT EXISTS ix_judge_norm ON judge_results(norm_md5)",
        "CREATE INDEX IF NOT EXISTS ix_judge_problem ON judge_results(problem, created_utc)",
        """CREATE TABLE IF NOT EXISTS board_rows (
             problem TEXT NOT NULL, taken_utc TEXT NOT NULL, rank INTEGER, team TEXT NOT NULL, job TEXT,
             speedup REAL, time_us REAL, screened_speedup REAL, screening TEXT, attempts INTEGER,
             submitted_utc TEXT, source TEXT, PRIMARY KEY (problem, taken_utc, team))""",
    ],
}


def has_fts(conn: sqlite3.Connection) -> bool:
    """True when the ``ideas_fts`` virtual table exists (fts5 compiled in)."""
    return conn.execute("SELECT 1 FROM sqlite_master WHERE name='ideas_fts'").fetchone() is not None


def connect(path: str | Path) -> sqlite3.Connection:
    """Open/create the ledger (WAL, busy_timeout 10 s, row_factory=sqlite3.Row) and migrate to
    SCHEMA_VERSION. Falls back to a LIKE-based search if fts5 is unavailable."""
    p = str(path)
    if p != ":memory:" and not p.startswith("file:"):
        pp = Path(p).expanduser()
        pp.parent.mkdir(parents=True, exist_ok=True)
        p = str(pp)
    conn = sqlite3.connect(p, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 10000")
    migrate(conn)
    return conn


def _schema_without_fts(sql: str) -> str:
    return re.sub(r"CREATE VIRTUAL TABLE[^;]*;", "", sql, flags=re.S | re.I)


def schema_version(conn: sqlite3.Connection) -> int:
    try:
        row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    except sqlite3.OperationalError:
        return 0
    return int(row[0]) if row else 0


def migrate(conn: sqlite3.Connection) -> int:
    """Apply schema.sql and numbered migrations; return the resulting version."""
    sql = SCHEMA_FILE.read_text(encoding="utf-8")
    try:
        conn.executescript(sql)
    except sqlite3.OperationalError as e:
        if "fts5" not in str(e).lower():
            raise
        conn.executescript(_schema_without_fts(sql))
    ver = schema_version(conn)
    for v in sorted(_MIGRATIONS):
        if v <= ver:
            continue
        for stmt in _MIGRATIONS[v]:
            try:
                conn.execute(stmt)
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise
        ver = v
        conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)", (str(ver),))
    if has_fts(conn):
        for t in _FTS_TRIGGERS:
            conn.execute(t)
    conn.commit()
    return ver


_READ_ACTIONS = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}
if hasattr(sqlite3, "SQLITE_RECURSIVE"):
    _READ_ACTIONS.add(sqlite3.SQLITE_RECURSIVE)


def readonly_query(conn: sqlite3.Connection, sql: str, params: tuple = (), limit: int = 1000) -> list[dict]:
    """`ledger query --sql`: only a single SELECT/WITH statement is accepted (authorizer denies writes)."""
    from kdev.contract import ErrorCode, KdevError

    s = sql.strip().rstrip(";").strip()
    if not re.match(r"(?is)^(select|with)\b", s) or ";" in s:
        raise KdevError(ErrorCode.INVALID_REQUEST, "only a single SELECT/WITH statement is allowed",
                        details={"field": "sql"}, recovery="pass one read-only SELECT")

    def auth(action: int, *_args: object) -> int:
        return sqlite3.SQLITE_OK if action in _READ_ACTIONS else sqlite3.SQLITE_DENY

    conn.set_authorizer(auth)
    try:
        cur = conn.execute(s, params)
        cols = [c[0] for c in cur.description or ()]
        rows = [dict(zip(cols, tuple(r))) for r in cur.fetchmany(limit)]
    except sqlite3.DatabaseError as e:
        raise KdevError(ErrorCode.INVALID_REQUEST, f"query rejected: {e}", details={"field": "sql"},
                        recovery="only read-only SELECTs over ledger tables are allowed") from e
    finally:
        conn.set_authorizer(None)
    return rows


def default_path(home: str | None = None) -> Path:
    """$KDEV_HOME/ledger.db (``home`` overrides). Uses P1's load_config when it works."""
    try:
        from kdev.config import load_config
        return Path(load_config(home=home).ledger_path)
    except Exception:  # noqa: BLE001 - config not ready / invalid: fall back to the environment
        base = home or os.environ.get("KDEV_HOME") or "~/.kdev"
        return Path(base).expanduser() / "ledger.db"


def open_default(home: str | None = None) -> sqlite3.Connection:
    """Connection to the configured ledger (created and migrated on first use)."""
    return connect(default_path(home))
