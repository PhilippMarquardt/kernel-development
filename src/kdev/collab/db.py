"""collab.db: the multi-agent message board and the documentation store. Owner: P9.

One SQLite file per kdev home (``$KDEV_HOME/collab.db``), WAL mode, same numbered-migration pattern as
``kdev.ledger.db`` (``_MIGRATIONS``, version recorded in ``meta.schema_version``). Many agents on one pod
write concurrently (and remote agents through ``kdev remote -- kdev board ...``): every write goes through
:func:`write_tx` = ``BEGIN IMMEDIATE`` + a long busy timeout + a retry loop on ``database is locked``.

Tables (v1):
- ``posts``: board messages (kind finding|claim|done|ask|answer|warn|decision), ``to_agent`` ('' = all),
  ``reply_to`` / ``thread_id``, ``refs`` (JSON list of typed refs), claims' ``scope`` (JSON list of
  files/shapes) and ``claim_state`` (open|done|released; '' for non-claims), ``expires_utc``.
- ``post_mentions(post_id, agent)``: ``--to`` target and ``@agent`` mentions (inbox).
- ``acks(post_id, agent)``; ``cursors(agent, last_post_id)`` (``board read --unread``).
- ``docs`` / ``doc_sections`` / ``doc_entries``: append-only documentation (notes|lessons|tracker|design).
  Entries are never updated in place except ``superseded_by`` (an entry replaced by a newer one; history kept).
- ``doc_exports(path, doc, md5, utc)``: last export per path (export refuses to overwrite hand edits).
- fts5 indexes ``posts_fts`` / ``entries_fts`` (LIKE fallback when fts5 is missing).
"""
from __future__ import annotations

import os
import random
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

SCHEMA_VERSION = 1
BUSY_TIMEOUT_MS = 30_000
RETRIES = 12

_FTS = [
    "CREATE VIRTUAL TABLE IF NOT EXISTS posts_fts USING fts5(body, agent, kind, problem, refs, "
    "content='posts', content_rowid='post_id')",
    """CREATE TRIGGER IF NOT EXISTS posts_fts_ai AFTER INSERT ON posts BEGIN
         INSERT INTO posts_fts(rowid, body, agent, kind, problem, refs)
         VALUES (new.post_id, new.body, new.agent, new.kind, coalesce(new.problem, ''), new.refs);
       END""",
    "CREATE VIRTUAL TABLE IF NOT EXISTS entries_fts USING fts5(body, section, author, refs, "
    "content='doc_entries', content_rowid='entry_id')",
    """CREATE TRIGGER IF NOT EXISTS entries_fts_ai AFTER INSERT ON doc_entries BEGIN
         INSERT INTO entries_fts(rowid, body, section, author, refs)
         VALUES (new.entry_id, new.body, new.section, new.author, new.refs);
       END""",
]

_MIGRATIONS: dict[int, list[str]] = {
    1: [
        """CREATE TABLE IF NOT EXISTS posts (
             post_id INTEGER PRIMARY KEY AUTOINCREMENT,
             created_utc TEXT NOT NULL, agent TEXT NOT NULL, kind TEXT NOT NULL, problem TEXT,
             to_agent TEXT NOT NULL DEFAULT '', reply_to INTEGER, thread_id INTEGER,
             refs TEXT NOT NULL DEFAULT '[]', scope TEXT NOT NULL DEFAULT '[]',
             claim_state TEXT NOT NULL DEFAULT '', expires_utc TEXT, closed_utc TEXT, closed_by INTEGER,
             body TEXT NOT NULL, source TEXT NOT NULL DEFAULT 'cli', host TEXT)""",
        "CREATE INDEX IF NOT EXISTS ix_posts_time ON posts(created_utc)",
        "CREATE INDEX IF NOT EXISTS ix_posts_thread ON posts(thread_id)",
        "CREATE INDEX IF NOT EXISTS ix_posts_claims ON posts(claim_state)",
        """CREATE TABLE IF NOT EXISTS post_mentions (
             post_id INTEGER NOT NULL, agent TEXT NOT NULL, PRIMARY KEY (post_id, agent))""",
        "CREATE INDEX IF NOT EXISTS ix_mentions_agent ON post_mentions(agent)",
        """CREATE TABLE IF NOT EXISTS acks (
             post_id INTEGER NOT NULL, agent TEXT NOT NULL, acked_utc TEXT NOT NULL,
             PRIMARY KEY (post_id, agent))""",
        "CREATE TABLE IF NOT EXISTS cursors (agent TEXT PRIMARY KEY, last_post_id INTEGER NOT NULL, utc TEXT)",
        """CREATE TABLE IF NOT EXISTS docs (
             doc TEXT PRIMARY KEY, kind TEXT NOT NULL, problem TEXT, title TEXT NOT NULL DEFAULT '',
             created_utc TEXT NOT NULL, source TEXT)""",
        """CREATE TABLE IF NOT EXISTS doc_sections (
             doc TEXT NOT NULL, section TEXT NOT NULL, heading TEXT NOT NULL, ord INTEGER NOT NULL,
             newest_first INTEGER NOT NULL DEFAULT 0, created_utc TEXT NOT NULL, PRIMARY KEY (doc, section))""",
        """CREATE TABLE IF NOT EXISTS doc_entries (
             entry_id INTEGER PRIMARY KEY AUTOINCREMENT, doc TEXT NOT NULL, section TEXT NOT NULL,
             created_utc TEXT NOT NULL, author TEXT NOT NULL DEFAULT '', body TEXT NOT NULL,
             refs TEXT NOT NULL DEFAULT '[]', raw INTEGER NOT NULL DEFAULT 0, source TEXT NOT NULL DEFAULT 'cli',
             superseded_by INTEGER, tags TEXT NOT NULL DEFAULT '')""",
        "CREATE INDEX IF NOT EXISTS ix_entries_doc ON doc_entries(doc, section, entry_id)",
        """CREATE TABLE IF NOT EXISTS doc_exports (
             path TEXT PRIMARY KEY, doc TEXT NOT NULL, md5 TEXT NOT NULL, utc TEXT NOT NULL)""",
    ],
}


def has_fts(conn: sqlite3.Connection) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE name='posts_fts'").fetchone() is not None


def schema_version(conn: sqlite3.Connection) -> int:
    try:
        row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    except sqlite3.OperationalError:
        return 0
    return int(row[0]) if row else 0


def _locked(e: sqlite3.OperationalError) -> bool:
    s = str(e).lower()
    return "locked" in s or "busy" in s


@contextmanager
def write_tx(conn: sqlite3.Connection, retries: int = RETRIES) -> Iterator[sqlite3.Connection]:
    """``BEGIN IMMEDIATE`` (takes the write lock up front, so two readers never deadlock upgrading), retried
    with jittered backoff while another process holds the lock; COMMIT on success, ROLLBACK on error."""
    for attempt in range(retries):
        try:
            conn.execute("BEGIN IMMEDIATE")
            break
        except sqlite3.OperationalError as e:
            if not _locked(e) or attempt == retries - 1:
                raise
            time.sleep(min(2.0, 0.05 * (2 ** attempt)) * (0.5 + random.random()))
    try:
        yield conn
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise


def migrate(conn: sqlite3.Connection) -> int:
    """Apply numbered migrations (+ fts5 when available); return the resulting version."""
    conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
    if schema_version(conn) >= SCHEMA_VERSION:
        return schema_version(conn)
    with write_tx(conn):
        ver = schema_version(conn)          # another process may have migrated meanwhile
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
        for stmt in _FTS:
            try:
                conn.execute(stmt)
            except sqlite3.OperationalError as e:
                if "fts5" not in str(e).lower() and "no such module" not in str(e).lower():
                    raise
                break
        conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)", (str(ver),))
    return ver


def connect(path: str | Path) -> sqlite3.Connection:
    """Open/create collab.db (WAL, busy_timeout 30 s, autocommit + explicit :func:`write_tx`) and migrate."""
    p = str(path)
    if p != ":memory:" and not p.startswith("file:"):
        pp = Path(p).expanduser()
        pp.parent.mkdir(parents=True, exist_ok=True)
        p = str(pp)
    last: Exception | None = None
    for attempt in range(RETRIES):
        conn = sqlite3.connect(p, timeout=BUSY_TIMEOUT_MS / 1000, isolation_level=None)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
            if p != ":memory:":
                conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            migrate(conn)
            return conn
        except sqlite3.OperationalError as e:
            conn.close()
            last = e
            if not _locked(e):
                raise
            time.sleep(0.05 * (attempt + 1) * (0.5 + random.random()))
    raise last  # type: ignore[misc]


def default_path(home: str | None = None) -> Path:
    """$KDEV_HOME/collab.db (``home`` overrides)."""
    try:
        from kdev.config import load_config
        return Path(load_config(home=home).home) / "collab.db"
    except Exception:  # noqa: BLE001 - config not ready / invalid: fall back to the environment
        base = home or os.environ.get("KDEV_HOME") or "~/.kdev"
        return Path(base).expanduser() / "collab.db"


def open_default(home: str | None = None) -> sqlite3.Connection:
    return connect(default_path(home))
