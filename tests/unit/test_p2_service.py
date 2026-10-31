"""P2: service, lock, queue, remote/sync. Placeholder; P2 replaces with real tests (fake compute_apps, tmp
lock path, tmp queue.db, sync planner)."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import kdev.service as service
from kdev.registry import build_registry


def test_service_commands_registered():
    names = set(build_registry().names())
    assert {"serve", "job submit", "job wait", "queue status", "lock run", "sync", "remote"} <= names


def test_queue_schema_is_valid_sql():
    sql = (Path(service.__file__).with_name("schema.sql")).read_text(encoding="utf-8")
    conn = sqlite3.connect(":memory:")
    conn.executescript(sql)
    tables = {r[0] for r in conn.execute("select name from sqlite_master where type='table'")}
    assert {"jobs", "events", "offenders", "heartbeat"} <= tables


def test_lock_run_not_exposed_over_mcp():
    assert build_registry().get("lock run").mcp is False
