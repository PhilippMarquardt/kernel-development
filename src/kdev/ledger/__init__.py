"""Experiment ledger (SQLite) and idea board. Owner: P4. Schema: schema.sql (frozen, PLAN 3.4)."""
from __future__ import annotations

from kdev.models import Check


def probe(ctx: object, problem: str | None = None) -> list[Check]:
    """Doctor checks: ledger.db opens, schema_version current, fts5 available, row counts. Never raises."""
    raise NotImplementedError("ledger.probe (P4)")
