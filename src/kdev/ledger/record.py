"""Ingest Results into the ledger. Owner: P4. Called by the daemon / direct mode after every job (P2) and by
`kdev ledger import`. Idempotent per run_id."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from kdev.contract import Result


def ingest_result(result_path: str | Path, conn: sqlite3.Connection | None = None,
                  meta: dict[str, Any] | None = None) -> str:
    """Read result.json (+ meta.json beside it) and upsert runs/artifacts/shape_results/scores/builds/
    kernels/calib rows; sets comparable=0 when fidelity opt-outs exist, exclusivity failed, toolchain
    mismatched or protocol != current. Returns run_id."""
    raise NotImplementedError("ledger.record.ingest_result (P4)")


def rows_from_result(result: Result, meta: dict[str, Any] | None = None) -> dict[str, list[dict[str, Any]]]:
    """Pure: Result -> {table: [row dicts]} (unit-tested without sqlite)."""
    raise NotImplementedError("ledger.record.rows_from_result (P4)")


def mark_protocol_incomparable(conn: sqlite3.Connection, current_protocol: str) -> int:
    """Protocol bump: set comparable=0 on rows of older protocols; return count."""
    raise NotImplementedError("ledger.record.mark_protocol_incomparable (P4)")
