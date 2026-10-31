"""Experiment ledger (SQLite) and idea board. Owner: P4. Schema: schema.sql (frozen, PLAN 3.4) + additive
migrations in ``kdev.ledger.db``."""
from __future__ import annotations

from kdev.models import Check


def probe(ctx: object, problem: str | None = None) -> list[Check]:
    """Doctor checks: ledger.db opens, schema_version current, fts5 available, row counts. Never raises."""
    out: list[Check] = []
    try:
        from kdev.ledger import db
        path = db.default_path(getattr(ctx, "home", None))
        conn = db.connect(path)
    except Exception as e:  # noqa: BLE001 - doctor must never raise
        return [Check("ledger.open", "FAIL", f"{type(e).__name__}: {e}", "check KDEV_HOME is writable")]
    try:
        out.append(Check("ledger.open", "PASS", str(path)))
        ver = db.schema_version(conn)
        out.append(Check("ledger.schema", "PASS" if ver == db.SCHEMA_VERSION else "WARN",
                         f"schema_version={ver} (want {db.SCHEMA_VERSION})",
                         "" if ver == db.SCHEMA_VERSION else "reopen with this kdev version to migrate"))
        out.append(Check("ledger.fts5", "PASS" if db.has_fts(conn) else "WARN",
                         "fts5 idea search" if db.has_fts(conn) else "fts5 unavailable: LIKE fallback"))
        counts = {t: conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
                  for t in ("runs", "judge_results", "ideas", "board_rows")}
        nc = conn.execute("SELECT count(*) FROM runs WHERE comparable=0").fetchone()[0]
        out.append(Check("ledger.rows", "PASS", " ".join(f"{k}={v}" for k, v in counts.items())
                         + f" not_comparable={nc}"))
        if problem:
            j = conn.execute("SELECT count(*) FROM judge_results WHERE problem=?", (problem,)).fetchone()[0]
            out.append(Check("ledger.problem", "PASS" if j else "WARN", f"{problem}: {j} judged runs",
                             "" if j else "judge a baseline first (LESSONS 2/3); `kdev ledger import` old JSONs"))
    except Exception as e:  # noqa: BLE001
        out.append(Check("ledger.query", "FAIL", f"{type(e).__name__}: {e}"))
    finally:
        conn.close()
    return out
