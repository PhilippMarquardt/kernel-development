"""Agent collaboration (P9): the message board (``kdev board``) and the documentation store (``kdev doc``),
both in ``$KDEV_HOME/collab.db``; ``kdev collab backup`` snapshots them. See docs/COLLAB.md."""
from __future__ import annotations

from kdev.models import Check


def probe(ctx: object, problem: str | None = None) -> list[Check]:
    """Doctor checks: collab.db opens (WAL), schema current, fts5, counts, open claims. Never raises."""
    try:
        from kdev.collab import db
        path = db.default_path(getattr(ctx, "home", None))
        conn = db.connect(path)
    except Exception as e:  # noqa: BLE001 - doctor must never raise
        return [Check("collab.open", "FAIL", f"{type(e).__name__}: {e}", "check KDEV_HOME is writable")]
    out: list[Check] = []
    try:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        ver = db.schema_version(conn)
        out.append(Check("collab.open", "PASS" if ver == db.SCHEMA_VERSION else "WARN",
                         f"{path} journal={mode} schema={ver}" + ("" if db.has_fts(conn) else " (no fts5: LIKE)")))
        n = {t: conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in ("posts", "docs", "doc_entries")}
        claims = conn.execute("SELECT count(*) FROM posts WHERE kind='claim' AND claim_state='open'").fetchone()[0]
        out.append(Check("collab.rows", "PASS", " ".join(f"{k}={v}" for k, v in n.items()) + f" open_claims={claims}"))
        bk = path.parent / "backup" / "manifest.json"
        if n["posts"] or n["docs"]:
            out.append(Check("collab.backup", "PASS" if bk.is_file() else "WARN",
                             f"last backup {bk}" if bk.is_file() else "no backup yet",
                             "" if bk.is_file() else "kdev collab backup (the coordinator's sync copies it)"))
    except Exception as e:  # noqa: BLE001
        out.append(Check("collab.query", "FAIL", f"{type(e).__name__}: {e}"))
    finally:
        conn.close()
    return out
