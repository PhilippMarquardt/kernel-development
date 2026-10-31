"""`kdev collab backup`: a consistent, copyable snapshot of the shared state. Owner: P9.

Pod storage is never persistent, and a live WAL database copied with tar/scp can be torn (the -wal file holds
recent commits). This writes, under ``$KDEV_HOME/backup/`` (or ``--out``):
- ``collab.db`` and ``ledger.db``: SQLite online-backup copies (consistent at one instant, WAL folded in);
- ``board.jsonl``: every post (one JSON object per line, oldest first);
- ``docs/<doc with / as __>.md``: every doc rendered (``kdev doc export`` format; ``kdev doc import`` reads it);
- ``manifest.json``: utc, counts and md5 of every file (verify after copying).
Each file is written to a temp name and renamed, so a concurrent copy never sees a half-written file.
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from kdev.util import atomic_write_json, atomic_write_text, md5_file


def _snapshot_db(src: Path, dst: Path) -> None:
    tmp = dst.with_name(f".{dst.name}.tmp{os.getpid()}")
    if tmp.exists():
        tmp.unlink()
    s = sqlite3.connect(f"file:{src.as_posix()}?mode=ro", uri=True, timeout=30)
    d = sqlite3.connect(str(tmp))
    try:
        s.backup(d)
    finally:
        d.close()
        s.close()
    os.replace(tmp, dst)


def backup(home: str | None = None, out: str | None = None) -> dict[str, Any]:
    """Write the snapshot (see module docstring); returns {dir, files{rel: {bytes, md5}}, posts, docs, entries}."""
    from kdev.collab import db, docs
    cpath = db.default_path(home)
    base = cpath.parent
    od = Path(out).expanduser() if out else base / "backup"
    od.mkdir(parents=True, exist_ok=True)
    (od / "docs").mkdir(exist_ok=True)
    conn = db.connect(cpath)
    try:
        _snapshot_db(cpath, od / "collab.db")
        ledger = base / "ledger.db"
        try:
            from kdev.ledger.db import default_path
            ledger = default_path(home)
        except Exception:  # noqa: BLE001
            pass
        if ledger.is_file():
            _snapshot_db(ledger, od / "ledger.db")
        rows = conn.execute("SELECT * FROM posts ORDER BY post_id").fetchall()
        lines = []
        for r in rows:
            d = dict(r)
            d["mentions"] = [x[0] for x in conn.execute("SELECT agent FROM post_mentions WHERE post_id=?",
                                                        (r["post_id"],))]
            lines.append(json.dumps(d, ensure_ascii=False, sort_keys=True))
        atomic_write_text(od / "board.jsonl", "\n".join(lines) + ("\n" if lines else ""))
        dl = docs.docs_list(conn)
        for d in dl:
            atomic_write_text(od / "docs" / (d["doc"].replace("/", "__") + ".md"), docs.render(conn, d["doc"]))
        n_entries = conn.execute("SELECT count(*) FROM doc_entries").fetchone()[0]
    finally:
        conn.close()
    files = {}
    for p in sorted(od.rglob("*")):
        if p.is_file() and p.name != "manifest.json" and not p.name.startswith("."):
            files[p.relative_to(od).as_posix()] = {"bytes": p.stat().st_size, "md5": md5_file(str(p))}
    from kdev.collab.board import utc_now
    rep = {"dir": str(od), "utc": utc_now(), "posts": len(rows), "docs": len(dl), "entries": int(n_entries),
           "files": files}
    atomic_write_json(od / "manifest.json", rep)
    return rep
