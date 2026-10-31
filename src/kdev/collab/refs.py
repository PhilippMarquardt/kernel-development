"""Typed evidence references shared by board posts and doc entries. Owner: P9.

A ref string is classified as (explicit prefixes win):
- ``run:ID`` or a run-dir id ``yyyymmdd-hhmmss-<tool>-...``          -> run (ledger ``runs``)
- ``idea:N`` / ``#N``                                                -> idea (ledger ``ideas``)
- ``judge:ID`` or 16 hex chars (okc job id)                          -> judge (ledger ``judge_results``)
- ``post:N``                                                         -> post (this board)
- ``http(s)://...``                                                  -> url
- ``file:PATH`` or an existing path (md5 recorded, path made absolute) -> file
- anything else                                                      -> text
:func:`describe` looks a ref up in the ledger (read-only; missing ledger = unknown) for compact display.
"""
from __future__ import annotations

import os
import re
import sqlite3
from typing import Any

KINDS = ("run", "idea", "judge", "post", "url", "file", "text")
_RUN = re.compile(r"^\d{8}-\d{6}-[\w.-]+$")
_JUDGE = re.compile(r"^[0-9a-f]{16}$")


def parse_ref(s: str) -> dict[str, str]:
    """``s`` -> {"kind", "ref"[, "md5"]} (see module docstring)."""
    s = s.strip()
    m = re.match(r"^(run|idea|judge|post|url|file|text):(.+)$", s)
    if m:
        kind, ref = m.group(1), m.group(2).strip()
    elif re.fullmatch(r"#\d+", s):
        kind, ref = "idea", s[1:]
    elif _RUN.match(s):
        kind, ref = "run", s
    elif _JUDGE.match(s):
        kind, ref = "judge", s
    elif re.match(r"^https?://", s):
        kind, ref = "url", s
    elif os.path.exists(s):
        kind, ref = "file", s
    else:
        kind, ref = "text", s
    out = {"kind": kind, "ref": ref}
    if kind == "file" and os.path.isfile(ref):
        from kdev.util import md5_file
        out["ref"] = os.path.abspath(ref)
        out["md5"] = md5_file(ref)
    return out


def parse_refs(items: list[str] | None) -> list[dict[str, str]]:
    """Repeatable/comma-separated ``--ref`` values -> typed refs (duplicates dropped, order kept)."""
    out: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for it in items or []:
        for part in str(it).split(","):
            if part.strip():
                r = parse_ref(part)
                if (r["kind"], r["ref"]) not in seen:
                    seen.add((r["kind"], r["ref"]))
                    out.append(r)
    return out


def short(r: dict[str, Any]) -> str:
    """One-token display: ``run:2026...``, ``#12``, ``judge:abc``, ``file:name.py@md5[:8]``."""
    k, v = r.get("kind"), str(r.get("ref", ""))
    if k == "idea":
        return f"#{v}"
    if k == "file":
        return f"file:{os.path.basename(v)}" + (f"@{r['md5'][:8]}" if r.get("md5") else "")
    if k == "text":
        return v[:40]
    return f"{k}:{v}"


def _ledger(home: str | None) -> sqlite3.Connection | None:
    try:
        from kdev.ledger.db import default_path
        p = default_path(home)
        if not p.is_file():
            return None
        c = sqlite3.connect(f"file:{p.as_posix()}?mode=ro", uri=True, timeout=5)
        c.row_factory = sqlite3.Row
        return c
    except Exception:  # noqa: BLE001 - lookups are best effort
        return None


def describe(refs: list[dict[str, Any]], home: str | None = None) -> list[str]:
    """Ledger facts for run/idea/judge refs (``run:R ab faster``, ``#12 limit/confirmed``, ``judge:J x1.53``);
    other refs as :func:`short`. Never raises."""
    c = _ledger(home) if any(r.get("kind") in ("run", "idea", "judge") for r in refs) else None
    out = []
    for r in refs:
        s = short(r)
        try:
            if c is not None and r["kind"] == "run":
                row = c.execute("SELECT tool, verdict, status FROM runs WHERE run_id=?", (r["ref"],)).fetchone()
                s += f" ({row['tool']} {row['verdict'] or row['status']})" if row else " (not in ledger)"
            elif c is not None and r["kind"] == "idea":
                row = c.execute("SELECT kind, status, title FROM ideas WHERE idea_id=?", (int(r["ref"]),)).fetchone()
                s += f" ({row['kind']}/{row['status']} {row['title'][:30]})" if row else " (unknown idea)"
            elif c is not None and r["kind"] == "judge":
                row = c.execute("SELECT problem, speedup FROM judge_results WHERE judge_id=?",
                                (r["ref"],)).fetchone()
                s += f" ({row['problem']} x{row['speedup']:.3f})" if row and row["speedup"] else ""
        except Exception:  # noqa: BLE001
            pass
        out.append(s)
    if c is not None:
        c.close()
    return out
