"""Run-dir / result plumbing shared by build, sass and sweep (P5, private).

Uses ``kdev.runs`` (P1) when implemented; falls back to an equivalent local implementation so P5 commands
work while P1 is in progress. Ledger ingestion (P4) is best effort.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
import secrets
from pathlib import Path
from typing import Any

from kdev.contract import Result


def home(ctx: Any = None) -> Path:
    h = getattr(ctx, "home", None) or os.environ.get("KDEV_HOME") or "~/.kdev"
    return Path(h).expanduser()


class _Run:
    def __init__(self, run_id: str, path: Path) -> None:
        self.run_id = run_id
        self.path = path

    def file(self, name: str) -> Path:
        return self.path / name


def new_run(ctx: Any, tool: str, slug: str = "") -> Any:
    try:
        from kdev.config import load_config  # noqa: PLC0415
        from kdev.runs import new_run as _nr  # noqa: PLC0415

        cfg = load_config(home=str(home(ctx)))
        return _nr(cfg, tool, slug)
    except NotImplementedError:
        pass
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", slug)[:40].strip("-")
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    base = f"{ts}-{tool.replace(' ', '_')}" + (f"-{slug}" if slug else "")
    root = home(ctx) / "runs"
    root.mkdir(parents=True, exist_ok=True)
    rid = base
    while (root / rid).exists():
        rid = f"{base}-{secrets.token_hex(2)}"
    (root / rid).mkdir()
    return _Run(rid, root / rid)


def write_json(run: Any, name: str, obj: Any) -> Path:
    p = Path(run.path) / name
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, p)
    return p


def write_text(run: Any, name: str, text: str) -> Path:
    p = Path(run.path) / name
    p.write_text(text, encoding="utf-8", newline="\n")
    return p


def finish(run: Any, result: Result, ingest: bool = True) -> Result:
    """Fill run_id/files, write result.json, ingest into the ledger (best effort)."""
    result.run_id = run.run_id
    result.files.setdefault("dir", str(run.path))
    result.files["result"] = str(Path(run.path) / "result.json")
    try:
        run.write_result(result)   # P1 RunDir
    except (AttributeError, NotImplementedError):
        write_json(run, "result.json", result.to_dict())
    if ingest:
        try:
            from kdev.ledger.record import ingest_result  # noqa: PLC0415

            ingest_result(result.files["result"])
        except Exception:  # noqa: BLE001 - ledger is optional until P4 lands
            pass
    return result
