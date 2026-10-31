"""Run-dir file helpers for the worker and engine (private to P3; independent of P1's RunDir methods so
the timing lane works before/without them). Accepts a kdev.runs.RunDir, a path or a str."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def run_path(run_dir: Any) -> Path:
    if run_dir is None:
        return Path(".")
    p = getattr(run_dir, "path", run_dir)
    return Path(p)


def run_id(run_dir: Any) -> str | None:
    rid = getattr(run_dir, "run_id", None)
    return rid or (run_path(run_dir).name if run_dir is not None else None)


def write_json(run_dir: Any, name: str, obj: Any) -> Path:
    p = run_path(run_dir) / name
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, default=str)
        f.write("\n")
    os.replace(tmp, p)
    return p


def append_jsonl(run_dir: Any, name: str, rows: list[Any]) -> Path:
    p = run_path(run_dir) / name
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":"), default=str) + "\n")
    return p


def write_text(run_dir: Any, name: str, text: str) -> Path:
    p = run_path(run_dir) / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8", newline="\n")
    return p


def log(run_dir: Any, msg: str) -> None:
    """Append a line to log.txt and echo to stderr (stdout stays reserved for the Result JSON)."""
    import sys
    import time

    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, file=sys.stderr, flush=True)
    try:
        p = run_path(run_dir) / "log.txt"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8", newline="\n") as f:
            f.write(line + "\n")
    except OSError:
        pass
