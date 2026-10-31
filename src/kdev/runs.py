"""Run directories and paged reads (`kdev show`). Owner: P1. See docs/PLAN.md sections 3.3 and 6.

Layout: $KDEV_HOME/runs/<yyyymmdd-hhmmss>-<tool>-<slug>/ with meta.json, result.json, raw.jsonl, log.txt and
tool artifacts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kdev.config import Config
from kdev.contract import Result


@dataclass
class RunDir:
    run_id: str
    path: Path

    def file(self, name: str) -> Path:
        return self.path / name

    def write_json(self, name: str, obj: Any) -> Path:
        """Atomic write (tmp + os.replace) of a JSON document."""
        raise NotImplementedError("RunDir.write_json (P1)")

    def append_jsonl(self, name: str, obj: Any) -> None:
        raise NotImplementedError("RunDir.append_jsonl (P1)")

    def write_result(self, result: Result) -> Path:
        """Write result.json; fills result.run_id, result.files['result'] and result.files['dir']."""
        raise NotImplementedError("RunDir.write_result (P1)")


@dataclass
class Page:
    """One page of a run section."""

    run_id: str
    section: str
    offset: int
    limit: int
    lines: list[str] = field(default_factory=list)
    total: int | None = None
    next_offset: int | None = None   # None = end


def new_run(cfg: Config, tool: str, slug: str = "") -> RunDir:
    """Create a new run dir (id = <yyyymmdd-hhmmss>-<tool>[-<slug>], uniquified)."""
    raise NotImplementedError("runs.new_run (P1)")


def open_run(cfg: Config, run_id: str) -> RunDir:
    """Find an existing run by id or unique prefix; KdevError(not_found) otherwise."""
    raise NotImplementedError("runs.open_run (P1)")


def list_runs(cfg: Config, tool: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
    raise NotImplementedError("runs.list_runs (P1)")


def show(cfg: Config, run_id: str, section: str = "summary", offset: int = 0, limit: int = 60) -> Page:
    """Paged read of a section: any key of result.files, or summary/meta/log/raw."""
    raise NotImplementedError("runs.show (P1)")
