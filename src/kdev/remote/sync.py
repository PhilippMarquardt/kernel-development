"""Bidirectional md5-verified sync pod <-> local. Owner: P2. (Pod storage is never persistent: M17.)

Rules: never overwrite a newer destination file; on conflict keep ``<name>.conflict-<ts>``; verify md5
after every copy; write a heartbeat file in cron mode. Ports scripts/sync_okc.sh logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from kdev.config import Config


@dataclass
class SyncPlanItem:
    rel_path: str
    action: str            # copy_to_local | copy_to_remote | conflict | skip_same
    src_md5: str = ""
    dst_md5: str = ""


@dataclass
class SyncReport:
    copied: int = 0
    conflicts: list[str] = field(default_factory=list)
    verified: int = 0
    failed: list[str] = field(default_factory=list)
    bytes: int = 0


def plan(local_index: dict[str, tuple[str, float]], remote_index: dict[str, tuple[str, float]],
         direction: str) -> list[SyncPlanItem]:
    """Pure planner over {rel_path: (md5, mtime)} indexes; direction pull|push|both."""
    raise NotImplementedError("sync.plan (P2)")


def sync(cfg: Config, direction: str = "pull", since_s: float | None = None, dry_run: bool = False
         ) -> SyncReport:
    raise NotImplementedError("sync.sync (P2)")
