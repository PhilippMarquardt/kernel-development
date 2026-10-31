"""Configuration (kdev.toml + environment). Owner: P1. See docs/PLAN.md section 3.10.

Search order: $KDEV_CONFIG, ./kdev.toml, $KDEV_HOME/kdev.toml. Env overrides: KDEV_HOME, KDEV_AGENT,
KDEV_NO_GPU, KDEV_GPU_TESTS, KDEV_LOCK_PATH, KDEV_PYTHON.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RemoteConfig:
    host: str = ""
    port: int = 22
    user: str = "root"
    key: str = ""
    remote_home: str = "/root/.kdev"


@dataclass
class Config:
    home: Path = field(default_factory=lambda: Path(os.environ.get("KDEV_HOME", "~/.kdev")).expanduser())
    lock_path: str = "/root/gpu.lock"            # shared with legacy gpu_run.sh
    offender_log: str = "/root/gpu_offenders.log"
    max_hold_s: int = 1800
    build_workers: int = 8
    foreign_wait_s: int = 600                     # gpu_run.sh aborts after 600 s of foreign contexts
    problems_root: str = ""
    default_problem: str = ""
    python: str = ""                              # worker interpreter ("" = sys.executable)
    deny_commands: list[str] = field(default_factory=lambda: ["okc submit", "okc check"])
    remote: RemoteConfig = field(default_factory=RemoteConfig)
    sync_paths: list[str] = field(default_factory=list)
    dashboard_out: str = ""
    board_enabled: bool = False

    @property
    def runs_dir(self) -> Path:
        return self.home / "runs"

    @property
    def ledger_path(self) -> Path:
        return self.home / "ledger.db"

    @property
    def queue_path(self) -> Path:
        return self.home / "queue.db"

    @property
    def cache_dir(self) -> Path:
        return self.home / "cache"


def load_config(home: str | None = None, path: str | None = None) -> Config:
    """Load kdev.toml (tomllib) + env overrides. Does not create directories."""
    raise NotImplementedError("config.load_config (P1)")


def no_gpu() -> bool:
    """True when KDEV_NO_GPU=1 (unit tests): GPU commands return capability_unavailable."""
    return os.environ.get("KDEV_NO_GPU", "") == "1"
