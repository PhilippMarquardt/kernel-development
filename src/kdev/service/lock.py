"""Exclusive-GPU lock with foreign-context detection (port of okc/tools/gpu_run.sh). Owner: P2.

Semantics: take an flock on ``cfg.lock_path`` (default /root/gpu.lock, shared with the legacy gpu_run.sh so
legacy scripts and kdev exclude each other), then wait until no OTHER live process holds a CUDA context
(nvidia-smi --query-compute-apps=pid; zombies 'Z*' ignored; own pid and children ignored). After 10 s of
waiting, log each offender once (queue.db offenders + legacy offender_log). Abort after
``cfg.foreign_wait_s`` (600 s) with gpu_busy_timeout. On Windows (no fcntl) use msvcrt locking or a
lock-file fallback so unit tests run.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Iterator

from kdev.config import Config


@dataclass
class ComputeApp:
    pid: int
    used_mib: int | None = None
    cmdline: str = ""
    state: str = ""          # ps stat ('Z' = zombie)


@dataclass
class LockGrant:
    """What the holder learns when the lock is granted."""

    exclusive: str            # ok | waited | fail | shared
    waited_s: float
    offenders: list[ComputeApp] = field(default_factory=list)
    lock_path: str = ""


ComputeAppsFn = Callable[[], list[ComputeApp]]


def compute_apps() -> list[ComputeApp]:
    """Live CUDA processes from nvidia-smi + /proc (cmdline, stat). Empty list if nvidia-smi is missing."""
    raise NotImplementedError("lock.compute_apps (P2)")


def foreign_apps(apps: list[ComputeApp], own_pids: set[int]) -> list[ComputeApp]:
    """Filter out own pids and zombies (pure; unit-testable)."""
    raise NotImplementedError("lock.foreign_apps (P2)")


@contextmanager
def exclusive(cfg: Config, *, timeout_s: float | None = None, job_id: str | None = None,
              apps_fn: ComputeAppsFn = compute_apps, poll_s: float = 2.0) -> Iterator[LockGrant]:
    """flock + wait for no foreign CUDA contexts; yields a LockGrant; KdevError(gpu_busy_timeout) on abort.
    The holder must not compile (enforced by callers via requires_prebuilt)."""
    raise NotImplementedError("lock.exclusive (P2)")
    yield  # pragma: no cover


def run_locked(cfg: Config, argv: list[str], timeout_s: float = 300) -> int:
    """`kdev lock run -- CMD`: deny-list check (okc submit/check -> quota_forbidden), then exclusive() and
    exec with timeout; prints GPU_EXCLUSIVE=OK waited=Ns to stderr like gpu_run.sh."""
    raise NotImplementedError("lock.run_locked (P2)")


def is_denied(cfg: Config, argv: list[str]) -> bool:
    """True if argv matches cfg.deny_commands (word-prefix match after basename of argv[0])."""
    raise NotImplementedError("lock.is_denied (P2)")
