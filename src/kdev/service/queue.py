"""SQLite job queue ($KDEV_HOME/queue.db, schema in schema.sql). Owner: P2.

Clients only INSERT queued jobs and cancel their own queued jobs; the daemon owns all other transitions.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any

from kdev.config import Config
from kdev.models import JobSpec, JobStatus


def connect(cfg: Config) -> sqlite3.Connection:
    """Open (and create from schema.sql) queue.db in WAL mode with a busy timeout."""
    raise NotImplementedError("queue.connect (P2)")


def enqueue(cfg: Config, spec: JobSpec) -> JobStatus:
    """Assign job_id/submitted_utc/lane if empty, insert as queued, return status with queue position."""
    raise NotImplementedError("queue.enqueue (P2)")


def get(cfg: Config, job_id: str) -> JobStatus:
    raise NotImplementedError("queue.get (P2)")


def list_jobs(cfg: Config, state: str | None = None, limit: int = 50) -> list[JobStatus]:
    raise NotImplementedError("queue.list_jobs (P2)")


def cancel(cfg: Config, job_id: str, agent: str = "") -> JobStatus:
    """Queued -> cancelled; running -> request cancellation (event) and let the daemon kill it."""
    raise NotImplementedError("queue.cancel (P2)")


def claim_next(conn: sqlite3.Connection, lane: str) -> JobSpec | None:
    """Daemon: atomically move the highest-priority oldest queued job of a lane to running."""
    raise NotImplementedError("queue.claim_next (P2)")


def finish(conn: sqlite3.Connection, job_id: str, state: str, *, exit_code: int | None = None,
           error_code: str | None = None, run_id: str | None = None, run_dir: str | None = None,
           waited_s: float | None = None) -> None:
    raise NotImplementedError("queue.finish (P2)")


def log_event(conn: sqlite3.Connection, kind: str, message: str, job_id: str | None = None) -> None:
    raise NotImplementedError("queue.log_event (P2)")


def log_offender(conn: sqlite3.Connection, pid: int, cmdline: str, *, gpu_mem_mib: int | None = None,
                 during_job: str | None = None, action: str = "waited") -> None:
    raise NotImplementedError("queue.log_offender (P2)")


def heartbeat(conn: sqlite3.Connection, *, gpu_lane_job: str | None, build_workers: int) -> None:
    raise NotImplementedError("queue.heartbeat (P2)")


def daemon_alive(cfg: Config, max_age_s: float = 15.0) -> bool:
    """True if the heartbeat row is fresher than max_age_s and its pid is alive."""
    raise NotImplementedError("queue.daemon_alive (P2)")


@dataclass
class QueueSnapshot:
    """Everything `queue status` and the dashboard show."""

    daemon: dict[str, Any] = field(default_factory=dict)
    running: list[JobStatus] = field(default_factory=list)
    queued: list[JobStatus] = field(default_factory=list)
    recent: list[JobStatus] = field(default_factory=list)
    offenders: list[dict[str, Any]] = field(default_factory=list)
    gpu: dict[str, Any] = field(default_factory=dict)


def snapshot(cfg: Config, recent: int = 20, offenders_since_s: float = 3 * 3600) -> QueueSnapshot:
    raise NotImplementedError("queue.snapshot (P2)")
