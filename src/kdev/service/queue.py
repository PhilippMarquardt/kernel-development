"""SQLite job queue ($KDEV_HOME/queue.db, schema in schema.sql). Owner: P2.

Clients only INSERT queued jobs and cancel their own queued jobs; the daemon owns all other transitions.
Connections are autocommit (``isolation_level=None``); multi-statement transitions use ``BEGIN IMMEDIATE``
so two daemons/clients can never claim the same job. WAL + busy timeout make concurrent readers safe.
"""
from __future__ import annotations

import json
import os
import socket
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kdev import __version__
from kdev.config import Config, no_gpu
from kdev.contract import ErrorCode, KdevError
from kdev.models import JOB_KINDS, JobSpec, JobStatus
from kdev.service._common import pid_alive
from kdev.util import new_id, utc_now

TERMINAL = ("done", "failed", "cancelled", "timeout")
STATES = ("queued", "running") + TERMINAL
_SCHEMA = Path(__file__).with_name("schema.sql")
_ORDER = "priority DESC, submitted_utc ASC, rowid ASC"


def connect(cfg: Config) -> sqlite3.Connection:
    """Open (and create from schema.sql) queue.db in WAL mode with a busy timeout."""
    path = Path(cfg.queue_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.executescript(_SCHEMA.read_text(encoding="utf-8"))
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


def _status(row: sqlite3.Row, position: int | None = None) -> JobStatus:
    return JobStatus(job_id=row["job_id"], state=row["state"], lane=row["lane"], kind=row["kind"],
                     agent=row["agent"] or "", position=position, run_id=row["run_id"], run_dir=row["run_dir"],
                     submitted_utc=row["submitted_utc"], started_utc=row["started_utc"],
                     finished_utc=row["finished_utc"], exit_code=row["exit_code"], error_code=row["error_code"],
                     waited_s=row["waited_s"])


def _position(conn: sqlite3.Connection, row: sqlite3.Row) -> int | None:
    if row["state"] != "queued":
        return None
    ids = [r[0] for r in conn.execute(f"SELECT job_id FROM jobs WHERE lane=? AND state='queued' ORDER BY {_ORDER}",
                                      (row["lane"],))]
    return ids.index(row["job_id"]) + 1 if row["job_id"] in ids else None


def _row(conn: sqlite3.Connection, job_id: str) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
    if row is None:
        # unique-prefix lookup, so `kdev job status j20261031-1430` works
        rows = conn.execute("SELECT * FROM jobs WHERE job_id LIKE ? LIMIT 2", (job_id + "%",)).fetchall()
        if len(rows) == 1:
            return rows[0]
        raise KdevError(ErrorCode.NOT_FOUND, f"unknown job {job_id!r}", recovery="`kdev job list` shows job ids")
    return row


def validate_spec(spec: JobSpec) -> None:
    if spec.kind not in JOB_KINDS:
        raise KdevError(ErrorCode.INVALID_REQUEST, f"spec.kind {spec.kind!r} not in {JOB_KINDS}")
    if spec.resolved_lane() not in ("build", "gpu"):
        raise KdevError(ErrorCode.INVALID_REQUEST, f"spec.lane {spec.lane!r} must be build|gpu")
    if spec.kind == "exec" and not (spec.params or {}).get("argv"):
        raise KdevError(ErrorCode.INVALID_REQUEST, "exec job needs params.argv")


def enqueue(cfg: Config, spec: JobSpec) -> JobStatus:
    """Assign job_id/submitted_utc/lane if empty, insert as queued, return status with queue position."""
    validate_spec(spec)
    spec.job_id = spec.job_id or new_id("j")
    spec.submitted_utc = spec.submitted_utc or utc_now()
    spec.lane = spec.resolved_lane()
    conn = connect(cfg)
    try:
        try:
            conn.execute("INSERT INTO jobs(job_id, lane, kind, state, priority, agent, spec_json, submitted_utc) "
                         "VALUES (?,?,?,?,?,?,?,?)",
                         (spec.job_id, spec.lane, spec.kind, "queued", int(spec.priority or 0), spec.agent,
                          json.dumps(spec.to_dict(), default=str), spec.submitted_utc))
        except sqlite3.IntegrityError as e:
            raise KdevError(ErrorCode.CONFLICT, f"job id {spec.job_id} already exists") from e
        log_event(conn, "submitted", f"{spec.kind} by {spec.agent or '?'}", spec.job_id)
        row = _row(conn, spec.job_id)
        return _status(row, _position(conn, row))
    finally:
        conn.close()


def get(cfg: Config, job_id: str) -> JobStatus:
    conn = connect(cfg)
    try:
        row = _row(conn, job_id)
        return _status(row, _position(conn, row))
    finally:
        conn.close()


def get_spec(cfg: Config, job_id: str) -> JobSpec:
    conn = connect(cfg)
    try:
        return JobSpec.from_dict(json.loads(_row(conn, job_id)["spec_json"]))
    finally:
        conn.close()


def list_jobs(cfg: Config, state: str | None = None, limit: int = 50) -> list[JobStatus]:
    if state and state not in STATES:
        raise KdevError(ErrorCode.INVALID_REQUEST, f"state {state!r} not in {STATES}")
    conn = connect(cfg)
    try:
        q, args = "SELECT * FROM jobs", []
        if state:
            q += " WHERE state=?"
            args.append(state)
        q += " ORDER BY submitted_utc DESC, rowid DESC LIMIT ?"
        args.append(int(limit))
        return [_status(r, _position(conn, r)) for r in conn.execute(q, args)]
    finally:
        conn.close()


def cancel(cfg: Config, job_id: str, agent: str = "") -> JobStatus:
    """Queued -> cancelled; running -> request cancellation (event) and let the daemon kill it."""
    conn = connect(cfg)
    try:
        row = _row(conn, job_id)
        job_id = row["job_id"]
        if agent and row["agent"] and agent != row["agent"]:
            raise KdevError(ErrorCode.CONFLICT, f"job {job_id} belongs to agent {row['agent']!r}, not {agent!r}",
                            recovery=f"cancel with --agent {row['agent']} if you really mean it")
        if row["state"] == "queued":
            cur = conn.execute("UPDATE jobs SET state='cancelled', finished_utc=?, error_code='cancelled' "
                               "WHERE job_id=? AND state='queued'", (utc_now(), job_id))
            if cur.rowcount:
                log_event(conn, "cancelled", f"by {agent or '?'} while queued", job_id)
        elif row["state"] == "running":
            log_event(conn, "cancel_request", f"by {agent or '?'}", job_id)
        row = _row(conn, job_id)
        return _status(row, _position(conn, row))
    finally:
        conn.close()


def cancel_requested(conn: sqlite3.Connection, job_id: str) -> bool:
    return conn.execute("SELECT 1 FROM events WHERE job_id=? AND kind='cancel_request' LIMIT 1",
                        (job_id,)).fetchone() is not None


def claim_next(conn: sqlite3.Connection, lane: str) -> JobSpec | None:
    """Daemon: atomically move the highest-priority oldest queued job of a lane to running."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute(f"SELECT job_id, spec_json FROM jobs WHERE lane=? AND state='queued' ORDER BY {_ORDER} "
                           "LIMIT 1", (lane,)).fetchone()
        if row is None:
            conn.execute("COMMIT")
            return None
        conn.execute("UPDATE jobs SET state='running', started_utc=? WHERE job_id=?", (utc_now(), row["job_id"]))
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    spec = JobSpec.from_dict(json.loads(row["spec_json"]))
    spec.job_id = row["job_id"]
    return spec


def claim(conn: sqlite3.Connection, job_id: str) -> bool:
    """Daemon/tests: move one specific queued job to running; False if it was not queued."""
    cur = conn.execute("UPDATE jobs SET state='running', started_utc=? WHERE job_id=? AND state='queued'",
                       (utc_now(), job_id))
    return cur.rowcount == 1


def set_running_info(conn: sqlite3.Connection, job_id: str, *, pid: int | None = None, run_id: str | None = None,
                     run_dir: str | None = None) -> None:
    conn.execute("UPDATE jobs SET pid=COALESCE(?, pid), run_id=COALESCE(?, run_id), run_dir=COALESCE(?, run_dir) "
                 "WHERE job_id=?", (pid, run_id, run_dir, job_id))


def finish(conn: sqlite3.Connection, job_id: str, state: str, *, exit_code: int | None = None,
           error_code: str | None = None, run_id: str | None = None, run_dir: str | None = None,
           waited_s: float | None = None) -> None:
    if state not in TERMINAL:
        raise ValueError(f"finish: {state!r} is not terminal")
    conn.execute("UPDATE jobs SET state=?, finished_utc=?, exit_code=?, error_code=?, "
                 "run_id=COALESCE(?, run_id), run_dir=COALESCE(?, run_dir), waited_s=COALESCE(?, waited_s) "
                 "WHERE job_id=?", (state, utc_now(), exit_code, error_code, run_id, run_dir, waited_s, job_id))
    log_event(conn, "finished", f"{state} exit={exit_code} {error_code or ''}".strip(), job_id)


def log_event(conn: sqlite3.Connection, kind: str, message: str, job_id: str | None = None) -> None:
    conn.execute("INSERT INTO events(ts_utc, job_id, kind, message) VALUES (?,?,?,?)",
                 (utc_now(), job_id, kind, message))


def log_offender(conn: sqlite3.Connection, pid: int, cmdline: str, *, gpu_mem_mib: int | None = None,
                 during_job: str | None = None, action: str = "waited") -> None:
    conn.execute("INSERT INTO offenders(ts_utc, pid, cmdline, gpu_mem_mib, during_job, action) VALUES (?,?,?,?,?,?)",
                 (utc_now(), int(pid), cmdline, gpu_mem_mib, during_job, action))
    log_event(conn, "offender", f"pid {pid} {action}: {cmdline[:120]}", during_job)


def heartbeat(conn: sqlite3.Connection, *, gpu_lane_job: str | None, build_workers: int) -> None:
    conn.execute("INSERT OR REPLACE INTO heartbeat(id, pid, host, ts_utc, version, gpu_lane_job, build_workers) "
                 "VALUES (1,?,?,?,?,?,?)", (os.getpid(), socket.gethostname(), utc_now(), __version__,
                                            gpu_lane_job, build_workers))


def clear_heartbeat(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM heartbeat WHERE id=1 AND pid=?", (os.getpid(),))


def _age_s(ts_utc: str) -> float:
    import datetime as dt
    try:
        t = dt.datetime.strptime(ts_utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
    except (TypeError, ValueError):
        return float("inf")
    return (dt.datetime.now(dt.timezone.utc) - t).total_seconds()


def heartbeat_row(cfg: Config) -> dict[str, Any] | None:
    if not Path(cfg.queue_path).exists():
        return None
    conn = connect(cfg)
    try:
        row = conn.execute("SELECT * FROM heartbeat WHERE id=1").fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def daemon_alive(cfg: Config, max_age_s: float = 15.0) -> bool:
    """True if the heartbeat row is fresher than max_age_s and its pid is alive."""
    hb = heartbeat_row(cfg)
    if not hb or _age_s(hb["ts_utc"]) > max_age_s:
        return False
    if hb.get("host") and hb["host"] != socket.gethostname():
        return True        # another host sharing KDEV_HOME: trust the fresh timestamp
    return pid_alive(hb.get("pid"))


@dataclass
class QueueSnapshot:
    """Everything `queue status` and the dashboard show."""

    daemon: dict[str, Any] = field(default_factory=dict)
    running: list[JobStatus] = field(default_factory=list)
    queued: list[JobStatus] = field(default_factory=list)
    recent: list[JobStatus] = field(default_factory=list)
    offenders: list[dict[str, Any]] = field(default_factory=list)
    gpu: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"daemon": self.daemon, "running": [j.to_dict() for j in self.running],
                "queued": [j.to_dict() for j in self.queued], "recent": [j.to_dict() for j in self.recent],
                "offenders": self.offenders, "gpu": self.gpu}


def offenders_since(cfg: Config, since_s: float | None = 3 * 3600) -> list[dict[str, Any]]:
    import datetime as dt
    conn = connect(cfg)
    try:
        if since_s is None:
            rows = conn.execute("SELECT * FROM offenders ORDER BY id DESC").fetchall()
        else:
            cut = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=since_s)).strftime("%Y-%m-%dT%H:%M:%SZ")
            rows = conn.execute("SELECT * FROM offenders WHERE ts_utc >= ? ORDER BY id DESC", (cut,)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def snapshot(cfg: Config, recent: int = 20, offenders_since_s: float = 3 * 3600) -> QueueSnapshot:
    snap = QueueSnapshot()
    hb = heartbeat_row(cfg)
    snap.daemon = dict(hb or {})
    snap.daemon["alive"] = daemon_alive(cfg)
    conn = connect(cfg)
    try:
        def rows(where: str, order: str, lim: int | None = None) -> list[JobStatus]:
            q = f"SELECT * FROM jobs WHERE {where} ORDER BY {order}" + (f" LIMIT {int(lim)}" if lim else "")
            return [_status(r, _position(conn, r)) for r in conn.execute(q)]
        snap.running = rows("state='running'", "started_utc")
        snap.queued = rows("state='queued'", "lane, " + _ORDER)
        snap.recent = rows("state IN ('done','failed','cancelled','timeout')", "finished_utc DESC", recent)
    finally:
        conn.close()
    snap.offenders = offenders_since(cfg, offenders_since_s)
    from kdev.service import lock
    gpu: dict[str, Any] = {"lock": lock.holder(cfg)}
    if no_gpu():
        gpu["skipped"] = "KDEV_NO_GPU=1"
    else:
        from kdev.service import smi
        gpu["state"] = smi.query_state().to_dict()
        gpu["foreign_apps"] = [a.__dict__ for a in lock.foreign_apps(lock.compute_apps(), {os.getpid()})]
    gpu["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    snap.gpu = gpu
    return snap
