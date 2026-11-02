"""`kdev serve`: the GPU job service daemon. Owner: P2. See docs/PLAN.md section 2.

- Build lane: a process pool of ``cfg.build_workers`` running build jobs with CUDA_VISIBLE_DEVICES="" (no
  CUDA context, never under the lock). Implemented by calling ``kdev.build.compile.build_job(spec, run_dir)``.
- GPU lanes: up to ``gpu_lanes`` jobs at a time (env KDEV_GPU_LANES, default 3). Safe because every worker takes
  the GPU lock itself: phased jobs prepare under the SHARED lock and time under the EXCLUSIVE one, legacy jobs
  wait for the exclusive lock (a lane spent waiting costs nothing). With one lane, a long ab blocked every
  short discover/check job behind it (pod 2026-11-01: 1 running + 6 queued at 18 % GPU). For each job: refuse if ``requires_prebuilt`` and
  ``kdev.build.cache.is_prebuilt(spec)`` is False (needs_build, unless allow_jit); then
  ``lock.exclusive()``; spawn ``<python> -m kdev.worker job.json --run-dir DIR`` with timeout; enforce
  max_hold_s (warn event at 80 %, kill at 100 %); record clocks before/after; mark the result not_exclusive if
  a foreign context appeared during the run (poll compute_apps while the worker runs).
- After each finished job: ``kdev.ledger.record.ingest_result(result.json)``.
- Heartbeat every 5 s. SIGTERM/SIGINT: stop claiming, let the running GPU job finish up to 60 s, then kill
  it and mark ``timeout`` (partial raw.jsonl survives). Queue persists across restarts; jobs left 'running'
  by a dead daemon are marked failed(internal) on start.

Implementation notes: build jobs are subprocesses (``python -m kdev.service._job``), not a multiprocessing
pool, so each one can be killed/timed out alone and a crashing compiler never takes the daemon down. The
build concurrency is capped by the cgroup CPU quota (RunPod: nproc=192 but ~20 usable CPUs) and 16.
A second daemon on the same KDEV_HOME refuses to start (flock on $KDEV_HOME/serve.lock). ``stop`` works by
a stop file ($KDEV_HOME/serve.stop, portable) plus SIGTERM on POSIX.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

from kdev.config import Config
from kdev.contract import ErrorCode
from kdev.service import _jobrun, queue
from kdev.service._common import build_worker_cap, pid_alive
from kdev.service.lock import _FileLock, _kill_tree
from kdev.util import utc_now

HEARTBEAT_S = 5.0
STALE_S = 30.0            # ensure(): no heartbeat for this long (or a dead pid) = the daemon is gone
# job-scoped env of a client that runs inside a kdev job: never inherited by a daemon it starts
_JOB_ENV = ("KDEV_JOB_ID", "KDEV_PHASED", "KDEV_PHASE_MODE", "KDEV_PHASE_GO", "KDEV_PHASE_SLOT_FD", "KDEV_PHASE_WHOLE",
            "KDEV_EXCLUSIVE", "KDEV_WAITED_S", "KDEV_LOG_CAPTURED")
_STOP = threading.Event()


@dataclass
class DaemonOptions:
    build_workers: int = 8
    gpu_lanes: int = int(os.environ.get("KDEV_GPU_LANES", "3") or 3)
    foreground: bool = True
    poll_s: float = 0.5
    python: str = ""          # worker interpreter ("" = cfg.python or sys.executable)


def _stop_file(cfg: Config) -> Path:
    return Path(cfg.home) / "serve.stop"


def _stopping(cfg: Config) -> bool:
    if not _STOP.is_set() and _stop_file(cfg).exists():
        _STOP.set()
    return _STOP.is_set()


def _recover(cfg: Config, conn) -> int:
    """Jobs left 'running' by a dead daemon -> failed(internal); kill a surviving orphan worker first."""
    n = 0
    for row in conn.execute("SELECT job_id, pid FROM jobs WHERE state='running'").fetchall():
        pid = row["pid"]
        if pid and pid_alive(pid) and sys.platform != "win32":
            try:
                os.killpg(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass
        queue.finish(conn, row["job_id"], "failed", error_code=ErrorCode.INTERNAL.value)
        queue.log_event(conn, "recovered", "daemon restarted while job was running", row["job_id"])
        n += 1
    return n


def serve(cfg: Config, opts: DaemonOptions) -> int:
    """Main loop; returns the exit code."""
    if opts.python:
        cfg.python = opts.python
    Path(cfg.home).mkdir(parents=True, exist_ok=True)
    guard = _FileLock(str(Path(cfg.home) / "serve.lock"))
    if not guard.try_acquire():
        print(f"kdev serve: another daemon already serves {cfg.home}", file=sys.stderr)
        guard.release()
        return 1
    _STOP.clear()
    try:
        _stop_file(cfg).unlink()
    except OSError:
        pass
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, lambda *_: _STOP.set())
            except (ValueError, OSError):
                pass
    workers = build_worker_cap(opts.build_workers)
    conn = queue.connect(cfg)
    builds: dict[str, _jobrun.BuildTask] = {}
    lanes = max(1, int(opts.gpu_lanes))
    gpu_threads: dict[str, threading.Thread] = {}
    try:
        n = _recover(cfg, conn)
        queue.log_event(conn, "serve_start", f"pid {os.getpid()} build_workers={workers} gpu_lanes={lanes} recovered={n}")
        print(f"kdev serve: pid {os.getpid()} home {cfg.home} build_workers={workers} lock {cfg.lock_path}",
              file=sys.stderr, flush=True)
        last_hb = 0.0
        while True:
            stopping = _stopping(cfg)
            if time.monotonic() - last_hb >= HEARTBEAT_S:
                queue.heartbeat(conn, gpu_lane_job=",".join(gpu_threads) or None, build_workers=workers)
                last_hb = time.monotonic()
            # build lane: reap, then fill free slots
            for job_id, task in list(builds.items()):
                try:
                    res = task.poll(cancel_fn=lambda j=job_id: queue.cancel_requested(conn, j), stop_fn=_STOP.is_set)
                except Exception as e:  # noqa: BLE001
                    traceback.print_exc()
                    queue.finish(conn, job_id, "failed", error_code=ErrorCode.INTERNAL.value)
                    queue.log_event(conn, "error", f"{type(e).__name__}: {e}", job_id)
                    del builds[job_id]
                    continue
                if res is not None:
                    queue.finish(conn, job_id, _jobrun.state_for(res), exit_code=res.exit_code,
                                 error_code=res.error.code if res.error else None, run_id=task.rd.run_id,
                                 run_dir=str(task.rd.path))
                    del builds[job_id]
            while not stopping and len(builds) < workers:
                spec = queue.claim_next(conn, "build")
                if spec is None:
                    break
                try:
                    task = _jobrun.BuildTask(cfg, spec, _jobrun.prepare_run(cfg, spec))
                    queue.set_running_info(conn, spec.job_id, run_id=task.rd.run_id, run_dir=str(task.rd.path))
                    task.start()
                    queue.set_running_info(conn, spec.job_id, pid=task.proc.pid if task.proc else None)
                    queue.log_event(conn, "started", "build lane", spec.job_id)
                    builds[spec.job_id] = task
                except Exception as e:  # noqa: BLE001
                    traceback.print_exc()
                    queue.finish(conn, spec.job_id, "failed", error_code=ErrorCode.INTERNAL.value)
                    queue.log_event(conn, "error", f"build start: {type(e).__name__}: {e}", spec.job_id)
            # GPU lanes: up to `lanes` jobs, each in a thread so heartbeats and builds continue; the worker's
            # own lock (phased shared/exclusive) serialises the timed sections.
            for jid, th in list(gpu_threads.items()):
                if not th.is_alive():
                    del gpu_threads[jid]
            while not stopping and len(gpu_threads) < lanes:
                spec = queue.claim_next(conn, "gpu")
                if spec is None:
                    break
                th = threading.Thread(target=run_gpu_job, args=(cfg, spec.job_id), daemon=True,
                                      name=f"gpu-{spec.job_id}")
                gpu_threads[spec.job_id] = th
                th.start()
            if stopping and not gpu_threads and not builds:
                break
            time.sleep(opts.poll_s)
    finally:
        for job_id, task in builds.items():
            if task.proc is not None:
                _kill_tree(task.proc)
        for th in gpu_threads.values():
            th.join(_jobrun.STOP_GRACE_S + 30)
        try:
            queue.log_event(conn, "serve_stop", f"pid {os.getpid()}")
            queue.clear_heartbeat(conn)
        finally:
            conn.close()
            try:
                _stop_file(cfg).unlink()
            except OSError:
                pass
            guard.release()
    return 0


def start_background(cfg: Config, opts: DaemonOptions) -> int:
    """Daemonise (setsid + nohup-style, log to $KDEV_HOME/serve.log); returns the pid."""
    Path(cfg.home).mkdir(parents=True, exist_ok=True)
    if queue.daemon_alive(cfg):
        hb = queue.heartbeat_row(cfg) or {}
        return int(hb.get("pid") or 0)
    argv = [sys.executable, "-m", "kdev", "serve", "run", "--foreground", "--build-workers", str(opts.build_workers),
            "--home", str(cfg.home)]
    env = {k: v for k, v in os.environ.items() if k not in _JOB_ENV}
    if env.get("CUDA_VISIBLE_DEVICES") == "":       # started from a build-lane child: the GPU lanes need CUDA
        env.pop("CUDA_VISIBLE_DEVICES")
    env["KDEV_HOME"] = str(cfg.home)
    env["KDEV_LOCK_PATH"] = cfg.lock_path
    src = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if opts.python or cfg.python:
        env["KDEV_PYTHON"] = opts.python or cfg.python
    log = open(Path(cfg.home) / "serve.log", "a", encoding="utf-8")
    log.write(f"\n[{utc_now()}] start: {' '.join(argv)}\n")
    log.flush()
    kw: dict = {"stdout": log, "stderr": subprocess.STDOUT, "stdin": subprocess.DEVNULL, "env": env,
                "cwd": str(cfg.home)}
    if sys.platform == "win32":
        kw["creationflags"] = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kw["start_new_session"] = True
    proc = subprocess.Popen(argv, **kw)
    log.close()
    t0 = time.monotonic()
    while time.monotonic() - t0 < 15:
        if queue.daemon_alive(cfg):
            break
        if proc.poll() is not None:
            break
        time.sleep(0.2)
    return proc.pid


def heartbeat_age_s(cfg: Config) -> float | None:
    """Seconds since the last heartbeat (None: no heartbeat row: never served here, or stopped cleanly)."""
    try:
        hb = queue.heartbeat_row(cfg)
    except Exception:  # noqa: BLE001
        return None
    return queue._age_s(hb["ts_utc"]) if hb else None


def ensure(cfg: Config, opts: DaemonOptions | None = None, *, stale_s: float = STALE_S,
           wait_s: float = 15.0) -> dict:
    """`kdev serve ensure`: start the daemon unless one is alive (heartbeat younger than ``stale_s`` and a live
    pid). Idempotent: concurrent callers serialise on $KDEV_HOME/serve.ensure.lock (the loser only waits for
    the heartbeat), and a second daemon on the same home refuses to start anyway (serve.lock).
    Returns {alive, started, pid, age_s}."""
    if queue.daemon_alive(cfg, max_age_s=stale_s):
        hb = queue.heartbeat_row(cfg) or {}
        return {"alive": True, "started": False, "pid": hb.get("pid"), "age_s": heartbeat_age_s(cfg)}
    age = heartbeat_age_s(cfg)
    Path(cfg.home).mkdir(parents=True, exist_ok=True)
    guard = _FileLock(str(Path(cfg.home) / "serve.ensure.lock"))
    started, pid = False, None
    try:
        if guard.try_acquire():
            if not queue.daemon_alive(cfg, max_age_s=stale_s):    # re-check under the guard
                pid = start_background(cfg, opts or DaemonOptions(foreground=False))
                started = True
        else:
            t0 = time.monotonic()
            while time.monotonic() - t0 < wait_s and not queue.daemon_alive(cfg, max_age_s=stale_s):
                time.sleep(0.2)
    finally:
        guard.release()
    alive = queue.daemon_alive(cfg, max_age_s=stale_s)
    if pid is None:
        pid = (queue.heartbeat_row(cfg) or {}).get("pid")
    return {"alive": alive, "started": started, "pid": pid, "age_s": age}


def stop(cfg: Config, timeout_s: float = 60) -> bool:
    """Ask the daemon to stop (stop file + SIGTERM); True once it is gone (or was not running)."""
    hb = queue.heartbeat_row(cfg) or {}
    pid = hb.get("pid")
    Path(cfg.home).mkdir(parents=True, exist_ok=True)
    _stop_file(cfg).write_text(utc_now(), encoding="utf-8")
    if pid and pid != os.getpid() and pid_alive(pid) and sys.platform != "win32":
        try:
            os.kill(int(pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        if not (pid and pid_alive(pid)):
            return True
        time.sleep(0.25)
    return False


def run_gpu_job(cfg: Config, job_id: str) -> None:
    """One GPU-lane job end to end (lock, worker subprocess, hold limits, ingest)."""
    conn = queue.connect(cfg)
    try:
        row = conn.execute("SELECT state FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            return
        if row["state"] == "queued" and not queue.claim(conn, job_id):
            return
        if row["state"] not in ("queued", "running"):
            return
        spec = queue.get_spec(cfg, job_id)
        spec.job_id = job_id
        rd = _jobrun.prepare_run(cfg, spec)
        queue.set_running_info(conn, job_id, run_id=rd.run_id, run_dir=str(rd.path))
        out = _jobrun.run_gpu(
            cfg, spec, rd,
            cancel_fn=lambda: queue.cancel_requested(conn, job_id), stop_fn=_STOP.is_set,
            on_pid=lambda pid: queue.set_running_info(conn, job_id, pid=pid),
            on_event=lambda kind, msg: queue.log_event(conn, kind, msg, job_id))
        for pid in out.offenders:
            queue.log_event(conn, "not_exclusive", f"foreign pid {pid} during run", job_id)
        res = out.result
        queue.finish(conn, job_id, out.state, exit_code=res.exit_code,
                     error_code=res.error.code if res.error else None, run_id=rd.run_id, run_dir=str(rd.path),
                     waited_s=out.waited_s)
    except Exception as e:  # noqa: BLE001  (never leave a job 'running')
        traceback.print_exc()
        try:
            queue.finish(conn, job_id, "failed", error_code=ErrorCode.INTERNAL.value)
            queue.log_event(conn, "error", f"{type(e).__name__}: {e}", job_id)
        except Exception:  # noqa: BLE001
            pass
    finally:
        conn.close()
