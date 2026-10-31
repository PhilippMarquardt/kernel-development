"""`kdev serve`: the GPU job service daemon. Owner: P2. See docs/PLAN.md section 2.

- Build lane: a process pool of ``cfg.build_workers`` running build jobs with CUDA_VISIBLE_DEVICES="" (no
  CUDA context, never under the lock). Implemented by calling ``kdev.build.compile.build_job(spec, run_dir)``.
- GPU lane: exactly one job at a time. For each job: refuse if ``requires_prebuilt`` and
  ``kdev.build.cache.is_prebuilt(spec)`` is False (needs_build, unless allow_jit); then
  ``lock.exclusive()``; spawn ``<python> -m kdev.worker job.json --run-dir DIR`` with timeout; enforce
  max_hold_s (warn event at 80 %, kill at 100 %); record clocks before/after; mark the result not_exclusive if
  a foreign context appeared during the run (poll compute_apps while the worker runs).
- After each finished job: ``kdev.ledger.record.ingest_result(result.json)``.
- Heartbeat every 5 s. SIGTERM/SIGINT: stop claiming, let the running GPU job finish up to 60 s, then kill
  it and mark ``timeout`` (partial raw.jsonl survives). Queue persists across restarts; jobs left 'running'
  by a dead daemon are marked failed(internal) on start.
"""
from __future__ import annotations

from dataclasses import dataclass

from kdev.config import Config


@dataclass
class DaemonOptions:
    build_workers: int = 8
    foreground: bool = True
    poll_s: float = 0.5
    python: str = ""          # worker interpreter ("" = cfg.python or sys.executable)


def serve(cfg: Config, opts: DaemonOptions) -> int:
    """Main loop; returns the exit code."""
    raise NotImplementedError("daemon.serve (P2)")


def start_background(cfg: Config, opts: DaemonOptions) -> int:
    """Daemonise (setsid + nohup-style, log to $KDEV_HOME/serve.log); returns the pid."""
    raise NotImplementedError("daemon.start_background (P2)")


def stop(cfg: Config, timeout_s: float = 60) -> bool:
    raise NotImplementedError("daemon.stop (P2)")


def run_gpu_job(cfg: Config, job_id: str) -> None:
    """One GPU-lane job end to end (lock, worker subprocess, hold limits, ingest)."""
    raise NotImplementedError("daemon.run_gpu_job (P2)")
