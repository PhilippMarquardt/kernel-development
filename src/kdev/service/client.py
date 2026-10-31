"""Client side of the service: submit/wait/status/cancel, with the direct (no-daemon) fallback. Owner: P2.

``kdev.registry.Context.submit`` (P1) calls :func:`submit_and_maybe_wait`.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from kdev.config import Config, no_gpu
from kdev.contract import ErrorCode, KdevError, KdevErrorInfo, Result
from kdev.models import JobSpec, JobStatus
from kdev.service import _jobrun, lock, queue


def _deny_check(cfg: Config, spec: JobSpec) -> None:
    if spec.kind == "exec" and lock.is_denied(cfg, list((spec.params or {}).get("argv") or [])):
        raise KdevError(ErrorCode.QUOTA_FORBIDDEN, "exec argv matches cfg.deny_commands (judge quota commands)",
                        recovery="judge checks only via `kdev judge pair --allow-check`; never submit")


def submit(cfg: Config, spec: JobSpec) -> JobStatus:
    """Enqueue (service must be alive) and return the status."""
    _deny_check(cfg, spec)
    if not queue.daemon_alive(cfg):
        raise KdevError(ErrorCode.CAPABILITY_UNAVAILABLE, "kdev serve is not running (no fresh heartbeat)",
                        recovery="start it with `kdev serve start`, or use --direct")
    return queue.enqueue(cfg, spec)


def _job_result(cfg: Config, st: JobStatus) -> Result:
    if st.run_dir:
        p = Path(st.run_dir) / "result.json"
        if p.exists():
            try:
                r = Result.from_dict(json.loads(p.read_text(encoding="utf-8")))
                r.job_id = st.job_id
                return r
            except Exception:  # noqa: BLE001
                pass
    code = st.error_code or (ErrorCode.CANCELLED.value if st.state == "cancelled" else
                             ErrorCode.TIMEOUT.value if st.state == "timeout" else
                             ErrorCode.INTERNAL.value if st.state == "failed" else None)
    r = Result(tool=st.kind, status="ok" if st.state == "done" else "error", verdict=st.state, job_id=st.job_id,
               run_id=st.run_id, summary={"state": st.state, "exit_code": st.exit_code})
    if code and st.state != "done":
        try:
            ErrorCode(code)
        except ValueError:
            code = ErrorCode.INTERNAL.value
        r.error = KdevErrorInfo(code, f"job {st.job_id} {st.state} without result.json",
                                recovery=f"kdev job logs {st.job_id}")
    if st.run_dir:
        r.files["dir"] = st.run_dir
    return r


def wait(cfg: Config, job_id: str, timeout_s: float | None = None, poll_s: float = 1.0) -> Result:
    """Block until done; return the job's Result (from result.json). On timeout return a Result with status
    'ok' and verdict 'pending' carrying job_id (the job keeps running)."""
    t0 = time.monotonic()
    while True:
        st = queue.get(cfg, job_id)
        if st.state in queue.TERMINAL:
            return _job_result(cfg, st)
        if timeout_s is not None and time.monotonic() - t0 >= timeout_s:
            where = f"position {st.position}" if st.position else st.state
            return Result(tool=st.kind, status="ok", verdict="pending", job_id=st.job_id, run_id=st.run_id,
                          summary={"state": st.state, "position": st.position, "waited_s": round(time.monotonic() - t0, 1)},
                          lines=[f"job: {st.job_id}  {where}"], next=[f"kdev job wait {st.job_id}"])
        time.sleep(poll_s)


def run_direct(cfg: Config, spec: JobSpec, timeout_s: float | None = None) -> Result:
    """No daemon: build-lane jobs run in-process; GPU jobs run the worker subprocess under lock.exclusive().

    (Build jobs run as a CUDA-free subprocess of this process, same code path as the daemon's build lane.)"""
    _deny_check(cfg, spec)
    spec = _jobrun.normalise(spec)
    if timeout_s:
        spec.timeout_s = int(timeout_s)
    rd = _jobrun.prepare_run(cfg, spec)
    if spec.lane == "build":
        task = _jobrun.BuildTask(cfg, spec, rd)
        task.start()
        return task.wait()
    out = _jobrun.run_gpu(cfg, spec, rd)
    return out.result


def submit_and_maybe_wait(cfg: Config, spec: JobSpec, *, detach: bool = False, direct: bool = False,
                          timeout_s: float | None = None) -> Result:
    """Routing rule of PLAN 2.2: daemon alive -> enqueue (+wait unless detach); else direct; KDEV_NO_GPU=1
    and a GPU lane -> capability_unavailable."""
    try:
        if no_gpu() and spec.resolved_lane() == "gpu":
            raise KdevError(ErrorCode.CAPABILITY_UNAVAILABLE, "KDEV_NO_GPU=1: GPU jobs are disabled",
                            recovery="unset KDEV_NO_GPU on a GPU host")
        if direct:
            return run_direct(cfg, spec, timeout_s)
        if queue.daemon_alive(cfg):
            st = submit(cfg, spec)
            if detach:
                return Result(tool=spec.kind, status="ok", verdict="queued", job_id=st.job_id,
                              summary={"job_id": st.job_id, "position": st.position, "lane": st.lane},
                              lines=[f"job: {st.job_id}  lane {st.lane}  position {st.position}"],
                              next=[f"kdev job wait {st.job_id}"])
            return wait(cfg, st.job_id, timeout_s)
        if detach:
            raise KdevError(ErrorCode.CAPABILITY_UNAVAILABLE, "--detach needs the service (kdev serve not running)",
                            recovery="`kdev serve start`, or drop --detach to run directly under the lock")
        return run_direct(cfg, spec)
    except KdevError as e:
        r = Result.from_error(spec.kind, e)
        r.job_id = spec.job_id or None
        return r
