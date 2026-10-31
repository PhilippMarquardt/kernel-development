"""Private: run one job end to end (shared by the daemon and client.run_direct). Owner: P2.

Build lane: ``<python> -m kdev.service._job JOB.json --run-dir DIR`` with CUDA_VISIBLE_DEVICES="" (no CUDA
context, never under the lock). GPU lane: needs_build gate -> lock -> ``<python> -m kdev.worker JOB.json
--run-dir DIR`` (or the raw argv of an ``exec`` job) with timeout / max-hold / cancel / foreign-context
polling -> fidelity patch -> result.json -> ledger ingest.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from kdev.config import Config, no_gpu
from kdev.contract import ErrorCode, Fidelity, KdevError, KdevErrorInfo, Result
from kdev.models import GpuState, JobSpec
from kdev.service import lock, smi
from kdev.service._common import descendants, new_run, write_json
from kdev.util import new_id, utc_now

TIMING_KINDS = ("time", "ab", "calib", "sweep", "timeline", "floor_calibrate", "profile")
STOP_GRACE_S = 60.0          # SIGTERM: let the running GPU job finish up to 60 s
FOREIGN_POLL_S = 2.0
HOLD_WARN_FRAC = 0.8

Flag = Callable[[], bool]


def _never() -> bool:
    return False


def normalise(spec: JobSpec) -> JobSpec:
    from kdev.service.queue import validate_spec
    validate_spec(spec)
    spec.job_id = spec.job_id or new_id("j")
    spec.submitted_utc = spec.submitted_utc or utc_now()
    spec.lane = spec.resolved_lane()
    return spec


def interpreter(cfg: Config) -> str:
    return cfg.python or os.environ.get("KDEV_PYTHON") or sys.executable


def child_env(cfg: Config, spec: JobSpec, *, no_cuda: bool) -> dict[str, str]:
    env = dict(os.environ)
    src = str(Path(__file__).resolve().parents[2])           # .../src (so kdev imports in the pinned python)
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["KDEV_HOME"] = str(cfg.home)
    env["KDEV_LOCK_PATH"] = cfg.lock_path
    env["KDEV_JOB_ID"] = spec.job_id
    if spec.agent:
        env["KDEV_AGENT"] = spec.agent
    if no_cuda:
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def prepare_run(cfg: Config, spec: JobSpec) -> Any:
    rd = new_run(cfg, spec.kind, spec.run_name or spec.job_id)
    write_json(Path(rd.path) / "job.json", spec.to_dict())
    return rd


def _log(rd: Any, msg: str) -> None:
    try:
        with open(Path(rd.path) / "log.txt", "a", encoding="utf-8") as f:
            f.write(f"[kdev {utc_now()}] {msg}\n")
    except OSError:
        pass


def _tail(path: Path, n: int = 12) -> list[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]
    except OSError:
        return []


def load_result(rd: Any, spec: JobSpec) -> Result | None:
    p = Path(rd.path) / "result.json"
    if not p.exists():
        return None
    try:
        return Result.from_dict(json.loads(p.read_text(encoding="utf-8")))
    except Exception as e:  # noqa: BLE001
        _log(rd, f"unreadable result.json: {type(e).__name__}: {e}")
        return None


def error_result(spec: JobSpec, rd: Any, code: ErrorCode, message: str, recovery: str = "",
                 details: dict[str, Any] | None = None) -> Result:
    r = Result(tool=spec.kind, status="error", verdict="error", job_id=spec.job_id,
               run_id=getattr(rd, "run_id", None),
               error=KdevErrorInfo(code.value, message, details or {}, recovery))
    if rd is not None:
        r.files["dir"] = str(rd.path)
        r.files["log"] = str(Path(rd.path) / "log.txt")
        tail = _tail(Path(rd.path) / "log.txt")
        if tail:
            r.error.details.setdefault("log_tail", tail)  # type: ignore[union-attr]
    return r


def save_result(cfg: Config, rd: Any, res: Result, *, ingest: bool = True) -> Path:
    p = Path(rd.path) / "result.json"
    res.run_id = res.run_id or rd.run_id
    res.files.setdefault("dir", str(rd.path))
    res.files.setdefault("result", str(p))
    if (Path(rd.path) / "log.txt").exists():
        res.files.setdefault("log", str(Path(rd.path) / "log.txt"))
    if (Path(rd.path) / "raw.jsonl").exists():
        res.files.setdefault("raw", str(Path(rd.path) / "raw.jsonl"))
    try:
        from kdev.runs import write_result_files
        write_result_files(rd.path, res)
    except ImportError:
        write_json(p, res.to_dict())
    if ingest:
        try:
            from kdev.ledger.record import ingest_result
            ingest_result(p)
        except Exception as e:  # noqa: BLE001  (P4 stub or ledger error must not lose the result)
            _log(rd, f"ledger ingest skipped: {type(e).__name__}: {e}")
    return p


def state_for(res: Result) -> str:
    if res.error is not None:
        if res.error.code == ErrorCode.TIMEOUT.value:
            return "timeout"
        if res.error.code == ErrorCode.CANCELLED.value:
            return "cancelled"
        return "failed"
    return "done" if res.status == "ok" else "failed"


# ---------------------------------------------------------------- build lane

@dataclass
class BuildTask:
    """A running build-lane subprocess (non-blocking; the daemon polls many of these)."""

    cfg: Config
    spec: JobSpec
    rd: Any
    proc: subprocess.Popen | None = None
    t0: float = 0.0
    killed: str = ""          # "" | timeout | cancelled | stopped
    _log_f: Any = None

    def start(self) -> None:
        self.t0 = time.monotonic()
        self._log_f = open(Path(self.rd.path) / "log.txt", "a", encoding="utf-8")
        argv = [interpreter(self.cfg), "-m", "kdev.service._job", str(Path(self.rd.path) / "job.json"),
                "--run-dir", str(self.rd.path)]
        self._log_f.write(f"[kdev {utc_now()}] build lane (CUDA_VISIBLE_DEVICES=''): {' '.join(argv)}\n")
        self._log_f.flush()
        self.proc = lock.spawn(argv, stdout=self._log_f, stderr=subprocess.STDOUT, cwd=str(self.rd.path),
                               env=child_env(self.cfg, self.spec, no_cuda=True))

    def poll(self, cancel_fn: Flag = _never, stop_fn: Flag = _never) -> Result | None:
        """None while running; the final Result (already saved + ingested) once finished."""
        assert self.proc is not None
        rc = self.proc.poll()
        if rc is None:
            el = time.monotonic() - self.t0
            why = ("timeout" if self.spec.timeout_s and el > self.spec.timeout_s else
                   "cancelled" if cancel_fn() else "stopped" if stop_fn() else "")
            if not why:
                return None
            self.killed = why
            lock._kill_tree(self.proc)
            rc = self.proc.returncode
        if self._log_f:
            self._log_f.close()
        res = self._finish(rc)
        save_result(self.cfg, self.rd, res)
        return res

    def _finish(self, rc: int | None) -> Result:
        if self.killed == "cancelled":
            return error_result(self.spec, self.rd, ErrorCode.CANCELLED, "build cancelled")
        if self.killed in ("timeout", "stopped"):
            why = f"exceeded timeout_s={self.spec.timeout_s}" if self.killed == "timeout" else "daemon stopped"
            return error_result(self.spec, self.rd, ErrorCode.TIMEOUT, f"build killed: {why}",
                                recovery="raise --timeout or resubmit")
        res = load_result(self.rd, self.spec)
        if res is None:
            return error_result(self.spec, self.rd, ErrorCode.INTERNAL,
                                f"build worker exited {rc} without result.json", recovery="see log.txt")
        res.job_id = self.spec.job_id
        return res

    def wait(self, poll_s: float = 0.5) -> Result:
        while True:
            try:
                r = self.poll()
            except KeyboardInterrupt:
                if self.proc:
                    lock._kill_tree(self.proc)
                raise
            if r is not None:
                return r
            time.sleep(poll_s)


# ---------------------------------------------------------------- GPU lane

def _clock(cfg: Config) -> GpuState:
    return GpuState(ts_utc=utc_now()) if no_gpu() else smi.query_state()


def _prebuilt_gate(spec: JobSpec) -> KdevError | None:
    """PLAN 8.2: no compile under the lock."""
    allow_jit = bool((spec.params or {}).get("allow_jit"))
    if spec.kind == "exec" or not spec.requires_prebuilt or allow_jit or not spec.files:
        return None
    try:
        from kdev.build.cache import is_prebuilt
        ok = is_prebuilt(spec)
    except NotImplementedError:
        return KdevError(ErrorCode.NEEDS_BUILD, "cannot verify a prebuilt cache entry (kdev.build.cache not "
                         "implemented yet); refusing to risk compiling under the GPU lock",
                         recovery="build first, or pass --allow-jit (recorded as an opt-out)")
    except Exception as e:  # noqa: BLE001
        return KdevError(ErrorCode.NEEDS_BUILD, f"prebuilt check failed: {type(e).__name__}: {e}",
                         recovery="run `kdev build FILE` first")
    if not ok:
        return KdevError(ErrorCode.NEEDS_BUILD, "no prebuilt cache hit for "
                         + ", ".join(f.path for f in spec.files), recovery="run `kdev build " +
                         " ".join(f.path for f in spec.files) + "` first (builds never take the GPU lock)")
    return None


@dataclass
class GpuOutcome:
    result: Result
    state: str
    waited_s: float = 0.0
    offenders: list[int] = field(default_factory=list)


def run_gpu(cfg: Config, spec: JobSpec, rd: Any, *, cancel_fn: Flag = _never, stop_fn: Flag = _never,
            on_pid: Callable[[int], None] | None = None, on_event: Callable[[str, str], None] | None = None,
            apps_fn: Callable[[], list[lock.ComputeApp]] | None = None, lock_timeout_s: float | None = None,
            ) -> GpuOutcome:
    """One GPU-lane job in this thread. Never raises KdevError (errors become Results)."""
    apps = apps_fn or (lambda: lock.compute_apps())
    ev = on_event or (lambda kind, msg: None)
    if spec.kind == "exec" and lock.is_denied(cfg, list(spec.params.get("argv") or [])):
        res = error_result(spec, rd, ErrorCode.QUOTA_FORBIDDEN, "deny-listed exec argv",
                           recovery="judge quota commands (cfg.deny_commands) are never run by the service")
        ev("refused", "deny-listed exec")
        save_result(cfg, rd, res)
        return GpuOutcome(res, "failed")
    gate = _prebuilt_gate(spec)
    if gate is not None:
        ev("refused", gate.message)
        res = error_result(spec, rd, gate.code, gate.message, gate.recovery)
        save_result(cfg, rd, res)
        return GpuOutcome(res, "failed")

    lock_wait = spec.timeout_s if lock_timeout_s is None else lock_timeout_s
    try:
        def progress(msg: str) -> None:
            _log(rd, msg)
            ev("waiting", msg)
            lock._stderr_progress(msg)

        with lock.exclusive_abortable(cfg, timeout_s=lock_wait, job_id=spec.job_id, apps_fn=apps,
                                      abort=lambda: cancel_fn() or stop_fn(),
                                      own_pids_fn=lambda: {os.getpid()}, priority=spec.priority,
                                      agent=spec.agent or "", progress=progress) as grant:
            _log(rd, f"GPU_EXCLUSIVE=OK waited={grant.waited_s:.0f}s exclusive={grant.exclusive}")
            ev("started", f"lock granted ({grant.exclusive}, waited {grant.waited_s:.1f}s)")
            out = _run_locked_child(cfg, spec, rd, grant, cancel_fn, stop_fn, on_pid, ev, apps)
        _attach_queue(out.result, grant)
    except KdevError as e:
        res = error_result(spec, rd, e.code, e.message, e.recovery, e.details)
        save_result(cfg, rd, res)
        return GpuOutcome(res, state_for(res))
    save_result(cfg, rd, out.result)
    return out


def _attach_queue(res: Result, grant: lock.LockGrant) -> None:
    """Lock wait facts on the Result: summary.lock + one output line when the wait was noticeable."""
    q = dict(grant.queue or {})
    q["waited_s"] = grant.waited_s
    try:
        res.summary = dict(res.summary or {})
        res.summary["lock"] = q
        if grant.waited_s >= 10:
            pos = q.get("position")
            eta = q.get("eta_s")
            res.lines = list(res.lines or []) + [
                f"lock: waited {grant.waited_s:.0f}s (queue position {pos if pos else '?'} of "
                f"{q.get('queue_len', '?')} at start, ETA then ~{(eta or 0) / 60:.1f} min"
                + (f", deferred to higher-priority kdev pid(s) {q['deferred_to']}" if q.get("deferred_to") else "")
                + ")"]
    except Exception:  # noqa: BLE001
        pass


def _child_argv(cfg: Config, spec: JobSpec, rd: Any) -> tuple[list[str], str]:
    if spec.kind == "exec":
        argv = lock.strip_dashdash(list(spec.params.get("argv") or []))
        return argv, str(spec.params.get("cwd") or rd.path)
    return [interpreter(cfg), "-m", "kdev.worker", str(Path(rd.path) / "job.json"), "--run-dir", str(rd.path)], \
        str(rd.path)


def _run_locked_child(cfg: Config, spec: JobSpec, rd: Any, grant: lock.LockGrant, cancel_fn: Flag, stop_fn: Flag,
                      on_pid: Callable[[int], None] | None, ev: Callable[[str, str], None],
                      apps: Callable[[], list[lock.ComputeApp]]) -> GpuOutcome:
    before = _clock(cfg)
    argv, cwd = _child_argv(cfg, spec, rd)
    max_hold = float(spec.max_hold_s or cfg.max_hold_s or 0)
    limit = min(x for x in (float(spec.timeout_s or 0) or float("inf"), max_hold or float("inf")))
    log_f = open(Path(rd.path) / "log.txt", "a", encoding="utf-8")
    log_f.write(f"[kdev {utc_now()}] gpu lane: {' '.join(argv)}\n")
    log_f.flush()
    t0 = time.monotonic()
    killed = ""
    warned = False
    violators: dict[int, lock.ComputeApp] = {}
    stop_t: float | None = None
    try:
        try:
            proc = lock.spawn(argv, lock_fd=getattr(grant, "_fd", None), stdout=log_f, stderr=subprocess.STDOUT,
                              cwd=cwd, env=dict(child_env(cfg, spec, no_cuda=False), KDEV_LOG_CAPTURED="1"))
        except (FileNotFoundError, PermissionError) as e:
            log_f.close()
            return GpuOutcome(error_result(spec, rd, ErrorCode.INVALID_REQUEST, f"cannot start {argv[0]}: {e}"),
                              "failed", grant.waited_s)
        if on_pid:
            on_pid(proc.pid)
        last_poll = 0.0
        try:
            while proc.poll() is None:
                el = time.monotonic() - t0
                if el > limit:
                    killed = "timeout"
                elif cancel_fn():
                    killed = "cancelled"
                elif stop_fn():
                    stop_t = stop_t or time.monotonic()
                    if time.monotonic() - stop_t > STOP_GRACE_S:
                        killed = "stopped"
                if killed:
                    ev("kill", f"{killed} after {el:.0f}s")
                    lock._kill_tree(proc)
                    break
                if max_hold and not warned and el > HOLD_WARN_FRAC * max_hold:
                    warned = True
                    ev("warn_hold", f"holding the GPU {el:.0f}s of max {max_hold:.0f}s")
                    log_f.write(f"[kdev {utc_now()}] WARNING: 80% of max_hold_s={max_hold:.0f}\n")
                    log_f.flush()
                if time.monotonic() - last_poll >= FOREIGN_POLL_S:
                    last_poll = time.monotonic()
                    own = {os.getpid(), proc.pid} | descendants(proc.pid)
                    for a in lock.foreign_apps(apps(), own):
                        if a.pid not in violators:
                            violators[a.pid] = a
                            lock._record_offender(cfg, a, spec.job_id, "violated")
                time.sleep(0.2)
        except BaseException:
            lock._kill_tree(proc)
            raise
        rc = proc.returncode
    finally:
        if not log_f.closed:
            log_f.close()
    after = _clock(cfg)
    el = time.monotonic() - t0

    if killed == "cancelled":
        res = error_result(spec, rd, ErrorCode.CANCELLED, f"cancelled after {el:.0f}s")
    elif killed:
        why = "daemon stopped (SIGTERM) and grace expired" if killed == "stopped" else \
            f"exceeded {'max_hold_s' if max_hold and limit == max_hold else 'timeout_s'}={limit:.0f}"
        res = error_result(spec, rd, ErrorCode.TIMEOUT, f"GPU job killed: {why}",
                           recovery="partial samples (raw.jsonl) are kept; raise --timeout or split the job")
    elif spec.kind == "exec":
        res = Result(tool="exec", status="ok" if rc == 0 else "fail", verdict="ok" if rc == 0 else "fail",
                     job_id=spec.job_id, summary={"returncode": rc, "argv": argv, "seconds": round(el, 2)},
                     lines=[f"exit {rc} after {el:.1f}s: {' '.join(argv)[:100]}"] + _tail(Path(rd.path) / "log.txt", 8))
    else:
        res = load_result(rd, spec) or error_result(
            spec, rd, ErrorCode.INTERNAL, f"worker exited {rc} without result.json", recovery="see log.txt")
    res.job_id = spec.job_id
    res.run_id = res.run_id or rd.run_id

    # fidelity: exclusivity + clocks, as observed by the service (the worker cannot see the lock)
    timing = spec.kind in TIMING_KINDS
    worker_clocks = bool(res.fidelity is not None and res.fidelity.clocks)
    if res.fidelity is not None or timing:
        fid = res.fidelity or Fidelity()
        fid.exclusive = "fail" if violators else grant.exclusive
        fid.waited_s = grant.waited_s
        if not fid.clocks:
            fid.clocks = {"before": before.to_dict(), "after": after.to_dict()}
        res.fidelity = fid
    # The service samples before the worker starts, i.e. usually the idle clock (345 MHz on H200): idle ->
    # boost is expected. Warn only on a drop (throttling), and only when the worker did not measure its own
    # warmed-up clocks (the timing engine does, and warns itself).
    if (not worker_clocks and smi.clocks_moved(before, after) and before.sm_mhz and after.sm_mhz
            and after.sm_mhz < before.sm_mhz):
        res.warnings.append(f"SM clock moved {before.sm_mhz}->{after.sm_mhz} MHz during the run")
    if violators:
        res.warnings.append("NOT EXCLUSIVE: foreign CUDA process(es) during the run: "
                            + ", ".join(f"{p} {a.cmdline[:50]}" for p, a in violators.items()))
        if timing and res.error is None:
            res.status, res.verdict = "error", "not_exclusive"
            res.error = KdevErrorInfo(ErrorCode.NOT_EXCLUSIVE.value, "exclusivity violated mid-run; result kept "
                                      "but not comparable", {"offenders": sorted(violators)},
                                      "re-run when `kdev queue offenders` is clear")
    return GpuOutcome(res, state_for(res), grant.waited_s, sorted(violators))
