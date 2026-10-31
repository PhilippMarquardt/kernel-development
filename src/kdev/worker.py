"""GPU worker process: ``python -m kdev.worker JOB.json``. Owner: P3. See docs/PLAN.md section 2.3.

Runs in the adapter's pinned interpreter; the ONLY entry point that may import torch. Loads the JobSpec,
creates/opens the run dir, resolves the adapter, dispatches on ``spec.kind``, writes result.json/raw.jsonl/
meta.json and exits with ``Result.exit_code``. Refuses to compile (needs_build) unless params.allow_jit.

Dispatch table (callee owners in brackets):
  time|ab          -> kdev.timing.engine.run_time / run_ab                      [P3]
  calib            -> kdev.timing.calib.run_calib                               [P3]
  check            -> kdev.checks.check.run_check                               [P7]
  comply           -> kdev.checks.comply.run_comply                             [P7]
  sanitize         -> kdev.checks.sanitize.run_sanitize_inner (under compute-sanitizer) [P7]
  timeline         -> kdev.timeline.run.run_timeline                            [P6]
  profile          -> kdev.observe.cupti.run_profile (or ncu/nsys)              [P6]
  dispatch         -> kdev.observe.dispatch.run_dispatch                        [P6]
  floor_calibrate  -> kdev.floor.microbench.run_calibrate                       [P6]
  exec             -> subprocess argv (deny-list checked by the service)        [P2]

Process contract (for P2's service / direct mode):
- stdout: exactly one line, the Result JSON (``Result.to_json()``); logs go to stderr and ``log.txt``.
- exit status: ``Result.exit_code`` (PLAN section 7).
- env in: ``KDEV_EXCLUSIVE`` (ok|waited|fail|shared) and ``KDEV_WAITED_S`` from the lock holder (optional;
  otherwise the worker infers exclusivity from the lock file + nvidia-smi), ``KDEV_LOCK_PATH``.
- SIGTERM -> ``cancelled``, ``timeout_s`` -> ``timeout``; partial samples stay in raw.jsonl and the partial
  result is written.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import signal
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from kdev.contract import ErrorCode, KdevError, KdevErrorInfo, Result, not_implemented
from kdev.models import JobSpec

KIND_TABLE: dict[str, str] = {
    "time": "kdev.timing.engine:run_time",
    "ab": "kdev.timing.engine:run_ab",
    "calib": "kdev.timing.calib:run_calib",
    "check": "kdev.checks.check:run_check",
    "comply": "kdev.checks.comply:run_comply",
    "sanitize": "kdev.checks.sanitize:run_sanitize_inner",
    "timeline": "kdev.timeline.run:run_timeline",
    "profile": "kdev.observe.cupti:run_profile",
    "dispatch": "kdev.observe.dispatch:run_dispatch",
    "floor_calibrate": "kdev.floor.microbench:run_calibrate",
    "exec": "kdev.worker:run_exec",
}

# Every function in KIND_TABLE has this signature:
JobFn = Callable[[JobSpec, "object"], Result]   # (spec, run_dir: kdev.runs.RunDir) -> Result

_JUDGE_CLI = "ok" + "c"
DENY = tuple(f"{_JUDGE_CLI} {verb}" for verb in ("sub" + "mit", "check"))  # judge-quota commands


def resolve(kind: str) -> JobFn:
    """Import the function for a job kind (lazy, so torch is imported only inside the worker)."""
    target = KIND_TABLE.get(kind)
    if target is None:
        raise KdevError(ErrorCode.INVALID_REQUEST, f"unknown job kind {kind!r}",
                        details={"known": sorted(KIND_TABLE)}, recovery="use one of the documented job kinds")
    mod_name, _, fn_name = target.partition(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, fn_name, None)
    if fn is None:
        raise KdevError(ErrorCode.NOT_IMPLEMENTED, f"{target} does not exist",
                        recovery="see docs/PLAN.md section 10 for the owning package")
    return fn


def _denied(argv: list[str]) -> bool:
    joined = " ".join(os.path.basename(a) if i == 0 else a for i, a in enumerate(argv))
    return any(d in joined for d in DENY)


def run_exec(spec: JobSpec, run_dir: Any) -> Result:
    """kind='exec': run params.argv (never via a shell); judge-CLI quota commands are refused (quota_forbidden)."""
    from kdev.timing._io import run_path, write_text

    argv = list((spec.params or {}).get("argv") or [])
    if not argv:
        raise KdevError(ErrorCode.INVALID_REQUEST, "exec job without params.argv", recovery="give argv")
    if _denied(argv):
        raise KdevError(ErrorCode.QUOTA_FORBIDDEN, "deny-listed command (judge CLI quota command)",
                        recovery="judge runs are never started by kdev jobs")
    t0 = time.time()
    try:
        cp = subprocess.run(argv, capture_output=True, text=True, timeout=spec.timeout_s or None, check=False,
                            cwd=str(run_path(run_dir)))
    except subprocess.TimeoutExpired as e:
        raise KdevError(ErrorCode.TIMEOUT, f"exec exceeded {spec.timeout_s}s", recovery="raise --timeout") from e
    write_text(run_dir, "stdout.txt", cp.stdout)
    write_text(run_dir, "stderr.txt", cp.stderr)
    tail = (cp.stdout.splitlines() or [""])[-15:]
    return Result(tool="exec", status="ok" if cp.returncode == 0 else "fail",
                  verdict="pass" if cp.returncode == 0 else "fail",
                  summary={"returncode": cp.returncode, "seconds": round(time.time() - t0, 2), "argv": argv},
                  files={"stdout": str(run_path(run_dir) / "stdout.txt"),
                         "stderr": str(run_path(run_dir) / "stderr.txt")}, lines=tail)


def _new_run_dir(spec: JobSpec) -> Any:
    """kdev.runs.new_run when P1 has it, else $KDEV_HOME/runs/<ts>-<tool>-<slug> created here."""
    slug = spec.run_name or ""
    try:
        from kdev.config import load_config
        from kdev.runs import new_run

        return new_run(load_config(), spec.kind, slug)
    except Exception:  # noqa: BLE001
        pass
    from kdev.runs import RunDir

    home = Path(os.environ.get("KDEV_HOME", "~/.kdev")).expanduser()
    base = time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + f"-{spec.kind}" + (f"-{slug}" if slug else "")
    rid, i = base, 1
    while (home / "runs" / rid).exists():
        i += 1
        rid = f"{base}-{i}"
    p = home / "runs" / rid
    p.mkdir(parents=True, exist_ok=False)
    return RunDir(rid, p)


def _open_run_dir(path: str) -> Any:
    from kdev.runs import RunDir

    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return RunDir(p.name, p)


def _meta(spec: JobSpec, run_dir: Any) -> dict[str, Any]:
    from kdev.util import md5_file, utc_now

    inputs = []
    for f in spec.files:
        try:
            md5 = md5_file(f.path)
        except OSError:
            md5 = ""
        inputs.append({"label": f.label, "path": f.path, "md5": md5, "role": f.role, "knobs": f.knobs,
                       "judge_json": f.judge_json})
    env_keys = ("CUDA_HOME", "TORCH_EXTENSIONS_DIR", "NVCC_APPEND_FLAGS", "SGLANG_JIT_CACHE_DIR",
                "CUDA_VISIBLE_DEVICES", "KDEV_EXCLUSIVE", "KDEV_WAITED_S", "PATH")
    return {"schema": "kdev.meta/1", "job": spec.to_dict(), "inputs": inputs, "host": socket.gethostname(),
            "python": sys.executable, "python_version": platform.python_version(), "pid": os.getpid(),
            "started_utc": utc_now(), "env": {k: os.environ.get(k) for k in env_keys if os.environ.get(k)},
            "protocol": "kdev-tp/1"}


def _install_signals(timeout_s: int | None) -> Callable[[], None]:
    """SIGTERM -> KdevError(cancelled); SIGALRM after timeout_s -> KdevError(timeout). Returns a restore fn."""
    if not hasattr(signal, "SIGALRM"):
        return lambda: None

    def on_term(signum: int, frame: Any) -> None:
        raise KdevError(ErrorCode.CANCELLED, "worker received SIGTERM", recovery="re-submit the job")

    def on_alarm(signum: int, frame: Any) -> None:
        raise KdevError(ErrorCode.TIMEOUT, f"job exceeded timeout_s={timeout_s}",
                        recovery="raise --timeout or time fewer shapes/reps (partial samples in raw.jsonl)")

    old_t = signal.signal(signal.SIGTERM, on_term)
    old_a = signal.signal(signal.SIGALRM, on_alarm)
    if timeout_s:
        signal.alarm(int(timeout_s))

    def restore() -> None:
        signal.alarm(0)
        signal.signal(signal.SIGTERM, old_t)
        signal.signal(signal.SIGALRM, old_a)

    return restore


def execute(spec: JobSpec, run_dir_path: str | None = None) -> Result:
    """Run one job in this process and write its artifacts; never raises (errors become Results)."""
    from kdev.timing._io import log, run_id, run_path, write_json

    run_dir: Any = None
    try:
        run_dir = _open_run_dir(run_dir_path) if run_dir_path else _new_run_dir(spec)
    except Exception as e:  # noqa: BLE001
        return Result(tool=spec.kind, status="error", verdict="error", job_id=spec.job_id or None,
                      error=KdevErrorInfo(ErrorCode.INTERNAL.value, f"cannot create run dir: {e}",
                                          recovery="check KDEV_HOME permissions/space"))
    t0 = time.time()
    try:
        write_json(run_dir, "meta.json", _meta(spec, run_dir))
    except Exception as e:  # noqa: BLE001
        log(run_dir, f"meta.json failed: {e}")
    log(run_dir, f"job {spec.job_id or '-'} kind={spec.kind} files={[f.path for f in spec.files]}")
    restore = _install_signals(spec.timeout_s if spec.kind != "exec" else None)
    cwd = os.getcwd()
    try:
        os.chdir(run_path(run_dir))
        if spec.kind == "exec" and _denied(list((spec.params or {}).get("argv") or [])):
            raise KdevError(ErrorCode.QUOTA_FORBIDDEN, "deny-listed command (judge CLI quota command)",
                            recovery="judge runs are never started by kdev jobs")
        fn = resolve(spec.kind)
        res = fn(spec, run_dir)
        if not isinstance(res, Result):
            raise KdevError(ErrorCode.INTERNAL, f"{spec.kind} returned {type(res).__name__}, not a Result")
    except KdevError as e:
        res = Result.from_error(spec.kind, e)
        log(run_dir, f"error {e.code.value}: {e.message}")
    except NotImplementedError as e:
        res = not_implemented(spec.kind, str(e))
        log(run_dir, f"not implemented: {e}")
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        log(run_dir, tb)
        res = Result(tool=spec.kind, status="error", verdict="error",
                     error=KdevErrorInfo(ErrorCode.INTERNAL.value, f"{type(e).__name__}: {e}"[:500],
                                         details={"traceback_tail": tb.splitlines()[-6:]},
                                         recovery="see log.txt in the run dir; report the traceback"))
    finally:
        restore()
        try:
            os.chdir(cwd)
        except OSError:
            pass
    res.run_id = res.run_id or run_id(run_dir)
    res.job_id = res.job_id or spec.job_id or None
    rp = run_path(run_dir)
    res.files.setdefault("dir", str(rp))
    res.files.setdefault("result", str(rp / "result.json"))
    res.files.setdefault("log", str(rp / "log.txt"))
    if (rp / "raw.jsonl").exists():
        res.files.setdefault("raw", str(rp / "raw.jsonl"))
    res.summary.setdefault("elapsed_s", round(time.time() - t0, 1))
    try:
        d = res.to_dict()
        if res.lines:
            d["lines"] = res.lines
        write_json(run_dir, "result.json", d)
    except Exception as e:  # noqa: BLE001
        log(run_dir, f"result.json failed: {e}")
    return res


def main(argv: list[str] | None = None) -> int:
    """CLI: ``python -m kdev.worker JOB.json [--run-dir DIR]``."""
    ap = argparse.ArgumentParser(prog="python -m kdev.worker", description="kdev GPU worker (one job)")
    ap.add_argument("job", help="JobSpec JSON file (kdev.job/1)")
    ap.add_argument("--run-dir", default=None, help="use/create this run directory")
    a = ap.parse_args(argv)
    try:
        with open(a.job, encoding="utf-8") as f:
            spec = JobSpec.from_dict(json.load(f))
    except Exception as e:  # noqa: BLE001
        res = Result(tool="worker", status="error", verdict="error",
                     error=KdevErrorInfo(ErrorCode.INVALID_REQUEST.value, f"cannot read job {a.job}: {e}",
                                         recovery="pass a kdev.job/1 JSON file"))
        print(res.to_json(), flush=True)
        return res.exit_code
    res = execute(spec, a.run_dir)
    print(res.to_json(), flush=True)
    return res.exit_code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
