"""The only wrapper around the `okc` CLI. Owner: P4.

Allowed read-only subcommands: status, leaderboard, jobs (list). `check` only when the caller passes the
private token from ``pair.run_pair`` (allow_check=True). `submit` is never allowed: there is no parameter
that enables it. ``withdraw`` (modifies the board) is forbidden too.

tests/unit/test_p4_safety.py enforces that no other module builds an ``okc`` argv.
"""
from __future__ import annotations

import json
import os
from typing import Any

from kdev.contract import ErrorCode, KdevError
from kdev.util import Runner, run

READONLY_SUBCOMMANDS = ("status", "leaderboard", "jobs")
FORBIDDEN_SUBCOMMANDS = ("submit", "withdraw")
OKC_BIN = "okc"
CHECK_TIMEOUT_S = 1500.0     # a judge run has 15 min in all, plus queueing


def _exec(argv: list[str], *, runner: Runner, timeout: float, cwd: str | None = None) -> dict[str, Any]:
    try:
        p = runner(argv, timeout=timeout, cwd=cwd) if cwd else runner(argv, timeout=timeout)
    except FileNotFoundError as e:
        raise KdevError(ErrorCode.CAPABILITY_UNAVAILABLE, "`okc` is not on PATH",
                        recovery="install the okc CLI or record saved JSONs with `kdev judge record`") from e
    except Exception as e:  # noqa: BLE001 - subprocess.TimeoutExpired and friends
        if type(e).__name__ == "TimeoutExpired":
            raise KdevError(ErrorCode.TIMEOUT, f"`{' '.join(argv[:2])}` timed out after {timeout:.0f} s") from e
        raise
    if p.returncode == 127 and not p.stdout:
        raise KdevError(ErrorCode.CAPABILITY_UNAVAILABLE, "`okc` is not on PATH",
                        details={"stderr": p.stderr[-300:]},
                        recovery="install the okc CLI or record saved JSONs with `kdev judge record`")
    out = (p.stdout or "").strip()
    try:
        d = json.loads(out) if out else {}
    except ValueError:
        d = {"text": out[-20000:]}
    if not isinstance(d, dict):
        d = {"items": d}
    if p.returncode != 0:
        d.setdefault("_okc_error", {"returncode": p.returncode, "stderr": (p.stderr or "")[-2000:]})
    return d


def okc_readonly(sub: str, args: list[str], *, runner: Runner = run, timeout: float = 60) -> dict[str, Any]:
    """Run `okc <sub> ... --json` for sub in READONLY_SUBCOMMANDS; KdevError(quota_forbidden) for anything
    else; KdevError(capability_unavailable) if `okc` is not on PATH."""
    if sub in FORBIDDEN_SUBCOMMANDS or sub not in READONLY_SUBCOMMANDS:
        raise KdevError(ErrorCode.QUOTA_FORBIDDEN, f"`okc {sub}` is not allowed through kdev",
                        details={"subcommand": sub, "allowed": list(READONLY_SUBCOMMANDS)},
                        recovery="kdev never submits; checks only via `kdev judge pair --allow-check`")
    bad = [a for a in args if a in FORBIDDEN_SUBCOMMANDS or a == "check"]
    if bad:
        raise KdevError(ErrorCode.QUOTA_FORBIDDEN, f"forbidden argument(s) {bad} for `okc {sub}`",
                        details={"args": args})
    argv = [OKC_BIN, sub, *[str(a) for a in args]]
    if "--json" not in argv:
        argv.append("--json")
    return _exec(argv, runner=runner, timeout=timeout)


def status(job_id: str, runner: Runner = run, *, wait: bool = False, timeout: float = 60) -> dict[str, Any]:
    """`okc status JOB --json` (``wait=True`` adds ``--wait``: blocks until the job finished)."""
    if not job_id or job_id.startswith("-"):
        raise KdevError(ErrorCode.INVALID_REQUEST, f"bad job id {job_id!r}", details={"field": "job_id"})
    args = [job_id] + (["--wait"] if wait else [])
    return okc_readonly("status", args, runner=runner, timeout=CHECK_TIMEOUT_S if wait else timeout)


def leaderboard(problem: str | None = None, runner: Runner = run) -> dict[str, Any]:
    """`okc leaderboard [PROBLEM] --json` (public board, read-only)."""
    return okc_readonly("leaderboard", [problem] if problem else [], runner=runner)


def jobs(runner: Runner = run) -> dict[str, Any]:
    return okc_readonly("jobs", [], runner=runner)


def _okc_check(path: str, problem: str, *, _allow: object, runner: Runner = run,
               wait: bool = True) -> dict[str, Any]:
    """Private: `okc check` of one file. ``_allow`` must be pair._ALLOW_CHECK (identity check), else
    quota_forbidden. Only judge.pair calls this. ``wait=False`` adds ``--no-wait`` (returns the job id)."""
    from kdev.judge import pair as _pair

    if _allow is not _pair._ALLOW_CHECK:  # noqa: SLF001 - the identity token is the gate
        raise KdevError(ErrorCode.QUOTA_FORBIDDEN, "`okc check` is only reachable via `kdev judge pair "
                        "--allow-check`", recovery="run `kdev judge pair FILE --base BASE --allow-check` (CLI)")
    if not os.path.isfile(path):
        raise KdevError(ErrorCode.NOT_FOUND, f"no such file: {path}", details={"path": path})
    ap = os.path.abspath(path)
    argv = [OKC_BIN, "check", os.path.basename(ap)] + ([] if wait else ["--no-wait"]) + ["--json"]
    d = _exec(argv, runner=runner, timeout=CHECK_TIMEOUT_S, cwd=os.path.dirname(ap))
    d.setdefault("_kdev", {"path": ap, "problem": problem})
    return d
