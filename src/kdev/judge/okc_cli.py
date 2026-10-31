"""The only wrapper around the `okc` CLI. Owner: P4.

Allowed read-only subcommands: status, leaderboard, jobs (list). `check` only when the caller passes the
private token from ``pair.run_pair`` (allow_check=True). `submit` is never allowed: there is no parameter
that enables it.
"""
from __future__ import annotations

from typing import Any

from kdev.util import Runner, run

READONLY_SUBCOMMANDS = ("status", "leaderboard", "jobs")
FORBIDDEN_SUBCOMMANDS = ("submit",)


def okc_readonly(sub: str, args: list[str], *, runner: Runner = run, timeout: float = 60) -> dict[str, Any]:
    """Run `okc <sub> ... --json` for sub in READONLY_SUBCOMMANDS; KdevError(quota_forbidden) for anything
    else; KdevError(capability_unavailable) if `okc` is not on PATH."""
    raise NotImplementedError("okc_cli.okc_readonly (P4)")


def status(job_id: str, runner: Runner = run) -> dict[str, Any]:
    raise NotImplementedError("okc_cli.status (P4)")


def leaderboard(problem: str | None = None, runner: Runner = run) -> dict[str, Any]:
    raise NotImplementedError("okc_cli.leaderboard (P4)")


def _okc_check(path: str, problem: str, *, _allow: object, runner: Runner = run) -> dict[str, Any]:
    """Private: `okc check` of one file. ``_allow`` must be pair._ALLOW_CHECK (identity check), else
    quota_forbidden. Only judge.pair calls this."""
    raise NotImplementedError("okc_cli._okc_check (P4)")
