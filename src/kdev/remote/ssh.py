"""ssh runner for `kdev remote -- kdev ...`. Owner: P2.

Uses the system ssh with ``-i key -p port -o BatchMode=yes -o ServerAliveInterval=30``. Commands are passed
as an argv list and shell-quoted once for the remote side.
"""
from __future__ import annotations

from kdev.config import RemoteConfig
from kdev.util import Proc, Runner, run


def ssh_argv(rc: RemoteConfig, remote_argv: list[str]) -> list[str]:
    """Build the local ssh argv (pure; unit-testable)."""
    raise NotImplementedError("ssh.ssh_argv (P2)")


def run_remote(rc: RemoteConfig, remote_argv: list[str], *, timeout: float | None = None,
               runner: Runner = run) -> Proc:
    raise NotImplementedError("ssh.run_remote (P2)")


def pull_run(rc: RemoteConfig, run_id: str, local_runs_dir: str, runner: Runner = run) -> str:
    """Copy one remote run dir to local (md5-verified); returns the local path."""
    raise NotImplementedError("ssh.pull_run (P2)")
