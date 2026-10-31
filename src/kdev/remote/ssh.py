"""ssh runner for `kdev remote -- kdev ...`. Owner: P2.

Uses the system ssh with ``-i key -p port -o BatchMode=yes -o ServerAliveInterval=30``. Commands are passed
as an argv list and shell-quoted once for the remote side.
"""
from __future__ import annotations

import os
import shlex
import tarfile
import tempfile
from pathlib import Path

from kdev.config import RemoteConfig
from kdev.contract import ErrorCode, KdevError
from kdev.util import Proc, Runner, md5_file, run

SSH_OPTS = ("-o", "BatchMode=yes", "-o", "ServerAliveInterval=30", "-o", "ConnectTimeout=20",
            "-o", "StrictHostKeyChecking=accept-new")


def _check(rc: RemoteConfig) -> None:
    if not rc.host:
        raise KdevError(ErrorCode.INVALID_REQUEST, "no remote host configured",
                        recovery="pass --host/--port or set remote.host/port/key in kdev.toml")


def _key_opts(rc: RemoteConfig) -> list[str]:
    return ["-i", os.path.expanduser(rc.key)] if rc.key else []


def target(rc: RemoteConfig) -> str:
    return f"{rc.user}@{rc.host}" if rc.user else rc.host


def ssh_argv(rc: RemoteConfig, remote_argv: list[str]) -> list[str]:
    """Build the local ssh argv (pure; unit-testable)."""
    _check(rc)
    return ["ssh", *_key_opts(rc), "-p", str(rc.port or 22), *SSH_OPTS, target(rc), shlex.join(remote_argv)]


def ssh_shell_argv(rc: RemoteConfig, script: str) -> list[str]:
    """ssh argv running a remote shell snippet verbatim (caller is responsible for quoting inside it)."""
    _check(rc)
    return ["ssh", *_key_opts(rc), "-p", str(rc.port or 22), *SSH_OPTS, target(rc), script]


def scp_argv(rc: RemoteConfig, src: str, dst: str, *, to_remote: bool) -> list[str]:
    """scp argv; the remote side of the copy is prefixed with user@host:."""
    _check(rc)
    r = f"{target(rc)}:"
    return ["scp", "-q", *_key_opts(rc), "-P", str(rc.port or 22), *SSH_OPTS,
            src if to_remote else r + src, r + dst if to_remote else dst]


def run_remote(rc: RemoteConfig, remote_argv: list[str], *, timeout: float | None = None,
               runner: Runner = run) -> Proc:
    return runner(ssh_argv(rc, remote_argv), timeout=timeout)


def remote_md5(rc: RemoteConfig, path: str, runner: Runner = run) -> str:
    p = runner(ssh_argv(rc, ["md5sum", path]), timeout=120)
    if p.returncode != 0 or not p.stdout.strip():
        raise KdevError(ErrorCode.NOT_FOUND, f"remote md5sum {path} failed: {p.stderr.strip()[:200]}")
    return p.stdout.split()[0]


def fetch_verified(rc: RemoteConfig, remote_path: str, local_path: str, runner: Runner = run,
                   expect_md5: str | None = None) -> str:
    """scp remote -> local .part, compare md5 with the remote md5sum, rename; returns the md5."""
    want = expect_md5 or remote_md5(rc, remote_path, runner)
    part = local_path + ".part"
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    # relative local name + cwd: a Windows drive letter ('C:\x') would otherwise parse as host 'C'
    p = runner(scp_argv(rc, remote_path, os.path.basename(part), to_remote=False), timeout=3600,
               cwd=os.path.dirname(os.path.abspath(part)))
    if p.returncode != 0:
        raise KdevError(ErrorCode.INTERNAL, f"scp {remote_path} failed: {p.stderr.strip()[:200]}")
    got = md5_file(part)
    if got != want:
        os.remove(part)
        raise KdevError(ErrorCode.CONFLICT, f"md5 mismatch for {remote_path}: remote {want} local {got}",
                        recovery="retry; the remote file may have changed during the copy")
    os.replace(part, local_path)
    return got


def push_verified(rc: RemoteConfig, local_path: str, remote_path: str, runner: Runner = run) -> str:
    """scp local -> remote, verify the remote md5sum; returns the md5."""
    want = md5_file(local_path)
    p = runner(scp_argv(rc, os.path.basename(local_path), remote_path, to_remote=True), timeout=3600,
               cwd=os.path.dirname(os.path.abspath(local_path)))
    if p.returncode != 0:
        raise KdevError(ErrorCode.INTERNAL, f"scp to {remote_path} failed: {p.stderr.strip()[:200]}")
    got = remote_md5(rc, remote_path, runner)
    if got != want:
        raise KdevError(ErrorCode.CONFLICT, f"md5 mismatch after upload of {local_path}: {want} vs {got}")
    return got


def safe_extract(tar_path: str, dest: str) -> list[str]:
    """Extract a tarball refusing absolute paths, '..' and links outside dest; returns member names."""
    dest_p = Path(dest).resolve()
    dest_p.mkdir(parents=True, exist_ok=True)
    names = []
    with tarfile.open(tar_path, "r:*") as tf:
        members = []
        for m in tf.getmembers():
            tgt = (dest_p / m.name).resolve()
            if m.name.startswith(("/", "\\")) or ".." in Path(m.name).parts or \
                    (tgt != dest_p and dest_p not in tgt.parents):
                raise KdevError(ErrorCode.INVALID_REQUEST, f"unsafe path in archive: {m.name}")
            if m.issym() or m.islnk() or m.isdev():
                continue
            members.append(m)
            names.append(m.name)
        try:
            tf.extractall(dest_p, members=members, filter="data")  # py>=3.12
        except TypeError:
            tf.extractall(dest_p, members=members)
    return names


def pull_run(rc: RemoteConfig, run_id: str, local_runs_dir: str, runner: Runner = run) -> str:
    """Copy one remote run dir to local (md5-verified); returns the local path."""
    if "/" in run_id or run_id.startswith(".") or not run_id:
        raise KdevError(ErrorCode.INVALID_REQUEST, f"bad run id {run_id!r}")
    rruns = rc.remote_home.rstrip("/") + "/runs"
    tgz = f"/tmp/kdev_pull_{run_id}.tgz"
    script = (f"cd {shlex.quote(rruns)} && test -d {shlex.quote(run_id)} && "
              f"tar -czf {shlex.quote(tgz)} {shlex.quote(run_id)} && md5sum {shlex.quote(tgz)}")
    p = runner(ssh_shell_argv(rc, script), timeout=600)
    if p.returncode != 0 or not p.stdout.strip():
        raise KdevError(ErrorCode.NOT_FOUND, f"remote run {run_id} not found under {rruns}",
                        details={"stderr": p.stderr[-300:]})
    want = p.stdout.split()[0]
    with tempfile.TemporaryDirectory(prefix="kdev_pull_") as td:
        local_tgz = os.path.join(td, f"{run_id}.tgz")
        fetch_verified(rc, tgz, local_tgz, runner, expect_md5=want)
        safe_extract(local_tgz, local_runs_dir)
    runner(ssh_argv(rc, ["rm", "-f", tgz]), timeout=60)
    return str(Path(local_runs_dir) / run_id)
