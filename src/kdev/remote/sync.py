"""Bidirectional md5-verified sync pod <-> local. Owner: P2. (Pod storage is never persistent: M17.)

Rules: never overwrite a newer destination file; on conflict keep ``<name>.conflict-<ts>``; verify md5
after every copy; write a heartbeat file in cron mode. Ports scripts/sync_okc.sh logic.

Paths: ``cfg.sync_paths`` (default runs, ledger.db, gpus), relative to KDEV_HOME locally and to
``remote.remote_home`` on the pod. Transfer = one tar.gz per direction, md5-verified as a whole (like
sync_okc.sh) and per file after extraction; SQLite dbs travel as consistent backup snapshots.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

from kdev.config import Config
from kdev.contract import ErrorCode, KdevError
from kdev.remote import _synchelper as helper
from kdev.remote import ssh
from kdev.util import Runner, md5_file, run, utc_now

DEFAULT_PATHS = ("runs", "ledger.db", "gpus")
MTIME_TOL_S = 2.0          # clock skew / fs resolution: closer mtimes with different md5 = conflict
_runner: Runner = run      # monkeypatched in tests


@dataclass
class SyncPlanItem:
    rel_path: str
    action: str            # copy_to_local | copy_to_remote | conflict | skip_same
    src_md5: str = ""
    dst_md5: str = ""


@dataclass
class SyncReport:
    copied: int = 0
    conflicts: list[str] = field(default_factory=list)
    verified: int = 0
    failed: list[str] = field(default_factory=list)
    bytes: int = 0


def plan(local_index: dict[str, tuple[str, float]], remote_index: dict[str, tuple[str, float]],
         direction: str) -> list[SyncPlanItem]:
    """Pure planner over {rel_path: (md5, mtime)} indexes; direction pull|push|both.

    ``conflict`` means: both sides changed (destination newer, or same mtime within MTIME_TOL_S); the
    incoming version is stored beside the destination as ``<name>.conflict-<ts>`` and the destination is
    untouched. The incoming side of a conflict is the remote for pull/both and the local for push.
    For pull/both, src_md5 is the remote md5; for push it is the local md5."""
    if direction not in ("pull", "push", "both"):
        raise KdevError(ErrorCode.INVALID_REQUEST, f"direction {direction!r} must be pull|push|both")
    items: list[SyncPlanItem] = []
    for rel in sorted(set(local_index) | set(remote_index)):
        loc, rem = local_index.get(rel), remote_index.get(rel)
        if loc and rem and loc[0] == rem[0]:
            items.append(SyncPlanItem(rel, "skip_same", rem[0], loc[0]))
            continue
        if rem and not loc:
            if direction in ("pull", "both"):
                items.append(SyncPlanItem(rel, "copy_to_local", rem[0], ""))
            continue
        if loc and not rem:
            if direction in ("push", "both"):
                items.append(SyncPlanItem(rel, "copy_to_remote", loc[0], ""))
            continue
        assert loc and rem
        dt = float(rem[1]) - float(loc[1])          # > 0: remote newer
        if direction == "pull":
            act = "copy_to_local" if dt > MTIME_TOL_S else "conflict"
            items.append(SyncPlanItem(rel, act, rem[0], loc[0]))
        elif direction == "push":
            act = "copy_to_remote" if -dt > MTIME_TOL_S else "conflict"
            items.append(SyncPlanItem(rel, act, loc[0], rem[0]))
        else:
            if dt > MTIME_TOL_S:
                items.append(SyncPlanItem(rel, "copy_to_local", rem[0], loc[0]))
            elif -dt > MTIME_TOL_S:
                items.append(SyncPlanItem(rel, "copy_to_remote", loc[0], rem[0]))
            else:
                items.append(SyncPlanItem(rel, "conflict", rem[0], loc[0]))
    return items


# ---------------------------------------------------------------- transports

class Transport(Protocol):
    def index(self, paths: list[str], since_s: float | None) -> dict[str, list]: ...
    def fetch(self, rels: list[str], out_tgz: str) -> str: ...                  # -> verified md5
    def push(self, tgz: str, manifest: dict[str, list], ts: str) -> dict: ...  # -> unpack report


class SshTransport:
    """The pod side via ssh/scp; runs _synchelper's source with the remote python3."""

    def __init__(self, cfg: Config, runner: Runner | None = None, python: str = "python3") -> None:
        self.rc = cfg.remote
        self.home = cfg.remote.remote_home
        self.runner = runner or _runner
        self.python = python
        self.src = Path(helper.__file__).read_text(encoding="utf-8")

    def _helper(self, *args: str, timeout: float = 1800) -> str:
        p = self.runner(ssh.ssh_argv(self.rc, [self.python, "-c", self.src, *args]), timeout=timeout)
        if p.returncode != 0:
            raise KdevError(ErrorCode.INTERNAL, f"remote sync helper {args[0]} failed: {p.stderr.strip()[-300:]}")
        return p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""

    def index(self, paths: list[str], since_s: float | None) -> dict[str, list]:
        out = self._helper("index", self.home, json.dumps(list(paths)), str(since_s or 0))
        return json.loads(out or "{}")

    def _tmp(self, name: str) -> str:
        return f"/tmp/kdev_sync_{os.getpid()}_{name}"

    def fetch(self, rels: list[str], out_tgz: str) -> str:
        with tempfile.TemporaryDirectory(prefix="kdev_sync_") as td:
            lst = os.path.join(td, "list.json")
            Path(lst).write_text(json.dumps(rels), encoding="utf-8")
            rlist, rtgz = self._tmp("list.json"), self._tmp("pull.tgz")
            ssh.push_verified(self.rc, lst, rlist, self.runner)
            want = self._helper("pack", self.home, rlist, rtgz)
            md5 = ssh.fetch_verified(self.rc, rtgz, out_tgz, self.runner, expect_md5=want)
        self.runner(ssh.ssh_argv(self.rc, ["rm", "-f", rlist, rtgz]), timeout=60)
        return md5

    def push(self, tgz: str, manifest: dict[str, list], ts: str) -> dict:
        with tempfile.TemporaryDirectory(prefix="kdev_sync_") as td:
            man = os.path.join(td, "manifest.json")
            Path(man).write_text(json.dumps(manifest), encoding="utf-8")
            rman, rtgz = self._tmp("manifest.json"), self._tmp("push.tgz")
            ssh.push_verified(self.rc, man, rman, self.runner)
            ssh.push_verified(self.rc, tgz, rtgz, self.runner)
            rep = json.loads(self._helper("unpack", self.home, rtgz, rman, ts) or "{}")
        self.runner(ssh.ssh_argv(self.rc, ["rm", "-f", rman, rtgz]), timeout=60)
        return rep


class LocalDirTransport:
    """A second directory standing in for the pod (tests, or a mounted backup disk)."""

    def __init__(self, home: str) -> None:
        self.home = home

    def index(self, paths: list[str], since_s: float | None) -> dict[str, list]:
        return helper.index(self.home, list(paths), since_s)

    def fetch(self, rels: list[str], out_tgz: str) -> str:
        return helper.pack(self.home, rels, out_tgz)

    def push(self, tgz: str, manifest: dict[str, list], ts: str) -> dict:
        return helper.unpack(self.home, tgz, manifest, ts)


def _transport(cfg: Config) -> Transport:
    if not cfg.remote.host:
        raise KdevError(ErrorCode.INVALID_REQUEST, "sync needs remote.host/port/key in kdev.toml",
                        recovery="configure [remote] in $KDEV_HOME/kdev.toml")
    return SshTransport(cfg)


# ---------------------------------------------------------------- sync

def sync_with(cfg: Config, transport: Transport, direction: str = "pull", since_s: float | None = None,
              dry_run: bool = False) -> tuple[SyncReport, list[SyncPlanItem]]:
    paths = list(cfg.sync_paths or DEFAULT_PATHS)
    home = str(cfg.home)
    Path(home).mkdir(parents=True, exist_ok=True)
    local_idx = helper.index(home, paths, since_s)
    remote_idx = transport.index(paths, since_s)
    items = plan({k: tuple(v) for k, v in local_idx.items()}, {k: tuple(v) for k, v in remote_idx.items()},
                 direction)
    rep = SyncReport()
    if dry_run:
        rep.conflicts = [i.rel_path for i in items if i.action == "conflict"]
        return rep, items
    ts = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    incoming_local = direction in ("pull", "both")
    to_local = [i for i in items if i.action == "copy_to_local" or (i.action == "conflict" and incoming_local)]
    to_remote = [i for i in items if i.action == "copy_to_remote" or (i.action == "conflict" and not incoming_local)]
    with tempfile.TemporaryDirectory(prefix="kdev_sync_") as td:
        if to_local:
            tgz = os.path.join(td, "pull.tgz")
            transport.fetch([i.rel_path for i in to_local], tgz)
            manifest = {i.rel_path: [remote_idx[i.rel_path][0], "conflict" if i.action == "conflict" else "copy",
                                     remote_idx[i.rel_path][1]] for i in to_local}
            _merge(rep, helper.unpack(home, tgz, manifest, ts))
        if to_remote:
            tgz = os.path.join(td, "push.tgz")
            helper.pack(home, [i.rel_path for i in to_remote], tgz)
            manifest = {i.rel_path: [local_idx[i.rel_path][0], "conflict" if i.action == "conflict" else "copy",
                                     local_idx[i.rel_path][1]] for i in to_remote}
            _merge(rep, transport.push(tgz, manifest, ts))
    _heartbeat(cfg, direction, rep)
    return rep, items


def _merge(rep: SyncReport, d: dict) -> None:
    rep.copied += int(d.get("copied", 0))
    rep.verified += int(d.get("verified", 0))
    rep.bytes += int(d.get("bytes", 0))
    rep.conflicts += list(d.get("conflicts", []))
    rep.failed += list(d.get("failed", []))


def _heartbeat(cfg: Config, direction: str, rep: SyncReport) -> None:
    try:
        p = Path(cfg.home) / "sync_heartbeat.json"
        p.write_text(json.dumps({"ts_utc": utc_now(), "direction": direction, "host": cfg.remote.host,
                                 "report": asdict(rep)}), encoding="utf-8")
    except OSError:
        pass


def sync(cfg: Config, direction: str = "pull", since_s: float | None = None, dry_run: bool = False
         ) -> SyncReport:
    rep, _ = sync_with(cfg, _transport(cfg), direction, since_s, dry_run)
    return rep


def verify_local_copy(path: str, md5: str) -> bool:
    return os.path.exists(path) and md5_file(path) == md5


__all__ = ["SyncPlanItem", "SyncReport", "plan", "sync", "sync_with", "SshTransport", "LocalDirTransport"]
