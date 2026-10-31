"""P2: remote - ssh argv, sync planner + md5-verified transfer (local-dir transport), bootstrap phases, probe."""
from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path

import pytest

import kdev.remote as remote
from kdev.config import Config, RemoteConfig
from kdev.contract import ErrorCode, KdevError
from kdev.remote import _synchelper as helper
from kdev.remote import bootstrap, ssh, sync
from kdev.util import Proc, md5_file


@pytest.fixture
def cfg(tmp_path) -> Config:
    c = Config(home=tmp_path / "local", lock_path=str(tmp_path / "gpu.lock"))
    c.remote = RemoteConfig(host="10.0.0.1", port=20860, key="~/.ssh/runpod_ed25519", remote_home="/root/.kdev")
    c.sync_paths = ["runs", "ledger.db"]
    c.home.mkdir()
    return c


def write(p: Path, text: str, mtime: float | None = None) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    if mtime is not None:
        os.utime(p, (mtime, mtime))


# ---------------------------------------------------------------- ssh

def test_ssh_argv(cfg):
    a = ssh.ssh_argv(cfg.remote, ["kdev", "queue", "status", "--json"])
    assert a[0] == "ssh" and a[a.index("-p") + 1] == "20860" and a[a.index("-i") + 1].endswith("runpod_ed25519")
    assert "BatchMode=yes" in a and "ServerAliveInterval=30" in a and a[-2] == "root@10.0.0.1"
    assert a[-1] == "kdev queue status --json"
    assert ssh.ssh_argv(cfg.remote, ["echo", "a b;rm -rf /"])[-1] == "echo 'a b;rm -rf /'"
    s = ssh.scp_argv(cfg.remote, "x.tgz", "/tmp/x.tgz", to_remote=True)
    assert s[0] == "scp" and s[s.index("-P") + 1] == "20860" and s[-1] == "root@10.0.0.1:/tmp/x.tgz"
    with pytest.raises(KdevError) as ei:
        ssh.ssh_argv(RemoteConfig(), ["ls"])
    assert ei.value.code == ErrorCode.INVALID_REQUEST


def test_fetch_verified_md5_mismatch(cfg, tmp_path):
    def runner(argv, **kw):
        if argv[0] == "scp":
            Path(kw["cwd"], argv[-1]).write_text("payload")
        return Proc(list(argv), 0, "", "")
    out = tmp_path / "got.tgz"
    with pytest.raises(KdevError) as ei:
        ssh.fetch_verified(cfg.remote, "/tmp/r.tgz", str(out), runner, expect_md5="0" * 32)
    assert ei.value.code == ErrorCode.CONFLICT and not out.exists() and not Path(str(out) + ".part").exists()
    import hashlib
    good = hashlib.md5(b"payload").hexdigest()
    assert ssh.fetch_verified(cfg.remote, "/tmp/r.tgz", str(out), runner, expect_md5=good) == good and out.exists()


def test_safe_extract_rejects_traversal(tmp_path):
    import io
    import tarfile
    tgz = tmp_path / "evil.tgz"
    with tarfile.open(tgz, "w:gz") as tf:
        data = b"x"
        ti = tarfile.TarInfo("../escape.txt")
        ti.size = len(data)
        tf.addfile(ti, io.BytesIO(data))
    with pytest.raises(KdevError):
        ssh.safe_extract(str(tgz), str(tmp_path / "dest"))
    assert not (tmp_path / "escape.txt").exists()


# ---------------------------------------------------------------- planner

def test_plan_rules():
    L = {"same": ("a", 100), "lonly": ("b", 100), "rnew": ("c", 100), "lnew": ("d", 200), "tie": ("e", 100)}
    R = {"same": ("a", 50), "ronly": ("f", 100), "rnew": ("g", 200), "lnew": ("h", 100), "tie": ("i", 101)}
    act = {d: {i.rel_path: i.action for i in sync.plan(L, R, d)} for d in ("pull", "push", "both")}
    assert act["pull"] == {"same": "skip_same", "ronly": "copy_to_local", "rnew": "copy_to_local",
                           "lnew": "conflict", "tie": "conflict"}
    assert act["push"] == {"same": "skip_same", "lonly": "copy_to_remote", "rnew": "conflict",
                           "lnew": "copy_to_remote", "tie": "conflict"}
    assert act["both"] == {"same": "skip_same", "lonly": "copy_to_remote", "ronly": "copy_to_local",
                           "rnew": "copy_to_local", "lnew": "copy_to_remote", "tie": "conflict"}
    with pytest.raises(KdevError):
        sync.plan({}, {}, "sideways")


# ---------------------------------------------------------------- transfer (LocalDirTransport = fake pod)

def test_sync_pull_push_conflict(cfg, tmp_path):
    pod = tmp_path / "pod"
    now = time.time()
    write(pod / "runs/r1/result.json", '{"a": 1}', now - 100)
    write(pod / "runs/r1/raw.jsonl", "1\n2\n", now - 100)
    write(pod / "runs/r2/NOTES.md", "pod newer", now)
    write(cfg.home / "runs/r2/NOTES.md", "local older", now - 1000)
    write(cfg.home / "runs/r3/mine.txt", "local only", now - 50)
    write(cfg.home / "runs/r4/x.txt", "local newer", now)
    write(pod / "runs/r4/x.txt", "pod older", now - 1000)
    write(pod / "runs/r1/skip.part", "partial")       # never synced
    t = sync.LocalDirTransport(str(pod))

    rep, items = sync.sync_with(cfg, t, "pull", dry_run=True)
    assert rep.copied == 0 and not (cfg.home / "runs/r1").exists() and rep.conflicts == ["runs/r4/x.txt"]

    rep, _ = sync.sync_with(cfg, t, "pull")
    assert rep.failed == [] and rep.verified == rep.copied == 4
    assert (cfg.home / "runs/r1/raw.jsonl").read_text() == "1\n2\n"
    assert (cfg.home / "runs/r2/NOTES.md").read_text() == "pod newer"          # older local overwritten
    assert (cfg.home / "runs/r4/x.txt").read_text() == "local newer"          # newer local kept ...
    conf = list((cfg.home / "runs/r4").glob("x.txt.conflict-*"))
    assert len(conf) == 1 and conf[0].read_text() == "pod older"               # ... incoming kept beside it
    assert not (cfg.home / "runs/r1/skip.part").exists()
    assert abs(os.path.getmtime(cfg.home / "runs/r1/result.json") - (now - 100)) < 2   # mtime preserved
    hb = json.loads((cfg.home / "sync_heartbeat.json").read_text())
    assert hb["direction"] == "pull" and hb["report"]["copied"] == 4

    rep, _ = sync.sync_with(cfg, t, "push")
    assert (pod / "runs/r3/mine.txt").read_text() == "local only"
    assert (pod / "runs/r4/x.txt").read_text() == "local newer"
    rep, items = sync.sync_with(cfg, t, "both")
    assert rep.copied == 0 and all(i.action == "skip_same" for i in items)    # converged, conflict copies ignored


def test_sync_since_and_sqlite_snapshot(cfg, tmp_path):
    pod = tmp_path / "pod"
    pod.mkdir()
    db = pod / "ledger.db"
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE t (x)")
    conn.executemany("INSERT INTO t VALUES (?)", [(i,) for i in range(100)])
    conn.commit()                                      # rows live in the WAL; the main file may be stale
    write(pod / "runs/old/a.txt", "old", time.time() - 7200)
    write(pod / "runs/new/b.txt", "new")
    rep, _ = sync.sync_with(cfg, sync.LocalDirTransport(str(pod)), "pull", since_s=3600)
    conn.close()
    assert not (cfg.home / "runs/old").exists() and (cfg.home / "runs/new/b.txt").exists()
    got = sqlite3.connect(cfg.home / "ledger.db")
    assert got.execute("SELECT count(*) FROM t").fetchone()[0] == 100
    got.close()


def test_unpack_detects_corruption(tmp_path):
    src = tmp_path / "src"
    write(src / "runs/a.txt", "hello")
    tgz = tmp_path / "x.tgz"
    helper.pack(str(src), ["runs/a.txt"], str(tgz))
    rep = helper.unpack(str(tmp_path / "dst"), str(tgz), {"runs/a.txt": ["0" * 32, "copy", 0]}, "ts")
    assert rep["failed"] == ["runs/a.txt"] and not (tmp_path / "dst/runs/a.txt").exists()
    rep = helper.unpack(str(tmp_path / "dst"), str(tgz), {"runs/a.txt": [md5_file(str(src / "runs/a.txt")), "copy",
                                                                          0]}, "ts")
    assert rep["verified"] == 1 and (tmp_path / "dst/runs/a.txt").read_text() == "hello"


def test_ssh_transport_commands(cfg):
    calls = []

    def runner(argv, **kw):
        calls.append(argv)
        return Proc(list(argv), 0, '{"runs/a": ["m", 1.0]}\n', "")
    t = sync.SshTransport(cfg, runner=runner)
    assert t.index(["runs"], None) == {"runs/a": ["m", 1.0]}
    a = calls[0]
    assert a[0] == "ssh" and a[-1].startswith("python3 -c ") and " index /root/.kdev " in a[-1]


# ---------------------------------------------------------------- bootstrap

def test_bootstrap_phases(cfg, monkeypatch):
    calls: list[list[str]] = []
    done = {"venv", "toolchain"}

    def runner(argv, **kw):
        calls.append(list(argv))
        cmd = argv[-1]
        if cmd.startswith("test -f"):
            return Proc(argv, 0 if any(f"/{p}.done" in cmd for p in done) else 1, "", "")
        if "md5sum" in cmd:
            return Proc(argv, 0, "", "")         # md5 mismatch -> kdev_install fails
        return Proc(argv, 0, "", "")
    monkeypatch.setattr(bootstrap, "_runner", runner)
    out = bootstrap.setup(cfg)
    assert out["venv"] == out["toolchain"] == "skipped"
    assert out["kdev_install"] == "failed" and out["flush_selftest"] == "skipped" and out["doctor"] == "skipped"
    with pytest.raises(KdevError):
        bootstrap.setup(cfg, phases=("nope",))


def test_bootstrap_selftest_goes_through_lock(cfg, monkeypatch):
    seen = []

    def runner(argv, **kw):
        seen.append(argv[-1])
        if argv[-1].startswith("test -f"):
            return Proc(argv, 1, "", "")
        return Proc(argv, 0, "READ_FLUSH_OK 12.0 us/read\n", "")
    monkeypatch.setattr(bootstrap, "_runner", runner)
    out = bootstrap.setup(cfg, phases=("flush_selftest", "doctor"))
    assert out == {"flush_selftest": "done", "doctor": "done"}
    assert any(" -m kdev lock run --timeout 120 -- " in c for c in seen)


# ---------------------------------------------------------------- probe / commands

def test_mount_fstype_and_probe(cfg, tmp_path):
    m = tmp_path / "mounts"
    m.write_text("overlay / overlay rw 0 0\nmfs#x /workspace fuse.mfs rw 0 0\n")
    assert remote._mount_fstype("/workspace/kdev", str(m)) == ("/workspace", "fuse.mfs")
    assert remote._mount_fstype("/root/.kdev", str(m)) == ("/", "overlay")

    class Ctx:
        config = cfg
    checks = remote.probe(Ctx())
    names = {c.name for c in checks}
    assert {"kdev home storage", "free disk", "ssh key"} <= names


def test_remote_command_parses_result(cfg, monkeypatch):
    from argparse import Namespace

    from kdev.registry import Context
    from kdev.remote import commands
    seen = {}

    def fake(rc, argv, **kw):
        seen["argv"] = argv
        return Proc(argv, 0, json.dumps({"schema": "kdev.result/1", "tool": "queue status", "status": "ok",
                                         "verdict": "idle", "version": "0.1.0", "protocol": "kdev-tp/1"}) + "\n", "")
    monkeypatch.setattr(ssh, "run_remote", fake)
    r = commands.cmd_remote(Namespace(host=None, port=None, pull=False, argv=["--", "kdev", "queue", "status"]),
                            Context(config=cfg))
    assert r.verdict == "idle" and seen["argv"][1:] == ["-m", "kdev", "queue", "status", "--json"]
