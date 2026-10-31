"""GPU lock queue view, ETA, hold observations and kdev priority gate (P2, lockq)."""
from __future__ import annotations

import json
import os
import time

from kdev.config import Config
from kdev.service import lockq


def _cfg(tmp_path):
    return Config(home=tmp_path / "home", lock_path=str(tmp_path / "gpu.lock"))


def test_holds_median_and_eta(tmp_path):
    cfg = _cfg(tmp_path)
    assert lockq.median_hold(cfg) == (lockq.DEFAULT_HOLD_S, 0)
    for k, s in (("1", 30), ("2", 90), ("3", 60), ("3", 61)):
        lockq.record_hold(cfg, k, s)
    med, n = lockq.median_hold(cfg)
    assert n == 3 and med == 61
    qv = lockq.QueueView(holders=[{"pid": 1, "cmdline": "gpu_run.sh x"}], median_hold_s=60,
                         waiters=[lockq.QEntry(10, 9, "a", 1.0, "gpu_run"), lockq.QEntry(os.getpid(), os.getpid(),
                                                                                         "kdev", 2.0, "kdev")])
    assert qv.position(os.getpid()) == 2 and qv.eta_s(2) == 90
    assert qv.eta_s(None) == 150
    d = lockq.describe(qv)
    assert "position 2/2" in d and "1 gpu_run.sh" in d and "ETA ~1.5 min" in d


def test_registry_priority_gate(tmp_path):
    cfg = _cfg(tmp_path)
    e = lockq.register(cfg, priority=5, job_id="j1", agent="a")
    assert e is not None
    # another (lower-priority) waiter sees us; we never see ourselves
    assert [d["job_id"] for d in lockq.higher_priority_waiting(cfg, 0, me=-1)] == ["j1"]
    assert lockq.higher_priority_waiting(cfg, 5, me=-1) == []
    assert lockq.higher_priority_waiting(cfg, 0) == []
    lockq.mark(e, "granted")
    assert lockq.higher_priority_waiting(cfg, 0, me=-1) == []
    lockq.unregister(e)
    assert lockq.registered(cfg) == []


def test_dead_registrations_are_dropped(tmp_path):
    cfg = _cfg(tmp_path)
    d = cfg.home / "lockq"
    d.mkdir(parents=True)
    (d / "999999.json").write_text(json.dumps({"pid": 999999, "priority": 9, "t": time.time(), "state": "waiting"}))
    assert lockq.higher_priority_waiting(cfg, 0) == []
    assert not (d / "999999.json").exists()


def test_view_orders_waiters(tmp_path):
    cfg = _cfg(tmp_path)
    e = lockq.register(cfg, priority=2, job_id="j", agent="me")
    qv = lockq.view(cfg, opener_fn=lambda p: [os.getpid(), 4242], waiting_fn=lambda p: p == os.getpid())
    assert [w.pid for w in qv.waiters] == [os.getpid()] and qv.waiters[0].kind == "kdev"
    assert qv.waiters[0].priority == 2 and [h["pid"] for h in qv.holders] == [4242]
    lockq.unregister(e)
