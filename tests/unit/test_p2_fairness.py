"""Board #314: shared-lock starvation (turnstile in service.phase) and the daemon watchdog (`kdev serve ensure`)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time

import pytest

from kdev.config import Config
from kdev.contract import ErrorCode, KdevError
from kdev.models import FileRef, JobSpec
from kdev.service import client, daemon, phase, queue

POSIX = sys.platform != "win32"
PY = sys.executable


# ------------------------------------------------------------------------------------------- pure policy
def test_starving_threshold_fifo_and_off(monkeypatch) -> None:
    now = 1000.0
    reg = [{"pid": 1, "state": "wait_slot", "t_starve": 900.0},
           {"pid": 2, "state": "wait_sh", "t_wait": 990.0, "t_starve": 960.0},
           {"pid": 3, "state": "wait_sh", "t_wait": 995.0},           # 5 s: not starving yet
           {"pid": 4, "state": "ready", "t_starve": 0.0}]             # not a waiter state
    assert [e["pid"] for e in phase.starving(reg, phase.SH_WAIT, now=now, limit_s=30)] == [2]
    assert [e["pid"] for e in phase.starving(reg, ("wait_slot", "wait_sh"), now=now, limit_s=30)] == [1, 2]
    assert phase.starving(reg, phase.SH_WAIT, now=now, limit_s=30, me=2) == []
    # FIFO among starvers: a waiter only yields to ones that started waiting before it
    assert [e["pid"] for e in phase.starving(reg, phase.SLOT_WAIT, now=now, limit_s=30, older_than=950.0)] == [1]
    assert phase.starving(reg, phase.SLOT_WAIT, now=now, limit_s=30, older_than=850.0) == []
    monkeypatch.setenv("KDEV_PHASE_STARVE_S", "0")                  # 0 = fairness off
    assert phase.starving(reg, phase.SH_WAIT, now=now) == []
    monkeypatch.setenv("KDEV_PHASE_STARVE_S", "30")
    assert [e["pid"] for e in phase.starving(reg, phase.SH_WAIT, now=now)] == [2]


def test_turnstile_yield_is_bounded() -> None:
    t = [0.0]
    reg = [{"pid": 7, "state": "wait_sh", "t_starve": time.time() - 100}]
    gate = phase.Turnstile(states=phase.SH_WAIT, max_s=120, reg_fn=lambda: reg, clock=lambda: t[0])
    assert gate.blocked() and gate.to == [7]
    t[0] = 119.0
    assert gate.blocked()
    t[0] = 120.0
    assert not gate.blocked()                    # bounded: the EX taker goes ahead after max_s
    gate.reset()
    assert gate.yielded_s == pytest.approx(120.0)
    reg.clear()                                  # the starving waiter got its slot
    assert not gate.blocked()
    assert not phase.Turnstile(states=phase.SH_WAIT, max_s=0, reg_fn=lambda: reg).blocked()


def test_must_defer_lifts_writer_preference_for_a_starving_waiter() -> None:
    ready = [{"pid": 7, "state": "ready"}]
    assert phase.must_defer(10.0, [], ready, me=100)                    # writer preference ...
    assert not phase.must_defer(10.0, [], ready, me=100, starving=True)  # ... but not forever
    # FIFO with an EX waiter (gpu_run.sh) that queued before us is kept: it is bounded by itself
    w = type("W", (), {"pid": 5, "owner_pid": 5, "started": 9.0})()
    assert phase.must_defer(10.0, [w], [], me=100, starving=True)


# ------------------------------------------------------------------------------------------- reproductions (POSIX)
@pytest.fixture
def fair_env(tmp_path, monkeypatch):
    from kdev.service import lockq

    monkeypatch.setattr(lockq, "view", lambda cfg, **kw: lockq.QueueView())
    monkeypatch.setenv("KDEV_HOME", str(tmp_path))
    monkeypatch.setenv("KDEV_PHASE_SLOTS", "1")
    monkeypatch.setenv("KDEV_PHASE_YIELD_S", "20")
    monkeypatch.setenv("KDEV_EXCLUSIVE", "")
    monkeypatch.setenv("KDEV_WAITED_S", "0")
    monkeypatch.setenv("KDEV_PHASED", "1")
    return tmp_path


def _sleepers(n: int) -> list[subprocess.Popen]:
    """Live pids for fake registry entries (entries of dead pids are dropped)."""
    return [subprocess.Popen(["sleep", "120"]) for _ in range(n)]


def _discover_wait(home, lp: str, give_up_s: float) -> float:
    """A discover job's worker: PhasedLock.prepare() (slot + SHARED). Returns the wait, raises on give-up."""
    t0 = time.monotonic()

    def stop() -> None:
        if time.monotonic() - t0 > give_up_s:
            raise KdevError(ErrorCode.GPU_BUSY_TIMEOUT, "starved")

    pl = phase.PhasedLock(lp, home, apps_fn=lambda: [], stop=stop)
    try:
        pl.prepare()
        return time.monotonic() - t0
    finally:
        for fd in (pl.slot_fd, pl.fd):
            if fd is not None and fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass
        pl.entry.remove()
        os.environ["KDEV_PHASED"] = "1"


def _ab_parents(home, n_parents: int, stop: threading.Event) -> list[threading.Thread]:
    """Other agents' judge-mode ab jobs: real _phased_children loops whose (fake) children each hold the single
    prepare slot ~0.3 s; the parent re-takes the slot the instant a child exits."""
    from kdev.timing import judge_mode as JM

    fake = home / "fakepy"
    fake.write_text(f"#!{PY}\nimport json, time\ntime.sleep(0.3)\n"
                    "print(json.dumps({'status': 'ok', 'verdict': 'same'}))\n")
    fake.chmod(0o755)
    files = [FileRef(path=str(home / f"{x}.py"), label=x) for x in "AB"]
    by = {f.label: f for f in files}

    def run(k: int) -> None:
        spec = JobSpec(kind="ab", files=files, params={"extra": {"judge_mode": True}}, job_id=f"ab{k}")
        rp = home / f"ab{k}"
        rp.mkdir()
        gen = JM._phased_children(spec, ["A", "B", "B", "A"] * 30, by, str(rp), time.time() + 600, lambda m: None,
                                  python=str(fake), free_mib=lambda: 10 ** 6)
        for _ in gen:
            if stop.is_set():
                gen.close()
                return

    ts = [threading.Thread(target=run, args=(k,), daemon=True) for k in range(n_parents)]
    for t in ts:
        t.start()
    return ts


@pytest.mark.skipif(not POSIX, reason="POSIX flock")
@pytest.mark.parametrize("fair", [False, True])
def test_slot_starvation_reproduced_and_fixed(fair_env, monkeypatch, fair) -> None:
    """Pod 2026-11-02: a discover sat in wait_slot while two agents' ab jobs cycled the prepare slot."""
    monkeypatch.setenv("KDEV_PHASE_STARVE_S", "1" if fair else "0")
    home = fair_env
    stop = threading.Event()
    ts = _ab_parents(home, 2, stop)
    try:
        time.sleep(0.8)                          # the ab jobs are cycling the slot
        if fair:
            waited = _discover_wait(home, str(home / "gpu.lock"), give_up_s=15)
            assert waited < 1 + 0.3 + 3          # starve threshold + one child + slack
        else:
            with pytest.raises(KdevError):
                _discover_wait(home, str(home / "gpu.lock"), give_up_s=5)
    finally:
        stop.set()
        for t in ts:
            t.join(30)


@pytest.mark.skipif(not POSIX, reason="POSIX flock")
@pytest.mark.parametrize("fair", [False, True])
def test_shared_starvation_by_back_to_back_timed_sections(fair_env, monkeypatch, fair) -> None:
    """Writer preference without a bound: two phased timers alternate EX sections (one is always 'ready'), so a
    SHARED waiter deferred forever. With the turnstile it gets SH after STARVE_S; timed sections stay EX."""
    import fcntl

    monkeypatch.setenv("KDEV_PHASE_STARVE_S", "1" if fair else "0")
    home = fair_env
    lp = str(home / "gpu.lock")
    stop = threading.Event()
    in_ex = [0]
    overlap = []
    mtx = threading.Lock()
    sleepers = _sleepers(2)

    def timer(pid: int) -> None:
        while not stop.is_set():
            pl = phase.PhasedLock(lp, home, apps_fn=lambda: [], stop=lambda: None)
            pl.entry = phase.Entry(home, pid=pid)
            pl.fd = os.open(lp, os.O_WRONLY | os.O_CREAT, 0o666)
            try:
                pl.exclusive()
                with mtx:
                    in_ex[0] += 1
                # strictly exclusive: nobody may hold SH while we time
                probe = os.open(lp, os.O_RDONLY)
                try:
                    fcntl.flock(probe, fcntl.LOCK_SH | fcntl.LOCK_NB)
                    overlap.append("shared during EX")
                except OSError:
                    pass
                finally:
                    os.close(probe)
                time.sleep(0.3)
            finally:
                pl.entry.set("ready")            # back in the queue at once (next ABBA process)
                os.close(pl.fd)

    ts = [threading.Thread(target=timer, args=(p.pid,), daemon=True) for p in sleepers]
    try:
        for t in ts:
            t.start()
        time.sleep(0.8)
        if fair:
            waited = _discover_wait(home, lp, give_up_s=15)
            assert waited < 1 + 2 * 0.3 + 3
        else:
            with pytest.raises(KdevError):
                _discover_wait(home, lp, give_up_s=5)
        assert in_ex[0] >= 2 and not overlap
    finally:
        stop.set()
        for t in ts:
            t.join(10)
        for p in sleepers:
            p.kill()
            p.wait()


# ------------------------------------------------------------------------------------------- daemon watchdog
@pytest.fixture
def cfg(tmp_path) -> Config:
    c = Config(home=tmp_path / "home", lock_path=str(tmp_path / "gpu.lock"),
               offender_log=str(tmp_path / "gpu_offenders.log"))
    c.home.mkdir(parents=True)
    return c


def _beat(cfg: Config, ts: str | None = None) -> None:
    conn = queue.connect(cfg)
    try:
        queue.heartbeat(conn, gpu_lane_job=None, build_workers=1)
        if ts:
            conn.execute("UPDATE heartbeat SET ts_utc=?", (ts,))
    finally:
        conn.close()


@pytest.fixture
def starts(cfg, monkeypatch) -> list[int]:
    """daemon.start_background replaced by a fake that heartbeats (as a fresh daemon would)."""
    calls: list[int] = []

    def fake(c, opts):
        calls.append(1)
        _beat(c)
        return 4242

    monkeypatch.setattr(daemon, "start_background", fake)
    return calls


def test_ensure_is_idempotent(cfg, starts) -> None:
    _beat(cfg)                                                   # alive
    assert daemon.ensure(cfg) == {"alive": True, "started": False, "pid": os.getpid(), "age_s": pytest.approx(0, abs=5)}
    assert starts == []
    _beat(cfg, "2020-01-01T00:00:00Z")                          # died long ago
    info = daemon.ensure(cfg)
    assert info["started"] and info["alive"] and info["pid"] == 4242 and starts == [1]
    assert not daemon.ensure(cfg)["started"] and starts == [1]   # second call: no second daemon


def test_ensure_waits_when_another_client_is_starting_it(cfg, starts) -> None:
    from kdev.service.lock import _FileLock

    g = _FileLock(str(cfg.home / "serve.ensure.lock"))
    assert g.try_acquire()
    try:
        info = daemon.ensure(cfg, wait_s=0.3)
    finally:
        g.release()
    assert not info["started"] and starts == []


def test_serve_ensure_cli(cfg, starts, monkeypatch) -> None:
    from kdev.cli import main

    monkeypatch.setenv("KDEV_HOME", str(cfg.home))
    assert main(["serve", "ensure", "--json"]) == 0
    assert starts == [1]
    assert main(["serve", "ensure", "--json"]) == 0
    assert starts == [1]


def _gpu_spec() -> JobSpec:
    return JobSpec(kind="exec", params={"argv": [PY, "-c", "pass"]}, agent="t")


def test_client_restarts_a_dead_daemon_before_submitting(cfg, starts, monkeypatch) -> None:
    monkeypatch.delenv("KDEV_NO_GPU", raising=False)
    monkeypatch.setenv("KDEV_AUTO_SERVE", "1")
    _beat(cfg, "2020-01-01T00:00:00Z")                          # a daemon served here and died
    r = client.submit_and_maybe_wait(cfg, _gpu_spec(), detach=True)
    assert r.verdict == "queued" and starts == [1]
    assert queue.get(cfg, r.job_id).state == "queued"
    # `kdev job submit` path
    assert client.submit(cfg, _gpu_spec()).state == "queued" and starts == [1]


def test_client_never_served_keeps_direct_path_and_opt_out(cfg, starts, monkeypatch) -> None:
    monkeypatch.delenv("KDEV_NO_GPU", raising=False)
    monkeypatch.setenv("KDEV_AUTO_SERVE", "1")
    r = client.submit_and_maybe_wait(cfg, _gpu_spec(), detach=True)   # no heartbeat row: laptop, never served
    assert r.error and r.error.code == ErrorCode.CAPABILITY_UNAVAILABLE.value and starts == []
    monkeypatch.setenv("KDEV_AUTO_SERVE", "0")
    _beat(cfg, "2020-01-01T00:00:00Z")
    with pytest.raises(KdevError):
        client.submit(cfg, _gpu_spec())
    assert starts == []


def test_wait_restarts_the_daemon_when_it_dies(cfg, monkeypatch) -> None:
    """11-01: the daemon died while clients waited on queued jobs; they hung forever."""
    monkeypatch.setenv("KDEV_AUTO_SERVE", "1")
    monkeypatch.setattr(client, "AUTO_SERVE_CHECK_S", 0.05)
    monkeypatch.setattr(daemon, "STALE_S", 0.5)
    _beat(cfg)
    st = queue.enqueue(cfg, _gpu_spec())
    _beat(cfg, "2020-01-01T00:00:00Z")                          # ... and then it died
    calls = []

    def fake(c, opts):                                           # the new daemon serves the queued job
        calls.append(1)
        _beat(c)
        conn = queue.connect(c)
        try:
            assert queue.claim(conn, st.job_id)
            queue.finish(conn, st.job_id, "done", exit_code=0)
        finally:
            conn.close()
        return 4343

    monkeypatch.setattr(daemon, "start_background", fake)
    r = client.wait(cfg, st.job_id, timeout_s=10, poll_s=0.02)
    assert r.verdict == "done" and calls == [1], r.to_dict()


def test_daemon_env_drops_job_scope(cfg, monkeypatch) -> None:
    seen = {}

    class P:
        pid = 1

        def poll(self):
            return 0

    def popen(argv, **kw):
        seen.update(kw["env"])
        return P()

    monkeypatch.setattr(daemon.subprocess, "Popen", popen)
    monkeypatch.setenv("KDEV_JOB_ID", "jx")
    monkeypatch.setenv("KDEV_PHASED", "held")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    daemon.start_background(cfg, daemon.DaemonOptions(foreground=False))
    assert "KDEV_JOB_ID" not in seen and "KDEV_PHASED" not in seen and "CUDA_VISIBLE_DEVICES" not in seen
    assert seen["KDEV_HOME"] == str(cfg.home)
    assert json.dumps(sorted(seen))                             # plain str env
