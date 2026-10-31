"""service.phase: phased GPU lock (prepare shared, timed section exclusive) + phased ab children."""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from kdev.models import FileRef, JobSpec
from kdev.service import phase

POSIX = sys.platform != "win32"


def W(pid: int, started: float, owner: int | None = None) -> SimpleNamespace:
    return SimpleNamespace(pid=pid, owner_pid=owner or pid, started=started)


def test_must_defer_writer_preference_and_fifo() -> None:
    me = 100
    assert not phase.must_defer(10.0, [], [], me=me)
    # a phased process ready to time -> wait (timed sections back to back)
    assert phase.must_defer(10.0, [], [{"pid": 7, "state": "ready"}], me=me)
    assert not phase.must_defer(10.0, [], [{"pid": me, "state": "ready"}], me=me)
    # EX waiter (gpu_run.sh) queued before us -> wait; after us -> go
    assert phase.must_defer(10.0, [W(5, 9.0)], [], me=me)
    assert not phase.must_defer(10.0, [W(5, 11.0)], [], me=me)
    # other phased processes in the flock queue (SH waiters / ready) never count as FIFO waiters
    assert not phase.must_defer(10.0, [W(8, 1.0)], [{"pid": 8, "state": "wait_sh"}], me=me)
    assert not phase.must_defer(10.0, [W(me, 1.0)], [], me=me)


def test_registry_staleness_and_idle(tmp_path) -> None:
    starts = {11: 500, 12: 600, 13: 700}
    sf = lambda pid: starts.get(pid, -1)  # noqa: E731
    for pid, st in ((11, "parked"), (12, "ready"), (13, "timing"), (14, "parked")):
        e = phase.Entry(tmp_path, pid=pid)
        e.doc["start"] = starts.get(pid, 1)
        e.set(st)
    # pid reuse: 13 now has another start time
    starts[13] = 701
    live = phase.entries(tmp_path, start_fn=sf)
    assert sorted(int(e["pid"]) for e in live) == [11, 12]
    assert not (tmp_path / "phase" / "procs" / "14.json").exists()     # dead pid removed
    assert phase.idle_pids(tmp_path, start_fn=sf) == {11, 12}
    assert not phase.kdev_holding(tmp_path, start_fn=sf)
    e = phase.Entry(tmp_path, pid=12)
    e.doc["start"] = 600
    e.set("timing")
    assert phase.kdev_holding(tmp_path, start_fn=sf)
    assert phase.idle_pids(tmp_path, start_fn=sf) == {11}


def test_enabled(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("KDEV_PHASED", raising=False)
    jm = {"extra": {"judge_mode": True}}
    assert phase.enabled(SimpleNamespace(kind="ab", params=jm))
    assert phase.enabled(SimpleNamespace(kind="time", params=jm))
    assert not phase.enabled(SimpleNamespace(kind="time", params={"extra": {}}))
    assert not phase.enabled(SimpleNamespace(kind="ab", params={"extra": {"judge_mode": True, "phased": False}}))
    assert phase.mode(SimpleNamespace(kind="discover", params={})) == "shared"
    assert phase.mode(SimpleNamespace(kind="check", params={})) == "shared"
    assert phase.mode(SimpleNamespace(kind="check", params={"extra": {"cold": True}})) is None
    assert phase.mode(SimpleNamespace(kind="ab", params=jm)) == "phased"
    for v in ("0", "held"):
        monkeypatch.setenv("KDEV_PHASED", v)
        assert not phase.enabled(SimpleNamespace(kind="ab", params=jm))
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delenv("KDEV_PHASED", raising=False)
    assert not phase.enabled(SimpleNamespace(kind="ab", params=jm))


@pytest.mark.skipif(not POSIX, reason="POSIX flock")
def test_prepare_shared_overlaps_timed_exclusive(tmp_path, monkeypatch) -> None:
    """Two preparers overlap (SH); a timed section (EX) starts only when no prepare runs, and a prepare
    cannot start while a timed section holds the lock."""
    from kdev.service import lockq

    monkeypatch.setattr(lockq, "view", lambda cfg, **kw: lockq.QueueView())
    monkeypatch.setenv("KDEV_EXCLUSIVE", "")
    monkeypatch.setenv("KDEV_WAITED_S", "0")
    lp = str(tmp_path / "gpu.lock")
    events: list[str] = []
    mtx = threading.Lock()

    def ev(s: str) -> None:
        with mtx:
            events.append(s)

    def mk(pid: int) -> phase.PhasedLock:
        pl = phase.PhasedLock(lp, tmp_path, apps_fn=lambda: [])
        pl.entry = phase.Entry(tmp_path, pid=pid)
        pl.entry.doc["start"] = None
        return pl

    a, b = mk(901), mk(902)
    a.slot_fd = b.slot_fd = -1
    a.prepare()
    b.prepare()                       # shared: does not block
    ev("both_prepared")
    done = threading.Event()

    def time_a() -> None:
        a.exclusive()                 # waits for b's SH
        ev("a_timing")
        done.wait(5)
        os.close(a.fd)                # end of a's "process"
        ev("a_done")

    t = threading.Thread(target=time_a)
    t.start()
    time.sleep(0.5)
    assert "a_timing" not in events   # b still holds SH
    import fcntl

    fcntl.flock(b.fd, fcntl.LOCK_UN)  # b leaves its prepare (e.g. parks)
    for _ in range(50):
        if "a_timing" in events:
            break
        time.sleep(0.05)
    assert "a_timing" in events
    # a new prepare must wait for the timed section
    c = mk(903)
    c.slot_fd = -1
    tc = threading.Thread(target=lambda: (c.prepare(), ev("c_prepared")))
    tc.start()
    time.sleep(0.5)
    assert "c_prepared" not in events
    done.set()
    t.join(5)
    tc.join(5)
    assert events.index("a_done") < events.index("c_prepared")


@pytest.mark.skipif(not POSIX, reason="pass_fds / flock")
def test_phased_children_keep_abba_timing_order(tmp_path, monkeypatch) -> None:
    """Children start concurrently (prepare overlap) but 'time' strictly in sequence order (go tokens)."""
    from kdev.timing import judge_mode as JM

    log = tmp_path / "order.log"
    fake = tmp_path / "fakepy"
    fake.write_text(f"""#!{sys.executable}
import json, os, sys, time
jp = sys.argv[3]; d = os.path.dirname(jp); i = json.load(open(jp))["params"]["extra"]["ab_proc"]
with open({str(log)!r}, "a") as f: f.write(f"start {{i}}\\n")
go = os.environ["KDEV_PHASE_GO"]
while not os.path.exists(go): time.sleep(0.02)
with open({str(log)!r}, "a") as f: f.write(f"time {{i}}\\n")
time.sleep(0.1)
print(json.dumps({{"status": "ok", "verdict": "same", "run_id": "p%d" % i}}))
""")
    fake.chmod(0o755)
    monkeypatch.setenv("KDEV_HOME", str(tmp_path))
    slots = iter(range(100))
    fds: list[int] = []

    def try_slot() -> int | None:
        next(slots)
        fd = os.open(os.devnull, os.O_RDONLY)
        fds.append(fd)
        return fd

    files = [FileRef(path=str(tmp_path / f"{x}.py"), label=x) for x in "AB"]
    spec = JobSpec(kind="ab", files=files, params={"extra": {"judge_mode": True}}, job_id="t")
    seq = JM.proc_sequence(["A", "B"])
    by = {f.label: f for f in files}
    msgs: list[str] = []
    got = []
    started_at_first_yield = None
    for g in JM._phased_children(spec, seq, by, str(tmp_path), time.time() + 600, msgs.append,
                                 try_slot=try_slot, python=str(fake), free_mib=lambda: 10 ** 6):
        if started_at_first_yield is None:
            started_at_first_yield = sum("started" in m for m in msgs)
        got.append(g)
    assert started_at_first_yield == 4          # all four processes were up while the first one timed
    assert [g[0] for g in got] == [0, 1, 2, 3] and [g[1] for g in got] == seq
    lines = log.read_text().split("\n")
    times = [ln for ln in lines if ln.startswith("time")]
    assert times == ["time 0", "time 1", "time 2", "time 3"]
    assert json.loads(json.dumps(got[0][2]))["run_id"] == "p0"


@pytest.mark.skipif(not POSIX, reason="POSIX flock")
def test_parked_process_holds_no_lock(tmp_path, monkeypatch) -> None:
    """Regression (pod 22:31): parked ABBA successors kept their SH lock, so the head never got EX and the
    whole GPU queue stalled. A parked process (waiting for its go token) must not block anyone's EX."""
    from kdev.service import lockq

    monkeypatch.setattr(lockq, "view", lambda cfg, **kw: lockq.QueueView())
    monkeypatch.setenv("KDEV_EXCLUSIVE", "")
    monkeypatch.setenv("KDEV_WAITED_S", "0")
    lp = str(tmp_path / "gpu.lock")

    def mk(pid: int, go: str | None) -> phase.PhasedLock:
        pl = phase.PhasedLock(lp, tmp_path, apps_fn=lambda: [], go_path=go)
        pl.entry = phase.Entry(tmp_path, pid=pid)
        pl.entry.doc["start"] = None
        pl.slot_fd = -1
        return pl

    head, succ = mk(911, None), mk(912, str(tmp_path / "go_succ"))
    head.prepare()
    succ.prepare()
    t = threading.Thread(target=succ.park, daemon=True)     # waits for its go token (never given here)
    t.start()
    time.sleep(0.3)
    head.park()
    got = threading.Event()
    threading.Thread(target=lambda: (head.exclusive(), got.set()), daemon=True).start()
    assert got.wait(5), "head could not get EX while its successor was parked"
    open(tmp_path / "go_succ", "w").close()
    t.join(5)


@pytest.mark.skipif(not POSIX, reason="POSIX flock")
def test_parked_gives_way_to_foreign_holder(tmp_path, monkeypatch) -> None:
    """gpu_run.sh holds the lock (EX, no kdev holder registered) and waits for every CUDA context: a parked
    process must exit (Requeue) instead of blocking it."""
    import fcntl

    from kdev.contract import KdevError
    from kdev.service import lockq

    monkeypatch.setattr(lockq, "view", lambda cfg, **kw: lockq.QueueView())
    monkeypatch.setattr(phase, "BAIL_S", 0.5)
    lp = str(tmp_path / "gpu.lock")
    pl = phase.PhasedLock(lp, tmp_path, apps_fn=lambda: [], go_path=str(tmp_path / "never"))
    pl.entry = phase.Entry(tmp_path, pid=921)
    pl.entry.doc["start"] = None
    pl.slot_fd = -1
    pl.prepare()
    pl.park_started = True
    foreign = os.open(lp, os.O_WRONLY)
    res: dict = {}

    def run() -> None:
        try:
            pl.park()
        except KdevError as e:
            res["e"] = e

    t = threading.Thread(target=run, daemon=True)
    t.start()
    time.sleep(0.3)
    fcntl.flock(foreign, fcntl.LOCK_EX)      # "gpu_run.sh" gets the lock once we parked
    t.join(5)
    os.close(foreign)
    assert "e" in res and res["e"].details.get(phase.REQUEUE_FLAG)
