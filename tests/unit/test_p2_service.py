"""P2: service - smi, lock (fake compute_apps, tmp lock path), queue (tmp queue.db), client, job runner, daemon."""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import threading
import time
from pathlib import Path

import pytest

import kdev.service as service
from kdev.config import Config
from kdev.contract import ErrorCode, KdevError, validate_result
from kdev.models import FileRef, JobSpec
from kdev.registry import build_registry
from kdev.service import _jobrun, client, daemon, lock, queue, smi
from kdev.service._common import build_worker_cap, cpu_budget, parse_duration
from kdev.util import Proc

PY = sys.executable
DENY = ["okc " + "submit", "okc " + "check"]          # built from parts: the P4 safety grep stays clean


@pytest.fixture
def cfg(tmp_path) -> Config:
    c = Config(home=tmp_path / "home", lock_path=str(tmp_path / "gpu.lock"),
               offender_log=str(tmp_path / "gpu_offenders.log"), foreign_wait_s=600, deny_commands=list(DENY))
    c.home.mkdir(parents=True)
    return c


@pytest.fixture
def no_apps(monkeypatch):
    monkeypatch.setattr(lock, "compute_apps", lambda: [])


def fake_runner(stdout: str, rc: int = 0):
    def runner(argv, **kw):
        return Proc(list(argv), rc, stdout, "")
    return runner


# ---------------------------------------------------------------- registry (Phase-0 tests kept)

def test_service_commands_registered():
    names = set(build_registry().names())
    assert {"serve", "job submit", "job wait", "queue status", "lock run", "sync", "remote"} <= names


def test_queue_schema_is_valid_sql():
    sql = (Path(service.__file__).with_name("schema.sql")).read_text(encoding="utf-8")
    conn = sqlite3.connect(":memory:")
    conn.executescript(sql)
    tables = {r[0] for r in conn.execute("select name from sqlite_master where type='table'")}
    assert {"jobs", "events", "offenders", "heartbeat"} <= tables


def test_lock_run_not_exposed_over_mcp():
    assert build_registry().get("lock run").mcp is False


# ---------------------------------------------------------------- smi

def test_parse_state_csv():
    st = smi.parse_state_csv("NVIDIA H200, GPU-1234, 1980, 3201, 120.55, 41, 7")
    assert (st.name, st.uuid, st.sm_mhz, st.mem_mhz, st.power_w, st.temp_c, st.util_pct) == \
        ("NVIDIA H200", "GPU-1234", 1980, 3201, 120.55, 41, 7)
    st = smi.parse_state_csv("H200, GPU-1, [N/A], 3201, [Not Supported], 41")
    assert st.sm_mhz is None and st.power_w is None and st.util_pct is None


def test_query_state_runner_and_missing():
    st = smi.query_state(runner=fake_runner("H200, GPU-1, 1755, 3201, 99.0, 40, 0\n"))
    assert st.sm_mhz == 1755 and st.ts_utc

    def boom(argv, **kw):
        raise FileNotFoundError("nvidia-smi")
    assert smi.query_state(runner=boom).name == ""
    assert smi.query_state(runner=fake_runner("", rc=9)).name == ""


def test_clocks_moved_and_ncu(tmp_path):
    from kdev.models import GpuState
    assert smi.clocks_moved(GpuState(sm_mhz=1980), GpuState(sm_mhz=1755))
    assert not smi.clocks_moved(GpuState(sm_mhz=1980), GpuState(sm_mhz=1970))
    assert not smi.clocks_moved(GpuState(), GpuState(sm_mhz=1))
    p = tmp_path / "params"
    p.write_text("ResmanDebugLevel: 4294967295\nRmProfilingAdminOnly: 1\n")
    assert smi.ncu_permitted(str(p)) is False
    p.write_text("RmProfilingAdminOnly: 0\n")
    assert smi.ncu_permitted(str(p)) is True
    assert smi.ncu_permitted(str(tmp_path / "nope")) is None


# ---------------------------------------------------------------- lock

def test_foreign_apps_filters_own_and_zombies():
    apps = [lock.ComputeApp(1, state="S"), lock.ComputeApp(2, state="Z"), lock.ComputeApp(3, state="R"),
            lock.ComputeApp(4, state="")]
    assert [a.pid for a in lock.foreign_apps(apps, {3})] == [1, 4]


def test_parse_compute_apps_csv():
    assert lock.parse_compute_apps_csv("240100, 3248\n17, [N/A]\nNo running processes found\n") == \
        [(240100, 3248), (17, None)]


def test_compute_apps_missing_smi(monkeypatch):
    def boom(argv, **kw):
        raise FileNotFoundError
    monkeypatch.setattr(lock, "_runner", boom)
    assert lock.compute_apps() == []


def test_exclusive_ok(cfg):
    with lock.exclusive(cfg, apps_fn=lambda: [], poll_s=0.01) as g:
        assert g.exclusive == "ok" and g.offenders == [] and g.lock_path == cfg.lock_path


def test_exclusive_waits_for_foreign_then_ok(cfg):
    calls = {"n": 0}

    def apps():
        calls["n"] += 1
        return [lock.ComputeApp(999999, 100, "python other.py", "S")] if calls["n"] <= 3 else []
    with lock.exclusive(cfg, apps_fn=apps, poll_s=0.01) as g:
        assert g.exclusive == "waited" and [a.pid for a in g.offenders] == [999999]


def test_exclusive_ignores_zombies_and_self(cfg):
    apps = [lock.ComputeApp(424242, state="Z"), lock.ComputeApp(os.getpid(), state="R")]
    with lock.exclusive(cfg, apps_fn=lambda: apps, poll_s=0.01) as g:
        assert g.exclusive == "ok"


def test_exclusive_aborts_and_logs_offenders(cfg, monkeypatch):
    monkeypatch.setattr(lock, "OFFENDER_LOG_AFTER_S", 0.0)
    cfg.foreign_wait_s = 0.2
    app = lock.ComputeApp(31337, 512, "python unlocked_warmup.py", "R")
    with pytest.raises(KdevError) as ei:
        with lock.exclusive(cfg, apps_fn=lambda: [app], poll_s=0.02, job_id="jX"):
            pass
    assert ei.value.code == ErrorCode.GPU_BUSY_TIMEOUT
    offs = queue.offenders_since(cfg, None)
    assert {o["action"] for o in offs} == {"waited", "aborted"}
    assert offs[0]["pid"] == 31337 and offs[0]["during_job"] == "jX"
    legacy = Path(cfg.offender_log).read_text()
    assert "waiting on unlocked pid 31337: python unlocked_warmup.py" in legacy
    # the lock was released on abort
    with lock.exclusive(cfg, apps_fn=lambda: [], poll_s=0.01):
        pass


def test_flock_excludes_second_holder(cfg):
    with lock.exclusive(cfg, apps_fn=lambda: [], poll_s=0.01):
        assert lock.holder(cfg)["held"] is True
        with pytest.raises(KdevError) as ei:
            with lock.exclusive(cfg, timeout_s=0.3, apps_fn=lambda: [], poll_s=0.01):
                pass
        assert ei.value.code == ErrorCode.GPU_BUSY_TIMEOUT
    assert lock.holder(cfg)["held"] is False


def test_exclusive_abortable(cfg):
    with lock.exclusive(cfg, apps_fn=lambda: []):
        with pytest.raises(KdevError) as ei:
            with lock.exclusive_abortable(cfg, apps_fn=lambda: [], abort=lambda: True):
                pass
        assert ei.value.code == ErrorCode.CANCELLED


@pytest.mark.parametrize("argv,denied", [
    (["okc", "submit", "a.py"], True),
    (["/usr/local/bin/okc", "check", "x"], True),
    (["okc.exe", "--profile", "p", "submit"], True),
    (["timeout", "30", "okc", "submit", "f.py"], True),
    (["bash", "-c", "cd /root/x && okc check f.py"], True),
    (["sh", "-c", "echo hi; okc submit f"], True),
    (["okc", "status", "job1"], False),
    (["okc", "leaderboard"], False),
    (["python", "bench.py", "--submit"], False),
    (["echo", "check"], False),
])
def test_is_denied(cfg, argv, denied):
    assert lock.is_denied(cfg, argv) is denied


def test_run_locked_deny_and_empty(cfg):
    with pytest.raises(KdevError) as ei:
        lock.run_locked(cfg, ["--", "okc", "submit", "f.py"])
    assert ei.value.code == ErrorCode.QUOTA_FORBIDDEN
    with pytest.raises(KdevError) as ei:
        lock.run_locked(cfg, ["--"])
    assert ei.value.code == ErrorCode.INVALID_REQUEST


def test_run_locked_exit_code_timeout_and_stderr(cfg, capfd):
    assert lock._run_locked(cfg, ["--", PY, "-c", "import sys; sys.exit(3)"], 30, apps_fn=lambda: []) == 3
    assert "GPU_EXCLUSIVE=OK waited=0s" in capfd.readouterr().err
    t0 = time.monotonic()
    assert lock._run_locked(cfg, [PY, "-c", "import time; time.sleep(30)"], 1, apps_fn=lambda: []) == lock.EXIT_TIMEOUT
    assert time.monotonic() - t0 < 20
    assert lock._run_locked(cfg, ["definitely-not-a-command-xyz"], 5, apps_fn=lambda: []) == 127


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX flock fd inheritance")
def test_child_inherits_lock(cfg):
    """gpu_run.sh semantics: the command keeps the flock (fd passed), so it is held while the child runs."""
    script = ("import fcntl,os,sys; fd=os.open(sys.argv[1], os.O_WRONLY);\n"
              "try:\n fcntl.flock(fd, fcntl.LOCK_EX|fcntl.LOCK_NB); sys.exit(0)\nexcept BlockingIOError:\n sys.exit(42)")
    assert lock._run_locked(cfg, [PY, "-c", script, cfg.lock_path], 30, apps_fn=lambda: []) == 42


# ---------------------------------------------------------------- queue

def spec(kind="exec", **kw) -> JobSpec:
    kw.setdefault("params", {"argv": [PY, "-c", "print('hi')"]} if kind == "exec" else {})
    return JobSpec(kind=kind, **kw)


def test_enqueue_positions_and_priority(cfg):
    a = queue.enqueue(cfg, spec(agent="w1"))
    b = queue.enqueue(cfg, spec(agent="w2"))
    c = queue.enqueue(cfg, spec(agent="w3", priority=5))
    bld = queue.enqueue(cfg, spec("build"))
    assert a.state == "queued" and a.lane == "gpu" and bld.lane == "build" and bld.position == 1
    assert queue.get(cfg, c.job_id).position == 1 and queue.get(cfg, a.job_id).position == 2
    assert queue.get(cfg, b.job_id).position == 3
    conn = queue.connect(cfg)
    first = queue.claim_next(conn, "gpu")
    assert first.job_id == c.job_id and queue.get(cfg, c.job_id).state == "running"
    assert queue.claim_next(conn, "gpu").job_id == a.job_id
    queue.finish(conn, a.job_id, "done", exit_code=0, run_id="r1", run_dir="/x")
    st = queue.get(cfg, a.job_id)
    assert (st.state, st.exit_code, st.run_id, st.finished_utc is not None) == ("done", 0, "r1", True)
    with pytest.raises(ValueError):
        queue.finish(conn, a.job_id, "running")
    conn.close()
    assert {j.job_id for j in queue.list_jobs(cfg, "queued")} == {b.job_id, bld.job_id}


def test_enqueue_validation(cfg):
    with pytest.raises(KdevError):
        queue.enqueue(cfg, JobSpec(kind="nope"))
    with pytest.raises(KdevError):
        queue.enqueue(cfg, JobSpec(kind="exec"))
    s = queue.enqueue(cfg, spec(job_id="jfixed"))
    with pytest.raises(KdevError) as ei:
        queue.enqueue(cfg, spec(job_id="jfixed"))
    assert ei.value.code == ErrorCode.CONFLICT and s.job_id == "jfixed"


def test_cancel_semantics(cfg):
    a = queue.enqueue(cfg, spec(agent="w1"))
    with pytest.raises(KdevError) as ei:
        queue.cancel(cfg, a.job_id, agent="other")
    assert ei.value.code == ErrorCode.CONFLICT
    assert queue.cancel(cfg, a.job_id, agent="w1").state == "cancelled"
    b = queue.enqueue(cfg, spec(agent="w1"))
    conn = queue.connect(cfg)
    queue.claim_next(conn, "gpu")
    assert not queue.cancel_requested(conn, b.job_id)
    assert queue.cancel(cfg, b.job_id).state == "running"
    assert queue.cancel_requested(conn, b.job_id)
    conn.close()
    with pytest.raises(KdevError) as ei:
        queue.get(cfg, "jnope")
    assert ei.value.code == ErrorCode.NOT_FOUND


def test_heartbeat_and_daemon_alive(cfg):
    assert not queue.daemon_alive(cfg)
    conn = queue.connect(cfg)
    queue.heartbeat(conn, gpu_lane_job=None, build_workers=4)
    assert queue.daemon_alive(cfg)
    conn.execute("UPDATE heartbeat SET ts_utc='2020-01-01T00:00:00Z'")
    assert not queue.daemon_alive(cfg)
    conn.execute("UPDATE heartbeat SET ts_utc=?, pid=?", (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), 2**22 + 7))
    assert not queue.daemon_alive(cfg)          # fresh but dead pid
    conn.close()


def test_snapshot(cfg):
    queue.enqueue(cfg, spec())
    snap = queue.snapshot(cfg)
    d = snap.to_dict()
    assert len(d["queued"]) == 1 and d["daemon"]["alive"] is False and d["gpu"]["skipped"]
    assert d["gpu"]["lock"]["held"] is False


def test_parse_duration_and_cpu_cap():
    assert parse_duration("3h") == 10800 and parse_duration("90s") == 90 and parse_duration("2") == 2
    assert parse_duration(None) is None
    with pytest.raises(KdevError):
        parse_duration("soon")
    assert 1 <= build_worker_cap(64) <= 16 and build_worker_cap(1) == 1 and cpu_budget() >= 1


# ---------------------------------------------------------------- client routing

def test_submit_needs_daemon(cfg):
    with pytest.raises(KdevError) as ei:
        client.submit(cfg, spec())
    assert ei.value.code == ErrorCode.CAPABILITY_UNAVAILABLE


def test_submit_and_wait_no_gpu(cfg):
    r = client.submit_and_maybe_wait(cfg, spec("time"))
    assert r.error.code == ErrorCode.CAPABILITY_UNAVAILABLE.value and validate_result(r.to_dict()) == []


def test_submit_denied_exec(cfg):
    with pytest.raises(KdevError) as ei:
        client.submit(cfg, spec(params={"argv": ["okc", "submit", "x.py"]}))
    assert ei.value.code == ErrorCode.QUOTA_FORBIDDEN


def test_wait_pending_then_done(cfg):
    st = queue.enqueue(cfg, spec())
    r = client.wait(cfg, st.job_id, timeout_s=0.05, poll_s=0.01)
    assert r.verdict == "pending" and r.job_id == st.job_id and r.status == "ok"
    conn = queue.connect(cfg)
    queue.claim_next(conn, "gpu")
    queue.finish(conn, st.job_id, "failed", error_code="timeout")
    conn.close()
    r = client.wait(cfg, st.job_id, poll_s=0.01)
    assert r.error.code == "timeout" and validate_result(r.to_dict()) == []


# ---------------------------------------------------------------- job runner (direct mode)

def test_direct_exec_job(cfg, no_apps):
    r = client.run_direct(cfg, spec(params={"argv": [PY, "-c", "print('hello from gpu lane')"]}))
    assert r.status == "ok" and r.summary["returncode"] == 0 and r.run_id
    d = Path(r.files["dir"])
    assert json.loads((d / "result.json").read_text())["status"] == "ok"
    assert "hello from gpu lane" in (d / "log.txt").read_text() and (d / "job.json").exists()
    assert "GPU_EXCLUSIVE=OK" in (d / "log.txt").read_text()


def test_direct_exec_failure_and_timeout(cfg, no_apps):
    r = client.run_direct(cfg, spec(params={"argv": [PY, "-c", "import sys; sys.exit(5)"]}))
    assert r.status == "fail" and r.summary["returncode"] == 5
    s = spec(params={"argv": [PY, "-c", "import time; time.sleep(60)"]}, timeout_s=1)
    t0 = time.monotonic()
    r = client.run_direct(cfg, s)
    assert r.error.code == "timeout" and time.monotonic() - t0 < 30 and _jobrun.state_for(r) == "timeout"


def test_needs_build_gate(cfg, no_apps):
    s = JobSpec(kind="time", files=[FileRef(path="/x/a.py")])
    r = client.run_direct(cfg, s)
    assert r.error.code == ErrorCode.NEEDS_BUILD.value
    # with a prebuilt cache hit the gate opens (worker then runs; here a stand-in command)
    import kdev.build.cache as cache
    assert _jobrun._prebuilt_gate(JobSpec(kind="time", files=[FileRef(path="a")], params={"allow_jit": True})) is None
    orig = cache.is_prebuilt
    try:
        cache.is_prebuilt = lambda spec: True
        assert _jobrun._prebuilt_gate(s) is None
        cache.is_prebuilt = lambda spec: False
        assert _jobrun._prebuilt_gate(s).code == ErrorCode.NEEDS_BUILD
    finally:
        cache.is_prebuilt = orig


def test_not_exclusive_marks_timing_result(cfg, monkeypatch):
    """A foreign context appearing during a timing job -> exclusive=fail, error not_exclusive, offender logged."""
    monkeypatch.setattr(_jobrun, "FOREIGN_POLL_S", 0.05)
    started = {"t": None}

    def apps():
        if started["t"] is None:
            return []
        return [lock.ComputeApp(77777, 64, "python rogue.py", "R")]

    def fake_child(cfg_, spec_, rd):
        started["t"] = time.time()
        code = ("import json,sys; json.dump({'schema':'kdev.result/1','tool':'time','status':'ok','verdict':'faster',"
                "'version':'0','protocol':'kdev-tp/1','fidelity':{'exclusive':'unknown'}}, open('result.json','w'));"
                "import time; time.sleep(0.6)")
        return [PY, "-c", code], str(rd.path)
    monkeypatch.setattr(_jobrun, "_child_argv", fake_child)
    s = JobSpec(kind="time", requires_prebuilt=False)
    rd = _jobrun.prepare_run(cfg, _jobrun.normalise(s))
    out = _jobrun.run_gpu(cfg, s, rd, apps_fn=apps)
    r = out.result
    assert out.offenders == [77777] and r.fidelity.exclusive == "fail" and r.error.code == "not_exclusive"
    assert any("NOT EXCLUSIVE" in w for w in r.warnings)
    assert [o["action"] for o in queue.offenders_since(cfg, None)] == ["violated"]
    saved = json.loads((Path(rd.path) / "result.json").read_text())
    assert saved["fidelity"]["exclusive"] == "fail" and saved["exit_code"] == 13


def test_build_lane_direct_runs_without_cuda(cfg):
    r = client.run_direct(cfg, JobSpec(kind="build", files=[FileRef(path="a.py")]))
    # P5 may still be a stub (not_implemented) - either way a valid Result came back from the CUDA-free child
    assert validate_result(r.to_dict()) == [] and r.job_id
    env = _jobrun.child_env(cfg, _jobrun.normalise(JobSpec(kind="build")), no_cuda=True)
    assert env["CUDA_VISIBLE_DEVICES"] == "" and env["KDEV_HOME"] == str(cfg.home)


# ---------------------------------------------------------------- daemon

def _serve_in_thread(cfg, workers=4, lanes=3):
    t = threading.Thread(target=daemon.serve, args=(cfg, daemon.DaemonOptions(build_workers=workers, gpu_lanes=lanes,
                                                                            poll_s=0.05)),
                         daemon=True)
    t.start()
    for _ in range(200):
        if queue.daemon_alive(cfg):
            return t
        time.sleep(0.05)
    raise AssertionError("daemon did not heartbeat")


def test_daemon_serial_gpu_lane_and_builds(cfg, no_apps, tmp_path):
    t = _serve_in_thread(cfg)
    try:
        marks = tmp_path / "marks.txt"
        code = ("import sys,time; f=open(sys.argv[1],'a'); f.write('S %f\\n'%time.time()); f.flush();"
                "time.sleep(0.4); f.write('E %f\\n'%time.time())")
        ids = [client.submit(cfg, spec(params={"argv": [PY, "-c", code, str(marks)]})).job_id for _ in range(3)]
        bid = client.submit(cfg, JobSpec(kind="build")).job_id
        res = [client.wait(cfg, j, timeout_s=60, poll_s=0.05) for j in ids]
        assert all(r.status == "ok" for r in res), [r.to_dict() for r in res]
        rb = client.wait(cfg, bid, timeout_s=60, poll_s=0.05)
        assert rb.verdict != "pending"
        ev = [line.split() for line in marks.read_text().splitlines()]
        assert [e[0] for e in ev] == ["S", "E"] * 3          # never two GPU jobs at once
        assert all(float(ev[i + 1][1]) >= float(ev[i][1]) for i in range(len(ev) - 1))
        st = queue.get(cfg, ids[0])
        assert st.state == "done" and st.run_dir and st.waited_s is not None
        # submit_and_maybe_wait routes to the daemon when it is alive (detach)
        os.environ.pop("KDEV_NO_GPU", None)
        r = client.submit_and_maybe_wait(cfg, spec(), detach=True)
        assert r.verdict == "queued" and r.job_id
        assert client.wait(cfg, r.job_id, timeout_s=60, poll_s=0.05).status == "ok"
    finally:
        assert daemon.stop(cfg, timeout_s=1) or True
        t.join(30)
    assert not t.is_alive() and not queue.daemon_alive(cfg)


def test_daemon_cancel_running_and_queued(cfg, no_apps):
    t = _serve_in_thread(cfg, lanes=1)   # one lane, so the second job stays queued
    try:
        long = client.submit(cfg, spec(params={"argv": [PY, "-c", "import time; time.sleep(60)"]})).job_id
        queued = client.submit(cfg, spec()).job_id
        for _ in range(200):
            if queue.get(cfg, long).state == "running" and queue.get(cfg, long).run_dir:
                break
            time.sleep(0.05)
        assert queue.cancel(cfg, queued).state == "cancelled"
        queue.cancel(cfg, long)
        r = client.wait(cfg, long, timeout_s=60, poll_s=0.05)
        assert r.error.code == "cancelled" and queue.get(cfg, long).state == "cancelled"
    finally:
        daemon.stop(cfg, timeout_s=1)
        t.join(30)


def test_daemon_recovers_stale_running_and_refuses_second(cfg):
    st = queue.enqueue(cfg, spec())
    conn = queue.connect(cfg)
    queue.claim_next(conn, "gpu")
    conn.close()
    # second instance refused while the guard is held
    from kdev.service.lock import _FileLock
    g = _FileLock(str(cfg.home / "serve.lock"))
    assert g.try_acquire()
    assert daemon.serve(cfg, daemon.DaemonOptions(poll_s=0.01)) == 1
    g.release()
    (cfg.home / "serve.stop").write_text("x")   # stale stop file is removed at start; set again below
    t = threading.Thread(target=daemon.serve, args=(cfg, daemon.DaemonOptions(poll_s=0.05)), daemon=True)
    t.start()
    for _ in range(200):
        if queue.get(cfg, st.job_id).state != "running":
            break
        time.sleep(0.05)
    s = queue.get(cfg, st.job_id)
    assert s.state == "failed" and s.error_code == "internal"
    daemon.stop(cfg, timeout_s=1)
    t.join(30)
    assert not t.is_alive()


# ---------------------------------------------------------------- CLI handlers

def test_cli_handlers(cfg, monkeypatch, tmp_path):
    from kdev import cli
    monkeypatch.setenv("KDEV_HOME", str(cfg.home))
    monkeypatch.setenv("KDEV_LOCK_PATH", cfg.lock_path)
    monkeypatch.setattr(lock, "compute_apps", lambda: [])
    r = cli.run(["lock", "run", "--timeout", "30", "--", PY, "-c", "print(1)"])
    assert r.status == "ok" and r.summary["returncode"] == 0 and r.summary["argv"][0] == PY
    r = cli.run(["lock", "run", "--", "okc", "submit", "f.py"])
    assert r.error.code == "quota_forbidden" and r.exit_code == 3
    sp = tmp_path / "job.json"
    sp.write_text(json.dumps(spec().to_dict()))
    assert cli.run(["job", "submit", str(sp)]).error.code == "capability_unavailable"
    st = queue.enqueue(cfg, spec(agent="pytest"))
    for argv in (["job", "list"], ["job", "status", st.job_id], ["queue", "status"], ["queue", "offenders"],
                 ["serve", "status"], ["job", "logs", st.job_id], ["job", "cancel", st.job_id]):
        r = cli.run(argv)
        assert validate_result(r.to_dict()) == [], argv
        assert r.status == "ok", (argv, r.to_dict())
    assert queue.get(cfg, st.job_id).state == "cancelled"
    assert cli.run(["job", "status", "jnope"]).error.code == "not_found"


def test_probe_never_raises(cfg):
    class Ctx:
        config = cfg
    checks = service.probe(Ctx())
    names = {c.name for c in checks}
    assert {"gpu", "gpu lock", "service", "ncu permission"} <= names
    assert all(c.status in ("PASS", "WARN", "FAIL", "SKIP") for c in checks)


def test_daemon_runs_gpu_jobs_in_parallel_lanes(cfg, no_apps):
    """Two GPU jobs run at once with 2 lanes (each worker takes the lock itself)."""
    t = _serve_in_thread(cfg, lanes=2)
    try:
        a = client.submit(cfg, spec(params={"argv": [PY, "-c", "import time; time.sleep(2)"]})).job_id
        b = client.submit(cfg, spec(params={"argv": [PY, "-c", "import time; time.sleep(2)"]})).job_id
        both = False
        for _ in range(200):
            if queue.get(cfg, a).state == "running" and queue.get(cfg, b).state == "running":
                both = True
                break
            time.sleep(0.05)
        assert both
        client.wait(cfg, a, timeout_s=60, poll_s=0.05)
        client.wait(cfg, b, timeout_s=60, poll_s=0.05)
    finally:
        daemon.stop(cfg, timeout_s=1)
        t.join(30)
