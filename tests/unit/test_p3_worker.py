"""P3: worker dispatch, artifact writing, error mapping, exec deny-list; never imports torch."""
from __future__ import annotations

import json
import sys

import pytest

from kdev import worker
from kdev.contract import ErrorCode, KdevError, Result, validate_result
from kdev.models import JobSpec


def _ok_job(spec, run_dir):
    from kdev.timing._io import append_jsonl

    append_jsonl(run_dir, "raw.jsonl", [{"x": 1}])
    return Result(tool=spec.kind, verdict="pass", summary={"n": 1})


def _err_job(spec, run_dir):
    raise KdevError(ErrorCode.NEEDS_BUILD, "not built", recovery="kdev build")


def _boom_job(spec, run_dir):
    raise RuntimeError("kaboom")


def _todo_job(spec, run_dir):
    raise NotImplementedError("x.y (P9)")


@pytest.fixture
def fake_kinds(monkeypatch):
    mod = sys.modules[__name__]
    monkeypatch.setitem(worker.KIND_TABLE, "fake_ok", f"{mod.__name__}:_ok_job")
    monkeypatch.setitem(worker.KIND_TABLE, "fake_err", f"{mod.__name__}:_err_job")
    monkeypatch.setitem(worker.KIND_TABLE, "fake_boom", f"{mod.__name__}:_boom_job")
    monkeypatch.setitem(worker.KIND_TABLE, "fake_todo", f"{mod.__name__}:_todo_job")


def test_resolve_known_and_unknown():
    assert worker.resolve("time").__name__ == "run_time"
    assert worker.resolve("calib").__name__ == "run_calib"
    with pytest.raises(KdevError) as e:
        worker.resolve("nope")
    assert e.value.code == ErrorCode.INVALID_REQUEST


def test_execute_writes_artifacts(fake_kinds, tmp_path):
    rd = tmp_path / "run1"
    res = worker.execute(JobSpec(kind="fake_ok", job_id="j1"), str(rd))
    assert res.status == "ok" and res.run_id == "run1" and res.job_id == "j1"
    d = json.loads((rd / "result.json").read_text())
    assert validate_result(d) == []
    meta = json.loads((rd / "meta.json").read_text())
    assert meta["job"]["kind"] == "fake_ok" and meta["protocol"] == "kdev-tp/1"
    assert (rd / "log.txt").exists() and res.files["raw"].endswith("raw.jsonl")


def test_execute_new_run_dir_under_home(fake_kinds, tmp_home):
    res = worker.execute(JobSpec(kind="fake_ok", run_name="x"))
    assert res.status == "ok"
    assert res.run_id.endswith("-fake_ok-x")
    assert (tmp_home / "runs" / res.run_id / "result.json").exists()


@pytest.mark.parametrize("kind,code,exit_code", [("fake_err", "needs_build", 4), ("fake_boom", "internal", 15),
                                                 ("fake_todo", "not_implemented", 14)])
def test_execute_errors_become_results(fake_kinds, tmp_path, kind, code, exit_code):
    res = worker.execute(JobSpec(kind=kind), str(tmp_path / kind))
    assert res.status == "error" and res.error.code == code and res.exit_code == exit_code
    assert validate_result(res.to_dict()) == []
    if kind == "fake_boom":
        assert "kaboom" in (tmp_path / kind / "log.txt").read_text()


def test_exec_deny_list(tmp_path):
    res = worker.execute(JobSpec(kind="exec", params={"argv": ["okc", "submit", "x.py"]}), str(tmp_path / "e"))
    assert res.error.code == "quota_forbidden" and res.exit_code == 3
    res = worker.execute(JobSpec(kind="exec", params={"argv": ["/usr/bin/okc", "check", "p", "x.py"]}),
                         str(tmp_path / "e2"))
    assert res.exit_code == 3


def test_exec_runs_argv(tmp_path):
    res = worker.execute(JobSpec(kind="exec", params={"argv": [sys.executable, "-c", "print('hi')"]}),
                         str(tmp_path / "e"))
    assert res.status == "ok" and res.summary["returncode"] == 0
    assert (tmp_path / "e" / "stdout.txt").read_text().strip() == "hi"


def test_main_reads_job_and_prints_one_json(fake_kinds, tmp_path, capsys):
    job = tmp_path / "job.json"
    job.write_text(json.dumps(JobSpec(kind="fake_ok").to_dict()))
    rc = worker.main([str(job), "--run-dir", str(tmp_path / "r")])
    out = capsys.readouterr().out.strip().splitlines()
    assert rc == 0 and len(out) == 1 and json.loads(out[0])["status"] == "ok"
    rc = worker.main([str(tmp_path / "missing.json")])
    assert rc == 2


def test_worker_and_timing_import_without_torch(monkeypatch):
    for blocked in ("torch", "numpy"):
        monkeypatch.setitem(sys.modules, blocked, None)
    import importlib

    for m in ("kdev.worker", "kdev.timing.engine", "kdev.timing.calib", "kdev.timing.flush",
              "kdev.timing.graph", "kdev.timing.overhead", "kdev.timing._env", "kdev.timing._direct",
              "kdev.timing.commands"):
        importlib.reload(importlib.import_module(m))


def test_time_job_without_gpu_is_capability_unavailable(tmp_path, monkeypatch):
    """With torch blocked the timing job fails cleanly (Result, not traceback)."""
    monkeypatch.setitem(sys.modules, "torch", None)
    pd = tmp_path / "p"
    pd.mkdir()
    (pd / "reference.py").write_text("")
    (pd / "baseline.py").write_text("")
    (pd / "a.py").write_text("")
    from kdev.models import FileRef

    res = worker.execute(JobSpec(kind="time", files=[FileRef(str(pd / "a.py"), "A")]), str(tmp_path / "r"))
    assert res.status == "error" and res.error.code in ("internal", "capability_unavailable")
    assert validate_result(res.to_dict()) == []
