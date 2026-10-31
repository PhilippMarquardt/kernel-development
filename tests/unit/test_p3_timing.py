"""P3: command registration and JobSpec building for time/ab/calib (no GPU; ctx.submit is captured)."""
from __future__ import annotations

from argparse import Namespace

import pytest

from kdev.contract import KdevError, Result
from kdev.models import JobSpec
from kdev.registry import Context, build_registry
from kdev.timing import commands as C
from kdev.timing.engine import FakeTimer
from kdev.timing.overhead import JUDGE_CONST_US


def test_timing_commands_registered():
    reg = build_registry()
    assert {"time", "ab", "calib"} <= set(reg.names())
    assert reg.get("ab").gpu is True
    assert "registry.todo" not in (reg.get("time").handler.__doc__ or "") and reg.get("time").handler is C.cmd_time


def test_defaults_are_judge_protocol():
    assert JUDGE_CONST_US == 4.0
    flush = next(a for a in build_registry().get("time").args if a.dest == "flush")
    assert flush.default == "read"
    ov = next(a for a in build_registry().get("time").args if a.dest == "overhead")
    assert ov.default == "judge"


def test_fake_timer_constructible():
    t = FakeTimer(means={"A": 20.0, "B": 19.0})
    assert t.labels() == ["A", "B"]


def test_judge_fixture_present(judge_json):
    recs = (judge_json.get("result") or judge_json)["records"]
    assert any(r.get("event") == "benchmark" for r in recs)


class _Ctx(Context):
    def __init__(self):
        super().__init__(agent="t", problem="muse_x")
        self.got = []

    def submit(self, spec, *, detach=False, direct=False, timeout_s=None):
        self.got.append((spec, detach, direct, timeout_s))
        return Result(tool=spec.kind, verdict="queued")


def _ns(**kw):
    base = dict(reps=40, seeds=None, flush="read", flush_mb=None, eager=False, overhead="judge",
                allow_jit=False, detach=False, direct=False, timeout=1800, priority=0, run_name="",
                shapes=None, order="abba", set=None, base="baseline")
    base.update(kw)
    return Namespace(**base)


def test_cmd_time_builds_judge_spec(tmp_path):
    f = tmp_path / "sub.py"
    f.write_text("x=1\n")
    ctx = _Ctx()
    res = C.cmd_time(_ns(file=str(f), flush_mb=512, direct=True, shapes="d1"), ctx)
    assert res.verdict == "queued"
    spec, detach, direct, _ = ctx.got[0]
    assert isinstance(spec, JobSpec) and spec.kind == "time" and spec.lane == "gpu" and direct
    assert spec.files[0].label == "A" and spec.files[0].md5 and spec.problem == "muse_x"
    assert spec.params["flush"] == "read" and spec.params["flush_bytes"] == 512 << 20
    assert spec.params["base"] == "baseline" and spec.params["graph"] is True and spec.params["shapes"] == "d1"
    assert spec.requires_prebuilt is True and spec.job_id.startswith("j")
    JobSpec.from_dict(spec.to_dict())


def test_cmd_time_errors(tmp_path):
    with pytest.raises(KdevError):
        C.cmd_time(_ns(file=str(tmp_path / "missing.py")), _Ctx())
    f = tmp_path / "s.py"
    f.write_text("")
    with pytest.raises(KdevError):
        C.cmd_time(_ns(file=str(f), base=str(tmp_path / "nobase.py")), _Ctx())
    with pytest.raises(KdevError):
        C.cmd_time(_ns(file=str(f), reps=1), _Ctx())


def test_cmd_ab_labels_knobs_and_optouts(tmp_path):
    a, b, c = (tmp_path / f"{n}.py" for n in "abc")
    for p in (a, b, c):
        p.write_text("")
    ctx = _Ctx()
    C.cmd_ab(_ns(files=[str(a), str(b), str(c)], set=["B:TILE=128", "C:FAST=true"], eager=True,
                 flush="write", allow_jit=True, order="rotate"), ctx)
    spec = ctx.got[0][0]
    assert [f.label for f in spec.files] == ["A", "B", "C"]
    assert spec.files[1].knobs == {"TILE": 128} and spec.files[2].knobs == {"FAST": True}
    assert spec.params["base"] == "none" and spec.params["graph"] is False and spec.params["order"] == "rotate"
    assert spec.params["allow_jit"] is True and spec.requires_prebuilt is False
    with pytest.raises(KdevError):
        C.cmd_ab(_ns(files=[str(a)]), _Ctx())
    with pytest.raises(KdevError):
        C.cmd_ab(_ns(files=[str(a), str(b)], set=["Z:K=1"]), _Ctx())


def test_parse_pair():
    assert C.parse_pair("/root/a.py:/root/j.json") == ("/root/a.py", "/root/j.json")
    assert C.parse_pair("a.py:-") == ("a.py", None)
    assert C.parse_pair("a.py") == ("a.py", None)
    assert C.parse_pair("C:\\x\\a.py:C:\\x\\j.json") == ("C:\\x\\a.py", "C:\\x\\j.json")


def test_cmd_calib(tmp_path, fixtures_dir):
    a = tmp_path / "a.py"
    a.write_text("")
    j = fixtures_dir / "judge" / "judge_moe.json"
    ctx = _Ctx()
    C.cmd_calib(_ns(pairs=[f"{a}:{j}", f"{a}:-"]), ctx)
    spec = ctx.got[0][0]
    assert spec.kind == "calib" and spec.files[0].judge_json.endswith("judge_moe.json")
    assert spec.files[1].judge_json is None and spec.params["base"] == "baseline"
    with pytest.raises(KdevError):
        C.cmd_calib(_ns(pairs=[f"{a}:-"]), _Ctx())
    with pytest.raises(KdevError):
        C.cmd_calib(_ns(pairs=[f"{a}:{tmp_path / 'nope.json'}"]), _Ctx())
