"""P3: timing engine, stats, schedule, calib. Placeholder; P3 replaces with FakeTimer-driven tests (ABBA
order, overhead subtraction, paired ratios, bootstrap CI, verdicts) and judge-JSON parsing."""
from __future__ import annotations

from kdev.registry import build_registry
from kdev.timing.engine import FakeTimer
from kdev.timing.overhead import JUDGE_CONST_US


def test_timing_commands_registered():
    names = set(build_registry().names())
    assert {"time", "ab", "calib"} <= names
    assert build_registry().get("ab").gpu is True


def test_defaults_are_judge_protocol():
    assert JUDGE_CONST_US == 4.0
    flush = next(a for a in build_registry().get("time").args if a.dest == "flush")
    assert flush.default == "read"


def test_fake_timer_constructible():
    t = FakeTimer(means={"A": 20.0, "B": 19.0})
    assert t.labels() == ["A", "B"]


def test_judge_fixture_present(judge_json):
    recs = (judge_json.get("result") or judge_json)["records"]
    assert any(r.get("event") == "benchmark" for r in recs)
