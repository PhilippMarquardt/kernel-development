"""P1: core, registry, CLI dispatch, output contract. Phase-0 tests cover the plumbing that already works."""
from __future__ import annotations

import json

from kdev import __version__
from kdev.cli import build_parser, main, run
from kdev.contract import EXIT_CODES, ErrorCode, Result, validate_result
from kdev.models import FileRef, JobSpec
from kdev.registry import build_registry

EXPECTED_COMMANDS = {
    "version", "doctor", "show", "runs list", "build", "check", "sanitize", "comply", "time", "ab", "calib",
    "sweep", "sass", "sass diff", "regs", "timeline", "floor", "floor calibrate", "profile", "dispatch",
    "serve", "job submit", "job status", "job wait", "job cancel", "job list", "job logs", "queue status",
    "queue offenders", "lock run", "ledger query", "ledger tried", "ledger best", "ledger import",
    "ledger export", "idea add", "idea list", "idea show", "idea claim", "idea release", "idea evidence",
    "idea resolve", "note", "judge prepare", "judge record", "judge ledger", "judge pair", "okc status",
    "okc leaderboard", "dashboard build", "dashboard watch", "sync", "remote", "remote setup", "mcp",
}


def test_registry_has_all_planned_commands():
    reg = build_registry()
    assert EXPECTED_COMMANDS <= set(reg.names())


def test_every_command_parser_builds():
    for cmd in build_registry().all():
        build_parser(cmd)  # raises on conflicting flags


def test_longest_prefix_resolution():
    reg = build_registry()
    cmd, rest = reg.resolve(["sass", "diff", "a.so", "b.so"])
    assert cmd.name == "sass diff" and rest == ["a.so", "b.so"]
    cmd, rest = reg.resolve(["sass", "x.so", "--loops"])
    assert cmd.name == "sass" and rest == ["x.so", "--loops"]


def test_version_command(capsys):
    assert main(["version"]) == 0
    assert __version__ in capsys.readouterr().out


def test_stub_returns_not_implemented_json(capsys):
    code = main(["ab", "a.py", "b.py", "--json"])
    assert code == EXIT_CODES[ErrorCode.NOT_IMPLEMENTED]
    d = json.loads(capsys.readouterr().out)
    assert d["error"]["code"] == "not_implemented"
    assert validate_result(d) == []


def test_unknown_command_is_invalid_request():
    r = None
    try:
        r = run(["frobnicate"])
    except Exception as e:  # KdevError
        assert getattr(e, "code", None) == ErrorCode.INVALID_REQUEST
    assert r is None


def test_result_roundtrip():
    r = Result(tool="time", verdict="faster", summary={"x": 1})
    d = r.to_dict()
    assert validate_result(d) == []
    assert Result.from_dict(d).to_dict() == d


def test_jobspec_roundtrip():
    s = JobSpec(kind="ab", files=[FileRef("/a.py", "A"), FileRef("/b.py", "B")], params={"reps": 40})
    assert JobSpec.from_dict(json.loads(json.dumps(s.to_dict()))) == s
    assert s.resolved_lane() == "gpu"
