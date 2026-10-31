"""P8: MCP server generated from the registry (PLAN 5). No GPU, no network; the SDK tests use the in-process
memory transport of the official `mcp` SDK (skipped if it is not installed)."""
from __future__ import annotations

import json
from argparse import Namespace

import pytest

from kdev.contract import Effects, ErrorCode, KdevError, Result
from kdev.mcp import schema as S
from kdev.mcp.server import RECOVERY, Dispatcher, build_tools
from kdev.registry import Arg, Context, Registry, build_registry


# ---------------------------------------------------------------------------------------------- fixtures

def _fake_registry(calls: list) -> Registry:
    reg = Registry()

    def echo(name):
        def h(args: Namespace, ctx: Context) -> Result:
            calls.append((name, dict(vars(args)), ctx.via))
            return Result(tool=name, verdict="ok", summary={"args": {k: v for k, v in vars(args).items()}})
        return h

    def ab(args: Namespace, ctx: Context) -> Result:
        calls.append(("ab", dict(vars(args)), ctx.via))
        return Result(tool="ab", verdict="queued", job_id="j1", run_id="r1", summary={"state": "queued"})

    state = {"n": 0}

    def job_wait(args: Namespace, ctx: Context) -> Result:
        state["n"] += 1
        if state["n"] < 2:
            raise KdevError(ErrorCode.TIMEOUT, "still running")
        return Result(tool="ab", verdict="faster", job_id=args.job_id, run_id="r1", summary={"state": "done"})

    def show(args: Namespace, ctx: Context) -> Result:
        return Result(tool="show", verdict="ok", run_id=args.run_id,
                      summary={"section": args.section, "offset": args.offset, "limit": args.limit})

    def big(args: Namespace, ctx: Context) -> Result:
        return Result(tool="big", run_id="rbig", shapes=[{"shape_id": f"s{i}", "pad": "x" * 200}
                                                        for i in range(500)])

    def boom(args: Namespace, ctx: Context) -> Result:
        raise NotImplementedError("x.y (P9)")

    def needs_build(args: Namespace, ctx: Context) -> Result:
        raise KdevError(ErrorCode.NEEDS_BUILD, "no prebuilt .so")

    reg.command("version", "print the version", echo("version"), package="T")
    reg.command("ab", "interleaved A/B timing", ab, gpu=True, effects=Effects(gpu="queued", writes=("run_dir",)),
                args=[Arg(("files",), "two or more files", type="path", nargs="+"),
                      Arg(("--reps",), "reps", type="int", default=40),
                      Arg(("--order",), "scheme", choices=("abba", "rotate"), default="abba"),
                      Arg(("--set",), "LABEL:K=V", action="append"),
                      Arg(("--eager",), "eager", action="store_true"),
                      Arg(("--defines",), "extra -D flags", default="")])
    reg.command("job wait", "wait for a job", job_wait, args=[Arg(("job_id",), "job"),
                                                              Arg(("--timeout",), "s", type="float", default=None)])
    reg.command("job status", "status", echo("job status"), args=[Arg(("job_id",), "job")])
    reg.command("show", "paged read", show, args=[Arg(("run_id",), "run"), Arg(("--section",), "s", default="summary"),
                                                  Arg(("--offset",), "o", type="int", default=0),
                                                  Arg(("--limit",), "l", type="int", default=60)])
    reg.command("ledger query", "query", echo("ledger query"),
                args=[Arg(("--tool",), "t"), Arg(("--sql",), "raw sql", mcp=False)])
    reg.command("big", "big output", big)
    reg.command("stub", "a stub", boom)
    reg.command("needs", "needs a build", needs_build)
    reg.command("judge pair", "pair check", echo("judge pair"), mcp=False,
                effects=Effects(network="okc_check", quota="explicit_flag"),
                args=[Arg(("candidate",), "f"), Arg(("--allow-check",), "x", action="store_true", mcp=False)])
    reg.command("sneaky", "registry forgot mcp=False", echo("sneaky"), effects=Effects(quota="explicit_flag"))
    reg.command("lock run", "raw", echo("lock run"), args=[Arg(("argv",), "cmd", nargs="...")])
    reg.command("job submit", "spec", echo("job submit"), args=[Arg(("spec",), "file")])
    return reg


@pytest.fixture
def fake():
    calls: list = []
    return Dispatcher(_fake_registry(calls), profile="full", sleep=lambda s: None), calls


# ---------------------------------------------------------------------------------------------- schemas

@pytest.mark.parametrize("profile", ["compact", "full"])
def test_real_registry_schemas_valid(profile):
    jsonschema = pytest.importorskip("jsonschema")
    tools = build_tools(build_registry(), profile)
    names = [t.name for t in tools]
    assert len(names) == len(set(names))
    for t in tools:
        jsonschema.Draft202012Validator.check_schema(t.input_schema)
        assert t.input_schema["type"] == "object" and t.input_schema["additionalProperties"] is False
        assert t.description and len(t.description) < 700, t.name


def test_full_profile_is_one_tool_per_exposed_command():
    reg = build_registry()
    tools = {t.name: t for t in build_tools(reg, "full")}
    exposed = [c for c in reg.all() if S.command_exposed(c)]
    assert set(tools) == {S.tool_name(c) for c in exposed}
    for c in exposed:
        t = tools[S.tool_name(c)]
        assert t.commands == {"": c.name}
        assert ("wait_s" in t.input_schema["properties"]) == c.gpu
        assert S.effects_tag(c) in t.description


def test_compact_profile_small_and_grouped():
    tools = build_tools(build_registry(), "compact")
    names = {t.name for t in tools}
    assert len(tools) <= 14
    assert {"kdev_doctor", "kdev_time", "kdev_ab", "kdev_show", "kdev_ledger", "kdev_idea", "kdev_job",
            "kdev_run"} <= names
    total = sum(len(json.dumps(t.input_schema)) + len(t.description) for t in tools)
    assert total < 16000          # ~4k tokens of permanent context


@pytest.mark.parametrize("profile", ["compact", "full"])
def test_hidden_commands_and_args_never_exposed(profile):
    reg = build_registry()
    tools = build_tools(reg, profile)
    blob = json.dumps([{"n": t.name, "d": t.description, "s": t.input_schema, "c": t.commands} for t in tools])
    for t in tools:
        assert "submit" not in t.name
        for cname in t.commands.values():
            assert cname not in S.HIDDEN_COMMANDS
            assert reg.get(cname).mcp
            assert reg.get(cname).effects.quota == "never"
        props = t.input_schema["properties"]
        for hidden in ("allow_check", "sql", "home", "json", "direct", "detach"):
            assert hidden not in props, (t.name, hidden)
    assert "judge pair" not in blob and "lock run" not in blob and "job submit" not in blob


def test_arg_schema_mapping():
    assert S.arg_schema(Arg(("--x",), "h", type="int", default=3)) == {"type": "integer", "default": 3,
                                                                       "description": "h"}
    assert S.arg_schema(Arg(("--f",), "flag", action="store_true"))["type"] == "boolean"
    s = S.arg_schema(Arg(("files",), "", type="path", nargs="+"))
    assert s == {"type": "array", "items": {"type": "string"}, "minItems": 1}
    s = S.arg_schema(Arg(("--ab",), "", nargs=2))
    assert s["minItems"] == s["maxItems"] == 2
    s = S.arg_schema(Arg(("--o",), "o", choices=("a", "b"), default="a"))
    assert s["enum"] == ["a", "b"] and s["default"] == "a"
    long = "x" * 80
    assert "description" not in S.arg_schema(Arg(("--y",), long), compact=True)
    assert S.arg_schema(Arg(("--y",), long), compact=False)["description"] == long
    assert S.is_required(Arg(("pos",), "")) and not S.is_required(Arg(("pos",), "", nargs="?"))


# ---------------------------------------------------------------------------------------------- dispatch

def test_typed_call_builds_cli_argv(fake):
    d, calls = fake
    r = d.call("kdev_ab", {"files": ["a.py", "-weird.py"], "reps": 7, "set": ["A:K=1", "B:K=2"], "eager": True,
                           "defines": "-DX=1 -DY", "agent": "w1"})
    assert r["status"] == "ok" and r["job_id"] == "j1"
    name, ns, via = calls[-1]
    assert via == "mcp"
    assert ns["files"] == ["a.py", "-weird.py"] and ns["reps"] == 7 and ns["set"] == ["A:K=1", "B:K=2"]
    assert ns["eager"] is True and ns["defines"] == "-DX=1 -DY" and ns["detach"] is True   # always detached
    assert ns["agent"] == "w1" and ns["order"] == "abba"


def test_result_identical_to_cli_json(fake):
    d, _ = fake
    cli = pytest.importorskip("kdev.cli")
    reg = d.reg
    via_mcp = d.call("kdev_ledger_query", {"tool": "ab"})
    via_cli = json.loads(cli.run(["ledger", "query", "--tool", "ab", "--json"], reg).to_json())
    via_cli["summary"]["args"].pop("json")
    via_mcp["summary"]["args"].pop("json")
    via_cli["summary"]["args"].pop("home", None)
    via_mcp["summary"]["args"].pop("home", None)
    via_cli["summary"]["args"].pop("quiet", None)
    via_mcp["summary"]["args"].pop("quiet", None)
    assert via_mcp == via_cli


def test_validation_errors_are_results(fake):
    d, _ = fake
    r = d.call("kdev_ab", {"files": ["a"], "order": "zigzag"})
    assert r["status"] == "error" and r["error"]["code"] == "invalid_request" and r["exit_code"] == 2
    assert "zigzag" in r["error"]["message"]
    r = d.call("kdev_ab", {})
    assert r["error"]["code"] == "invalid_request" and "files" in r["error"]["message"]
    r = d.call("kdev_ledger_query", {"sql": "DELETE FROM runs"})
    assert r["error"]["code"] == "invalid_request" and "sql" in r["error"]["message"]
    r = d.call("kdev_nope", {})
    assert r["error"]["code"] == "invalid_request"


def test_error_codes_get_recovery_hints(fake):
    d, _ = fake
    r = d.call("kdev_needs", {})
    assert r["error"]["code"] == "needs_build" and r["exit_code"] == 4
    assert r["error"]["recovery"] == RECOVERY["needs_build"]
    r = d.call("kdev_stub", {})
    assert r["error"]["code"] == "not_implemented" and r["error"]["recovery"]
    assert set(RECOVERY) == {c.value for c in ErrorCode}


def test_hidden_and_quota_commands_unreachable(fake):
    d, calls = fake
    names = set(d.tools)
    assert not names & {"kdev_judge_pair", "kdev_sneaky", "kdev_lock_run", "kdev_job_submit"}
    d2 = Dispatcher(d.reg, profile="compact")
    for cmd, argv in (("judge pair", ["a.py", "--allow-check"]), ("sneaky", []), ("lock run", ["okc", "x"]),
                      ("job submit", ["s.json"])):
        r = d2.call("kdev_run", {"command": cmd, "argv": argv})
        assert r["status"] == "error", cmd
        assert r["error"]["code"] in ("quota_forbidden", "invalid_request"), cmd
    assert d2.call("kdev_run", {"command": "judge pair", "argv": []})["error"]["code"] == "quota_forbidden"
    assert not [c for c in calls if c[0] in ("judge pair", "sneaky", "lock run", "job submit")]


def test_generic_run_rejects_hidden_flags(fake):
    d, calls = fake
    d2 = Dispatcher(d.reg, profile="compact")
    r = d2.call("kdev_run", {"command": "ledger query", "argv": ["--sql", "select 1"]})
    assert r["error"]["code"] == "invalid_request"
    r = d2.call("kdev_run", {"command": "version", "argv": ["--home", "/elsewhere"]})
    assert r["error"]["code"] == "invalid_request" and "--home" in r["error"]["message"]
    r = d2.call("kdev_run", {"command": "ab", "argv": ["a", "b", "--direct"]})
    assert r["error"]["code"] == "invalid_request"
    r = d2.call("kdev_run", {"command": "version"})
    assert r["status"] == "ok"
    listing = d2.call("kdev_run", {"command": "help"})
    assert "version" in listing["summary"]["commands"] and "judge pair" not in listing["summary"]["commands"]


def test_gpu_wait_polls_job_and_reports_progress(fake):
    d, _ = fake
    seen = []
    r = d.call("kdev_ab", {"files": ["a", "b"], "wait_s": 30}, progress=lambda done, tot, msg: seen.append(msg))
    assert r["status"] == "ok" and r["verdict"] == "faster" and r["summary"]["state"] == "done"
    assert seen and "j1" in seen[-1]


def test_large_output_is_paged_not_inlined(fake):
    d, _ = fake
    r = d.call("kdev_big", {})
    assert r["shapes"] == []
    assert any("kdev_show run_id=rbig" in w for w in r["warnings"])
    assert len(json.dumps(r)) <= d.inline_max_bytes


def test_resource_uri_maps_to_show(fake):
    d, _ = fake
    r = d.read_resource("kdev://runs/20261031-ab-x/raw?offset=60&limit=30")
    assert r["summary"] == {"section": "raw", "offset": 60, "limit": 30} and r["run_id"] == "20261031-ab-x"
    assert d.read_resource("http://evil/x")["error"]["code"] == "invalid_request"


def test_compact_group_actions(fake):
    reg = build_registry()
    d = Dispatcher(reg, profile="compact")
    r = d.call("kdev_job", {"action": "bogus"})
    assert r["error"]["code"] == "invalid_request"
    r = d.call("kdev_ledger", {"action": "query", "sql": "select 1"})
    assert r["error"]["code"] == "invalid_request"


def test_mcp_tools_command_lists_profile():
    from kdev.mcp.commands import cmd_mcp_tools
    r = cmd_mcp_tools(Namespace(profile="compact"), Context())
    assert r.status == "ok" and r.summary["tools"] and r.summary["total_bytes"] > 0


# ---------------------------------------------------------------------------------------------- SDK (in-process)

def test_in_process_mcp_client(fake):
    pytest.importorskip("mcp")
    import anyio
    from mcp.shared.memory import create_connected_server_and_client_session

    from kdev.mcp.server import build_server

    d, _ = fake
    server = build_server(dispatcher=d)
    progress_msgs: list = []

    async def go():
        async with create_connected_server_and_client_session(server) as client:
            init_tools = await client.list_tools()
            names = {t.name for t in init_tools.tools}
            assert "kdev_ab" in names and "kdev_judge_pair" not in names
            ab = next(t for t in init_tools.tools if t.name == "kdev_ab")
            assert ab.inputSchema["properties"]["wait_s"]["maximum"] == 120
            res = await client.call_tool("kdev_version", {})
            assert not res.isError
            body = json.loads(res.content[0].text)
            assert body["schema"] == "kdev.result/1" and res.structuredContent == body
            res = await client.call_tool("kdev_needs", {})
            assert res.isError and res.structuredContent["error"]["code"] == "needs_build"
            res = await client.call_tool("kdev_ab", {"files": ["a", "b"], "bogus": 1})
            assert res.isError                       # schema validation (additionalProperties false)

            async def on_progress(p, total, msg):
                progress_msgs.append(msg)

            res = await client.call_tool("kdev_ab", {"files": ["a", "b"], "wait_s": 5},
                                         progress_callback=on_progress)
            assert res.structuredContent["verdict"] == "faster"
            tpl = await client.list_resource_templates()
            assert tpl.resourceTemplates[0].uriTemplate == "kdev://runs/{run_id}/{section}"
            rr = await client.read_resource("kdev://runs/r1/log")
            assert json.loads(rr.contents[0].text)["summary"]["section"] == "log"

    anyio.run(go)
    assert progress_msgs


def test_build_server_without_sdk(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "mcp", None)
    monkeypatch.setitem(sys.modules, "mcp.types", None)
    from kdev.mcp.server import build_server
    with pytest.raises(KdevError) as ei:
        build_server()
    assert ei.value.code == ErrorCode.CAPABILITY_UNAVAILABLE and "kdev[mcp]" in ei.value.recovery
