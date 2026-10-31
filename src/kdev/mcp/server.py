"""MCP stdio server generated from the registry (docs/PLAN.md section 5, docs/MCP.md). Owner: P8.

Design (REA style):
- Tools are generated from ``kdev.registry`` - the CLI contract. Each tool call turns its arguments into an
  argv for that command's own argparse parser (``kdev.cli.build_parser``), so validation, types, choices and
  defaults are exactly the CLI's, then calls the same handler with ``Context(via='mcp')``. The result is
  ``Result.to_dict()`` - byte-identical to ``kdev <cmd> --json`` - unless it exceeds the inline budget, in
  which case bulky members (``shapes``, ``dispatch``) are dropped with an explicit warning that names the
  ``kdev_show`` call that pages them (run ids are the evidence ids).
- Profiles: ``compact`` (default, ~12 grouped tools, short descriptions) and ``full`` (one tool per command).
- GPU tools always enqueue detached and return ``{job_id, run_id}`` unless ``wait_s`` (<= 120) is given, in
  which case the server polls ``job wait`` in chunks and sends progress notifications.
- Errors are Results with a stable ``error.code``; an empty ``recovery`` is filled from :data:`RECOVERY`.
- Exposure rules live in :mod:`kdev.mcp.schema` (judge pair / --allow-check, lock run, serve, remote, sync,
  job submit, ledger --sql are never reachable; nothing here can submit to a judge).
- Resources: ``kdev://runs/{run_id}/{section}?offset=N&limit=M`` = ``kdev show``.

Only :func:`build_server` / :func:`main` import the ``mcp`` SDK (the low-level ``mcp.server.lowlevel.Server``
is used rather than FastMCP because the input schemas are generated, not derived from Python signatures).
Everything else (``build_tools``, ``Dispatcher``) is SDK-free and unit-testable.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import io
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import parse_qs, unquote, urlparse

from kdev import __version__
from kdev.contract import ErrorCode, KdevError, Result, not_implemented
from kdev.mcp.schema import (MAX_WAIT_S, all_args, arg_exposed, arg_schema, command_exposed, command_schema,
                             effects_tag, exposed_args, is_array, is_flag, is_required, tool_name, wait_schema)
from kdev.registry import Command, Context, Registry, build_registry

COMPACT_TOOLS = ("doctor", "build", "check", "time", "ab", "sass", "timeline", "ledger", "idea", "job",
                 "board", "doc", "show", "run")

# compact tool -> (description, {action: command}); action "" = single-command tool.
COMPACT_GROUPS: dict[str, tuple[str, dict[str, str]]] = {
    "doctor": ("", {"": "doctor"}),
    "build": ("", {"": "build"}),
    "check": ("", {"": "check"}),
    "time": ("", {"": "time"}),
    "ab": ("", {"": "ab"}),
    "sass": ("SASS/regs analysis of a build", {"analyze": "sass", "diff": "sass diff", "regs": "regs"}),
    "timeline": ("", {"": "timeline"}),
    "ledger": ("experiment ledger (read-mostly)", {"query": "ledger query", "tried": "ledger tried",
                                                   "best": "ledger best", "judged": "judge ledger",
                                                   "import": "ledger import", "export": "ledger export"}),
    "idea": ("idea board", {"list": "idea list", "show": "idea show", "add": "idea add", "claim": "idea claim",
                            "release": "idea release", "evidence": "idea evidence", "resolve": "idea resolve"}),
    "job": ("GPU job queue", {"status": "job status", "wait": "job wait", "list": "job list",
                              "logs": "job logs", "cancel": "job cancel", "queue": "queue status",
                              "offenders": "queue offenders"}),
    "board": ("agent message board (P9)", {"post": "board post", "read": "board read", "inbox": "board inbox",
                                            "ack": "board ack", "search": "board search", "thread": "board thread",
                                            "claims": "board claims", "release": "board release"}),
    "doc": ("shared docs: notes/lessons/tracker (append-only, P9)",
            {"append": "doc append", "show": "doc show", "toc": "doc toc", "list": "doc list",
             "search": "doc search", "checklist": "doc checklist"}),
    "show": ("", {"": "show"}),
}

INLINE_MAX_BYTES = 24_000
WAIT_CHUNK_S = 10.0
FINAL_JOB_STATES = ("done", "failed", "cancelled", "timeout")

RECOVERY: dict[str, str] = {
    "check_failed": "correctness failed: read kdev_show run_id=<run> section=log, fix, re-run kdev_check",
    "invalid_request": "fix the named field (tool schema lists valid params; kdev_run command=help lists commands)",
    "quota_forbidden": "not available over MCP / would use a judge quota; ask the coordinator to run it via the CLI",
    "needs_build": "run kdev_build on the file(s) first, then retry",
    "toolchain_mismatch": "rebuild with the pinned toolchain (kdev_doctor shows the pin), then retry",
    "gpu_busy_timeout": "GPU lock/foreign contexts did not clear: kdev_job action=queue, retry later",
    "capability_unavailable": "backend missing: use the fallback named in the message (kdev_doctor lists backends)",
    "build_failed": "read the compiler errors: kdev_show run_id=<run> section=log",
    "timeout": "partial samples kept (kdev_show run_id=<run>); retry with a larger timeout or fewer shapes",
    "not_found": "check the id: kdev_run command='runs list', kdev_job action=list, kdev_idea action=list",
    "conflict": "owned by someone else or changed: re-read (kdev_idea action=show) and retry",
    "cancelled": "the job was cancelled; resubmit if still needed",
    "not_exclusive": "timing not exclusive (comparable=0): re-run when kdev_job action=queue shows no offenders",
    "not_implemented": "this command is still a stub; use another tool or the CLI",
    "internal": "kdev bug (traceback on the server's stderr); report it with the arguments used",
}

INSTRUCTIONS = (
    "kdev: judge-faithful GPU kernel tooling. Every tool returns the kdev.result/1 envelope (same as "
    "`kdev <cmd> --json`): status ok|fail|error, verdict, summary, files, warnings, next, error{code,recovery}. "
    "run_id is the evidence id: page large outputs with kdev_show (section, offset, limit; follow next_offset) "
    "or resource kdev://runs/{run_id}/{section}. GPU tools enqueue and return job_id; pass wait_s<=120 or poll "
    "kdev_job action=wait. Common params: problem, agent (attribution); GPU tools also timeout (job s), "
    "priority, run_name (run-dir slug), shapes (shape-id filter). Ledger first: kdev_ledger action=tried "
    "before timing anything. Never claim a kernel is 'at the limit' without a sass run and a sweep run as "
    "evidence. This server cannot submit or check on the judge."
)

ProgressFn = Callable[[float, float | None, str], None]


@dataclass
class ToolSpec:
    """One MCP tool: name, description, JSON input schema, and the command(s) it dispatches to."""

    name: str
    description: str
    input_schema: dict[str, Any]
    commands: dict[str, str] = field(default_factory=dict)  # action -> command name; "" = single command
    generic: bool = False                                   # kdev_run: command + argv
    read_only: bool = False
    open_world: bool = False


# ---------------------------------------------------------------------------------------------- tool table


def _single_tool(cmd: Command, name: str, compact: bool) -> ToolSpec:
    desc = f"{cmd.help} {effects_tag(cmd)}"
    if cmd.gpu:
        desc += " GPU job: returns job_id; wait_s<=120 to wait."
    return ToolSpec(name, desc, command_schema(cmd, compact), {"": cmd.name},
                    read_only=_read_only(cmd), open_world=cmd.effects.network != "none")


def _read_only(cmd: Command) -> bool:
    return not cmd.effects.writes and cmd.effects.gpu == "none"


def _merge_prop(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    if a.get("type") != b.get("type") or a.get("items", {}).get("type") != b.get("items", {}).get("type"):
        return {}                                            # conflicting types: validated server-side
    out = dict(a)
    for key in ("enum",):
        if key in a or key in b:
            out.pop(key, None)                               # enum differs per action: validated server-side
    if "items" in out:
        out["items"] = {k: v for k, v in out["items"].items() if k != "enum"}
    out.pop("default", None)
    out.pop("minItems", None)
    out.pop("maxItems", None)
    return out


def _short(text: str, n: int = 48) -> str:
    """First clause of a help string, at most n chars (compact tool descriptions)."""
    t = text.split(";")[0].split(" (")[0].strip()
    return t if len(t) <= n else t[: n - 1].rstrip() + "~"


def _group_tool(name: str, title: str, actions: dict[str, Command], compact: bool) -> ToolSpec:
    props: dict[str, Any] = {"action": {"type": "string", "enum": list(actions)}}
    parts = []
    for act, cmd in actions.items():
        sig = []
        for a in exposed_args(cmd):
            s = arg_schema(a, compact)
            s.pop("description", None)
            if a.dest in props and a.dest != "action":
                props[a.dest] = _merge_prop(props[a.dest], s)
            else:
                props[a.dest] = s
            if a.positional or is_required(a):
                sig.append(a.dest + ("!" if is_required(a) else ""))
        parts.append(f"{act}({','.join(sig)}): {_short(cmd.help)}")
    writes = sorted({w for c in actions.values() for w in c.effects.writes})
    eff = f"[gpu=none{' writes=' + ','.join(writes) if writes else ''} quota=never]"
    desc = f"{title} {eff}. actions: " + "; ".join(parts) + ". (! = required; other params optional)"
    schema = {"type": "object", "properties": props, "required": ["action"], "additionalProperties": False}
    return ToolSpec(name, desc, schema, {a: c.name for a, c in actions.items()},
                    read_only=all(_read_only(c) for c in actions.values()),
                    open_world=any(c.effects.network != "none" for c in actions.values()))


def _run_tool() -> ToolSpec:
    schema = {"type": "object", "properties": {
        "command": {"type": "string", "description": "e.g. 'floor', 'runs list'; 'help' lists commands"},
        "argv": {"type": "array", "items": {"type": "string"}, "description": "CLI args after the command"},
        "wait_s": wait_schema()}, "required": ["command"], "additionalProperties": False}
    return ToolSpec("kdev_run", "any other exposed kdev command, CLI-style argv; same JSON result as --json "
                    "[effects per command]", schema, generic=True)


def build_tools(reg: Registry, profile: str = "compact") -> list[ToolSpec]:
    """The tool table for ``profile`` (compact|full). Pure: no SDK import."""
    if profile not in ("compact", "full"):
        raise KdevError(ErrorCode.INVALID_REQUEST, f"unknown MCP profile {profile!r}",
                        recovery="use --profile compact or --profile full")
    names = set(reg.names())
    if profile == "full":
        return [_single_tool(c, tool_name(c), compact=False) for c in reg.all() if command_exposed(c)]
    tools: list[ToolSpec] = []
    for short, (title, actions) in COMPACT_GROUPS.items():
        cmds = {a: reg.get(c) for a, c in actions.items() if c in names and command_exposed(reg.get(c))}
        if not cmds:
            continue
        if list(cmds) == [""]:
            tools.append(_single_tool(cmds[""], "kdev_" + short, compact=True))
        else:
            cmds.pop("", None)
            tools.append(_group_tool("kdev_" + short, title, cmds, compact=True))
    tools.append(_run_tool())
    return tools


# ---------------------------------------------------------------------------------------------- dispatcher


def _err(tool: str, code: ErrorCode, message: str, recovery: str = "", **details: Any) -> Result:
    return Result.from_error(tool, KdevError(code, message, details=details or None, recovery=recovery))


def _to_argv(cmd: Command, values: dict[str, Any]) -> list[str]:
    """Typed tool arguments -> argv for ``cmd``'s CLI parser (options first, positionals last)."""
    opts: list[str] = []
    pos: list[str] = []
    for a in exposed_args(cmd):
        if a.dest not in values or values[a.dest] is None:
            continue
        v = values[a.dest]
        if a.positional:
            pos.extend(str(x) for x in (v if isinstance(v, list) else [v]))
            continue
        flag = next(f for f in a.flags if f.startswith("--"))
        if is_flag(a):
            if v is True or str(v).lower() in ("1", "true", "yes"):
                opts.append(flag)
        elif a.action == "append":
            opts.extend(f"{flag}={x}" for x in (v if isinstance(v, list) else [v]))
        elif is_array(a):
            items = [str(x) for x in (v if isinstance(v, list) else [v])]
            opts.append(flag)
            opts.extend(items)
        else:
            opts.append(f"{flag}={v}")
    if any(p.startswith("-") for p in pos):
        return opts + ["--"] + pos
    return opts + pos


_ARGPARSE_TYPES = {"str": str, "path": str, "int": int, "float": float}


def _fallback_parser(cmd: Command) -> argparse.ArgumentParser:
    """Mirror of kdev.cli.build_parser, used only if the CLI module cannot be imported."""
    p = argparse.ArgumentParser(prog=f"kdev {cmd.name}", description=cmd.help)
    for a in all_args(cmd):
        kw: dict[str, Any] = {"help": a.help}
        if a.action == "store_true":
            kw["action"] = "store_true"
        else:
            if a.action == "append":
                kw["action"] = "append"
            kw["type"] = _ARGPARSE_TYPES.get(a.type, str)
            if a.choices is not None:
                kw["choices"] = list(a.choices)
            if a.nargs == "...":
                kw["nargs"] = argparse.REMAINDER
            elif a.nargs is not None:
                kw["nargs"] = a.nargs
        if not a.positional:
            kw["default"] = bool(a.default) if a.action == "store_true" else a.default
            kw["dest"] = a.dest
            if a.required:
                kw["required"] = True
        elif a.nargs in ("?", "*") and a.default is not None:
            kw["default"] = a.default
        p.add_argument(*a.flags, **kw)
    return p


def _cli_parser(cmd: Command) -> argparse.ArgumentParser:
    try:
        from kdev.cli import build_parser
    except Exception as e:  # noqa: BLE001 - CLI module broken/missing: same Args, local mirror
        print(f"kdev.mcp: kdev.cli unavailable ({type(e).__name__}: {e}); using the fallback parser",
              file=sys.stderr)
        return _fallback_parser(cmd)
    return build_parser(cmd)


def _parse(cmd: Command, argv: list[str]) -> argparse.Namespace:
    """Parse with the CLI's own parser, minus every arg MCP may not set. argparse errors -> invalid_request."""
    visible = dataclasses.replace(cmd, args=[a for a in cmd.args if arg_exposed(cmd.name, a)])
    parser = _cli_parser(visible)
    buf = io.StringIO()
    try:
        with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(buf):
            ns = parser.parse_args(argv)
    except SystemExit:
        msg = buf.getvalue().strip().splitlines()
        raise KdevError(ErrorCode.INVALID_REQUEST, msg[-1] if msg else f"bad arguments for {cmd.name}",
                        details={"argv": argv}, recovery=RECOVERY["invalid_request"]) from None
    # globals / GPU flags that MCP must not set (e.g. --home, --direct) arrive through the shared parser:
    for a in all_args(cmd):
        if arg_exposed(cmd.name, a) or a.dest in ("json", "detach") or not hasattr(ns, a.dest):
            continue
        default = bool(a.default) if a.action == "store_true" else a.default
        if getattr(ns, a.dest) != default:
            raise KdevError(ErrorCode.QUOTA_FORBIDDEN if a.dest == "allow_check" else ErrorCode.INVALID_REQUEST,
                            f"--{a.dest.replace('_', '-')} is not available over MCP",
                            recovery="drop the flag (use the CLI for it)")
    # hidden command args keep their CLI defaults, so handlers see the same Namespace shape as the CLI's
    for a in cmd.args:
        if not arg_exposed(cmd.name, a) and not hasattr(ns, a.dest):
            setattr(ns, a.dest, [] if a.nargs == "..." else (bool(a.default) if a.action == "store_true"
                                                              else a.default))
    return ns


class Dispatcher:
    """Executes MCP tool calls against the registry. SDK-free; ``call`` returns the Result dict."""

    def __init__(self, reg: Registry | None = None, profile: str = "compact", *,
                 inline_max_bytes: int = INLINE_MAX_BYTES, sleep: Callable[[float], None] = time.sleep) -> None:
        self.reg = reg or build_registry()
        self.profile = profile
        self.tools: dict[str, ToolSpec] = {t.name: t for t in build_tools(self.reg, profile)}
        self.inline_max_bytes = inline_max_bytes
        self.sleep = sleep

    # -- public ------------------------------------------------------------------------------------------
    def call(self, name: str, arguments: dict[str, Any] | None = None,
             progress: ProgressFn | None = None) -> dict[str, Any]:
        arguments = dict(arguments or {})
        try:
            result = self._call(name, arguments, progress)
        except KdevError as e:
            result = Result.from_error(name, e)
        except Exception as e:  # noqa: BLE001 - never a bare traceback to the client
            traceback.print_exc(file=sys.stderr)
            result = _err(name, ErrorCode.INTERNAL, f"{type(e).__name__}: {e}")
        return self._finish(result)

    def read_resource(self, uri: str) -> dict[str, Any]:
        """kdev://runs/{run_id}/{section}?offset=N&limit=M -> the `show` Result dict."""
        u = urlparse(uri)
        parts = [unquote(p) for p in (u.netloc + u.path).split("/") if p]
        if u.scheme != "kdev" or len(parts) < 2 or parts[0] != "runs":
            return self._finish(_err("show", ErrorCode.INVALID_REQUEST, f"bad resource uri {uri!r}",
                                     recovery="use kdev://runs/{run_id}/{section}"))
        q = {k: v[-1] for k, v in parse_qs(u.query).items()}
        args: dict[str, Any] = {"run_id": parts[1], "section": parts[2] if len(parts) > 2 else "summary"}
        for k in ("offset", "limit"):
            if k in q:
                args[k] = q[k]
        return self.call_command("show", args)

    def call_command(self, command: str, values: dict[str, Any], progress: ProgressFn | None = None) -> dict:
        try:
            r = self._typed(command, values, progress)
        except KdevError as e:
            r = Result.from_error(command, e)
        return self._finish(r)

    # -- internals ---------------------------------------------------------------------------------------
    def _call(self, name: str, arguments: dict[str, Any], progress: ProgressFn | None) -> Result:
        tool = self.tools.get(name)
        if tool is None:
            raise KdevError(ErrorCode.INVALID_REQUEST, f"unknown tool {name!r}",
                            recovery="list tools again; profile=" + self.profile)
        if tool.generic:
            return self._generic(arguments, progress)
        if list(tool.commands) == [""]:
            return self._typed(tool.commands[""], arguments, progress)
        action = arguments.pop("action", None)
        if action not in tool.commands:
            raise KdevError(ErrorCode.INVALID_REQUEST, f"{name}: action must be one of {sorted(tool.commands)}",
                            details={"action": action}, recovery="set action")
        return self._typed(tool.commands[action], arguments, progress)

    def _command(self, command: str) -> Command:
        try:
            cmd = self.reg.get(command)
        except KeyError:
            raise KdevError(ErrorCode.INVALID_REQUEST, f"unknown command {command!r}",
                            recovery="kdev_run command=help lists the exposed commands") from None
        if not command_exposed(cmd):
            quota = cmd.effects.quota != "never" or cmd.effects.network == "okc_check"
            raise KdevError(ErrorCode.QUOTA_FORBIDDEN if quota else ErrorCode.INVALID_REQUEST,
                            f"`{command}` is not exposed over MCP", recovery=RECOVERY["quota_forbidden"])
        return cmd

    def _typed(self, command: str, values: dict[str, Any], progress: ProgressFn | None) -> Result:
        cmd = self._command(command)
        allowed = {a.dest for a in exposed_args(cmd)} | ({"wait_s"} if cmd.gpu else set())
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise KdevError(ErrorCode.INVALID_REQUEST, f"{command}: unknown or hidden parameter(s) {unknown}",
                            details={"allowed": sorted(allowed)}, recovery="drop them; see the tool schema")
        missing = [a.dest for a in exposed_args(cmd) if is_required(a) and values.get(a.dest) in (None, [], "")]
        if missing:
            raise KdevError(ErrorCode.INVALID_REQUEST, f"{command}: missing required {missing}",
                            recovery="add the missing parameter(s)")
        wait_s = values.pop("wait_s", 0) or 0
        return self._execute(cmd, _parse(cmd, _to_argv(cmd, values)), float(wait_s), progress)

    def _generic(self, arguments: dict[str, Any], progress: ProgressFn | None) -> Result:
        command = " ".join(str(arguments.get("command", "")).split())
        if command in ("", "help", "list", "--help"):
            cmds = {c.name: c.help for c in self.reg.all() if command_exposed(c)}
            return Result(tool="mcp", verdict="ok", summary={"commands": cmds},
                          lines=[f"{k}: {v}" for k, v in cmds.items()])
        argv = arguments.get("argv") or []
        if not isinstance(argv, list):
            raise KdevError(ErrorCode.INVALID_REQUEST, "argv must be a list of strings", recovery="pass a list")
        cmd = self._command(command)
        ns = _parse(cmd, [str(x) for x in argv])
        return self._execute(cmd, ns, float(arguments.get("wait_s", 0) or 0), progress)

    def _execute(self, cmd: Command, ns: argparse.Namespace, wait_s: float, progress: ProgressFn | None) -> Result:
        if cmd.gpu:
            ns.detach = True   # never block an MCP call on a GPU job; waiting is explicit and bounded
        ctx = Context(json=True, agent=getattr(ns, "agent", None) or os.environ.get("KDEV_AGENT", "") or "mcp",
                      home=None, problem=getattr(ns, "problem", None), quiet=True, via="mcp")
        result = self._invoke(cmd, ns, ctx)
        wait_s = max(0.0, min(float(wait_s), MAX_WAIT_S))
        if cmd.gpu and wait_s > 0 and result.job_id and result.status != "error":
            result = self._wait(result, wait_s, progress)
        return result

    @staticmethod
    def _invoke(cmd: Command, ns: argparse.Namespace, ctx: Context) -> Result:
        try:
            return cmd.handler(ns, ctx)
        except NotImplementedError as e:
            return not_implemented(cmd.name, str(e) or cmd.name)
        except KdevError as e:
            return Result.from_error(cmd.name, e)

    def _wait(self, submitted: Result, wait_s: float, progress: ProgressFn | None) -> Result:
        job_id = submitted.job_id
        t0 = time.monotonic()
        state = "queued"
        while True:
            elapsed = time.monotonic() - t0
            remaining = wait_s - elapsed
            if remaining <= 0:
                break
            try:
                waited = self._typed("job wait", {"job_id": job_id, "timeout": min(WAIT_CHUNK_S, remaining)}, None)
            except KdevError as e:
                waited = Result.from_error("job wait", e)
            state = str(waited.summary.get("state") or waited.verdict or state)
            if progress is not None:
                progress(round(time.monotonic() - t0, 1), wait_s, f"job {job_id}: {state}")
            if state in FINAL_JOB_STATES:
                return waited
            if waited.status == "error" and waited.error and waited.error.code != ErrorCode.TIMEOUT.value:
                break                                     # job wait unavailable: stop polling, keep job id
        submitted.warnings.append(f"job {job_id} still {state} after {wait_s:.0f}s")
        submitted.next.insert(0, f"kdev_job action=wait job_id={job_id}")
        return submitted

    def _finish(self, result: Result) -> dict[str, Any]:
        if result.error is not None and not result.error.recovery:
            result.error.recovery = RECOVERY.get(result.error.code, RECOVERY["internal"])
        d = json.loads(json.dumps(result.to_dict(), default=str))
        size = len(json.dumps(d, separators=(",", ":")))
        if size <= self.inline_max_bytes:
            return d
        run = d.get("run_id") or "<run_id>"
        for key in ("shapes", "dispatch", "summary"):
            if not d.get(key):
                continue
            n = len(d[key])
            nbytes = len(json.dumps(d[key], separators=(",", ":")))
            d[key] = [] if key == "shapes" else ({} if key == "summary" else None)
            d["warnings"].append(f"{key}: {n} entries ({nbytes} B) omitted over MCP; "
                                 f"read with kdev_show run_id={run} section=result")
            if len(json.dumps(d, separators=(",", ":"))) <= self.inline_max_bytes:
                break
        return d


# ---------------------------------------------------------------------------------------------- SDK layer


def build_server(profile: str = "compact", dispatcher: Dispatcher | None = None) -> Any:
    """Return a configured ``mcp.server.lowlevel.Server`` (imports the mcp SDK; KdevError
    (capability_unavailable) with a 'pip install kdev[mcp]' recovery if missing)."""
    try:
        import anyio
        import mcp.types as types
        from mcp.server.lowlevel import Server
        from mcp.server.lowlevel.helper_types import ReadResourceContents
    except ImportError as e:
        raise KdevError(ErrorCode.CAPABILITY_UNAVAILABLE, f"mcp SDK not installed ({e})",
                        recovery="pip install 'kdev[mcp]'") from None

    disp = dispatcher or Dispatcher(profile=profile)
    server: Any = Server("kdev", version=__version__, instructions=INSTRUCTIONS)

    @server.list_tools()
    async def _list_tools() -> list[types.Tool]:
        return [types.Tool(name=t.name, description=t.description, inputSchema=t.input_schema,
                           annotations=types.ToolAnnotations(readOnlyHint=t.read_only, destructiveHint=False,
                                                             openWorldHint=t.open_world))
                for t in disp.tools.values()]

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        progress = None
        try:
            rc = server.request_context
            token = rc.meta.progressToken if rc.meta is not None else None
        except LookupError:
            rc, token = None, None
        if token is not None and rc is not None:
            session = rc.session

            def progress(done: float, total: float | None, message: str) -> None:
                with contextlib.suppress(Exception):
                    anyio.from_thread.run(session.send_progress_notification, token, done, total, message)

        d = await anyio.to_thread.run_sync(lambda: disp.call(name, arguments or {}, progress))
        text = json.dumps(d, separators=(",", ":"), default=str)
        return types.CallToolResult(content=[types.TextContent(type="text", text=text)], structuredContent=d,
                                    isError=d.get("status") == "error")

    @server.list_resource_templates()
    async def _templates() -> list[types.ResourceTemplate]:
        return [types.ResourceTemplate(uriTemplate="kdev://runs/{run_id}/{section}", name="run-section",
                                       description="paged run section (= kdev show); ?offset=N&limit=M",
                                       mimeType="application/json")]

    @server.list_resources()
    async def _resources() -> list[types.Resource]:
        return []

    @server.read_resource()
    async def _read(uri: Any) -> list[ReadResourceContents]:
        d = await anyio.to_thread.run_sync(lambda: disp.read_resource(str(uri)))
        return [ReadResourceContents(content=json.dumps(d, separators=(",", ":"), default=str),
                                     mime_type="application/json")]

    server.kdev_dispatcher = disp
    return server


def main(argv: list[str] | None = None) -> int:
    """`kdev mcp [--profile compact|full]` / `python -m kdev.mcp`: serve over stdio until EOF."""
    p = argparse.ArgumentParser(prog="kdev mcp", description="kdev MCP stdio server")
    p.add_argument("--profile", choices=("compact", "full"), default=os.environ.get("KDEV_MCP_PROFILE", "compact"))
    args = p.parse_args(sys.argv[1:] if argv is None else argv)
    try:
        server = build_server(args.profile)
    except KdevError as e:
        print(Result.from_error("mcp", e).to_json())
        return Result.from_error("mcp", e).exit_code
    import anyio
    from mcp.server.stdio import stdio_server

    async def _serve() -> None:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    anyio.run(_serve)
    return 0
