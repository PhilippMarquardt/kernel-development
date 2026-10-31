"""Registry Arg -> JSON schema (compact by default) and the MCP exposure rules. Owner: P8.

Rules: positional and optional Args map to properties named ``Arg.dest``; type str/path -> string, int ->
integer, float -> number, bool/store_true -> boolean, nargs '+'/'*'/N/append -> array; choices -> enum;
Args with ``mcp=False`` and commands with ``mcp=False`` are omitted. GPU commands get ``wait_s`` (0..120,
default 0 = return job id immediately). Compact profile drops per-property descriptions longer than 60 chars.

Exposure is decided here, in one place, and is *stricter* than the registry flags (defence in depth, PLAN 5
and 8.1): a command is hidden if the registry says ``mcp=False``, if it is on :data:`HIDDEN_COMMANDS`, or if
its declared effects could consume a judge quota (``quota != 'never'`` or ``network == 'okc_check'``).
An Arg is hidden if ``mcp=False``, if it is a REMAINDER argv, or if it is listed in :data:`HIDDEN_ARGS`.
"""
from __future__ import annotations

from typing import Any

from kdev.registry import GLOBAL_ARGS, GPU_ARGS, Arg, Command

# Never exposed over MCP, whatever the registry says (PLAN 5 "Not exposed over MCP").
HIDDEN_COMMANDS = frozenset({
    "judge pair",            # the only path to a judge check (quota); CLI-only behind --allow-check
    "lock run",              # raw command under the GPU lock
    "serve", "serve start", "serve stop", "serve run", "serve status",
    "remote", "remote setup",
    "sync", "sync push", "sync pull", "sync both",
    "job submit",            # raw JobSpec files (could carry an exec job); agents use the typed GPU tools
    "mcp", "mcp tools",
    "dashboard watch", "dashboard serve",   # endless loops
})
# (command, dest) pairs never exposed; "*" = every command.
HIDDEN_ARGS = frozenset({
    ("judge pair", "allow_check"),
    ("ledger query", "sql"),
    ("*", "json"), ("*", "home"), ("*", "quiet"),       # transport-level globals
    ("*", "detach"), ("*", "direct"),                    # MCP decides detach via wait_s; direct would block
})
MAX_WAIT_S = 120
COMPACT_DESC_MAX = 60

_JSON_TYPES = {"str": "string", "path": "string", "int": "integer", "float": "number", "bool": "boolean"}


def command_exposed(cmd: Command) -> bool:
    """True if ``cmd`` may appear as (or inside) an MCP tool."""
    if not cmd.mcp or cmd.name in HIDDEN_COMMANDS:
        return False
    eff = cmd.effects
    if getattr(eff, "quota", "never") != "never" or getattr(eff, "network", "none") == "okc_check":
        return False
    return True


def arg_exposed(cmd_name: str, arg: Arg) -> bool:
    if not arg.mcp or arg.nargs == "...":
        return False
    d = arg.dest
    return (cmd_name, d) not in HIDDEN_ARGS and ("*", d) not in HIDDEN_ARGS


def all_args(cmd: Command) -> list[Arg]:
    """Command args + global args (+ GPU args) in CLI order; a command arg shadows a global of the same dest."""
    out: list[Arg] = []
    seen: set[str] = set()
    for a in list(cmd.args) + GLOBAL_ARGS + (GPU_ARGS if cmd.gpu else []):
        if a.dest not in seen:
            seen.add(a.dest)
            out.append(a)
    return out


def exposed_args(cmd: Command) -> list[Arg]:
    """The subset of :func:`all_args` that MCP may set."""
    return [a for a in all_args(cmd) if arg_exposed(cmd.name, a)]


def is_array(arg: Arg) -> bool:
    return arg.action == "append" or arg.nargs in ("+", "*") or isinstance(arg.nargs, int)


def is_flag(arg: Arg) -> bool:
    return arg.action == "store_true" or arg.type == "bool"


def _jsonable(v: Any) -> bool:
    return isinstance(v, (str, int, float, bool, list, tuple))


def arg_schema(arg: Arg, compact: bool = True) -> dict[str, Any]:
    """JSON schema of one Arg (see module docstring)."""
    if is_flag(arg):
        s: dict[str, Any] = {"type": "boolean"}
        if arg.default:
            s["default"] = True
    else:
        item: dict[str, Any] = {"type": _JSON_TYPES.get(arg.type, "string")}
        if arg.choices is not None:
            item["enum"] = list(arg.choices)
        if is_array(arg):
            s = {"type": "array", "items": item}
            if arg.nargs == "+":
                s["minItems"] = 1
            if isinstance(arg.nargs, int):
                s["minItems"] = s["maxItems"] = arg.nargs
        else:
            s = item
        if arg.default is not None and arg.default != "" and _jsonable(arg.default):
            s["default"] = list(arg.default) if isinstance(arg.default, tuple) else arg.default
    desc = (arg.help or "").strip()
    if desc and (not compact or len(desc) <= COMPACT_DESC_MAX):
        s["description"] = desc
    return s


def is_required(arg: Arg) -> bool:
    if arg.positional:
        return arg.nargs not in ("?", "*") and arg.default is None
    return bool(arg.required)


def wait_schema() -> dict[str, Any]:
    return {"type": "number", "minimum": 0, "maximum": MAX_WAIT_S, "default": 0,
            "description": "seconds to wait for the GPU job (0 = return job_id now)"}


def command_schema(cmd: Command, compact: bool = True) -> dict[str, Any]:
    """Input schema of the MCP twin of ``cmd`` (object, additionalProperties false)."""
    props: dict[str, Any] = {}
    required: list[str] = []
    shared = GLOBAL_ARGS + GPU_ARGS
    for a in exposed_args(cmd):
        props[a.dest] = arg_schema(a, compact)
        if compact and a in shared and a not in cmd.args:
            props[a.dest].pop("description", None)       # documented once in the server instructions
        if is_required(a):
            required.append(a.dest)
    if cmd.gpu:
        props["wait_s"] = wait_schema()
    s: dict[str, Any] = {"type": "object", "properties": props, "additionalProperties": False}
    if required:
        s["required"] = required
    return s


def effects_tag(cmd: Command) -> str:
    """'[gpu=queued writes=run_dir,ledger quota=never]' (REA 'truthful effects')."""
    e = cmd.effects
    parts = [f"gpu={e.gpu}"]
    if e.writes:
        parts.append("writes=" + ",".join(e.writes))
    if e.network != "none":
        parts.append(f"net={e.network}")
    parts.append(f"quota={e.quota}")
    return "[" + " ".join(parts) + "]"


def tool_name(cmd: Command) -> str:
    """'sass diff' -> 'kdev_sass_diff'."""
    return "kdev_" + cmd.name.replace(" ", "_").replace("-", "_")
