"""Registry Arg -> JSON schema (compact by default). Owner: P8.

Rules: positional and optional Args map to properties named ``Arg.dest``; type str/path -> string, int ->
integer, float -> number, bool/store_true -> boolean, nargs '+'/'*'/append -> array; choices -> enum;
Args with ``mcp=False`` and commands with ``mcp=False`` are omitted. GPU commands get ``wait_s`` (0..120,
default 0 = return job id immediately). Compact profile drops per-property descriptions longer than 60 chars.
"""
from __future__ import annotations

from typing import Any

from kdev.registry import Arg, Command


def arg_schema(arg: Arg, compact: bool = True) -> dict[str, Any]:
    raise NotImplementedError("mcp.schema.arg_schema (P8)")


def command_schema(cmd: Command, compact: bool = True) -> dict[str, Any]:
    raise NotImplementedError("mcp.schema.command_schema (P8)")


def tool_name(cmd: Command) -> str:
    """'sass diff' -> 'kdev_sass_diff'."""
    return "kdev_" + cmd.name.replace(" ", "_").replace("-", "_")
