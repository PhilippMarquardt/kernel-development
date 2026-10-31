"""MCP stdio server generated from the registry (docs/PLAN.md section 5). Owner: P8.

Implementation sketch: ``from mcp.server.fastmcp import FastMCP``; for each exposed Command register a tool
whose handler builds an argparse Namespace from the tool arguments (defaults from the Arg list), sets
``Context(via='mcp')`` and calls the same handler as the CLI; returns ``Result.to_dict()`` trimmed to the
summary + files (large content only via kdev_show). Profiles: compact (default, ~12 grouped tools) | full.
Resources: kdev://runs/{run_id}/{section}. Never exposes commands/args with mcp=False (judge pair
--allow-check, lock run, serve, remote, sync, ledger --sql).
"""
from __future__ import annotations

from typing import Any

COMPACT_TOOLS = ("doctor", "build", "check", "time", "ab", "sass", "timeline", "ledger", "idea", "job",
                 "show", "run")


def build_server(profile: str = "compact") -> Any:
    """Return a configured FastMCP instance (imports the mcp SDK; KdevError(capability_unavailable) with a
    'pip install kdev[mcp]' recovery if missing)."""
    raise NotImplementedError("mcp.server.build_server (P8)")


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError("mcp.server.main (P8)")
