"""CLI: `kdev mcp` (start the MCP stdio server). Owner: P8. Must not import the mcp SDK at import time."""
from __future__ import annotations

from kdev.registry import Arg, Registry, todo


def register(reg: Registry) -> None:
    reg.command("mcp", "run the MCP stdio server (pip install kdev[mcp])", todo("mcp.server.main (P8)"),
                args=[Arg(("--profile",), "compact|full", choices=("compact", "full"), default="compact")],
                mcp=False, package="P8")
