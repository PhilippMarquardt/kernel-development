"""CLI: `kdev mcp` (start the MCP stdio server), `kdev mcp tools` (inspect the generated tool table).
Owner: P8. Must not import the mcp SDK at import time."""
from __future__ import annotations

import json
from argparse import Namespace

from kdev.contract import Result
from kdev.registry import Arg, Context, Registry


def cmd_mcp(args: Namespace, ctx: Context) -> Result:
    """Serve MCP over stdio; exits the process when the client disconnects (stdout belongs to the protocol,
    so nothing is printed afterwards)."""
    from kdev.mcp.server import build_server, main

    build_server(args.profile)          # fail fast with capability_unavailable if the SDK is missing
    raise SystemExit(main(["--profile", args.profile]))


def cmd_mcp_tools(args: Namespace, ctx: Context) -> Result:
    """List the MCP tools of a profile with their schema size (approx. permanent context cost)."""
    from kdev.mcp.server import build_tools
    from kdev.registry import build_registry

    tools = build_tools(build_registry(), args.profile)
    rows = []
    total = 0
    for t in tools:
        size = len(json.dumps({"name": t.name, "description": t.description, "inputSchema": t.input_schema}))
        total += size
        rows.append({"name": t.name, "bytes": size, "read_only": t.read_only, "description": t.description})
    lines = [f"OK  mcp tools  profile={args.profile}  tools={len(tools)}  ~{total // 4} tokens"]
    lines += [f"  {r['name']:22s} {r['bytes']:5d} B  {r['description'][:70]}" for r in rows[:15]]
    if len(rows) > 15:
        lines.append(f"  ... {len(rows) - 15} more tools (use --json)")
    return Result(tool="mcp tools", verdict="ok",
                  summary={"profile": args.profile, "tools": rows, "total_bytes": total}, lines=lines)


def register(reg: Registry) -> None:
    prof = Arg(("--profile",), "compact|full", choices=("compact", "full"), default="compact")
    reg.command("mcp", "run the MCP stdio server (pip install kdev[mcp])", cmd_mcp, args=[prof],
                mcp=False, package="P8")
    reg.command("mcp tools", "list the generated MCP tools and their context cost", cmd_mcp_tools, args=[prof],
                mcp=False, package="P8")
