"""MCP server over the command registry (official `mcp` SDK, FastMCP, stdio). Owner: P8.

Install with ``pip install kdev[mcp]``. Importing this package must NOT import the SDK (the CLI imports
kdev.mcp.commands at startup); only server.serve() imports ``mcp``.
"""
