"""P8: dashboard, MCP server, skill. Placeholder; P8 replaces with schema-generation tests (every exposed
command -> valid JSON schema; mcp=False commands absent) and a render smoke test."""
from __future__ import annotations

from pathlib import Path

from kdev.mcp.schema import tool_name
from kdev.registry import build_registry


def test_p8_commands_registered():
    assert {"dashboard build", "mcp"} <= set(build_registry().names())


def test_tool_names_unique():
    names = [tool_name(c) for c in build_registry().all()]
    assert len(names) == len(set(names))


def test_skill_file_present():
    root = Path(__file__).resolve().parents[2]
    assert (root / "skills" / "kdev" / "SKILL.md").exists()
