"""P4: safety rule 8.1 - kdev never submits; `okc check` only through judge.pair with --allow-check.
P4 strengthens these (e.g. AST scan for subprocess argv containing 'okc')."""
from __future__ import annotations

import re
from pathlib import Path

import kdev
from kdev.registry import build_registry

SRC = Path(kdev.__file__).parent
ALLOWED_OKC_FILES = {"judge/okc_cli.py", "judge/pair.py", "judge/__init__.py", "judge/commands.py",
                     "config.py", "service/lock.py", "dashboard/board.py", "registry.py"}


def test_no_submit_code_path():
    pat = re.compile(r"""["']okc["']\s*,\s*["']submit["']|okc\s+submit""")
    offenders = []
    for p in SRC.rglob("*.py"):
        rel = p.relative_to(SRC).as_posix()
        if rel in ALLOWED_OKC_FILES:
            continue
        if pat.search(p.read_text(encoding="utf-8")):
            offenders.append(rel)
    assert not offenders


def test_allow_check_not_exposed_over_mcp():
    cmd = build_registry().get("judge pair")
    assert cmd.mcp is False
    assert all(a.mcp is False for a in cmd.args if "--allow-check" in a.flags)
