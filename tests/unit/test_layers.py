"""P1: every kdev module imports without torch / the mcp SDK (torch only inside worker-run functions)."""
from __future__ import annotations

import importlib
import pkgutil
import sys

import kdev


def _all_modules():
    for m in pkgutil.walk_packages(kdev.__path__, prefix="kdev."):
        if m.name in ("kdev.__main__", "kdev.mcp.__main__"):
            continue
        yield m.name


def test_import_all_without_torch_or_mcp(monkeypatch):
    for blocked in ("torch", "mcp", "numpy"):
        monkeypatch.setitem(sys.modules, blocked, None)  # import -> ImportError
    for name in list(sys.modules):
        if name.startswith("kdev."):
            monkeypatch.delitem(sys.modules, name)
    failures = []
    for name in _all_modules():
        try:
            importlib.import_module(name)
        except Exception as e:  # noqa: BLE001
            failures.append(f"{name}: {type(e).__name__}: {e}")
    assert not failures, "\n".join(failures)
