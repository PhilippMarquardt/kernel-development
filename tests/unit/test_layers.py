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
    saved = dict(sys.modules)
    try:
        _import_all(monkeypatch)
    finally:  # restore the exact module table so later tests see one copy of every class
        monkeypatch.undo()
        for name in [n for n in sys.modules if n.startswith("kdev") and n not in saved]:
            del sys.modules[name]
        sys.modules.update({k: v for k, v in saved.items() if k.startswith("kdev")})
        for name, mod in saved.items():  # package attributes (kdev.contract etc.) back to the old modules
            if name.startswith("kdev.") and mod is not None:
                parent, _, child = name.rpartition(".")
                if parent in sys.modules:
                    setattr(sys.modules[parent], child, mod)


def _import_all(monkeypatch):
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
