"""P7: adapters, check, sanitize, comply. Placeholder; P7 replaces with okc adapter tests against a tiny fake
problem dir (fake reference.py without torch) and classification tests."""
from __future__ import annotations

from kdev.adapters.base import Adapter, Shape, TimingProtocol
from kdev.adapters.registry import ADAPTERS
from kdev.registry import build_registry


def test_p7_commands_registered():
    assert {"check", "sanitize", "comply"} <= set(build_registry().names())


def test_adapter_interface_frozen_names():
    for meth in ("shapes", "tests", "make_inputs", "call", "check", "score", "judge_cases", "build_hook"):
        assert hasattr(Adapter, meth)
    assert {"okc", "cu_main", "kernelbench"} <= set(ADAPTERS)
    assert TimingProtocol().overhead_us == 4.0
    assert Shape("d1/sliding", {"batch": 1}).weight == 1.0
