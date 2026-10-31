"""P6: timeline, floor, GPU profiles, occupancy, observe backends. Placeholder; P6 replaces with decoder,
roofline/skeleton and occupancy tests."""
from __future__ import annotations

import sys
from pathlib import Path

import kdev
from kdev.registry import build_registry

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib


def test_p6_commands_registered():
    assert {"timeline", "floor", "floor calibrate", "regs", "profile", "dispatch"} <= set(build_registry().names())


def test_h200_profile_parses():
    p = Path(kdev.__file__).parent / "data" / "gpus" / "h200.toml"
    d = tomllib.loads(p.read_text(encoding="utf-8"))
    assert d["schema"] == "kdev.gpu/1" and d["sm_count"] == 132
    assert d["measured"]["graph_fixed_us"]["value"] == 4.0


def test_marks_header_present():
    assert (Path(kdev.__file__).parent / "timeline" / "marks.cuh").exists()
