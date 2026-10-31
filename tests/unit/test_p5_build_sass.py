"""P5: build/toolchain/ptxas, SASS analyser, sweep. Placeholder; P5 replaces with parser/audit tests on the
fixtures (PDL hoist positive case: LDG.E.CONSTANT at 0x0080 before ACQBULK in vjc_k_L1_F0_M1.sass)."""
from __future__ import annotations

from kdev.registry import build_registry
from kdev.sass.audits import AUDITS
from kdev.sass.classes import CLASSES


def test_build_sass_commands_registered():
    assert {"build", "sass", "sass diff", "sweep"} <= set(build_registry().names())


def test_sass_fixture_has_pdl_hoist_case(sample_sass):
    acq = sample_sass.find("ACQBULK")
    ldg = sample_sass.find("LDG.E.CONSTANT")
    assert 0 <= ldg < acq


def test_ptxas_fixture(sample_ptxas):
    assert "Used" in sample_ptxas and "registers" in sample_ptxas


def test_audit_and_class_names():
    assert {"pdl", "spills", "serialize", "expect", "repeat"} == set(AUDITS)
    assert "mma" in CLASSES and "ldl_stl" in CLASSES
