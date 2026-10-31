"""`kdev doctor`: health checks aggregated from each package's ``probe()``. Owner: P1.

Probe providers (each returns ``list[kdev.models.Check]`` and must never raise or need a GPU context):
``kdev.service.probe`` (GPU visible, lock health, foreign CUDA processes, ncu permission),
``kdev.build.probe`` (toolchain pin vs found, TORCH_EXTENSIONS_DIR keying, thrust headers),
``kdev.observe.probe`` (ncu/nsys/cupti/nvbit/sanitizer availability), ``kdev.ledger.probe`` (db
writable, schema version), ``kdev.remote.probe`` (KDEV_HOME not on a network/mfs path, free disk).
"""
from __future__ import annotations

from kdev.contract import Result
from kdev.models import Check
from kdev.registry import Context

PROBE_MODULES = ("kdev.service", "kdev.build", "kdev.observe", "kdev.ledger", "kdev.remote")


def collect_checks(ctx: Context, problem: str | None = None) -> list[Check]:
    """Import each PROBE_MODULES entry and call ``probe(ctx, problem)``; a probe that raises becomes a FAIL
    line naming the module."""
    raise NotImplementedError("doctor.collect_checks (P1)")


def run_doctor(ctx: Context, problem: str | None = None, fix_hints: bool = False) -> Result:
    """PASS/WARN/FAIL summary; status fail if any FAIL."""
    raise NotImplementedError("doctor.run_doctor (P1)")
