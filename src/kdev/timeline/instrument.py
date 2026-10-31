"""Source instrumentation for timelines. Owner: P6.

Finds ``// @kdev:mark NAME`` annotations in CUDA sources (standalone .cu or load_inline strings inside a .py),
assigns ids, inserts ``KD_MARK(id);`` at those lines, adds the marks.cuh include + the kd_marks_buf
definition + a host ``kd_marks_init`` binding, and returns the instrumented copy + id->name map.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Instrumented:
    path: str                       # instrumented copy (in the run dir)
    marks: dict[int, str] = field(default_factory=dict)
    defines: list[str] = field(default_factory=lambda: ["-DKDEV_MARKS"])
    warnings: list[str] = field(default_factory=list)


def find_annotations(source: str) -> list[tuple[int, str]]:
    """[(line_no, name)] of '// @kdev:mark NAME' (pure)."""
    raise NotImplementedError("instrument.find_annotations (P6)")


def instrument_source(source: str) -> tuple[str, dict[int, str]]:
    """Pure transform of one CUDA source string."""
    raise NotImplementedError("instrument.instrument_source (P6)")


def instrument_file(path: str, out_dir: str) -> Instrumented:
    raise NotImplementedError("instrument.instrument_file (P6)")
