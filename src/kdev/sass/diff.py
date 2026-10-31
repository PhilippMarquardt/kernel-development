"""Build-vs-build SASS diff. Owner: P5.

Matches kernels by demangled name (fallback: closest histogram), then reports per kernel: instr count, regs,
class histogram delta, hot-loop size/mix delta, and audit findings that appear/disappear.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from kdev.sass.parse import Kernel


@dataclass
class KernelDiff:
    name_a: str
    name_b: str
    instr_a: int
    instr_b: int
    regs_a: int | None = None
    regs_b: int | None = None
    class_delta: dict[str, float] = field(default_factory=dict)
    loop_delta: dict[str, float] = field(default_factory=dict)
    findings_new: list[str] = field(default_factory=list)
    findings_gone: list[str] = field(default_factory=list)


def diff_kernels(a: list[Kernel], b: list[Kernel], kernel_regex: str | None = None) -> list[KernelDiff]:
    raise NotImplementedError("diff.diff_kernels (P5)")
