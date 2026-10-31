"""Disassembly wrappers: cuobjdump -sass / -elf / -ptx and nvdisasm -cfg -lineinfo. Owner: P5."""
from __future__ import annotations

from kdev.util import Runner, run


def extract_cubins(path: str, out_dir: str, runner: Runner = run) -> list[str]:
    """`cuobjdump -xelf all` from a .so/.o/fatbin (or return [path] for a .cubin)."""
    raise NotImplementedError("disasm.extract_cubins (P5)")


def sass_text(path: str, *, kernel_regex: str | None = None, runner: Runner = run) -> str:
    """`cuobjdump -sass` text (optionally filtered by -fun)."""
    raise NotImplementedError("disasm.sass_text (P5)")


def nvdisasm_cfg(cubin: str, *, lineinfo: bool = True, runner: Runner = run) -> str | None:
    """`nvdisasm -cfg [-g] cubin` dot text; None if nvdisasm is unavailable (callers fall back to
    kdev.sass.cfg back-edge detection on plain SASS)."""
    raise NotImplementedError("disasm.nvdisasm_cfg (P5)")
