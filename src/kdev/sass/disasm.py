"""Disassembly wrappers: cuobjdump -sass / -xelf / -res-usage and nvdisasm -cfg / -g. Owner: P5.

Tool lookup: explicit ``tool=`` argument, else the pinned/detected CUDA tree (``$KDEV_CUDA_HOME``, ``$CUDA_HOME``),
else PATH, else the bare name (so a fake runner in tests sees 'cuobjdump').
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from kdev.contract import ErrorCode, KdevError
from kdev.util import Runner, run

_EXTRACT = re.compile(r"Extracting ELF file\s+\d+:\s*(\S+)")
_FUNC = re.compile(r"^\s*Function\s*:\s*(\S+)", re.M)


def tool_path(name: str, tool: str | None = None) -> str:
    if tool:
        return tool
    from kdev.build.toolchain import find_tool  # noqa: PLC0415

    return find_tool(name) or name


def _run(argv: list[str], runner: Runner, cwd: str | None = None, what: str = "") -> str:
    try:
        p = runner(argv, timeout=600, cwd=cwd) if cwd else runner(argv, timeout=600)
    except FileNotFoundError as e:
        raise KdevError(ErrorCode.CAPABILITY_UNAVAILABLE, f"{argv[0]} not found",
                        recovery="install the CUDA toolkit (cuobjdump/nvdisasm) or set KDEV_CUDA_HOME") from e
    if p.returncode != 0:
        if p.returncode == 127 or "not found" in (p.stderr or "").lower() and not p.stdout:
            raise KdevError(ErrorCode.CAPABILITY_UNAVAILABLE, f"{argv[0]} unavailable: {p.stderr.strip()[:200]}",
                            recovery="install the CUDA toolkit or set KDEV_CUDA_HOME")
        raise KdevError(ErrorCode.INVALID_REQUEST, f"{what or argv[0]} failed (rc {p.returncode}): "
                        f"{(p.stderr or p.stdout).strip()[:300]}", details={"argv": argv})
    return p.stdout


def extract_cubins(path: str, out_dir: str, runner: Runner = run, *, tool: str | None = None) -> list[str]:
    """`cuobjdump -xelf all` from a .so/.o/fatbin into out_dir (or return [path] for a .cubin)."""
    if path.endswith(".cubin"):
        return [path]
    os.makedirs(out_dir, exist_ok=True)
    out = _run([tool_path("cuobjdump", tool), "-xelf", "all", os.path.abspath(path)], runner, cwd=out_dir,
               what="cuobjdump -xelf")
    files = [str(Path(out_dir) / m) for m in _EXTRACT.findall(out)]
    if not files:
        files = sorted(str(p) for p in Path(out_dir).glob("*.cubin"))
    return files


def filter_functions(text: str, kernel_regex: str | None) -> str:
    """Keep only the 'Function : X' sections whose name matches ``kernel_regex`` (header lines kept)."""
    if not kernel_regex:
        return text
    rx = re.compile(kernel_regex)
    out: list[str] = []
    keep = True
    for line in text.splitlines():
        m = _FUNC.match(line)
        if m:
            keep = bool(rx.search(m.group(1)))
        if keep:
            out.append(line)
    return "\n".join(out) + "\n"


def sass_text(path: str, *, kernel_regex: str | None = None, runner: Runner = run, tool: str | None = None) -> str:
    """`cuobjdump -sass` text of a .so/.cubin/.o (optionally only functions matching ``kernel_regex``)."""
    out = _run([tool_path("cuobjdump", tool), "-sass", path], runner, what="cuobjdump -sass")
    return filter_functions(out, kernel_regex)


def res_usage(path: str, *, runner: Runner = run, tool: str | None = None) -> dict[str, dict[str, int]]:
    """`cuobjdump -res-usage`: {function: {'REG': n, 'STACK': n, 'SHARED': n, 'LOCAL': n, ...}} ({} on error)."""
    try:
        out = _run([tool_path("cuobjdump", tool), "-res-usage", path], runner, what="cuobjdump -res-usage")
    except KdevError:
        return {}
    return parse_res_usage(out)


def parse_res_usage(text: str) -> dict[str, dict[str, int]]:
    res: dict[str, dict[str, int]] = {}
    cur = None
    for line in text.splitlines():
        m = re.match(r"\s*Function\s+(\S+?):?\s*$", line)
        if m:
            cur = m.group(1)
            res[cur] = {}
            continue
        if cur is not None:
            for k, v in re.findall(r"([A-Z_]+(?:\[\d+\])?):(\d+)", line):
                res[cur][k] = int(v)
    return res


def nvdisasm_cfg(cubin: str, *, lineinfo: bool = True, runner: Runner = run, tool: str | None = None) -> str | None:
    """`nvdisasm -cfg [-g] cubin` dot text; None if nvdisasm is unavailable (callers fall back to
    kdev.sass.cfg back-edge detection on plain SASS)."""
    argv = [tool_path("nvdisasm", tool), "-cfg", *(["-g"] if lineinfo else []), cubin]
    try:
        p = runner(argv, timeout=600)
    except (FileNotFoundError, OSError):
        return None
    if p.returncode != 0:
        if lineinfo:   # cubins without line info make -g fail on some versions
            return nvdisasm_cfg(cubin, lineinfo=False, runner=runner, tool=tool)
        return None
    return p.stdout


def nvdisasm_sass(cubin: str, *, lineinfo: bool = True, runner: Runner = run, tool: str | None = None) -> str | None:
    """`nvdisasm -c [-g] cubin`: SASS with '//## File "...", line N' comments (parsed into Instr.line).
    None if nvdisasm is unavailable."""
    argv = [tool_path("nvdisasm", tool), "-c", *(["-g"] if lineinfo else []), cubin]
    try:
        p = runner(argv, timeout=600)
    except (FileNotFoundError, OSError):
        return None
    if p.returncode != 0:
        if lineinfo:
            return nvdisasm_sass(cubin, lineinfo=False, runner=runner, tool=tool)
        return None
    return p.stdout
