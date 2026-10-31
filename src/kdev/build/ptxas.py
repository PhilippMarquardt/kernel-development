"""ptxas -v / nvcc warning parser. Owner: P5. Fixtures: tests/fixtures/ptxas/vjc.ptxas, v130_attention.log,
tests/fixtures/build/px_c7515.ptxas.

Recognises 'ptxas info : Compiling entry function '<mangled>' for 'sm_90a'', 'Function properties for X' +
'N bytes stack frame, N bytes spill stores, N bytes spill loads', 'Used N registers, used B barriers, S bytes
smem, C bytes cmem[0]', and the wgmma serialisation messages C7510-C7515 (ptxas prints them as 'ptxas info'
or 'ptxas warning' *before* the entry they name; kept verbatim and attributed via the quoted function name).
Names can be demangled with cu++filt (:func:`demangle`).
"""
from __future__ import annotations

import re

from kdev.models import KernelInfo
from kdev.util import Runner, run

_ENTRY = re.compile(r"Compiling entry function '([^']+)' for '([^']+)'")
_PROPS = re.compile(r"Function properties for (\S+)")
_STACK = re.compile(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads")
_USED = re.compile(r"Used (\d+) registers")
_BARS = re.compile(r"used (\d+) barriers")
_SMEM = re.compile(r"(\d+) bytes smem")
_CMEM = re.compile(r"(\d+) bytes (cmem\[\d+\])")
_SER = re.compile(r"\((C751\d)\)\s*(.*)")
_SER_FN = re.compile(r"in the function '([^']+)'")
_WARN_CODE = re.compile(r"ptxas (?:warning|info)\s*:\s*\((C\d{4})\)")


def parse_ptxas(text: str) -> list[KernelInfo]:
    """One KernelInfo per 'Compiling entry function' (order of appearance; a kernel compiled for two archs
    appears twice with an '@sm_xx' suffix on the second). Serialisation messages naming the kernel are
    appended verbatim to ``warnings``; other ptxas warnings with a code are attached to the entry being
    compiled."""
    kernels: list[KernelInfo] = []
    by_name: dict[str, KernelInfo] = {}
    cur: KernelInfo | None = None
    props_for: str | None = None
    pending_ser: list[tuple[str, str]] = []   # (function, line) seen before the entry was compiled
    for raw in (text or "").splitlines():
        line = raw.strip()
        m = _ENTRY.search(line)
        if m:
            name, arch = m.group(1), m.group(2)
            key = name if name not in by_name else f"{name}@{arch}"
            cur = KernelInfo(name=key)
            kernels.append(cur)
            by_name[key] = cur
            props_for = None
            continue
        m = _SER.search(line)
        if m and "ptxas" in line:
            fn = _SER_FN.search(line)
            if fn:
                pending_ser.append((fn.group(1), line))
            elif cur is not None:
                cur.warnings.append(line)
            continue
        m = _PROPS.search(line)
        if m:
            props_for = m.group(1)
            continue
        m = _STACK.search(line)
        if m:
            if cur is not None and props_for == cur.name.split("@")[0]:
                cur.stack, cur.spill_store, cur.spill_load = (int(m.group(i)) for i in (1, 2, 3))
            props_for = None
            continue
        m = _USED.search(line)
        if m and cur is not None:
            cur.regs = int(m.group(1))
            b = _BARS.search(line)
            if b:
                cur.barriers = int(b.group(1))
            s = _SMEM.search(line)
            if s:
                cur.smem_static = int(s.group(1))
            for val, bank in _CMEM.findall(line):
                cur.cmem[bank] = int(val)
            continue
        m = _WARN_CODE.search(line)
        if m and cur is not None:
            cur.warnings.append(line)
    for fn, line in pending_ser:
        k = by_name.get(fn)
        if k is not None:
            k.warnings.append(line)
    return kernels


def serialization_warnings(text: str) -> list[tuple[str, str]]:
    """[(code e.g. 'C7515', message)] in order of appearance (message verbatim after the code)."""
    out: list[tuple[str, str]] = []
    for line in (text or "").splitlines():
        m = _SER.search(line)
        if m and "ptxas" in line:
            out.append((m.group(1), m.group(2).strip()))
    return out


def serialization_by_kernel(text: str) -> dict[str, list[str]]:
    """{mangled kernel name: [codes]} for C751x messages that name their function."""
    out: dict[str, list[str]] = {}
    for line in (text or "").splitlines():
        m = _SER.search(line)
        fn = _SER_FN.search(line)
        if m and fn and "ptxas" in line:
            out.setdefault(fn.group(1), []).append(m.group(1))
    return out


def demangle(names: list[str], runner: Runner = run, tool: str = "cu++filt") -> dict[str, str]:
    """{mangled: demangled} via cu++filt (then c++filt); identity when neither is available."""
    if not names:
        return {}
    for t in (tool, "c++filt"):
        try:
            p = runner([t, *names], timeout=30)
        except Exception:  # noqa: BLE001
            continue
        if p.returncode == 0:
            outs = p.stdout.splitlines()
            if len(outs) == len(names):
                return dict(zip(names, outs))
    return {n: n for n in names}
