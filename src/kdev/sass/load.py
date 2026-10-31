"""Resolve a `kdev sass` target into parsed kernels + ptxas facts. Owner: P5.

Targets: a SASS text file (.sass/.txt/.nvd: cuobjdump or nvdisasm output), a .cubin, a .so/.o/.fatbin, or a
built submission (.py/.cu: the latest ``kdev build`` of that path from the cache manifest; ``needs_build``
otherwise). Line info comes from ``nvdisasm -g`` on the cubins when available.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from kdev.contract import ErrorCode, KdevError
from kdev.models import KernelInfo
from kdev.sass import disasm
from kdev.sass.parse import Kernel, parse_sass
from kdev.util import Runner, md5_file, run

TEXT_SUFFIXES = (".sass", ".txt", ".nvd", ".asm")


@dataclass
class Target:
    path: str
    kind: str                                   # text | cubin | binary | build
    kernels: list[Kernel] = field(default_factory=list)
    infos: dict[str, KernelInfo] = field(default_factory=dict)   # by mangled name
    ptxas_text: str = ""
    sass_text: str = ""
    warnings: list[str] = field(default_factory=list)


def _attach(t: Target) -> None:
    for k in t.kernels:
        info = t.infos.get(k.name)
        if info is not None and info.regs is not None:
            k.regs = info.regs


def _from_binary(path: str, runner: Runner, lineinfo: bool = True) -> tuple[str, dict[str, KernelInfo]]:
    texts = []
    infos: dict[str, KernelInfo] = {}
    if path.endswith(".cubin") and lineinfo:
        t = disasm.nvdisasm_sass(path, runner=runner)
        if t:
            texts.append(t)
    if not texts:
        texts.append(disasm.sass_text(path, runner=runner))
    for name, r in disasm.res_usage(path, runner=runner).items():
        infos[name] = KernelInfo(name=name, regs=r.get("REG"), stack=r.get("STACK", 0),
                                 smem_static=r.get("SHARED", 0))
    return "\n".join(texts), infos


def load_target(path: str, *, runner: Runner = run, home: str | None = None, lineinfo: bool = True) -> Target:
    if not os.path.exists(path):
        raise KdevError(ErrorCode.NOT_FOUND, f"no such file: {path}")
    low = path.lower()
    if low.endswith(TEXT_SUFFIXES):
        txt = Path(path).read_text(encoding="utf-8", errors="replace")
        t = Target(path=path, kind="text", sass_text=txt, kernels=parse_sass(txt))
        side = Path(path).with_suffix(".ptxas")
        if side.is_file():
            from kdev.build.ptxas import parse_ptxas  # noqa: PLC0415

            t.ptxas_text = side.read_text(encoding="utf-8", errors="replace")
            t.infos = {k.name: k for k in parse_ptxas(t.ptxas_text)}
        _attach(t)
        return t
    if low.endswith((".py", ".cu")):
        return _from_build(path, runner, home, lineinfo)
    txt, infos = _from_binary(path, runner, lineinfo)
    t = Target(path=path, kind="cubin" if low.endswith(".cubin") else "binary", sass_text=txt,
               kernels=parse_sass(txt), infos=infos)
    _attach(t)
    return t


def _from_build(path: str, runner: Runner, home: str | None, lineinfo: bool) -> Target:
    from kdev.build import cache  # noqa: PLC0415

    man = cache.latest_manifest_for(path, home)
    if not man or not man.get("exts"):
        raise KdevError(ErrorCode.NEEDS_BUILD, f"no cached build of {path}",
                        recovery=f"run `kdev build {path}` first")
    t = Target(path=path, kind="build")
    if man.get("md5") and os.path.isfile(path) and md5_file(path) != man["md5"]:
        t.warnings.append(f"STALE: {os.path.basename(path)} changed since its last build; rerun kdev build")
    if man.get("knobs"):
        t.warnings.append(f"build knobs: {man['knobs']}")
    texts: list[str] = []
    ptx: list[str] = []
    for e in man["exts"]:
        d = Path(e["dir"])
        info_p = d / "kdev_info.json"
        info = {}
        if info_p.is_file():
            info = json.loads(info_p.read_text(encoding="utf-8"))
            for k in info.get("kernels", []):
                ki = KernelInfo.from_dict(k)
                t.infos[ki.name] = ki
        log = d / "build.log"
        if log.is_file():
            ptx.append(log.read_text(encoding="utf-8", errors="replace"))
        got = False
        if lineinfo:
            for c in info.get("cubins", []):
                if os.path.isfile(c):
                    s = disasm.nvdisasm_sass(c, runner=runner)
                    if s:
                        texts.append(s)
                        got = True
        if not got and (d / "sass.txt").is_file():
            texts.append((d / "sass.txt").read_text(encoding="utf-8", errors="replace"))
            got = True
        if not got:
            t.warnings.append(f"no SASS for ext {e['name']} (cuobjdump unavailable at build time?)")
    t.sass_text = "\n".join(texts)
    t.ptxas_text = "\n".join(ptx)
    t.kernels = parse_sass(t.sass_text)
    _attach(t)
    return t
