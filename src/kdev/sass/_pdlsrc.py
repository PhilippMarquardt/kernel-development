"""PDL audit attribution (P5, private): which kernel parameter each pre-wait global load reads, and the
per-file allowlist.

Attribution is a backward register slice over the instruction stream (address order, CFG ignored: an
approximation that is right for the straight-line prologue code where pre-wait loads live). It follows the
address registers of the load through arithmetic until it reaches a parameter-bank constant
``c[0x0][OFF]`` (OFF >= the arch's parameter base, 0x210 on sm_70-sm_90, 0x380 on sm_100+) or another load
(pointer chase: ``block_table[seq_lens[b]]`` reports both roots). Parameter offsets are named from the
source: the ``__global__`` signature of the kernel, and for a by-value struct parameter (``Params p``) the
struct's member layout (C rules for scalars, pointers and fixed arrays). Unknown types stop the layout:
later members are reported as ``param+0xNN``.

Allowlist (per file, with a justification; loads whose every root is allowed are not errors):

    // kdev:pdl-allow seq_lens,kv_indptr -- read-only inputs, never written by the previous kernel
    # kdev:pdl-allow p.page_table: written once at prepare time

in the submission or any of its CUDA sources, or the same lines (without the comment marker) in a sidecar
file ``<file>.pdl-allow``. A name matches the full root name (``p.seq_lens``), its last component
(``seq_lens``) or the raw offset (``param+0x8``). An entry without a reason is ignored (warned).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

from kdev.sass.parse import Instr, Kernel

_CBANK = re.compile(r"c\[0x0\]\[(0x[0-9a-fA-F]+)\]")
_ALLOW = re.compile(r"kdev:pdl-allow\s+([^\s:]+(?:\s*,\s*[^\s:,]+)*)\s*(?:--|:)?\s*(.*)$")
_SIDE_ALLOW = re.compile(r"^\s*(?:kdev:pdl-allow\s+)?([^\s:#]+(?:\s*,\s*[^\s:,]+)*)\s*(?:--|:)\s*(.+)$")
LOADS = frozenset({"LDG", "LD", "ULDG", "LDGSTS", "UTMALDG", "UBLKCP", "LDGMC", "UTMAPF", "ATOMG", "ATOM"})
_TOOLKIT = ("/include/", "\\include\\", "cuda/std", "/cccl/", "cutlass/", "cute/")
MAX_STEPS = 600

# C type sizes (bytes) for the layout of params / by-value structs
_SIZES = {
    "bool": 1, "char": 1, "signed char": 1, "unsigned char": 1, "int8_t": 1, "uint8_t": 1,
    "short": 2, "unsigned short": 2, "int16_t": 2, "uint16_t": 2, "half": 2, "__half": 2, "__nv_bfloat16": 2,
    "nv_bfloat16": 2, "bf16": 2,
    "int": 4, "unsigned": 4, "unsigned int": 4, "int32_t": 4, "uint32_t": 4, "float": 4, "half2": 4,
    "__half2": 4, "__nv_bfloat162": 4,
    "long": 8, "unsigned long": 8, "long long": 8, "unsigned long long": 8, "int64_t": 8, "uint64_t": 8,
    "size_t": 8, "double": 8, "intptr_t": 8, "uintptr_t": 8, "float2": 8, "int2": 8, "uint2": 8,
    "float4": 16, "int4": 16, "uint4": 16, "CUtensorMap": 64,
}
_ALIGN_OVERRIDE = {"CUtensorMap": 64}


def param_base(arch: str) -> int:
    m = re.search(r"sm_(\d+)", arch or "")
    return 0x380 if m and int(m.group(1)) >= 100 else 0x210


# ------------------------------------------------------------------------------------------- slicing
def _addr_regs(ins: Instr) -> list[str]:
    """Registers forming the global address of a load (the operand after ``desc[..]``, else the first
    bracketed source operand)."""
    from kdev.sass._dataflow import regs_in  # noqa: PLC0415

    ops = ins.operands
    cand = [o for o in ops if "desc[" in o]
    if not cand:
        brs = [o for o in ops if "[" in o]
        if ins.base in ("LDGSTS",) and len(brs) >= 2:
            cand = brs[1:2]
        elif ins.base in ("ATOMG", "ATOM", "UTMALDG", "UTMAPF", "UBLKCP"):
            cand = brs[-1:] if ins.base != "ATOMG" else brs[:1]
        else:
            cand = brs[-1:] if brs else []
    out: list[str] = []
    for o in cand:
        o2 = re.sub(r"desc\[[^\]]*\]", "", o)
        out += [r for r in regs_in(o2) if r not in ("RZ", "URZ")]
    return out


@dataclass
class LoadRoot:
    addr: int
    offsets: list[int] = field(default_factory=list)      # parameter byte offsets reached
    pointers: list[int] = field(default_factory=list)     # of those, read as 64-bit values (base pointers)
    via: list[int] = field(default_factory=list)          # addresses of loads the address depends on
    line: str | None = None                                # user-source file:line of the address math
    names: list[str] = field(default_factory=list)
    allowed: str | None = None                             # justification when every root is allowlisted


def slice_load(instrs: list[Instr], pos: int, pbase: int, max_steps: int = MAX_STEPS) -> LoadRoot:
    """Backward slice of the address of ``instrs[pos]``."""
    from kdev.sass._dataflow import split_defs  # noqa: PLC0415

    ld = instrs[pos]
    root = LoadRoot(addr=ld.addr)
    ptr_line: str | None = None             # line of the first base-pointer read (the C expression)
    fallback_line = ld.line if ld.line and not _toolkit(ld.line) else None
    work: list[tuple[str, int]] = [(r, pos) for r in _addr_regs(ld)]
    for c in _CBANK.findall(" ".join(ld.operands)):
        if int(c, 16) >= pbase:
            root.offsets.append(int(c, 16) - pbase)
    seen: set[tuple[str, int]] = set()
    steps = 0
    while work and steps < max_steps:
        reg, start = work.pop()
        if (reg, start) in seen:
            continue
        seen.add((reg, start))
        for j in range(start - 1, -1, -1):
            steps += 1
            ins = instrs[j]
            _, defs, uses = split_defs(ins)
            if reg not in defs:
                continue
            consts = [int(c, 16) for c in _CBANK.findall(" ".join(ins.operands))]
            wide = ".64" in ins.opcode or ".128" in ins.opcode
            for c in consts:
                if c >= pbase:
                    root.offsets.append(c - pbase)
                    if wide:
                        root.pointers.append(c - pbase)
                        if ins.line and not _toolkit(ins.line) and ptr_line is None:
                            ptr_line = ins.line
            if fallback_line is None and ins.line and not _toolkit(ins.line):
                fallback_line = ins.line
            if ins.base in LOADS or ins.base in ("LDS", "LDSM", "LDL"):
                if ins.base in LOADS:
                    root.via.append(ins.addr)
                    work += [(r, j) for r in _addr_regs(ins)]
                break
            work += [(r, j) for r in uses if not r.startswith(("P", "UP")) and r not in ("RZ", "URZ")]
            if not ins.pred:            # an unpredicated def kills the value; a predicated one may not
                break
    root.offsets = sorted(set(root.offsets))
    root.pointers = sorted(set(root.pointers))
    root.via = sorted(set(root.via))
    root.line = ptr_line or fallback_line
    return root


def _toolkit(line: str) -> bool:
    return any(t in line for t in _TOOLKIT)


# ------------------------------------------------------------------------------------------- naming
def _strip_comments(src: str) -> str:
    src = re.sub(r"/\*.*?\*/", " ", src, flags=re.S)
    return re.sub(r"//[^\n]*", " ", src)


def _norm_type(t: str) -> str:
    t = re.sub(r"\b(const|volatile|__restrict__|__restrict|restrict|struct|typename|__grid_constant__)\b", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"^(std::|::)", "", t)
    return t


def _size_align(t: str, structs: dict[str, list[tuple[str, str, int]]], depth: int = 0) -> tuple[int, int] | None:
    t = _norm_type(t)
    if "*" in t or "&" in t:
        return 8, 8
    if t in _SIZES:
        s = _SIZES[t]
        return s, _ALIGN_OVERRIDE.get(t, min(s, 16))
    base = t.split("::")[-1].split("<")[0].strip()
    if base in _SIZES:
        s = _SIZES[base]
        return s, _ALIGN_OVERRIDE.get(base, min(s, 16))
    if base in structs and depth < 3:
        _, size, align, ok = _layout(structs[base], structs, depth + 1)
        return (size, align) if ok else None
    return None


def _layout(members: list[tuple[str, str, int]], structs: dict[str, list[tuple[str, str, int]]], depth: int = 0
            ) -> tuple[list[tuple[int, int, str, str]], int, int, bool]:
    """([(offset, size, name, type)], total size, alignment, complete). Stops at the first member of an
    unknown type (complete=False; the known prefix is returned)."""
    off, al_max, out = 0, 1, []
    for typ, name, count in members:
        sa = _size_align(typ, structs, depth)
        if sa is None:
            return out, off, al_max, False
        s, a = sa
        off = (off + a - 1) // a * a
        out.append((off, s * count, name, typ))
        off += s * count
        al_max = max(al_max, a)
    size = (off + al_max - 1) // al_max * al_max
    return out, size, al_max, True


def _members(body: str) -> list[tuple[str, str, int]]:
    out = []
    for decl in body.split(";"):
        d = decl.strip()
        if not d or "(" in d or d.startswith(("static ", "template", "using ", "typedef", "public", "private")):
            continue
        d = re.sub(r"=[^,]*", "", d)               # default member initialisers
        d = re.sub(r"\balignas\s*\([^)]*\)", "", d)
        m = re.match(r"^(.*?)([A-Za-z_]\w*(?:\s*\[[^\]]*\])*)\s*((?:,\s*\**\s*[A-Za-z_]\w*(?:\s*\[[^\]]*\])*)*)$", d,
                     re.S)
        if not m:
            continue
        typ = m.group(1).strip()
        names = [m.group(2)] + [x.strip() for x in m.group(3).split(",") if x.strip()]
        for i, nm in enumerate(names):
            t = typ
            stars = re.match(r"^(\**)\s*(.*)$", nm)
            if i > 0 and stars and stars.group(1):
                t, nm = typ + stars.group(1), stars.group(2)
            arr = re.findall(r"\[([^\]]*)\]", nm)
            nm = re.sub(r"\s*\[[^\]]*\]", "", nm).strip()
            count = 1
            for a in arr:
                try:
                    count *= int(a, 0)
                except ValueError:
                    count = 0
            if not typ or not nm or count == 0:
                continue
            out.append((t, nm, count))
    return out


def parse_structs(src: str) -> dict[str, list[tuple[str, str, int]]]:
    s = _strip_comments(src)
    out: dict[str, list[tuple[str, str, int]]] = {}
    for m in re.finditer(r"\b(?:struct|class)\s+(?:alignas\s*\([^)]*\)\s*)?(\w+)\s*(?::[^{;]*)?\{", s):
        i, depth = m.end(), 1
        while i < len(s) and depth:
            depth += {"{": 1, "}": -1}.get(s[i], 0)
            i += 1
        body = s[m.end():i - 1]
        body = re.sub(r"\{[^{}]*\}", ";", body)          # nested function bodies / initialisers
        out.setdefault(m.group(1), _members(body))
    for m in re.finditer(r"typedef\s+struct\s*\{([^}]*)\}\s*(\w+)\s*;", s):
        out.setdefault(m.group(2), _members(m.group(1)))
    return out


def kernel_params(src: str, short: str) -> list[tuple[str, str]] | None:
    """[(type, name)] of ``__global__ ... short(...)`` in ``src`` (first definition; ``__launch_bounds__(..)``
    and templates before the name are fine)."""
    s = _strip_comments(src)
    for m in re.finditer(r"\b" + re.escape(short) + r"\s*\(", s):
        head = s[max(0, m.start() - 600):m.start()]
        cut = max(head.rfind(";"), head.rfind("}"))
        if "__global__" not in head[cut + 1:]:
            continue
        i, depth = m.end(), 1
        while i < len(s) and depth:
            depth += {"(": 1, ")": -1}.get(s[i], 0)
            i += 1
        rest = s[i:i + 200].lstrip()
        if not rest.startswith("{") and not rest.startswith("__"):
            continue                        # a declaration / call, not the definition
        plist = s[m.end():i - 1].strip()
        if plist in ("", "void"):
            return []
        out = []
        for p in _split_top(plist):
            p = re.sub(r"=.*$", "", p).strip()
            mm = re.match(r"^(.*?)([A-Za-z_]\w*)\s*(\[[^\]]*\])?$", p, re.S)
            if mm:
                out.append(((mm.group(1) + ("*" if mm.group(3) else "")).strip(), mm.group(2)))
        return out
    return None


def _split_top(s: str) -> list[str]:
    out, depth, cur = [], 0, []
    for ch in s:
        if ch in "<([":
            depth += 1
        elif ch in ">)]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    return [x for x in out if x.strip()]


def short_kernel_name(mangled: str) -> str:
    """Unqualified kernel name from an Itanium mangled name (or the name itself)."""
    if not mangled.startswith("_Z"):
        return mangled
    nested = mangled.startswith("_ZN")
    s, pos, best = mangled, (3 if nested else 2), ""
    while pos < len(s):
        m = re.match(r"\d+", s[pos:])
        if m:
            n = int(m.group(0))
            pos += len(m.group(0))
            best = s[pos:pos + n]
            pos += n
            if not nested or (pos < len(s) and s[pos] in "IE"):
                break
            continue
        if s[pos] in "IE":
            break
        pos += 1
    return best


@dataclass
class ParamMap:
    fields: list[tuple[int, int, str]] = field(default_factory=list)   # (offset, size, name)
    known_until: int | None = None                                       # offsets beyond are unnamed
    note: str = ""

    def name(self, off: int) -> str:
        for o, s, n in self.fields:
            if o <= off < o + max(s, 1):
                return n if off == o else f"{n}+{off - o:#x}"
        return f"param+{off:#x}"


def param_map(sources: dict[str, str], mangled: str) -> ParamMap:
    short = short_kernel_name(mangled)
    structs: dict[str, list[tuple[str, str, int]]] = {}
    for txt in sources.values():
        for k, v in parse_structs(txt).items():
            structs.setdefault(k, v)
    params = None
    for txt in sources.values():
        params = kernel_params(txt, short)
        if params is not None:
            break
    if params is None:
        return ParamMap(note=f"signature of {short or mangled[:40]} not found in sources")
    pm = ParamMap()
    off = 0
    for typ, name in params:
        sa = _size_align(typ, structs)
        if sa is None:
            pm.known_until = off
            pm.note = f"unknown type {typ!r}"
            return pm
        s, a = sa
        off = (off + a - 1) // a * a
        base = _norm_type(typ).split("::")[-1].split("<")[0].strip()
        if "*" not in typ and base in structs:
            lay, _, _, ok = _layout(structs[base], structs)
            for mo, ms, mn, _ in lay:
                pm.fields.append((off + mo, ms, f"{name}.{mn}"))
            if not ok:
                pm.note = f"struct {base}: layout known up to its first unknown member"
        else:
            pm.fields.append((off, s, name))
        off += s
    return pm


# ------------------------------------------------------------------------------------------- allowlist
def parse_allow(texts: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    """{name: reason} from ``kdev:pdl-allow`` lines in sources (and sidecar files, keyed '*.pdl-allow')."""
    allow: dict[str, str] = {}
    warns: list[str] = []
    for path, txt in texts.items():
        side = path.endswith(".pdl-allow")
        for ln in txt.splitlines():
            m = (_SIDE_ALLOW.match(ln) if side and not ln.strip().startswith("#") else None) or \
                (_ALLOW.search(ln) if "kdev:pdl-allow" in ln else None)
            if not m:
                continue
            names = [n.strip() for n in m.group(1).split(",") if n.strip()]
            reason = m.group(2).strip().strip('"\'').rstrip("*/").strip()
            if not reason:
                warns.append(f"pdl-allow {','.join(names)} in {os.path.basename(path)} has no justification: ignored")
                continue
            for n in names:
                allow[n] = reason
    return allow, warns


def match_allow(name: str, allow: dict[str, str]) -> str | None:
    if not allow:
        return None
    base = re.sub(r"\+0x[0-9a-f]+$", "", name)
    for cand in (name, base, base.split(".")[-1]):
        if cand in allow:
            return allow[cand]
    return None


# ------------------------------------------------------------------------------------------- context
@dataclass
class PdlContext:
    sources: dict[str, str] = field(default_factory=dict)    # path -> text (CUDA sources + the submission)
    allow: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _read(path: str, limit: int = 8 << 20) -> str | None:
    try:
        if os.path.isfile(path) and os.path.getsize(path) <= limit:
            with open(path, encoding="utf-8", errors="replace") as f:
                return f.read()
    except OSError:
        return None
    return None


def build_context(target: str | None, kernels: list[Kernel], extra_sources: list[str] | None = None) -> PdlContext:
    """Sources = the target file, its ``.pdl-allow`` sidecar, every user source named in the line info,
    and ``extra_sources``."""
    ctx = PdlContext()
    paths: list[str] = []
    if target:
        paths += [target, target + ".pdl-allow", os.path.splitext(target)[0] + ".pdl-allow"]
    for k in kernels:
        for ins in k.instrs:
            if ins.line and not _toolkit(ins.line):
                p = ins.line.rsplit(":", 1)[0]
                if p not in paths:
                    paths.append(p)
    paths += list(extra_sources or [])
    for p in paths:
        t = _read(p)
        if t is not None:
            ctx.sources[p] = t
    ctx.allow, ctx.warnings = parse_allow(ctx.sources)
    return ctx


def source_line(ctx: PdlContext | None, line: str | None) -> str:
    if not line or ctx is None:
        return ""
    path, _, n = line.rpartition(":")
    txt = ctx.sources.get(path)
    if txt is None or not n.isdigit():
        return ""
    lines = txt.splitlines()
    i = int(n) - 1
    return lines[i].strip()[:100] if 0 <= i < len(lines) else ""


def _is_ptr(pm: ParamMap, off: int) -> bool:
    return any(o == off and s == 8 for o, s, _ in pm.fields)


def attribute(kernel: Kernel, loads: list[Instr], ctx: PdlContext | None) -> list[LoadRoot]:
    """LoadRoot (names, source line, allow verdict) for each of ``loads``."""
    pos = {i.addr: n for n, i in enumerate(kernel.instrs)}
    pbase = param_base(kernel.arch)
    pm = param_map(ctx.sources, kernel.name) if ctx and ctx.sources else ParamMap()
    out = []
    for ld in loads:
        r = slice_load(kernel.instrs, pos[ld.addr], pbase)
        # the base pointers; scalar params (sizes, strides) only feed the index
        offs = [o for o in r.offsets if _is_ptr(pm, o)] if pm.fields else r.pointers
        r.names = [pm.name(o) for o in offs] or [f"?{pm.name(o)}" for o in r.offsets]   # ? = no base pointer found
        if r.via and not r.offsets:
            r.names = ["(loaded pointer)"]
        if not r.names:
            r.names = ["(unresolved)"]
        reasons = [match_allow(n, ctx.allow if ctx else {}) for n in r.names]
        if reasons and all(reasons) and r.names != ["(unresolved)"]:
            r.allowed = "; ".join(dict.fromkeys(x for x in reasons if x))
        out.append(r)
    return out


def summarize(roots: list[LoadRoot], ctx: PdlContext | None, limit: int = 4) -> tuple[str, list[dict[str, Any]]]:
    """('p.seq_lens x6 (cuda.cu:2444 `...`), p.kv_indptr x3', details) grouped by root names."""
    groups: dict[str, list[LoadRoot]] = {}
    for r in roots:
        groups.setdefault(" + ".join(r.names), []).append(r)
    parts, details = [], []
    for name, rs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        ln = next((r.line for r in rs if r.line), None)
        src = source_line(ctx, ln)
        loc = (os.path.basename(ln.rsplit(":", 1)[0]) + ":" + ln.rsplit(":", 1)[1]) if ln else ""
        parts.append(f"{name} x{len(rs)}" + (f" ({loc}" + (f" `{src[:60]}`" if src else "") + ")" if loc else ""))
        details.append({"root": name, "count": len(rs), "addrs": [hex(r.addr) for r in rs[:16]], "line": ln,
                        "source": src, "via_loads": sorted({hex(v) for r in rs for v in r.via})[:8],
                        "allowed": rs[0].allowed})
    s = ", ".join(parts[:limit]) + (f", +{len(parts) - limit} more" if len(parts) > limit else "")
    return s, details
