"""SASS text parser. Owner: P5.

Input: cuobjdump -sass text ('Function : <name>' headers, lines '/*0080*/  @P0 LDG.E.CONSTANT R100, desc[..]
;  /* 0x... */'), nvdisasm text ('.text.<name>:' sections, '.L_x_N:' labels, branch targets '`(.L_x_N)',
'//## File "f.cu", line N' line info), or bare instruction lines (split.py output: one unnamed kernel).
Output: :class:`Kernel` with ordered :class:`Instr`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Instr:
    addr: int
    opcode: str            # full mnemonic with modifiers, e.g. 'LDG.E.CONSTANT'
    base: str              # base opcode, e.g. 'LDG'
    operands: list[str] = field(default_factory=list)
    pred: str = ""         # '@P0', '@!P1' or ''
    text: str = ""         # original line (without encoding)
    target: int | None = None   # branch target address for BRA/BRX/JMP/CALL
    line: str | None = None     # source file:line when -lineinfo is available


@dataclass
class Kernel:
    name: str
    instrs: list[Instr] = field(default_factory=list)
    arch: str = ""
    regs: int | None = None


BRANCHES = frozenset({"BRA", "BRX", "JMP", "JMX", "CALL", "RET", "EXIT", "BREAK", "KILL", "BPT"})
_INSTR = re.compile(r"^\s*/\*([0-9a-fA-F]{4,})\*/\s+(.*?)\s*;")
_FUNC = re.compile(r"^\s*Function\s*:\s*(\S+)")
_ARCH = re.compile(r"(?:code for|arch\s*=|\.target)\s*(sm_\w+)")
_TEXT = re.compile(r"^\s*\.text\.([^:\s]+):\s*$")
_SECTION = re.compile(r"^\s*\.section\s+\.text\.([^,\s]+)")
_LABEL = re.compile(r"^\s*(\.L[\w.]+|[A-Za-z_$][\w$.]*):\s*$")
_LINEINFO = re.compile(r'//##\s*File\s+"([^"]*)",\s*line\s+(\d+)')
_HEX = re.compile(r"^-?0x[0-9a-fA-F]+$")
_LABEL_REF = re.compile(r"`\(([^)]+)\)")
_TARGET_HEX = re.compile(r"(?<![\w\[+])0x([0-9a-fA-F]+)\s*$")


def split_operands(s: str) -> list[str]:
    """Split on commas that are not inside [] / {} / ()."""
    out, depth, cur = [], 0, []
    for ch in s:
        if ch in "[{(":
            depth += 1
        elif ch in "]})":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        out.append("".join(cur).strip())
    return out


def parse_instr(addr: int, body: str, line: str | None = None) -> Instr:
    """One instruction from its text (without address and ';')."""
    text = body.strip()
    pred = ""
    rest = text
    m = re.match(r"^(@!?U?P[T\d]+)\s+(.*)$", rest)
    if m:
        pred, rest = m.group(1), m.group(2)
    parts = rest.split(None, 1)
    opcode = parts[0] if parts else ""
    operands = split_operands(parts[1]) if len(parts) > 1 else []
    base = opcode.split(".")[0]
    return Instr(addr=addr, opcode=opcode, base=base, operands=operands, pred=pred, text=text, line=line)


def _branch_target(ins: Instr, labels: dict[str, int]) -> int | None:
    if ins.base not in ("BRA", "JMP", "CALL", "BSSY", "BREAK", "SSY", "PBK", "PCNT", "CAL", "BRK"):
        return None
    if not ins.operands:
        return None
    last = ins.operands[-1].strip()
    m = _LABEL_REF.search(last)
    if m:
        return labels.get(m.group(1))
    m = _TARGET_HEX.search(last)
    if m and not last.startswith(("R", "UR", "c[")):
        return int(m.group(1), 16)
    return None


def parse_sass(text: str) -> list[Kernel]:
    """All kernels in a cuobjdump/nvdisasm SASS listing (in order). A listing without headers yields one
    kernel named '' (the split.py per-kernel files)."""
    kernels: list[Kernel] = []
    cur: Kernel | None = None
    arch = ""
    pending_labels: list[str] = []
    labels: dict[str, int] = {}
    line_info: str | None = None
    unresolved: list[tuple[Kernel, dict[str, int]]] = []

    def start(name: str) -> Kernel:
        nonlocal labels
        labels = {}
        k = Kernel(name=name, arch=arch)
        kernels.append(k)
        unresolved.append((k, labels))
        return k

    for raw in (text or "").splitlines():
        m = _INSTR.match(raw)
        if m:
            if cur is None:
                cur = start("")
            addr = int(m.group(1), 16)
            for lb in pending_labels:
                labels[lb] = addr
            pending_labels = []
            cur.instrs.append(parse_instr(addr, m.group(2), line_info))
            continue
        m = _FUNC.match(raw)
        if m:
            cur = start(m.group(1))
            line_info = None
            continue
        m = _TEXT.match(raw)
        if m:
            name = m.group(1)
            if cur is None or cur.name != name or cur.instrs:
                cur = start(name)
            line_info = None
            continue
        m = _SECTION.match(raw)
        if m:
            pending_labels = []
            continue
        m = _LINEINFO.search(raw)
        if m:
            line_info = f"{m.group(1).replace(chr(92) * 2, chr(92))}:{m.group(2)}"
            continue
        m = _LABEL.match(raw)
        if m:
            pending_labels.append(m.group(1))
            continue
        m = _ARCH.search(raw)
        if m:
            arch = m.group(1)
            if cur is not None and not cur.instrs:
                cur.arch = arch
    for k, lbls in unresolved:
        if not k.arch:
            k.arch = arch
        for ins in k.instrs:
            ins.target = _branch_target(ins, lbls)
    return [k for k in kernels if k.instrs]


_REG = re.compile(r"^(-|!|~|\|)?(U?R\d+|URZ|RZ|U?P\d+|UPT|PT|SR_\w+|SRZ|B\d+)")


def _norm_operand(op: str) -> str:
    o = op.replace(".reuse", "").strip()
    o = re.sub(r"c\[0x[0-9a-fA-F]+\]\s*\[[^\]]*\]", "c[]", o)
    o = re.sub(r"desc\[[^\]]*\]", "desc[]", o)
    o = re.sub(r"\bUR\d+\b", "UR", o)
    o = re.sub(r"\bR\d+\b", "R", o)
    o = re.sub(r"\bUP\d\b", "UP", o)
    o = re.sub(r"\bP\d\b", "P", o)
    o = re.sub(r"\bB\d+\b", "B", o)
    o = re.sub(r"-?0x[0-9a-fA-F]+|(?<![\w.])-?\d+(?:\.\d+)?(?:e[+-]?\d+)?(?![\w])", "imm", o)
    return o


def normalize(instr: Instr) -> str:
    """Register-agnostic form for repeated-work detection: opcode + operand shapes (R12 -> R, 0x10 -> imm,
    c[0x0][0x218] -> c[]), predicate kept as presence only."""
    p = "@P " if instr.pred else ""
    return p + instr.opcode + " " + ", ".join(_norm_operand(o) for o in instr.operands)


def by_name(kernels: list[Kernel], regex: str | None) -> list[Kernel]:
    """Kernels whose name matches ``regex`` (search); all when regex is None/empty."""
    if not regex:
        return kernels
    rx = re.compile(regex)
    return [k for k in kernels if rx.search(k.name)]
