"""SASS text parser. Owner: P5.

Input: cuobjdump -sass text ('Function : <name>' headers, lines '/*0080*/  @P0 LDG.E.CONSTANT R100, desc[..]
;  /* 0x... */'). Output: :class:`Kernel` with ordered :class:`Instr`.
"""
from __future__ import annotations

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


def parse_sass(text: str) -> list[Kernel]:
    raise NotImplementedError("parse.parse_sass (P5)")


def normalize(instr: Instr) -> str:
    """Register-agnostic form for repeated-work detection: opcode + operand shapes (R12 -> R, 0x10 -> imm,
    c[0x0][0x218] -> c[]), predicate kept as presence only."""
    raise NotImplementedError("parse.normalize (P5)")
