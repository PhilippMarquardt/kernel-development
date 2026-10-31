"""Opcode classes for histograms. Owner: P5.

Classes (first match wins): mma (HGMMA, HMMA, IMMA, QGMMA, UTCMMA, ...), tensor_mem (UTMALDG, UTMASTG,
UBLKCP, LDSM, STSM), fp32 (FFMA, FADD, FMUL, FMNMX, FSETP, ...), fp16x2 (HFMA2, HADD2, HMUL2, ...),
mufu (MUFU), cvt (F2F, F2I, I2F, F2FP, I2FP), ldg (LDG), stg (STG), lds (LDS), sts (STS), ldl_stl (LDL,
STL = spills/local), ldc (LDC, ULDC), atom (ATOM, ATOMG, ATOMS, RED, REDG), sync (BAR, MEMBAR, ACQBULK,
SYNCS, WARPSYNC, ERRBAR, DEPBAR, FENCE), int (IMAD, IADD3, LOP3, SHF, LEA, ISETP, VIADD, IMNMX, ...),
uniform (U* not matched above), move (MOV, SEL, PRMT, SHFL, S2R, S2UR, CS2R), control (BRA, BRX, EXIT,
RET, CALL, BSSY, BSYNC, NOP, YIELD), other.
"""
from __future__ import annotations

from collections import Counter

from kdev.sass.parse import Instr

CLASSES: tuple[str, ...] = ("mma", "tensor_mem", "fp32", "fp16x2", "mufu", "cvt", "ldg", "stg", "lds", "sts",
                            "ldl_stl", "ldc", "atom", "sync", "int", "uniform", "move", "control", "other")

_SETS: list[tuple[str, frozenset[str]]] = [
    ("mma", frozenset({"HGMMA", "HMMA", "IMMA", "QGMMA", "IGMMA", "BGMMA", "DMMA", "BMMA", "UTCMMA", "UTCHMMA",
                       "UTCQMMA", "UTCIMMA", "UTCOMMA", "OMMA", "QMMA", "HMNMX2_MMA"})),
    ("tensor_mem", frozenset({"UTMALDG", "UTMASTG", "UTMAPF", "UTMACCTL", "UTMACMDFLUSH", "UTMAREDG", "UBLKCP",
                              "UBLKRED", "UBLKPF", "LDSM", "STSM", "UTCBAR", "UTCCP", "UTCLDTM", "LDTM", "STTM",
                              "UTCATOMSWS", "LDGSTS"})),
    ("fp32", frozenset({"FFMA", "FADD", "FMUL", "FMNMX", "FSETP", "FSEL", "FSET", "FCHK", "FRND", "FSWZADD",
                        "FFMA32I", "FADD32I", "FMUL32I", "FMNMX3", "FADD2", "FMUL2", "FFMA2", "FCMP"})),
    ("fp16x2", frozenset({"HFMA2", "HADD2", "HMUL2", "HMNMX2", "HSETP2", "HSET2", "HFMA2_MMA", "HADD2_F32",
                          "HFMA2.MMA"})),
    ("mufu", frozenset({"MUFU"})),
    ("cvt", frozenset({"F2F", "F2I", "I2F", "F2FP", "I2FP", "I2I", "F2IP", "FRND"})),
    ("ldg", frozenset({"LDG", "LD", "ULDG", "LDGMC"})),
    ("stg", frozenset({"STG", "ST"})),
    ("lds", frozenset({"LDS", "ULDS"})),
    ("sts", frozenset({"STS", "USTS"})),
    ("ldl_stl", frozenset({"LDL", "STL"})),
    ("ldc", frozenset({"LDC", "ULDC"})),
    ("atom", frozenset({"ATOM", "ATOMG", "ATOMS", "RED", "REDG", "REDAS", "REDUX", "ATOMSWS"})),
    ("sync", frozenset({"BAR", "MEMBAR", "ACQBULK", "SYNCS", "WARPSYNC", "ERRBAR", "DEPBAR", "FENCE", "LDGDEPBAR",
                        "ARRIVES", "CCTL", "CCTLL", "UCGABAR_ARV", "UCGABAR_WAIT", "WARPGROUP", "UCGABAR", "MEMBAR",
                        "ELECT", "VOTE", "VOTEU", "MATCH", "ENDCOLLECTIVE", "PREEXIT", "ACQSHMINIT", "USETMAXREG",
                        "SETMAXREG", "NANOSLEEP", "B2R", "R2B"})),
    ("int", frozenset({"IMAD", "IADD3", "IADD", "LOP3", "LOP", "SHF", "SHL", "SHR", "LEA", "ISETP", "VIADD",
                       "IMNMX", "VIMNMX", "VIADDMNMX", "VIMNMX3", "IABS", "POPC", "FLO", "BREV", "BMSK", "IDP",
                       "IDP4A", "IMUL", "XMAD", "ISCADD", "PLOP3", "P2R", "R2P", "IADD32I", "LOP32I", "IMAD32I",
                       "ISET", "BFE", "BFI", "IMNMX3", "IADD2", "ICMP"})),
    ("move", frozenset({"MOV", "MOV32I", "SEL", "PRMT", "SHFL", "S2R", "S2UR", "CS2R", "R2UR", "UMOV", "MOVM"})),
    ("control", frozenset({"BRA", "BRX", "JMP", "JMX", "EXIT", "RET", "CALL", "BSSY", "BSYNC", "NOP", "YIELD",
                           "BREAK", "KILL", "BPT", "WARPSYNC.ALL", "BMOV", "RPCMOV", "LEPC", "JMXU", "CALL.ABS"})),
]
_LOOKUP: dict[str, str] = {}
for _cls, _ops in _SETS:
    for _o in _ops:
        _LOOKUP.setdefault(_o, _cls)
_LOOKUP["UMOV"] = "uniform"   # uniform datapath ops (U* not listed above) are 'uniform'


def classify_opcode(base: str) -> str:
    """Class of a base opcode ('LDG', 'HGMMA', 'UIADD3')."""
    c = _LOOKUP.get(base)
    if c is not None:
        return c
    if base.startswith("U") and len(base) > 1:
        return "uniform"
    return "other"


def classify(instr: Instr) -> str:
    return classify_opcode(instr.base)


def histogram(instrs: list[Instr], per: float | None = None) -> dict[str, float]:
    """Counts per class (optionally divided by ``per`` elements, e.g. the weights per loop iteration).
    Keys are the classes present, in :data:`CLASSES` order. NOPs are not counted (padding)."""
    c = Counter(classify(i) for i in instrs if i.base != "NOP")
    div = float(per) if per else 1.0
    return {k: (c[k] / div if per else float(c[k])) for k in CLASSES if c.get(k)}


def opcode_counts(instrs: list[Instr], *, full: bool = False) -> dict[str, int]:
    """Per-opcode counts (base opcodes, or full mnemonics with ``full``), most common first; NOPs dropped."""
    c = Counter((i.opcode if full else i.base) for i in instrs if i.base != "NOP")
    return dict(c.most_common())
