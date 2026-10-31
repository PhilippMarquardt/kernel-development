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

from kdev.sass.parse import Instr

CLASSES: tuple[str, ...] = ("mma", "tensor_mem", "fp32", "fp16x2", "mufu", "cvt", "ldg", "stg", "lds", "sts",
                            "ldl_stl", "ldc", "atom", "sync", "int", "uniform", "move", "control", "other")


def classify(instr: Instr) -> str:
    raise NotImplementedError("classes.classify (P5)")


def histogram(instrs: list[Instr], per: float | None = None) -> dict[str, float]:
    """Counts per class (optionally divided by ``per`` elements, e.g. the weights per loop iteration)."""
    raise NotImplementedError("classes.histogram (P5)")
