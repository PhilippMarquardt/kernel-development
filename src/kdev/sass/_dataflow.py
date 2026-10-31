"""Approximate SASS register def/use and local value numbering (P5, private).

Good enough for the repeated-work / loop-invariant audits, not a verifier: widths come from opcode
modifiers (.64/.128/.WIDE/LDSM .x4/HMMA), the first operand is the destination unless the opcode has none,
predicate operands right after it are destinations too (ISETP P0, PT, ...; IADD3 R0, P0, ...).
"""
from __future__ import annotations

import re

from kdev.sass.classes import classify
from kdev.sass.parse import Instr

_NO_DEST = frozenset({"STG", "STS", "STL", "ST", "RED", "REDG", "REDAS", "BAR", "BRA", "BRX", "JMP", "JMX", "EXIT",
                      "RET", "CALL", "KILL", "BPT", "MEMBAR", "ACQBULK", "NOP", "BSSY", "BSYNC", "WARPSYNC",
                      "ERRBAR", "DEPBAR", "FENCE", "CCTL", "CCTLL", "UTMALDG", "UTMASTG", "UTMAPF", "UTMACCTL",
                      "UTMACMDFLUSH", "UTMAREDG", "UBLKCP", "UBLKRED", "UBLKPF", "STSM", "LDGSTS", "ARRIVES",
                      "PREEXIT", "YIELD", "LDGDEPBAR", "UTCBAR", "UTCCP", "BREAK", "NANOSLEEP", "USETMAXREG",
                      "SETMAXREG", "ENDCOLLECTIVE", "STTM", "WARPGROUP", "UCGABAR", "UCGABAR_ARV",
                      "UCGABAR_WAIT", "ACQSHMINIT", "BMOV"})
_IMPURE = frozenset({"S2R", "S2UR", "CS2R", "SHFL", "VOTE", "VOTEU", "MATCH", "ELECT", "REDUX", "ATOM", "ATOMG",
                     "ATOMS", "LDL", "SYNCS", "B2R", "R2B", "LEPC", "P2R", "R2P"})
_PURE_S2R = re.compile(r"SR_(TID|CTAID|LANEID|NTID|NCTAID|SWINHI|CgaCtaId|GRIDID|VIRTID|SMID)", re.I)
_REG_TOKEN = re.compile(r"\b(UR\d+|R\d+|UP\d|P\d)(\.(?:64|128))?\b")
COPY_OPS = frozenset({"MOV", "UMOV", "MOV32I"})


def width(instr: Instr) -> int:
    m = instr.opcode
    if instr.base in ("HMMA",):
        return 4 if ".F32" in m else 2
    if instr.base == "LDSM":
        return 4 if (".4" in m or ".X4" in m.upper()) else (2 if (".2" in m or ".X2" in m.upper()) else 1)
    if ".128" in m:
        return 4
    if ".64" in m or ".WIDE" in m or instr.base in ("DADD", "DMUL", "DFMA", "CS2R") and ".32" not in m:
        return 2
    return 1


def _expand(tok: str, suffix: str | None, w: int = 1) -> list[str]:
    n = 2 if suffix == ".64" else (4 if suffix == ".128" else w)
    m = re.match(r"(U?R)(\d+)$", tok)
    if not m or n == 1:
        return [tok]
    return [f"{m.group(1)}{int(m.group(2)) + j}" for j in range(n)]


def regs_in(op: str) -> list[str]:
    out: list[str] = []
    for tok, suf in _REG_TOKEN.findall(op.replace(".reuse", "")):
        out += _expand(tok, suf or None)
    return out


def is_pred(op: str) -> bool:
    return bool(re.fullmatch(r"!?U?P(\d|T)", op.strip()))


def split_defs(instr: Instr) -> tuple[int, list[str], list[str]]:
    """(number of leading destination operands, defined registers, used registers)."""
    ops = instr.operands
    defs: list[str] = []
    uses: list[str] = []
    k0 = 0
    if classify(instr) == "mma" and ops:   # accumulators are read-modify-write
        defs += regs_in(ops[0])
    elif instr.base not in _NO_DEST and ops:
        m = _REG_TOKEN.fullmatch(ops[0].replace(".reuse", "").strip())
        if m:
            defs += _expand(m.group(1), m.group(2), width(instr))
            k0 = 1
            while k0 < min(len(ops), 3) and len(ops) - k0 > 1 and is_pred(ops[k0]) and                     not ops[k0].strip().startswith("!"):
                defs.append(ops[k0].strip())
                k0 += 1
    for op in ops[k0:]:
        uses += regs_in(op)
    if instr.pred:
        uses += regs_in(instr.pred.lstrip("@!"))
    bad = ("RZ", "URZ", "PT", "UPT")
    return k0, [r for r in defs if r not in bad], [r for r in uses if r not in bad]


def defs_uses(instr: Instr) -> tuple[list[str], list[str]]:
    """(defined registers, used registers). RZ/URZ/PT/UPT are never reported."""
    _, d, u = split_defs(instr)
    return d, u


def is_pure(instr: Instr) -> bool:
    """No side effects and deterministic given its register inputs (+ memory epoch for loads)."""
    cls = classify(instr)
    if cls in ("mma", "tensor_mem", "stg", "sts", "atom", "sync", "control", "ldl_stl"):
        return False
    if instr.base in _NO_DEST:
        return False
    if instr.base in ("S2R", "S2UR", "CS2R"):
        return bool(_PURE_S2R.search(" ".join(instr.operands)))
    if instr.base in _IMPURE:
        return False
    return True


def mem_kind(instr: Instr) -> str:
    """'' for non-loads, 'const' (LDC/ULDC/LDG.CONSTANT), 'shared' (LDS), 'global' (other LDG/LD)."""
    if instr.base in ("LDC", "ULDC") or (instr.base in ("LDG", "ULDG") and ".CONSTANT" in instr.opcode):
        return "const"
    if instr.base in ("LDS", "ULDS", "LDSM"):
        return "shared"
    if instr.base in ("LDG", "LD", "ULDG", "LDGMC"):
        return "global"
    return ""


def _shape(op: str, vn: dict[str, int]) -> str:
    o = op.replace(".reuse", "")

    def sub(m: re.Match) -> str:
        regs = _expand(m.group(1), m.group(2) or None)
        return "<" + ",".join(f"v{vn.get(r, r)}" for r in regs) + ">"

    return _REG_TOKEN.sub(sub, o)


def value_numbers(body: list[Instr]) -> list[tuple[Instr, Instr]]:
    """Local value numbering over ``body`` in order. Returns (duplicate, first) pairs: pure instructions
    with at least one register input that recompute a value already computed earlier in the body (same
    opcode, same input values, same immediates, same memory epoch for loads). Copies propagate values."""
    vn: dict[str, int] = {}
    nxt = [0]

    def fresh() -> int:
        nxt[0] += 1
        return nxt[0]

    def val(r: str) -> int:
        if r not in vn:
            vn[r] = fresh()
        return vn[r]

    table: dict[tuple, tuple[list[int], Instr]] = {}
    epoch = {"shared": 0, "global": 0, "const": 0}
    dups: list[tuple[Instr, Instr]] = []
    for ins in body:
        if ins.base == "NOP":
            continue
        k0, defs, uses = split_defs(ins)
        cls = classify(ins)
        if cls in ("sts", "sync", "atom") or ins.base in ("LDGSTS", "STSM", "UBLKCP", "UTMALDG"):
            epoch["shared"] += 1
        if cls in ("stg", "atom") or ins.base in ("MEMBAR", "ACQBULK", "FENCE"):
            epoch["global"] += 1
        if ins.base in COPY_OPS and not ins.pred and len(ins.operands) == 2 and defs and \
                len(regs_in(ins.operands[1])) == 1:
            vn[defs[0]] = val(regs_in(ins.operands[1])[0])
            continue
        if ins.base == "IMAD" and ".MOV" in ins.opcode and not ins.pred and len(ins.operands) == 4 and defs and \
                ins.operands[1] == "RZ" and ins.operands[2] == "RZ" and len(regs_in(ins.operands[3])) == 1:
            vn[defs[0]] = val(regs_in(ins.operands[3])[0])
            continue
        if not is_pure(ins) or not defs:
            for r in defs:
                vn[r] = fresh()
            continue
        for r in uses:
            val(r)
        mk = mem_kind(ins)
        key = (ins.opcode, ins.pred and _shape(ins.pred, vn),
               tuple(_shape(o, vn) for o in ins.operands[k0:]),
               epoch.get(mk, 0) if mk else -1,
               tuple(vn.get(r, 0) for r in defs) if ins.pred else ())
        hit = table.get(key)
        if hit is not None and uses:
            dups.append((ins, hit[1]))
            for r, v in zip(defs, hit[0]):
                vn[r] = v
            continue
        vals = [fresh() for _ in defs]
        for r, v in zip(defs, vals):
            vn[r] = v
        if hit is None:
            table[key] = (vals, ins)
    return dups


def loop_invariants(body: list[Instr]) -> list[Instr]:
    """Pure, unpredicated instructions of a loop body whose register inputs are all loop-invariant (never
    defined in the body, or defined once earlier in the body by an invariant instruction). Loads only from
    the constant bank / LDG.CONSTANT count. Constants without register inputs are skipped (remat)."""
    defs_at: dict[str, list[int]] = {}
    du = [defs_uses(i) for i in body]
    for k, (d, _) in enumerate(du):
        for r in d:
            defs_at.setdefault(r, []).append(k)
    inv: set[int] = set()
    out: list[Instr] = []
    for k, ins in enumerate(body):
        d, u = du[k]
        if ins.base == "NOP" or ins.pred or not d or not u or not is_pure(ins):
            continue
        mk = mem_kind(ins)
        if mk and mk != "const":
            continue
        ok = True
        for r in u:
            dk = defs_at.get(r, [])
            if not dk:
                continue
            if len(dk) == 1 and dk[0] in inv and dk[0] < k:
                continue
            ok = False
            break
        if ok and all(len(defs_at.get(r, [])) == 1 for r in d):
            inv.add(k)
            out.append(ins)
    return out
