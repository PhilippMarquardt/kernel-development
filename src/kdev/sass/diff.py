"""Build-vs-build SASS diff. Owner: P5.

Matches kernels by name (exact, then with the per-build anonymous-namespace hashes ``_GLOBAL__N__<hash>`` /
``_INTERNAL_<hash>`` stripped), fallback: closest class histogram. Reports per kernel: instr count, regs,
class histogram delta, hot-loop size/mix delta, and audit findings that appear/disappear.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from kdev.sass.audits import run_audits
from kdev.sass.classes import histogram
from kdev.sass.loops import hot_loops
from kdev.sass.parse import Kernel, by_name

DIFF_AUDITS = ("pdl", "spills", "repeat", "invariant")


@dataclass
class KernelDiff:
    name_a: str
    name_b: str
    instr_a: int
    instr_b: int
    regs_a: int | None = None
    regs_b: int | None = None
    class_delta: dict[str, float] = field(default_factory=dict)
    loop_delta: dict[str, float] = field(default_factory=dict)
    findings_new: list[str] = field(default_factory=list)
    findings_gone: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def stable_name(name: str) -> str:
    """Kernel name without per-translation-unit hashes (they change with every source edit)."""
    n = re.sub(r"_GLOBAL__N__[0-9a-f]+_\d+_\w+?_cu_[0-9a-f]{8}", "_GLOBAL__N_", name)
    n = re.sub(r"_INTERNAL_[0-9a-f]+_\d+_\w+?_cu_[0-9a-f]{8}", "_INTERNAL_", n)
    n = re.sub(r"\d+(_GLOBAL__N_|_INTERNAL_)", r"\1", n)
    return n


def _count(k: Kernel) -> int:
    return sum(1 for i in k.instrs if i.base != "NOP")


def _l1(a: dict[str, float], b: dict[str, float]) -> float:
    return sum(abs(a.get(c, 0.0) - b.get(c, 0.0)) for c in set(a) | set(b))


def match_kernels(a: list[Kernel], b: list[Kernel]) -> list[tuple[Kernel | None, Kernel | None]]:
    pairs: list[tuple[Kernel | None, Kernel | None]] = []
    rest_b = list(b)
    rest_a: list[Kernel] = []
    for ka in a:
        hit = next((kb for kb in rest_b if kb.name == ka.name), None) or \
            next((kb for kb in rest_b if stable_name(kb.name) == stable_name(ka.name)), None)
        if hit is not None:
            pairs.append((ka, hit))
            rest_b.remove(hit)
        else:
            rest_a.append(ka)
    for ka in rest_a:
        if not rest_b:
            pairs.append((ka, None))
            continue
        ha = histogram(ka.instrs)
        kb = min(rest_b, key=lambda k: _l1(ha, histogram(k.instrs)))
        pairs.append((ka, kb))
        rest_b.remove(kb)
    pairs += [(None, kb) for kb in rest_b]
    return pairs


def _loop_summary(k: Kernel) -> dict[str, float]:
    h = hot_loops(k, top=1)
    if not h:
        return {}
    return {"n_instr": float(h[0].loop.n_instr), **h[0].histogram}


def diff_kernels(a: list[Kernel], b: list[Kernel], kernel_regex: str | None = None) -> list[KernelDiff]:
    out: list[KernelDiff] = []
    for ka, kb in match_kernels(by_name(a, kernel_regex), by_name(b, kernel_regex)):
        ha = histogram(ka.instrs) if ka else {}
        hb = histogram(kb.instrs) if kb else {}
        cd = {c: hb.get(c, 0.0) - ha.get(c, 0.0) for c in dict.fromkeys([*ha, *hb])
              if hb.get(c, 0.0) != ha.get(c, 0.0)}
        la = _loop_summary(ka) if ka else {}
        lb = _loop_summary(kb) if kb else {}
        ld = {c: lb.get(c, 0.0) - la.get(c, 0.0) for c in dict.fromkeys([*la, *lb])
              if lb.get(c, 0.0) != la.get(c, 0.0)}
        fa = {f.key(): f.message for f in run_audits(ka, list(DIFF_AUDITS)) if f.severity != "info"} if ka else {}
        fb = {f.key(): f.message for f in run_audits(kb, list(DIFF_AUDITS)) if f.severity != "info"} if kb else {}
        out.append(KernelDiff(name_a=ka.name if ka else "", name_b=kb.name if kb else "",
                              instr_a=_count(ka) if ka else 0, instr_b=_count(kb) if kb else 0,
                              regs_a=ka.regs if ka else None, regs_b=kb.regs if kb else None,
                              class_delta=cd, loop_delta=ld,
                              findings_new=[fb[k] for k in fb if k not in fa],
                              findings_gone=[fa[k] for k in fa if k not in fb]))
    return out
