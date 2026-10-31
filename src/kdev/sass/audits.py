"""SASS audits. Owner: P5. Each returns a list of :class:`Finding` (empty = pass).

- pdl: in kernels containing ACQBULK (griddepcontrol.wait), any global load (LDG*, LD.E*, ULDG, UTMALDG,
  LDGSTS) reachable on a path from entry before the first ACQBULK (M9; LESSONS 11). Severity error.
- spills: LDL/STL anywhere (warn) and inside loops (error), plus ptxas spill bytes.
- serialize: ptxas C7510-C7515 warnings for the kernel (error: wgmma serialised).
- expect: classes/opcodes that should be present (e.g. HGMMA) but are absent -> DCE suspicion (M15).
- repeat: repeated-work detector: normalised k-gram windows (k >= 4) that occur >= 2 times inside one hot-loop
  body with identical operand shapes, ranked by (k x occurrences); reports address ranges (redundant
  decode / address math recomputed per element).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from kdev.models import KernelInfo
from kdev.sass.parse import Kernel

AUDITS = ("pdl", "spills", "serialize", "expect", "repeat")


@dataclass
class Finding:
    audit: str
    severity: str           # info | warn | error
    kernel: str
    message: str
    addrs: list[int] = field(default_factory=list)
    excerpt: list[str] = field(default_factory=list)   # <= 8 SASS lines


def audit_pdl(kernel: Kernel) -> list[Finding]:
    raise NotImplementedError("audits.audit_pdl (P5)")


def audit_spills(kernel: Kernel, info: KernelInfo | None = None) -> list[Finding]:
    raise NotImplementedError("audits.audit_spills (P5)")


def audit_serialize(kernel_name: str, ptxas_text: str) -> list[Finding]:
    raise NotImplementedError("audits.audit_serialize (P5)")


def audit_expect(kernel: Kernel, expected: list[str]) -> list[Finding]:
    raise NotImplementedError("audits.audit_expect (P5)")


def audit_repeat(kernel: Kernel, k_min: int = 4, min_count: int = 2) -> list[Finding]:
    raise NotImplementedError("audits.audit_repeat (P5)")


def run_audits(kernel: Kernel, names: list[str], *, info: KernelInfo | None = None, ptxas_text: str = "",
               expected: list[str] | None = None) -> list[Finding]:
    raise NotImplementedError("audits.run_audits (P5)")
