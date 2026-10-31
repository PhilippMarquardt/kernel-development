"""SASS audits. Owner: P5. Each returns a list of :class:`Finding` (empty = pass).

- pdl: in kernels containing ACQBULK (griddepcontrol.wait), any global load (LDG*, LD.E*, ULDG, UTMALDG,
  LDGSTS) reachable on a path from entry before the first ACQBULK (M9; LESSONS 11). Severity error; loads on a
  path that never reaches ACQBULK are a warn. Each load is attributed to the kernel parameter (struct member)
  its address comes from (``_pdlsrc``); loads whose roots are all on the file's ``kdev:pdl-allow NAME -- why``
  list are info.
- spills: LDL/STL anywhere (warn) and inside loops (error), plus ptxas spill bytes.
- serialize: ptxas C7510-C7515 warnings for the kernel (error: wgmma serialised).
- expect: classes/opcodes that should be present (e.g. HGMMA) but are absent -> DCE suspicion (M15).
- repeat: repeated-work detector inside hot loops: (a) local value numbering finds pure instructions that
  recompute a value already computed in the same iteration (same opcode + same input values: redundant
  decode / address math, the KPERM 'scale-pair decode repeated 4x per k16 group' pattern) -> warn; (b)
  normalised k-gram windows (k >= 4) that occur >= 2 times in one loop body (unrolled copies; shows the
  per-element instruction cost) -> info, ranked by (k x occurrences).
- invariant: uniform/loop-invariant recompute: pure instructions in a loop whose inputs are all defined outside
  the loop (hoistable per-iteration work, e.g. U* address math recomputed every trip) -> info/warn.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field

from kdev.build.ptxas import serialization_warnings
from kdev.models import KernelInfo
from kdev.sass import _dataflow as df
from kdev.sass.cfg import build_cfg, can_reach, find_loops, loop_instrs, reachable_avoiding
from kdev.sass.classes import CLASSES, classify
from kdev.sass.parse import Instr, Kernel, normalize

AUDITS = ("pdl", "spills", "serialize", "expect", "repeat", "invariant")
DEFAULT_AUDITS = ("pdl", "spills", "serialize", "repeat", "invariant")

GLOBAL_LOADS = frozenset({"LDG", "LD", "ULDG", "UTMALDG", "LDGSTS", "UBLKCP", "LDGMC", "UTMAPF", "ATOMG", "ATOM"})
_SER_FN = re.compile(r"in the function '([^']+)'")


@dataclass
class Finding:
    audit: str
    severity: str           # info | warn | error
    kernel: str
    message: str
    addrs: list[int] = field(default_factory=list)
    excerpt: list[str] = field(default_factory=list)   # <= 8 SASS lines
    details: list[dict] = field(default_factory=list)   # audit-specific (pdl: per-root groups)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["addrs"] = [hex(a) for a in self.addrs[:32]]
        return d

    def key(self) -> str:
        """Address-free identity (for build-vs-build diffs)."""
        msg = self.message
        if self.audit == "pdl":           # the attribution part (root names, lines) is not identity
            msg = re.split(r" read |: ", msg)[0]
        return f"{self.audit}:{self.severity}:" + re.sub(r"0x[0-9a-f]+|\d+", "#", msg)


def _ex(instrs: list[Instr], n: int = 8) -> list[str]:
    return [f"/*{i.addr:04x}*/ {i.text}" + (f"   // {i.line}" if i.line else "") for i in instrs[:n]]


def audit_pdl(kernel: Kernel, ctx: object | None = None) -> list[Finding]:
    """Pre-wait global loads, attributed to the kernel parameter they read (``_pdlsrc``). Loads whose every
    root is on the file's ``kdev:pdl-allow`` list are reported as info with the justification."""
    from kdev.sass import _pdlsrc  # noqa: PLC0415

    acq = {i.addr for i in kernel.instrs if i.base == "ACQBULK"}
    if not acq:
        return []
    cfg = build_cfg(kernel)
    before = reachable_avoiding(kernel, cfg, acq)
    reach = can_reach(kernel, cfg, acq)
    loads = [i for i in kernel.instrs if i.base in GLOBAL_LOADS and i.addr in before]
    hoisted = [i for i in loads if i.addr in reach]
    never = [i for i in loads if i.addr not in reach]
    out: list[Finding] = []
    first = min(acq)
    if hoisted:
        roots = _pdlsrc.attribute(kernel, hoisted, ctx)  # type: ignore[arg-type]
        bad = [i for i, r in zip(hoisted, roots) if not r.allowed]
        ok = [r for r in roots if r.allowed]
        if bad:
            rb = [r for r in roots if not r.allowed]
            what, det = _pdlsrc.summarize(rb, ctx)  # type: ignore[arg-type]
            out.append(Finding("pdl", "error", kernel.name,
                               f"{len(bad)} global load(s) before griddepcontrol.wait (ACQBULK @0x{first:04x}) read "
                               f"{what}: producer data read before the wait races the previous kernel (LESSONS 11); "
                               f"move it after the wait, or allowlist a read-only input with "
                               f"`// kdev:pdl-allow NAME -- why`",
                               [i.addr for i in bad], _ex(bad), det))
        if ok:
            what, det = _pdlsrc.summarize(ok, ctx)  # type: ignore[arg-type]
            reasons = "; ".join(dict.fromkeys(r.allowed or "" for r in ok))
            out.append(Finding("pdl", "info", kernel.name,
                               f"{len(ok)} pre-wait load(s) allowlisted: {what} ({reasons[:120]})",
                               [r.addr for r in ok], [], det))
    if never:
        roots = _pdlsrc.attribute(kernel, never, ctx)  # type: ignore[arg-type]
        what, det = _pdlsrc.summarize(roots, ctx)  # type: ignore[arg-type]
        out.append(Finding("pdl", "warn", kernel.name,
                           f"{len(never)} global load(s) on paths that exit without griddepcontrol.wait: {what}",
                           [i.addr for i in never], _ex(never), det))
    return out


def audit_spills(kernel: Kernel, info: KernelInfo | None = None) -> list[Finding]:
    out: list[Finding] = []
    local = [i for i in kernel.instrs if i.base in ("LDL", "STL")]
    cfg = build_cfg(kernel)
    seen: set[int] = set()
    loops = sorted(find_loops(kernel, cfg) if local else [], key=lambda lp: (len(lp.blocks), lp.head))
    for lp in loops:   # innermost (smallest) first: each LDL/STL is reported once, in its innermost loop
        head = lp.head
        body = [i for i in loop_instrs(kernel, cfg, lp) if i.base in ("LDL", "STL")]
        new = [i for i in body if i.addr not in seen]
        seen.update(i.addr for i in body)
        if not new:
            continue
        c = Counter(i.base for i in new)
        out.append(Finding("spills", "error", kernel.name,
                           f"local memory in loop @0x{head:04x}: {c.get('LDL', 0)} LDL + {c.get('STL', 0)} STL per "
                           f"iteration (spills or a dynamically indexed register array)",
                           [i.addr for i in new], _ex(new)))
    rest = [i for i in local if i.addr not in seen]
    if rest:
        c = Counter(i.base for i in rest)
        out.append(Finding("spills", "warn", kernel.name,
                           f"local memory outside loops: {c.get('LDL', 0)} LDL + {c.get('STL', 0)} STL",
                           [i.addr for i in rest], _ex(rest)))
    if info is not None and (info.spill_store or info.spill_load):
        out.append(Finding("spills", "warn", kernel.name,
                           f"ptxas: {info.spill_store} bytes spill stores, {info.spill_load} bytes spill loads, "
                           f"{info.stack} bytes stack (regs {info.regs})"))
    return out


def audit_serialize(kernel_name: str, ptxas_text: str) -> list[Finding]:
    """C751x messages naming ``kernel_name`` (exact mangled name, or substring), or all when the name is ''."""
    out: list[Finding] = []
    lines = [ln for ln in (ptxas_text or "").splitlines() if re.search(r"\(C751\d\)", ln)]
    for code, msg in serialization_warnings("\n".join(lines)):
        fn = _SER_FN.search(msg)
        name = fn.group(1) if fn else ""
        if kernel_name and name and kernel_name != name and kernel_name not in name and name not in kernel_name:
            continue
        out.append(Finding("serialize", "error", kernel_name or name,
                           f"ptxas {code}: {msg[:300]}"))
    return out


def _expect_present(kernel: Kernel, item: str) -> bool:
    it = item.strip()
    if it.lower() in CLASSES:
        return any(classify(i) == it.lower() for i in kernel.instrs)
    up = it.upper()
    return any(i.base == up or i.opcode == up or i.opcode.startswith(up + ".") for i in kernel.instrs)


def audit_expect(kernel: Kernel, expected: list[str]) -> list[Finding]:
    missing = [e for e in expected if e and not _expect_present(kernel, e)]
    if not missing:
        return []
    return [Finding("expect", "error", kernel.name,
                    f"expected {', '.join(missing)} absent from SASS: dead-code eliminated or wrong path "
                    f"(M15: check the benchmark really computes)")]


def _structural_repeats(body: list[Instr], k_min: int, min_count: int, k_max: int = 32,
                        max_report: int = 3) -> list[tuple[int, int, list[int]]]:
    """[(k, count, start indices)] of repeated normalised windows, chosen greedily by k x count over all
    window sizes, non-overlapping with each other."""
    seq = [normalize(i) for i in body]
    bases = [i.base for i in body]
    n = len(seq)
    cands: list[tuple[int, int, list[int]]] = []
    for k in range(k_min, min(k_max, n // max(min_count, 1)) + 1):
        pos: dict[tuple, list[int]] = {}
        for s in range(0, n - k + 1):
            pos.setdefault(tuple(seq[s:s + k]), []).append(s)
        for starts in pos.values():
            if len(starts) < min_count or len(set(bases[starts[0]:starts[0] + k])) < 2:
                continue
            chosen, last = [], -k
            for s in starts:
                if s >= last + k:
                    chosen.append(s)
                    last = s
            if len(chosen) >= min_count:
                cands.append((k * len(chosen), k, chosen))
    covered = [False] * n
    found: list[tuple[int, int, list[int]]] = []
    for _, k, chosen in sorted(cands, key=lambda c: (-c[0], -c[1], c[2][0])):
        free = [s for s in chosen if not any(covered[s:s + k])]
        if len(free) < min_count or k * len(free) < k * len(chosen) // 2:
            continue
        for s in free:
            for j in range(s, s + k):
                covered[j] = True
        found.append((k, len(free), free))
        if len(found) >= max_report:
            break
    return found


def _loops_for_repeat(kernel: Kernel, max_loops: int = 3):
    from kdev.sass.loops import all_loops  # noqa: PLC0415

    return [h for h in all_loops(kernel) if h.innermost][:max_loops]


def audit_repeat(kernel: Kernel, k_min: int = 4, min_count: int = 2) -> list[Finding]:
    out: list[Finding] = []
    for h in _loops_for_repeat(kernel):
        body = [i for i in h.instrs if i.base != "NOP"]
        dups = df.value_numbers(body)
        if dups:
            c = Counter(classify(d) for d, _ in dups)
            pct = 100.0 * len(dups) / max(1, len(body))
            sev = "warn" if (len(dups) >= 4 or pct >= 2.0) else "info"
            pairs: list[Instr] = []
            for d, f in dups[:4]:
                pairs += [f, d]
            mix = ", ".join(f"{k} {v}" for k, v in c.most_common(6))
            out.append(Finding("repeat", sev, kernel.name,
                               f"loop @0x{h.loop.head:04x}: {len(dups)} instr(s) ({pct:.1f}% of {len(body)}) recompute "
                               f"a value already computed in the same iteration ({mix}); CSE/hoist them "
                               f"(redundant decode or address math)",
                               [d.addr for d, _ in dups], _ex(pairs)))
        for k, cnt, starts in _structural_repeats(body, k_min, min_count):
            w = body[starts[0]:starts[0] + k]
            hist = Counter(classify(i) for i in w)
            mix = ", ".join(f"{a} {b}" for a, b in hist.most_common(5))
            out.append(Finding("repeat", "info", kernel.name,
                               f"loop @0x{h.loop.head:04x}: a {k}-instr window repeats {cnt}x (unrolled copies; "
                               f"{k * cnt}/{len(body)} instrs; per copy: {mix})",
                               [body[s].addr for s in starts], _ex(w)))
    return out


def audit_invariant(kernel: Kernel, min_count: int = 2) -> list[Finding]:
    out: list[Finding] = []
    for h in _loops_for_repeat(kernel):
        body = [i for i in h.instrs if i.base != "NOP"]
        inv = df.loop_invariants(body)
        if len(inv) < min_count:
            continue
        c = Counter(classify(i) for i in inv)
        uni = sum(1 for i in inv if i.base.startswith("U"))
        sev = "warn" if len(inv) >= max(4, len(body) // 20) else "info"
        mix = ", ".join(f"{k} {v}" for k, v in c.most_common(6))
        out.append(Finding("invariant", sev, kernel.name,
                           f"loop @0x{h.loop.head:04x}: {len(inv)} loop-invariant instr(s) recomputed every "
                           f"iteration ({uni} uniform; {mix}); hoist unless it is deliberate rematerialisation",
                           [i.addr for i in inv], _ex(inv)))
    return out


def parse_audit_names(spec: str | list[str] | None) -> tuple[list[str], list[str]]:
    """'pdl,spills,expect=HGMMA|LDS' -> (['pdl', 'spills', 'expect'], ['HGMMA', 'LDS']). 'all' = every audit."""
    if spec is None:
        return list(DEFAULT_AUDITS), []
    items = spec.split(",") if isinstance(spec, str) else list(spec)
    names: list[str] = []
    expected: list[str] = []
    for it in (x.strip() for x in items):
        if not it:
            continue
        if it == "all":
            names += [a for a in AUDITS if a != "expect"]
            continue
        if it.startswith("expect"):
            _, _, val = it.partition("=")
            expected += [v for v in re.split(r"[|+/ ]", val) if v]
            names.append("expect")
            continue
        if it not in AUDITS:
            from kdev.contract import ErrorCode, KdevError  # noqa: PLC0415

            raise KdevError(ErrorCode.INVALID_REQUEST, f"unknown audit {it!r}",
                            recovery=f"use a comma list of {', '.join(AUDITS)} (expect=OPCODE)")
        names.append(it)
    return list(dict.fromkeys(names)), expected


def run_audits(kernel: Kernel, names: list[str], *, info: KernelInfo | None = None, ptxas_text: str = "",
               expected: list[str] | None = None, pdl_ctx: object | None = None) -> list[Finding]:
    names, exp2 = parse_audit_names(list(names))
    expected = list(expected or []) + exp2
    out: list[Finding] = []
    for n in names:
        if n == "pdl":
            out += audit_pdl(kernel, pdl_ctx)
        elif n == "spills":
            out += audit_spills(kernel, info)
        elif n == "serialize":
            if ptxas_text:
                out += audit_serialize(kernel.name, ptxas_text)
            elif info is not None:
                out += [Finding("serialize", "error", kernel.name, f"ptxas {w.strip()[:300]}")
                        for w in info.warnings if re.search(r"\(C751\d\)", w)]
        elif n == "expect":
            out += audit_expect(kernel, expected)
        elif n == "repeat":
            out += audit_repeat(kernel)
        elif n == "invariant":
            out += audit_invariant(kernel)
    rank = {"error": 0, "warn": 1, "info": 2}
    return sorted(out, key=lambda f: (rank.get(f.severity, 3), f.audit))
