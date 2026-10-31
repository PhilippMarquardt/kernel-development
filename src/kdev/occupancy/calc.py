"""Occupancy calculator (pure). Owner: P6.

Inputs from ptxas (kdev.build.ptxas.parse_ptxas) / cuobjdump -res-usage: regs, smem static, dynamic smem,
block size, launch bounds, setmaxnreg groups. Output per kernel: CTAs/SM by each limiter (regs, smem,
warps, CTA slots), the binding limiter, warps/SMSP, and the register rule e.g. "5 warps/SMSP caps regs at
96" (regs per thread <= 65536 / (4 * warps_per_smsp * 32) rounded down to the allocation granule of 8).

Register model (Hopper): registers are allocated per warp in units of 8 regs/thread (256 per warp) out of
each SM sub-partition's (SMSP) quarter of the register file; a CTA's warps are spread over the 4 SMSPs, so
``n`` CTAs of ``w`` warps fit when ceil(n * w / 4) warps per SMSP fit the quarter.

setmaxnreg model (PTX ISA: ``setmaxnreg.inc`` blocks until the CTA's register pool can satisfy it): the pool
is the CTA's launch allocation (threads x launch regs); after all groups' setmaxnreg the sum of threads x
regs must fit that pool, otherwise the increasing warps wait forever (a GPU hang, measured on H200: 640
threads at 96 launch regs -> producer 24 / consumers 112 max; 120 hangs).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from kdev.floor.gpus import GpuProfile


@dataclass
class OccupancyReport:
    kernel: str
    regs: int
    block: int
    smem_bytes: int
    ctas_per_sm: dict[str, int] = field(default_factory=dict)   # {regs, smem, warps, slots}
    limiter: str = ""
    active_ctas: int = 0
    warps_per_smsp: float = 0.0
    reg_cap_for_next_warp: int | None = None
    notes: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)             # launch would fail / setmaxnreg would hang
    setmaxnreg: dict | None = None                               # pool analysis when groups are given

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


def _granule(profile: GpuProfile) -> int:
    return int(profile.limit("reg_alloc_granule", 8))


def _round_up(x: int, g: int) -> int:
    return ((x + g - 1) // g) * g


def max_regs_for_warps(profile: GpuProfile, warps_per_smsp: int) -> int:
    """Largest regs/thread such that ``warps_per_smsp`` warps fit one SMSP's register quarter."""
    if warps_per_smsp <= 0:
        return int(profile.limit("regs_per_thread_max", 255))
    g = _granule(profile)
    per_smsp = profile.regs_per_sm // 4
    r = (per_smsp // (warps_per_smsp * 32)) // g * g
    return int(min(r, profile.limit("regs_per_thread_max", 255)))


def _ctas_by_regs(profile: GpuProfile, regs: int, warps: int) -> int:
    alloc = _round_up(max(regs, 1), _granule(profile)) * 32          # regs per warp
    per_smsp_warps = (profile.regs_per_sm // 4) // alloc
    if per_smsp_warps <= 0:
        return 0
    n = 0
    while math.ceil((n + 1) * warps / 4) <= per_smsp_warps and n < 1024:
        n += 1
    return n


def setmaxnreg_check(profile: GpuProfile, launch_regs: int, block: int, groups: list[tuple[int, int]]) -> dict:
    """Pool analysis of setmaxnreg groups [(threads, regs_after)]: errors (hang/illegal), warnings and the max
    feasible regs per group given the others."""
    lo = int(profile.limit("setmaxnreg_min", 24))
    hi = int(profile.limit("setmaxnreg_max", 256))
    g = _granule(profile)
    pool = block * _round_up(launch_regs, g)
    errors: list[str] = []
    warnings: list[str] = []
    for th, r in groups:
        if r % 8 or not lo <= r <= hi:
            errors.append(f"setmaxnreg {r}: must be a multiple of 8 in [{lo}, {hi}] (illegal instruction)")
        if th % 128:
            warnings.append(f"group of {th} threads is not whole warpgroups (setmaxnreg acts per warp; "
                            "keep groups warpgroup-aligned)")
    total_threads = sum(th for th, _ in groups)
    if total_threads != block:
        warnings.append(f"groups cover {total_threads} threads, block is {block}: remaining warps keep "
                        f"{launch_regs} regs")
    after = sum(th * r for th, r in groups) + max(0, block - total_threads) * _round_up(launch_regs, g)
    if after > pool:
        errors.append(f"HANG: after setmaxnreg the CTA needs {after} regs but its launch pool is {pool} "
                      f"({block} thr x {_round_up(launch_regs, g)}); setmaxnreg.inc waits forever")
    max_feasible = []
    for i, (th, r) in enumerate(groups):
        others = after - th * r
        m = ((pool - others) // th) // 8 * 8 if th else 0
        max_feasible.append(int(max(0, min(m, hi))))
    per_smsp = 0
    for th, r in groups:
        per_smsp += math.ceil(th / 32 / 4) * 32 * r
    if per_smsp > profile.regs_per_sm // 4:
        errors.append(f"per-SMSP register quarter exceeded after setmaxnreg ({per_smsp} > {profile.regs_per_sm // 4})")
    return {"pool": pool, "after": after, "headroom": pool - after, "groups": [list(x) for x in groups],
            "max_feasible": max_feasible, "errors": errors, "warnings": warnings}


def occupancy(profile: GpuProfile, kernel: str, regs: int, block: int, smem_bytes: int,
              setmaxnreg: list[tuple[int, int]] | None = None) -> OccupancyReport:
    """CTAs/SM per limiter for one kernel (regs/thread at launch, threads/CTA, smem bytes/CTA incl. static)."""
    rep = OccupancyReport(kernel=kernel, regs=int(regs), block=int(block), smem_bytes=int(smem_bytes))
    tmax = int(profile.limit("threads_per_cta_max", 1024))
    rmax = int(profile.limit("regs_per_thread_max", 255))
    smax = int(profile.limit("smem_per_cta_max", profile.smem_per_sm))
    reserved = int(profile.limit("smem_reserved_per_cta", 0))
    if block <= 0 or block > tmax:
        rep.errors.append(f"block {block} outside 1..{tmax}")
    if regs > rmax:
        rep.errors.append(f"{regs} regs/thread > {rmax}")
    if smem_bytes > smax:
        rep.errors.append(f"smem {smem_bytes} B > per-CTA max {smax} B (launch fails)")
    warps = max(1, math.ceil(block / 32))
    by_regs = _ctas_by_regs(profile, regs, warps)
    per_cta_smem = smem_bytes + reserved
    by_smem = profile.smem_per_sm // per_cta_smem if per_cta_smem > 0 and profile.smem_per_sm else profile.max_ctas_per_sm
    by_warps = profile.max_warps_per_sm // warps
    by_slots = profile.max_ctas_per_sm
    rep.ctas_per_sm = {"regs": int(by_regs), "smem": int(by_smem), "warps": int(by_warps), "slots": int(by_slots)}
    if rep.errors:
        rep.active_ctas = 0
    else:
        rep.active_ctas = int(min(rep.ctas_per_sm.values()))
    rep.limiter = min(("regs", "smem", "warps", "slots"), key=lambda k: rep.ctas_per_sm[k])
    if rep.active_ctas == 0 and not rep.errors:
        rep.errors.append(f"0 CTAs/SM: {rep.limiter} cannot fit one CTA (launch fails: too many resources)")
    rep.warps_per_smsp = rep.active_ctas * warps / 4
    wps = math.ceil(max(rep.active_ctas, 1) * warps / 4)
    cap_now = max_regs_for_warps(profile, wps)
    rep.notes.append(f"{wps} warps/SMSP caps regs at {cap_now}")
    nxt = rep.active_ctas + 1
    nxt_wps = math.ceil(nxt * warps / 4)
    cap_next = max_regs_for_warps(profile, nxt_wps)
    other = min(by_smem, by_warps, by_slots)
    if nxt <= other:
        rep.reg_cap_for_next_warp = cap_next
        if rep.limiter == "regs":
            rep.notes.append(f"{nxt} CTAs/SM ({nxt_wps} warps/SMSP) needs <= {cap_next} regs (now {regs})")
    else:
        rep.notes.append(f"{nxt} CTAs/SM impossible regardless of regs ({min(('smem', by_smem), ('warps', by_warps), ('slots', by_slots), key=lambda x: x[1])[0]} limit)")
    if per_cta_smem and rep.active_ctas:
        left = profile.smem_per_sm - rep.active_ctas * per_cta_smem
        rep.notes.append(f"smem/SM used {rep.active_ctas * per_cta_smem} B, free {left} B")
    if setmaxnreg:
        sm = setmaxnreg_check(profile, regs, block, list(setmaxnreg))
        rep.setmaxnreg = sm
        rep.errors.extend(sm["errors"])
        rep.notes.extend(sm["warnings"])
        mf = ", ".join(f"{th}thr<= {m}" for (th, _), m in zip(setmaxnreg, sm["max_feasible"]))
        rep.notes.append(f"setmaxnreg pool {sm['pool']} after {sm['after']} (headroom {sm['headroom']}); max per "
                         f"group given the others: {mf}")
    return rep


# ---- input parsing helpers (pure) -------------------------------------------------------------------------

_RES_RE = re.compile(r"Function\s+(\S+?)\s*:\s*(.*)")


def parse_res_usage(text: str) -> list[dict]:
    """``cuobjdump -res-usage`` -> [{kernel, regs, smem_static, stack, local}]."""
    out = []
    cur: str | None = None
    for line in text.splitlines():
        s = line.strip()
        m = _RES_RE.match(s)
        rest = s
        if m:
            cur, rest = m.group(1), m.group(2)
        fields = dict(re.findall(r"([A-Z]+(?:\[\d+\])?):(\d+)", rest))
        if cur is None or "REG" not in fields:
            continue                  # CUDA >= 12 prints the fields on the line after 'Function NAME:'
        out.append({"kernel": cur, "regs": int(fields["REG"]), "smem_static": int(fields.get("SHARED", 0)),
                    "stack": int(fields.get("STACK", 0)), "local": int(fields.get("LOCAL", 0))})
        cur = None
    return out


_PTX_FN = re.compile(r"Compiling entry function '([^']+)'")
_PTX_REGS = re.compile(r"Used (\d+) registers(?:, used (\d+) barriers)?(?:, (\d+) bytes smem)?")


def parse_ptxas_minimal(text: str) -> list[dict]:
    """Fallback ptxas -v parser (kernel, regs, smem_static); kdev.build.ptxas.parse_ptxas is preferred."""
    out: list[dict] = []
    cur = None
    for line in text.splitlines():
        m = _PTX_FN.search(line)
        if m:
            cur = m.group(1)
            continue
        m = _PTX_REGS.search(line)
        if m and cur:
            out.append({"kernel": cur, "regs": int(m.group(1)), "smem_static": int(m.group(3) or 0)})
            cur = None
    return out


def launch_bounds(source: str) -> list[int]:
    """Distinct first arguments of ``__launch_bounds__(N, ...)`` in a source text."""
    vals = re.findall(r"__launch_bounds__\s*\(\s*(\d+)", source)
    return sorted({int(v) for v in vals})


def parse_groups(spec: str) -> list[tuple[int, int]]:
    """'128:24,512:112' -> [(128, 24), (512, 112)]."""
    out = []
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        th, _, r = part.partition(":")
        out.append((int(th), int(r)))
    return out
