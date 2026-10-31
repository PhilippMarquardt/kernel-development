"""CLI: regs. Owner: P6.

Static only (no GPU). Kernel facts come from, in order: a ptxas -v log (``.log/.txt/.ptxas``), a built
``.so``/``.cubin`` (``cuobjdump -res-usage``), or a source file built compile-only through ``kdev.build``
(P5). Block size: ``--block``, else the file's single ``__launch_bounds__`` value.
"""
from __future__ import annotations

import os
import re
import shutil
from argparse import Namespace
from typing import Any

from kdev.contract import Effects, ErrorCode, KdevError, Result
from kdev.registry import Arg, Context, Registry
from kdev.util import Runner, run

P = "P6"
LOG_SUFFIXES = (".log", ".txt", ".ptxas")


def _cuobjdump() -> str:
    p = shutil.which("cuobjdump")
    if p:
        return p
    for home in (os.environ.get("CUDA_HOME", ""), "/root/cuda130w", "/usr/local/cuda"):
        c = os.path.join(home, "bin", "cuobjdump") if home else ""
        if c and os.path.isfile(c):
            return c
    return "cuobjdump"


def _from_ptxas_text(text: str) -> list[dict[str, Any]]:
    from kdev.occupancy.calc import parse_ptxas_minimal
    try:
        from kdev.build.ptxas import parse_ptxas
        infos = parse_ptxas(text)
        return [{"kernel": k.name, "regs": k.regs or 0, "smem_static": k.smem_static, "spill_store": k.spill_store,
                 "spill_load": k.spill_load} for k in infos if k.regs is not None]
    except NotImplementedError:
        return parse_ptxas_minimal(text)


def kernels_for(target: str, runner: Runner = run, problem: str | None = None) -> tuple[list[dict[str, Any]], str]:
    """[{kernel, regs, smem_static, ...}] and the source label used."""
    if not os.path.isfile(target):
        raise KdevError(ErrorCode.NOT_FOUND, f"no such file {target}")
    low = target.lower()
    if low.endswith(LOG_SUFFIXES):
        with open(target, encoding="utf-8", errors="replace") as f:
            return _from_ptxas_text(f.read()), "ptxas log"
    if low.endswith((".so", ".cubin", ".fatbin", ".o")):
        from kdev.occupancy.calc import parse_res_usage
        p = runner([_cuobjdump(), "-res-usage", target], timeout=120)
        if p.returncode != 0:
            raise KdevError(ErrorCode.CAPABILITY_UNAVAILABLE, f"cuobjdump -res-usage failed: {p.stderr.strip()[:200]}",
                            recovery="put the CUDA toolkit bin/ on PATH or pass a ptxas -v log")
        return parse_res_usage(p.stdout), "cuobjdump -res-usage"
    try:
        from kdev.build.compile import build_file
        rep = build_file(target, problem=problem)
    except NotImplementedError as e:
        raise KdevError(ErrorCode.CAPABILITY_UNAVAILABLE, "kdev build is not available for source targets yet",
                        recovery="pass the built .so/.cubin or a ptxas -v log") from e
    if rep.errors:
        raise KdevError(ErrorCode.BUILD_FAILED, f"build failed: {rep.errors[0][:200]}",
                        recovery=f"kdev build {target} --verbose")
    out = []
    for ext in rep.exts:
        for k in ext.kernels:
            if k.regs is not None:
                out.append({"kernel": k.name, "regs": k.regs, "smem_static": k.smem_static,
                            "spill_store": k.spill_store, "spill_load": k.spill_load})
    return out, "kdev build (ptxas -v)"


def cmd_regs(args: Namespace, ctx: Context) -> Result:
    from kdev.floor._text import table
    from kdev.floor.gpus import load_profile
    from kdev.occupancy.calc import launch_bounds, occupancy, parse_groups
    prof = load_profile(args.gpu or os.environ.get("KDEV_GPU") or "h200", home=ctx.home)
    target = os.path.abspath(os.path.expanduser(args.target))
    kernels, source = kernels_for(target, problem=ctx.problem)
    if args.kernel:
        rx = re.compile(args.kernel)
        kernels = [k for k in kernels if rx.search(k["kernel"])]
    if not kernels:
        raise KdevError(ErrorCode.NOT_FOUND, f"no kernels found in {target}" + (f" matching {args.kernel!r}"
                                                                              if args.kernel else ""))
    block = args.block
    notes: list[str] = []
    if block is None and not target.lower().endswith(LOG_SUFFIXES + (".so", ".cubin", ".fatbin", ".o")):
        try:
            with open(target, encoding="utf-8", errors="replace") as f:
                lb = launch_bounds(f.read())
        except OSError:
            lb = []
        if len(lb) == 1:
            block = lb[0]
            notes.append(f"block {block} from __launch_bounds__")
        elif lb:
            notes.append(f"several __launch_bounds__ values {lb}: pass --block")
    if block is None:
        raise KdevError(ErrorCode.INVALID_REQUEST, "block size unknown", recovery="pass --block N (threads per CTA)")
    groups = parse_groups(args.setmaxnreg) if getattr(args, "setmaxnreg", None) else None
    reps = [occupancy(prof, k["kernel"], int(k["regs"]), int(block), int(k.get("smem_static", 0)) + int(args.smem or 0),
                      groups) for k in kernels]
    rows = []
    for r, k in zip(reps, kernels):
        c = r.ctas_per_sm
        rows.append([r.kernel[:48], r.regs, r.block, r.smem_bytes, f"{c['regs']}/{c['smem']}/{c['warps']}/{c['slots']}",
                     r.active_ctas, r.limiter, f"{r.warps_per_smsp:g}", r.reg_cap_for_next_warp or "-",
                     f"{k.get('spill_store', 0)}/{k.get('spill_load', 0)}" if "spill_store" in k else "-"])
    lines = [f"gpu {prof.name}: {prof.regs_per_sm} regs/SM, {prof.smem_per_sm} B smem/SM; source: {source}"] + notes
    lines += table(["kernel", "regs", "block", "smem", "ctas r/s/w/slot", "active", "limiter", "warps/SMSP",
                    "regs_for_+1", "spill st/ld"], rows)
    for r in reps[:4]:
        lines += [f"{r.kernel[:40]}: {n}" for n in r.notes[:3]]
    errors = [f"{r.kernel[:40]}: {e}" for r in reps for e in r.errors]
    status = "fail" if errors else "ok"
    return Result(tool="regs", status=status, verdict="error" if errors else "ok",
                  summary={"gpu": prof.name, "kernels": [r.to_dict() for r in reps]}, lines=lines, warnings=errors)


def register(reg: Registry) -> None:
    reg.command("regs", "static occupancy per kernel: regs/smem/limiter/warps-per-SMSP rule",
                cmd_regs,
                args=[Arg(("target",), "built .so/.cubin, ptxas -v log, or source (built via kdev build)",
                          type="path"),
                      Arg(("--kernel",), "kernel regex", default=None),
                      Arg(("--block",), "threads per block (if not in launch bounds)", type="int", default=None),
                      Arg(("--smem",), "dynamic smem bytes", type="int", default=0),
                      Arg(("--setmaxnreg",), "groups THREADS:REGS,... e.g. 128:24,512:112 (hang check)",
                          default=None),
                      Arg(("--gpu",), "GPU profile", default=None)],
                effects=Effects(gpu="none"), package=P)
