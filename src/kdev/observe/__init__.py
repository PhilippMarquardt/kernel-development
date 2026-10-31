"""Runtime observation backends: CUPTI (torch.profiler), ncu, nsys, NVBit; dispatch probe. Owner: P6.

Backends report availability through ``probe()``; tools pick the best available and name it in the Result
(REA: missing backend = capability_unavailable, never a crash).
"""
from __future__ import annotations

import glob
import importlib.util
import os
import shutil

from kdev.models import Check


def _cupti_check() -> Check:
    spec = importlib.util.find_spec("torch")          # top-level lookup only: never imports torch
    if spec is None or not spec.origin:
        return Check("cupti", "WARN", "torch not importable in this interpreter",
                     "profile/dispatch run in the worker's pinned interpreter (config `python`)")
    root = os.path.dirname(os.path.dirname(spec.origin))
    libs = glob.glob(os.path.join(root, "torch", "lib", "libcupti*")) + \
        glob.glob(os.path.join(root, "nvidia", "cuda_cupti", "lib", "libcupti*"))
    detail = f"torch.profiler at {os.path.dirname(spec.origin)}" + (f", {os.path.basename(libs[0])}" if libs else "")
    return Check("cupti", "PASS", detail)


def probe(ctx: object, problem: str | None = None) -> list[Check]:
    """Doctor checks: ncu present + permitted (ERR_NVGPUCTRPERM / RmProfilingAdminOnly), nsys present,
    compute-sanitizer present, NVBit present (NVBIT_PATH), torch.profiler CUPTI importable. Never raises."""
    out: list[Check] = []
    try:
        from kdev.observe import ncu
        ok, why = ncu.available()
        if ok:
            out.append(Check("ncu", "PASS", why))
        elif "not found" in why:
            out.append(Check("ncu", "SKIP", why, "kdev profile falls back to CUPTI (torch.profiler)"))
        else:
            out.append(Check("ncu", "WARN", why, "kdev profile falls back to CUPTI; ncu needs CAP_SYS_ADMIN or "
                                                 "RmProfilingAdminOnly=0 on the host"))
    except Exception as e:  # noqa: BLE001
        out.append(Check("ncu", "WARN", f"probe failed: {e}"))
    try:
        from kdev.observe import nsys
        out.append(Check("nsys", "PASS" if nsys.available() else "SKIP",
                         nsys.find_nsys() or "nsys not found", "" if nsys.available() else "optional"))
    except Exception as e:  # noqa: BLE001
        out.append(Check("nsys", "WARN", f"probe failed: {e}"))
    cs = shutil.which("compute-sanitizer")
    if not cs:
        for home in (os.environ.get("CUDA_HOME", ""), "/usr/local/cuda", "/root/cuda130w"):
            c = os.path.join(home, "bin", "compute-sanitizer") if home else ""
            if c and os.path.isfile(c):
                cs = c
                break
    out.append(Check("compute-sanitizer", "PASS" if cs else "WARN", cs or "not found",
                     "" if cs else "kdev sanitize needs it (CUDA toolkit bin/)"))
    try:
        from kdev.observe import nvbit
        p = nvbit.tool_path()
        out.append(Check("nvbit", "PASS" if p else "SKIP", p or "NVBIT_PATH unset or instr_count.so missing",
                         "" if p else "optional"))
    except Exception as e:  # noqa: BLE001
        out.append(Check("nvbit", "WARN", f"probe failed: {e}"))
    try:
        out.append(_cupti_check())
    except Exception as e:  # noqa: BLE001
        out.append(Check("cupti", "WARN", f"probe failed: {e}"))
    return out
