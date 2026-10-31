"""Compile-only builds with the pinned judge toolchain (M2-M4, M20). Owner: P5."""
from __future__ import annotations

import os
from pathlib import Path

from kdev.models import Check


def probe(ctx: object, problem: str | None = None) -> list[Check]:
    """Doctor checks: toolchain pin vs found (nvcc/ptxas/cuobjdump/nvdisasm versions+paths), torch/triton vs
    pin, TORCH_EXTENSIONS_DIR keyed per toolchain, thrust/CCCL headers present. Never raises."""
    out: list[Check] = []
    try:
        from kdev.build.toolchain import check_tools, find_lock, load_lock, detect  # noqa: PLC0415

        lock = find_lock(problem)
        if lock:
            pin = load_lock(lock)
            out.append(Check("toolchain.pin", "PASS", f"{lock}: nvcc V{pin.nvcc_version} ({pin.source})"))
        else:
            pin = detect()
            out.append(Check("toolchain.pin", "WARN", f"no toolchain.lock; detected V{pin.nvcc_version or '?'}",
                             hint="write <problem>/toolchain.lock with the judge's nvcc (LESSONS 9: 13.0.88)"))
        for tool, st, detail in check_tools(pin):
            out.append(Check(f"toolchain.{tool}", st, detail,
                             hint="" if st == "PASS" else "install/point cuda_home at the pinned CUDA tree"))
        if pin.cuda_home:
            inc = Path(pin.cuda_home) / "include"
            hdrs = [inc / "thrust" / "version.h", inc / "cub" / "version.cuh"]
            alt = list((Path(pin.cuda_home) / "targets").glob("*/include/thrust/version.h")) if \
                (Path(pin.cuda_home) / "targets").is_dir() else []
            ok = all(h.is_file() for h in hdrs) or bool(alt)
            out.append(Check("toolchain.cccl", "PASS" if ok else "WARN",
                             "thrust/cub headers present" if ok else f"thrust/cub headers missing under {inc}",
                             hint="" if ok else "M20: reinstall nvidia-cuda-cccl / fix the include symlink"))
        ted = os.environ.get("TORCH_EXTENSIONS_DIR")
        if ted:
            keyed = pin.nvcc_version and pin.nvcc_version.split(".")[0] in ted
            out.append(Check("toolchain.extensions_dir", "PASS" if keyed else "WARN", ted,
                             hint="" if keyed else "kdev build keys its own cache by toolchain; a shared "
                             "TORCH_EXTENSIONS_DIR not keyed by nvcc version reuses stale .so files (M3)"))
        for k, v in (("torch", pin.torch), ("triton", pin.triton)):
            if v:
                out.append(Check(f"toolchain.{k}", "SKIP", f"pin {v} (checked in the worker interpreter)"))
    except Exception as e:  # noqa: BLE001 - doctor never raises
        out.append(Check("toolchain", "FAIL", f"{type(e).__name__}: {e}"))
    return out
