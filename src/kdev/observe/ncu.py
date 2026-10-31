"""Nsight Compute wrapper (only when permitted). Owner: P6.

Permission: ncu needs GPU performance counters. On our pods they are blocked (``ERR_NVGPUCTRPERM``:
``RmProfilingAdminOnly=1`` in /proc/driver/nvidia/params and the container lacks CAP_SYS_ADMIN, even as
root). ``available()`` checks the binary, the driver parameter and the capability without touching the GPU;
``is_perm_error()`` recognises the error in the output of a real run, so callers fall back to CUPTI
(kdev.observe.cupti) with ``capability_unavailable: ncu (ERR_NVGPUCTRPERM)`` named in the Result.
"""
from __future__ import annotations

import csv
import io
import os
import shutil

from kdev.util import Runner, run

PERM_MARKERS = ("ERR_NVGPUCTRPERM", "does not have permission to access NVIDIA GPU Performance Counters",
                "RmProfilingAdminOnly")
CAP_SYS_ADMIN_BIT = 21


def find_ncu() -> str | None:
    p = shutil.which("ncu")
    if p:
        return p
    for home in (os.environ.get("CUDA_HOME", ""), "/usr/local/cuda", "/root/cuda130w"):
        if home:
            for c in (os.path.join(home, "bin", "ncu"), os.path.join(home, "nsight-compute", "ncu")):
                if os.path.isfile(c):
                    return c
    return None


def _profiling_admin_only(params_path: str) -> bool | None:
    try:
        with open(params_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("RmProfilingAdminOnly"):
                    return line.split(":", 1)[1].strip() not in ("0", "")
    except OSError:
        return None
    return None


def _has_cap_sys_admin(status_path: str) -> bool | None:
    try:
        with open(status_path, encoding="utf-8") as f:
            for line in f:
                if line.startswith("CapEff:"):
                    return bool(int(line.split()[1], 16) >> CAP_SYS_ADMIN_BIT & 1)
    except (OSError, ValueError, IndexError):
        return None
    return None


def available(runner: Runner = run, *, params_path: str = "/proc/driver/nvidia/params",
              status_path: str = "/proc/self/status", ncu_path: str | None = None) -> tuple[bool, str]:
    """(ok, reason): binary present and counters permitted; reason e.g. 'ERR_NVGPUCTRPERM'."""
    exe = ncu_path or find_ncu()
    if not exe:
        return False, "ncu not found"
    try:
        p = runner([exe, "--version"], timeout=30)
        if p.returncode != 0:
            return False, f"ncu --version failed: {(p.stderr or p.stdout).strip()[:80]}"
    except (OSError, Exception) as e:  # noqa: BLE001 - runner may raise TimeoutExpired etc.
        return False, f"ncu not runnable: {e}"
    admin_only = _profiling_admin_only(params_path)
    if admin_only:
        cap = _has_cap_sys_admin(status_path)
        if not cap:
            return False, "ERR_NVGPUCTRPERM (RmProfilingAdminOnly=1, no CAP_SYS_ADMIN)"
    return True, "ok" if admin_only is not None else "ok (driver params unreadable: permission unverified)"


def is_perm_error(text: str) -> bool:
    return any(m in text for m in PERM_MARKERS)


SECTION_ALIASES = {"sol": "SpeedOfLight", "occupancy": "Occupancy", "warps": "WarpStateStats",
                   "mem": "MemoryWorkloadAnalysis", "launch": "LaunchStats", "compute": "ComputeWorkloadAnalysis",
                   "sched": "SchedulerStats", "source": "SourceCounters"}


def ncu_argv(target_argv: list[str], metrics: str = "SpeedOfLight,Occupancy,WarpStateStats",
             kernel_regex: str | None = None, out: str = "ncu.ncu-rep", *, ncu: str = "ncu",
             launch_count: int | None = 50) -> list[str]:
    """ncu command line: named sections, boost clocks kept (``--clock-control none``, judge-like), caches
    flushed per replayed kernel (``--cache-control all``), all child processes."""
    argv = [ncu, "--target-processes", "all", "--clock-control", "none", "--cache-control", "all", "-f", "-o", out]
    for s in (x.strip() for x in (metrics or "").split(",")):
        if s:
            argv += ["--section", SECTION_ALIASES.get(s.lower(), s)]
    if kernel_regex:
        argv += ["--kernel-name", f"regex:{kernel_regex}"]
    if launch_count:
        argv += ["--launch-count", str(launch_count)]
    return argv + list(target_argv)


def ncu_csv_argv(report: str, ncu: str = "ncu") -> list[str]:
    return [ncu, "--import", report, "--csv", "--page", "details"]


def parse_csv(text: str) -> list[dict]:
    """`ncu --csv --page details` -> rows."""
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.startswith('"ID"') or ln.startswith("ID,")), None)
    if start is None:
        return []
    rows = []
    for r in csv.DictReader(io.StringIO("\n".join(lines[start:]))):
        if not r.get("Kernel Name"):
            continue
        rows.append({k: (v if v is not None else "") for k, v in r.items() if k is not None})
    return rows


def summarize_rows(rows: list[dict]) -> list[dict]:
    """Per (ID, kernel): {id, kernel, grid, block, metrics{section/metric: 'value unit'}} in launch order."""
    out: dict[str, dict] = {}
    for r in rows:
        key = r.get("ID", "")
        d = out.setdefault(key, {"id": key, "kernel": r.get("Kernel Name", ""), "grid": r.get("Grid Size", ""),
                                 "block": r.get("Block Size", ""), "metrics": {}})
        m = f"{r.get('Section Name', '')}/{r.get('Metric Name', '')}"
        unit = r.get("Metric Unit", "")
        d["metrics"][m] = f"{r.get('Metric Value', '')} {unit}".strip()
    return list(out.values())
