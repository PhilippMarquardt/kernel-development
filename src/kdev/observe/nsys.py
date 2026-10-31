"""Nsight Systems wrapper (if present): kernel trace without metrics. Owner: P6.

nsys needs no counter permission (CUPTI activity only), so it works where ncu is blocked. Kernel times are
from ``nsys stats --report cuda_gpu_kern_sum``.
"""
from __future__ import annotations

import csv
import io
import os
import shutil

from kdev.util import Runner, run


def find_nsys() -> str | None:
    p = shutil.which("nsys")
    if p:
        return p
    for home in (os.environ.get("CUDA_HOME", ""), "/usr/local/cuda", "/opt/nvidia/nsight-systems"):
        c = os.path.join(home, "bin", "nsys") if home else ""
        if c and os.path.isfile(c):
            return c
    return None


def available(runner: Runner = run) -> bool:
    exe = find_nsys()
    if not exe:
        return False
    try:
        return runner([exe, "--version"], timeout=30).returncode == 0
    except Exception:  # noqa: BLE001
        return False


def nsys_argv(target_argv: list[str], out: str = "nsys") -> list[str]:
    """nsys profile --trace=cuda --sample=none --cpuctxsw=none -o out ..."""
    return ["nsys", "profile", "--trace=cuda", "--sample=none", "--cpuctxsw=none", "--force-overwrite=true",
            "-o", out] + list(target_argv)


def stats_argv(report: str) -> list[str]:
    return ["nsys", "stats", "--report", "cuda_gpu_kern_sum", "--format", "csv", "--force-export=true", report]


def _f(v: str) -> float:
    try:
        return float(v.replace(",", ""))
    except (ValueError, AttributeError):
        return 0.0


def parse_kern_sum(csv_text: str) -> list[dict]:
    """`nsys stats --report cuda_gpu_kern_sum --format csv` -> rows
    {name, time_pct, total_us, count, avg_us, med_us, min_us, max_us} sorted by total time."""
    lines = csv_text.splitlines()
    start = next((i for i, ln in enumerate(lines) if "Time (%)" in ln or "Time(%)" in ln), None)
    if start is None:
        return []
    out = []
    for r in csv.DictReader(io.StringIO("\n".join(lines[start:]))):
        r = {(k or "").strip(): (v or "").strip() for k, v in r.items()}
        name = r.get("Name", "")
        if not name:
            continue
        g = lambda *ks: next((r[k] for k in ks if k in r), "0")  # noqa: E731
        out.append({"name": name, "time_pct": _f(g("Time (%)", "Time(%)")),
                    "total_us": _f(g("Total Time (ns)", "Total Time")) / 1000.0,
                    "count": int(_f(g("Instances", "Calls"))),
                    "avg_us": _f(g("Avg (ns)", "Average", "Avg")) / 1000.0,
                    "med_us": _f(g("Med (ns)", "Median", "Med")) / 1000.0,
                    "min_us": _f(g("Min (ns)", "Minimum", "Min")) / 1000.0,
                    "max_us": _f(g("Max (ns)", "Maximum", "Max")) / 1000.0})
    out.sort(key=lambda d: -d["total_us"])
    return out
