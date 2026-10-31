"""nvidia-smi queries (clocks, power, temperature, utilisation, GPU identity). Owner: P2.

All functions take ``runner`` so tests feed canned CSV (tests/fixtures/smi/).
"""
from __future__ import annotations

from pathlib import Path

from kdev.models import GpuState
from kdev.util import Runner, run, utc_now

QUERY_FIELDS = "name,uuid,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu"
NVIDIA_PARAMS = "/proc/driver/nvidia/params"


def query_state(gpu: int = 0, runner: Runner = run) -> GpuState:
    """nvidia-smi --query-gpu=name,uuid,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu
    --format=csv,noheader,nounits. Missing nvidia-smi -> GpuState() with empty fields (never raises)."""
    try:
        p = runner(["nvidia-smi", f"--id={gpu}", f"--query-gpu={QUERY_FIELDS}", "--format=csv,noheader,nounits"],
                   timeout=20)
    except Exception:  # noqa: BLE001  (FileNotFoundError, TimeoutExpired, ...)
        return GpuState(ts_utc=utc_now())
    if p.returncode != 0 or not p.stdout.strip():
        return GpuState(ts_utc=utc_now())
    st = parse_state_csv(p.stdout.strip().splitlines()[0])
    st.ts_utc = utc_now()
    return st


def _num(s: str, cast: type) -> int | float | None:
    s = s.strip()
    if not s or s.startswith("[") or s.upper() in ("N/A", "NA"):
        return None
    try:
        return int(float(s)) if cast is int else float(s)
    except ValueError:
        return None


def parse_state_csv(text: str) -> GpuState:
    """Pure parser for the CSV line above."""
    parts = [x.strip() for x in text.strip().split(",")]
    parts += [""] * (7 - len(parts))
    return GpuState(name=parts[0], uuid=parts[1], sm_mhz=_num(parts[2], int), mem_mhz=_num(parts[3], int),
                    power_w=_num(parts[4], float), temp_c=_num(parts[5], int), util_pct=_num(parts[6], int))


def clocks_moved(before: GpuState, after: GpuState, tol: float = 0.03) -> bool:
    """True if SM clock moved more than tol (fidelity warning)."""
    if not before.sm_mhz or not after.sm_mhz:
        return False
    return abs(after.sm_mhz - before.sm_mhz) / before.sm_mhz > tol


def ncu_permitted(params_path: str = NVIDIA_PARAMS) -> bool | None:
    """Parse /proc/driver/nvidia/params RmProfilingAdminOnly (None if unknown)."""
    try:
        text = Path(params_path).read_text()
    except OSError:
        return None
    for line in text.splitlines():
        if line.strip().startswith("RmProfilingAdminOnly"):
            val = line.split(":", 1)[-1].strip()
            if val in ("0", "1"):
                return val == "0"
    return None
