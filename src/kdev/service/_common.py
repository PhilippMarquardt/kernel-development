"""Private helpers shared by kdev.service and kdev.remote (P2): config resolution, run dirs, pids, CPUs.

These tolerate P1 stubs (``load_config``/``runs.new_run`` raising NotImplementedError) so P2 works and is
testable before P1 lands; once P1 is implemented the P1 functions are used.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from kdev.config import Config

_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def get_config(ctx: Any = None) -> Config:
    """ctx.config if set, else kdev.config.load_config(home=ctx.home), else Config() + env overrides."""
    cfg = getattr(ctx, "config", None)
    if isinstance(cfg, Config):
        return cfg
    home = getattr(ctx, "home", None)
    try:
        from kdev.config import load_config
        cfg = load_config(home=home)
    except NotImplementedError:
        cfg = Config()
        if home:
            cfg.home = Path(home).expanduser()
        if os.environ.get("KDEV_LOCK_PATH"):
            cfg.lock_path = os.environ["KDEV_LOCK_PATH"]
        if os.environ.get("KDEV_PYTHON"):
            cfg.python = os.environ["KDEV_PYTHON"]
        if os.environ.get("KDEV_OFFENDER_LOG"):
            cfg.offender_log = os.environ["KDEV_OFFENDER_LOG"]
    if ctx is not None:
        try:
            ctx.config = cfg
        except Exception:  # noqa: BLE001
            pass
    return cfg


def parse_duration(text: str | float | int | None) -> float | None:
    """'3h' -> 10800.0, '90s', '30m', '2d', plain number = seconds; None/'' -> None."""
    if text is None or text == "":
        return None
    if isinstance(text, (int, float)):
        return float(text)
    m = re.fullmatch(r"\s*([0-9]*\.?[0-9]+)\s*([smhd]?)\s*", str(text))
    if not m:
        from kdev.contract import ErrorCode, KdevError
        raise KdevError(ErrorCode.INVALID_REQUEST, f"bad duration {text!r}", recovery="use e.g. 90s, 30m, 3h, 2d")
    return float(m.group(1)) * _UNITS[m.group(2) or "s"]


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=1, default=str))


class _RunDirFallback:
    """Minimal RunDir used only while kdev.runs.new_run is a P1 stub."""

    def __init__(self, run_id: str, path: Path) -> None:
        self.run_id = run_id
        self.path = path

    def file(self, name: str) -> Path:
        return self.path / name


def new_run(cfg: Config, tool: str, slug: str = "") -> Any:
    """kdev.runs.new_run, or a compatible fallback (same id format) while P1 is a stub."""
    try:
        from kdev.runs import new_run as _p1_new_run
        return _p1_new_run(cfg, tool, slug)
    except NotImplementedError:
        pass
    import datetime as dt
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", slug).strip("-")[:40]
    base = f"{ts}-{tool}" + (f"-{slug}" if slug else "")
    cfg.runs_dir.mkdir(parents=True, exist_ok=True)
    run_id, n = base, 1
    while True:
        p = cfg.runs_dir / run_id
        try:
            p.mkdir()
            return _RunDirFallback(run_id, p)
        except FileExistsError:
            n += 1
            run_id = f"{base}-{n}"


def pid_alive(pid: int | None) -> bool:
    """True if a process with this pid exists and is not a zombie. Never sends a real signal."""
    if not pid or pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes
        k32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        h = k32.OpenProcess(0x1000, False, int(pid))   # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
        k32.CloseHandle(h)
        return bool(ok) and code.value == 259          # STILL_ACTIVE
    st = proc_state(pid)
    if st:
        return not st.startswith(("Z", "X"))
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def proc_state(pid: int) -> str:
    """Single-letter state from /proc/<pid>/stat ('' if unknown/absent)."""
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return ""
    rp = text.rfind(")")
    return text[rp + 2: rp + 3] if rp >= 0 else ""


def proc_cmdline(pid: int, limit: int = 300) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\0", b" ").decode("utf-8", "replace").strip()[:limit]


def descendants(pid: int) -> set[int]:
    """All descendant pids of ``pid`` from /proc (empty off Linux)."""
    children: dict[int, list[int]] = {}
    proc = Path("/proc")
    if not proc.is_dir():
        return set()
    for d in proc.iterdir():
        if not d.name.isdigit():
            continue
        try:
            text = (d / "stat").read_text()
        except OSError:
            continue
        rp = text.rfind(")")
        parts = text[rp + 2:].split()
        if len(parts) > 1:
            children.setdefault(int(parts[1]), []).append(int(d.name))
    out: set[int] = set()
    stack = [pid]
    while stack:
        for c in children.get(stack.pop(), []):
            if c not in out:
                out.add(c)
                stack.append(c)
    return out


def cpu_budget() -> int:
    """CPUs actually usable: cgroup quota (v2 cpu.max, v1 cfs_quota/period), affinity, cpu_count.

    RunPod containers report nproc=192 while the cgroup quota is ~20 CPUs; oversubscribing compiles there
    slows every build and starves the timing lane's host thread."""
    candidates: list[float] = []
    try:
        q, p = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        if q != "max":
            candidates.append(int(q) / int(p))
    except (OSError, ValueError):
        pass
    try:
        q1 = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        p1 = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if q1 > 0 and p1 > 0:
            candidates.append(q1 / p1)
    except (OSError, ValueError):
        pass
    try:
        candidates.append(len(os.sched_getaffinity(0)))  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        candidates.append(os.cpu_count() or 1)
    return max(1, int(min(candidates)))


def build_worker_cap(requested: int, hard_cap: int = 16) -> int:
    """Concurrent build jobs: min(requested, 16, usable CPUs - 2 reserved for the GPU lane/daemon)."""
    return max(1, min(int(requested or 1), hard_cap, max(1, cpu_budget() - 2)))
