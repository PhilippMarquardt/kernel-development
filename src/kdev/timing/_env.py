"""Worker-side environment probes for timing fidelity (private to P3): clocks, foreign CUDA contexts,
lock state, toolchain notes of loaded extensions, and the no-compile (JIT) guard.

Everything here degrades gracefully: a probe that cannot run returns None/'unknown' and never raises.
torch is imported lazily inside functions only.
"""
from __future__ import annotations

import contextlib
import os
import re
import subprocess
from typing import Any, Iterator

from kdev.contract import ErrorCode, KdevError

SO_NOTE_RE = re.compile(rb"Cuda compilation tools, release [0-9.]+, V([0-9.]+)")
SMI_FIELDS = "clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,name,uuid"


def _run(argv: list[str], timeout: float = 20.0) -> str:
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False).stdout
    except Exception:  # noqa: BLE001
        return ""


def gpu_state() -> dict[str, Any]:
    """nvidia-smi snapshot as a dict (P2's smi.query_state when implemented, else a direct query)."""
    try:
        from kdev.service import smi

        st = smi.query_state()
        return st.to_dict()
    except Exception:  # noqa: BLE001
        pass
    out = _run(["nvidia-smi", f"--query-gpu={SMI_FIELDS}", "--format=csv,noheader,nounits"]).strip()
    if not out:
        return {}
    parts = [p.strip() for p in out.splitlines()[0].split(",")]

    def num(i: int, f: type = int) -> Any:
        try:
            return f(float(parts[i]))
        except (ValueError, IndexError):
            return None

    from kdev.util import utc_now

    return {"sm_mhz": num(0), "mem_mhz": num(1), "power_w": num(2, float), "temp_c": num(3),
            "util_pct": num(4), "name": parts[5] if len(parts) > 5 else "",
            "uuid": parts[6] if len(parts) > 6 else "", "ts_utc": utc_now()}


def clocks_moved(before: dict[str, Any], after: dict[str, Any], tol: float = 0.03) -> bool:
    a, b = before.get("sm_mhz"), after.get("sm_mhz")
    if not a or not b:
        return False
    return abs(b - a) / a > tol


def _is_zombie(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as f:
            return f.read().rsplit(")", 1)[1].split()[0] == "Z"
    except Exception:  # noqa: BLE001
        return False


def _ancestors() -> set[int]:
    pids, p = set(), os.getpid()
    for _ in range(64):
        pids.add(p)
        try:
            with open(f"/proc/{p}/stat", encoding="utf-8") as f:
                p = int(f.read().rsplit(")", 1)[1].split()[1])
        except Exception:  # noqa: BLE001
            break
        if p <= 1:
            break
    return pids


def foreign_contexts() -> list[int] | None:
    """PIDs of other live processes holding a CUDA context (zombies, self and ancestors ignored);
    None when nvidia-smi is unavailable."""
    out = _run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"])
    if not out and not _have_smi():
        return None
    mine = _ancestors()
    pids = []
    for line in out.splitlines():
        try:
            pid = int(line.strip())
        except ValueError:
            continue
        if pid in mine or _is_zombie(pid) or not os.path.exists(f"/proc/{pid}"):
            continue
        pids.append(pid)
    return pids


def _have_smi() -> bool:
    return bool(_run(["nvidia-smi", "-L"]))


def lock_held(path: str | None = None) -> bool | None:
    """True if the shared GPU lock file is currently held by someone (normally our parent: gpu_run.sh or
    kdev's lock), False if it is free, None if unknown (no fcntl / no file)."""
    path = path or os.environ.get("KDEV_LOCK_PATH") or "/root/gpu.lock"
    try:
        import fcntl
    except ImportError:
        return None
    if not os.path.exists(path):
        return None
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return None
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return True
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def exclusivity(start_foreign: list[int] | None, end_foreign: list[int] | None,
                mid_foreign: list[int] | None = None) -> tuple[str, float]:
    """(exclusive, waited_s). The service/flock parent may export KDEV_EXCLUSIVE (ok|waited|fail|shared)
    and KDEV_WAITED_S; otherwise: lock held by someone + no foreign context at start/mid/end -> 'ok';
    foreign context seen -> 'fail'; lock free -> 'shared'; nothing known -> 'unknown'."""
    waited = float(os.environ.get("KDEV_WAITED_S", "0") or 0)
    seen = [p for lst in (start_foreign, mid_foreign, end_foreign) if lst for p in lst]
    if seen:
        return "fail", waited
    env = os.environ.get("KDEV_EXCLUSIVE", "").lower()
    if env in ("ok", "waited", "fail", "shared"):
        return env, waited
    held = lock_held()
    if held is False:
        return "shared", waited
    if held and start_foreign is not None and end_foreign is not None:
        return "ok", waited
    return "unknown", waited


def loaded_extension_sos() -> list[str]:
    """Shared objects mapped into this process that look like JIT/torch extensions (not site-packages)."""
    try:
        with open("/proc/self/maps", encoding="utf-8") as f:
            paths = {line.split()[-1] for line in f if ".so" in line and "/" in line}
    except Exception:  # noqa: BLE001
        return []
    ext_dirs = [d for d in (os.environ.get("TORCH_EXTENSIONS_DIR"), os.environ.get("SGLANG_JIT_CACHE_DIR"))
                if d]
    out = []
    for p in sorted(paths):
        if any(p.startswith(d) for d in ext_dirs) or "torch_extensions" in p or "/ext1" in p:
            out.append(p)
    return out


def so_notes(path: str) -> list[str]:
    """nvcc versions recorded in a .so ("Cuda compilation tools, release 13.0, V13.0.88" -> "13.0.88")."""
    try:
        from kdev.build import toolchain

        notes = toolchain.so_notes(path)
        vers = set()
        for n in notes:
            m = re.search(r"V([0-9]+\.[0-9]+\.[0-9]+)", n)
            if m:
                vers.add(m.group(1))
        if vers or not notes:
            return sorted(vers)
    except Exception:  # noqa: BLE001
        pass
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        return []
    return sorted({m.group(1).decode() for m in SO_NOTE_RE.finditer(data)})


def nvcc_version() -> tuple[str, str]:
    """(CUDA_HOME, nvcc version like '13.0.88') of the toolchain torch's cpp_extension would use."""
    try:
        from torch.utils.cpp_extension import CUDA_HOME
    except Exception:  # noqa: BLE001
        CUDA_HOME = os.environ.get("CUDA_HOME", "")  # noqa: N806
    out = _run([os.path.join(CUDA_HOME or "", "bin", "nvcc"), "--version"])
    m = re.search(r"V([0-9]+\.[0-9]+\.[0-9]+)", out)
    return CUDA_HOME or "", (m.group(1) if m else "")


def toolchain_pin(problem_dir: str | None, extra: dict[str, Any]) -> str | None:
    """Pinned nvcc version: params.extra.toolchain_pin, else the problem's toolchain.lock (P5), else
    $KDEV_TOOLCHAIN_PIN, else None."""
    if extra.get("toolchain_pin"):
        return str(extra["toolchain_pin"])
    try:
        from kdev.build import toolchain

        lock = toolchain.find_lock(problem_dir)
        if lock:
            return toolchain.load_lock(lock).nvcc_version or None
    except Exception:  # noqa: BLE001
        pass
    return os.environ.get("KDEV_TOOLCHAIN_PIN") or None


def toolchain_fidelity(problem_dir: str | None, extra: dict[str, Any]) -> dict[str, Any]:
    """Fidelity.toolchain: {nvcc, cuda_home, pin, so_notes, sos, match}. ``match`` is False when any loaded
    extension .so or the active nvcc differs from the pin; omitted (unknown) without a pin."""
    home, nv = nvcc_version()
    sos = loaded_extension_sos()
    notes = sorted({v for p in sos for v in so_notes(p)})
    pin = toolchain_pin(problem_dir, extra)
    d: dict[str, Any] = {"nvcc": nv, "cuda_home": home, "pin": pin, "so_notes": notes,
                         "n_ext_so": len(sos), "torch_extensions_dir": os.environ.get("TORCH_EXTENSIONS_DIR", "")}
    if pin:
        d["match"] = all(v == pin for v in notes) and (not nv or nv == pin)
    return d


@contextlib.contextmanager
def no_compile_guard(enabled: bool = True) -> Iterator[list[str]]:
    """Refuse to compile inside the GPU lane (M4): torch cpp_extension builds that ninja reports as having
    work to do raise KdevError(needs_build). Up-to-date cached builds pass (ninja 'no work to do').
    Yields the list of build directories that were checked."""
    checked: list[str] = []
    if not enabled:
        yield checked
        return
    try:
        import torch.utils.cpp_extension as ce
    except Exception:  # noqa: BLE001
        yield checked
        return
    orig = getattr(ce, "_run_ninja_build", None)
    if orig is None:
        yield checked
        return

    def guarded(build_directory: str, verbose: bool, error_prefix: str) -> None:
        checked.append(build_directory)
        out = _run(["ninja", "-C", build_directory, "-n"], timeout=60)
        if "no work to do" not in out:
            raise KdevError(ErrorCode.NEEDS_BUILD,
                            f"extension in {build_directory} is not prebuilt (ninja has work to do)",
                            details={"build_directory": build_directory, "ninja": out[-400:]},
                            recovery="run `kdev build FILE` first (or pass --allow-jit, logged as an opt-out)")
        orig(build_directory, verbose, error_prefix)

    ce._run_ninja_build = guarded
    try:
        yield checked
    finally:
        ce._run_ninja_build = orig
