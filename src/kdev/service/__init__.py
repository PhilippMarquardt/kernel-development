"""GPU job service (`kdev serve`), exclusive lock, nvidia-smi helpers. Owner: P2.

Modules: lock (flock + foreign-context wait, gpu_run.sh semantics), smi (nvidia-smi queries), queue (SQLite
queue.db), daemon (build lane + serial GPU lane), client (submit/wait/status/cancel), commands (CLI).
"""
from __future__ import annotations

import os

from kdev.models import Check


def probe(ctx: object, problem: str | None = None) -> list[Check]:
    """Doctor checks: GPU visible (nvidia-smi), lock file writable, foreign CUDA processes, daemon heartbeat,
    ncu permission (RmProfilingAdminOnly in /proc/driver/nvidia/params). Never raises."""
    checks: list[Check] = []

    def guard(name: str, fn) -> None:
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            checks.append(Check(name, "FAIL", f"probe error: {type(e).__name__}: {e}"))

    from kdev.config import no_gpu
    from kdev.service._common import get_config
    cfg = get_config(ctx)

    def gpu() -> None:
        if no_gpu():
            checks.append(Check("gpu", "SKIP", "KDEV_NO_GPU=1"))
            return
        from kdev.service import smi
        st = smi.query_state()
        if st.name:
            checks.append(Check("gpu", "PASS", f"{st.name} sm {st.sm_mhz} MHz {st.temp_c} C util {st.util_pct}%"))
        else:
            checks.append(Check("gpu", "FAIL", "nvidia-smi missing or no GPU visible",
                                hint="run on the GPU pod, or set KDEV_NO_GPU=1 for GPU-free work"))

    def lock_file() -> None:
        path = cfg.lock_path
        parent = os.path.dirname(path) or "."
        if not os.path.isdir(parent):
            checks.append(Check("gpu lock", "WARN", f"{parent} does not exist (lock {path})",
                                hint="set lock_path in kdev.toml (legacy gpu_run.sh uses /root/gpu.lock)"))
            return
        try:
            with open(path, "a"):
                pass
        except OSError as e:
            checks.append(Check("gpu lock", "FAIL", f"{path} not writable: {e}"))
            return
        from kdev.service.lock import holder
        h = holder(cfg)
        detail = f"{path} " + (f"held by pid {h.get('pid', '?')} {h.get('cmdline', '')[:60]}" if h.get("held") else "free")
        checks.append(Check("gpu lock", "PASS", detail))

    def foreign() -> None:
        if no_gpu():
            return
        from kdev.service import lock
        apps = lock.foreign_apps(lock.compute_apps(), {os.getpid()})
        if not apps:
            checks.append(Check("foreign CUDA", "PASS", "no other live CUDA process"))
            return
        h = lock.holder(cfg)
        status = "PASS" if h.get("held") else "WARN"      # a context under a held lock is the holder's job
        checks.append(Check("foreign CUDA", status, ", ".join(f"{a.pid} {a.cmdline[:50]}" for a in apps[:3]),
                            hint="" if status == "PASS" else "unlocked GPU work corrupts timings: use kdev lock run"))

    def daemon() -> None:
        from kdev.service import queue
        if queue.daemon_alive(cfg):
            hb = queue.heartbeat_row(cfg) or {}
            checks.append(Check("service", "PASS", f"kdev serve pid {hb.get('pid')} hb {hb.get('ts_utc')}"))
        else:
            checks.append(Check("service", "WARN", "kdev serve not running: GPU commands run in --direct mode",
                                hint="kdev serve start"))

    def ncu() -> None:
        from kdev.service import smi
        ok = smi.ncu_permitted()
        if ok is None:
            checks.append(Check("ncu permission", "SKIP", "no /proc/driver/nvidia/params"))
        elif ok:
            checks.append(Check("ncu permission", "PASS", "RmProfilingAdminOnly=0"))
        else:
            checks.append(Check("ncu permission", "WARN", "RmProfilingAdminOnly=1: ncu counters blocked",
                                hint="use kdev profile --backend cupti / timeline instead"))

    for name, fn in (("gpu", gpu), ("gpu lock", lock_file), ("foreign CUDA", foreign), ("service", daemon),
                     ("ncu permission", ncu)):
        guard(name, fn)
    return checks
