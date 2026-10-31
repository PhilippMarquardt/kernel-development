"""GPU job service (`kdev serve`), exclusive lock, nvidia-smi helpers. Owner: P2.

Modules: lock (flock + foreign-context wait, gpu_run.sh semantics), smi (nvidia-smi queries), queue (SQLite
queue.db), daemon (build lane + serial GPU lane), client (submit/wait/status/cancel), commands (CLI).
"""
from __future__ import annotations

from kdev.models import Check


def probe(ctx: object, problem: str | None = None) -> list[Check]:
    """Doctor checks: GPU visible (nvidia-smi), lock file writable, foreign CUDA processes, daemon heartbeat,
    ncu permission (RmProfilingAdminOnly in /proc/driver/nvidia/params). Never raises."""
    raise NotImplementedError("service.probe (P2)")
