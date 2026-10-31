"""Adapter lookup. Owner: P7.

Resolution of ``--problem P`` / a file path: (1) explicit adapter:problem ('okc:muse_moe_w4a16'); (2) a
problem dir containing reference.py + TASK.md -> okc; (3) a .cu with main() -> cu_main; (4) a file
defining Model/ModelNew -> kernelbench; (5) the directory of the submission file walking upwards;
(6) config default_problem. Unknown -> KdevError(invalid_request).
"""
from __future__ import annotations

from kdev.adapters.base import Adapter

ADAPTERS = {
    "okc": "kdev.adapters.okc:OkcAdapter",
    "cu_main": "kdev.adapters.cu_main:CuMainAdapter",
    "kernelbench": "kdev.adapters.kernelbench:KernelBenchAdapter",
}


def get_adapter(problem: str | None = None, file: str | None = None, adapter: str | None = None) -> Adapter:
    raise NotImplementedError("adapters.registry.get_adapter (P7)")


def detect(file: str) -> tuple[str, str] | None:
    """(adapter name, problem dir) for a submission path, or None."""
    raise NotImplementedError("adapters.registry.detect (P7)")
