"""Standalone .cu with main() adapter (times itself through kdev/bench.cuh; later). Owner: P7.
v1: build + sass/regs/sanitize support only; timing via the program's own output is a stretch goal."""
from __future__ import annotations


class CuMainAdapter:
    name = "cu_main"

    def __init__(self, path: str) -> None:
        raise NotImplementedError("CuMainAdapter (P7)")
