"""NVBit (optional): instruction counts without counter permission. Owner: P6. v1 = probe only."""
from __future__ import annotations


def available() -> bool:
    """NVBIT_PATH set and the instr_count tool .so present."""
    raise NotImplementedError("nvbit.available (P6)")
