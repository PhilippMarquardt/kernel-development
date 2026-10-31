"""NVBit (optional): instruction counts without counter permission. Owner: P6. v1 = probe only."""
from __future__ import annotations

import os

TOOL_CANDIDATES = ("tools/instr_count/instr_count.so", "instr_count.so", "lib/instr_count.so")


def tool_path() -> str | None:
    root = os.environ.get("NVBIT_PATH", "")
    if not root:
        return None
    for rel in TOOL_CANDIDATES:
        p = os.path.join(root, rel)
        if os.path.isfile(p):
            return p
    return None


def available() -> bool:
    """NVBIT_PATH set and the instr_count tool .so present."""
    return tool_path() is not None
