"""Nsight Systems wrapper (if present): kernel trace without metrics. Owner: P6."""
from __future__ import annotations

from kdev.util import Runner, run


def available(runner: Runner = run) -> bool:
    raise NotImplementedError("nsys.available (P6)")


def nsys_argv(target_argv: list[str], out: str = "nsys") -> list[str]:
    """nsys profile --trace=cuda --sample=none --cpuctxsw=none -o out ..."""
    raise NotImplementedError("nsys.nsys_argv (P6)")


def parse_kern_sum(csv_text: str) -> list[dict]:
    """`nsys stats --report cuda_gpu_kern_sum --format csv` -> rows."""
    raise NotImplementedError("nsys.parse_kern_sum (P6)")
