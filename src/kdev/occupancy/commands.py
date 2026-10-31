"""CLI: regs. Owner: P6."""
from __future__ import annotations

from kdev.contract import Effects
from kdev.registry import Arg, Registry, todo

P = "P6"


def register(reg: Registry) -> None:
    reg.command("regs", "static occupancy per kernel: regs/smem/limiter/warps-per-SMSP rule",
                todo("occupancy.calc.occupancy (P6)"),
                args=[Arg(("target",), "built .py, .so or .cubin", type="path"),
                      Arg(("--kernel",), "kernel regex", default=None),
                      Arg(("--block",), "threads per block (if not in launch bounds)", type="int", default=None),
                      Arg(("--smem",), "dynamic smem bytes", type="int", default=0),
                      Arg(("--gpu",), "GPU profile", default=None)],
                effects=Effects(gpu="none"), package=P)
