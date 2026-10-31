"""CLI: sweep. Owner: P5."""
from __future__ import annotations

from kdev.contract import Effects
from kdev.registry import Arg, Registry, todo

P = "P5"


def register(reg: Registry) -> None:
    reg.command("sweep", "knob grid: parallel prebuild, quick check, batched interleaved ab vs base",
                todo("sweep.grid.run_sweep (P5)"),
                args=[Arg(("file",), "file with knobs (@override dict / -D defines)", type="path"),
                      Arg(("--grid",), "'K=a,b K2=c,d' or cublaslt:SHAPE", required=True),
                      Arg(("--base",), "base file (default: unmodified file)", type="path", default=None),
                      Arg(("--build-parallel",), "parallel builds", type="int", default=8),
                      Arg(("--top",), "top K points", type="int", default=5),
                      Arg(("--quick-check",), "1-seed check before timing", action="store_true")],
                effects=Effects(gpu="queued", writes=("run_dir", "cache", "ledger")), gpu=True,
                long_running=True, package=P)
