"""CLI: profile, dispatch. Owner: P6."""
from __future__ import annotations

from kdev.contract import Effects
from kdev.registry import Arg, Registry, todo

P = "P6"
FX = Effects(gpu="queued", writes=("run_dir", "ledger"))


def register(reg: Registry) -> None:
    reg.command("profile", "top kernels by time (ncu if permitted, else CUPTI; nsys if present)",
                todo("observe.cupti.run_profile (P6)"),
                args=[Arg(("file",), "submission", type="path"),
                      Arg(("--backend",), "auto|ncu|nsys|cupti", choices=("auto", "ncu", "nsys", "cupti"),
                          default="auto"),
                      Arg(("--metrics",), "ncu section set", default=None),
                      Arg(("--nopdl",), "disable PDL so per-kernel times separate", action="store_true")],
                effects=FX, gpu=True, long_running=True, package=P)
    reg.command("dispatch", "which kernels launch per scored shape (CUPTI names x count x grid)",
                todo("observe.dispatch.run_dispatch (P6)"),
                args=[Arg(("file",), "submission", type="path")],
                effects=FX, gpu=True, long_running=True, package=P)
