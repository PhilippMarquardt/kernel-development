"""CLI: sweep. Owner: P5."""
from __future__ import annotations

from argparse import Namespace

from kdev.contract import Effects, Result
from kdev.registry import Arg, Context, Registry

P = "P5"


def cmd_sweep(args: Namespace, ctx: Context) -> Result:
    from kdev.sweep.grid import parse_grid, run_sweep  # noqa: PLC0415

    params = {}
    if getattr(args, "shapes", None):
        params["shapes"] = args.shapes
    return run_sweep(args.file, parse_grid(args.grid), base=args.base, top=args.top,
                     build_parallel=args.build_parallel, quick_check=args.quick_check, ctx=ctx,
                     dry_run=args.dry_run, params=params,
                     drop=tuple(x for x in ("build_failed", "spills", "c751x")
                                if not (args.keep_spills and x == "spills")))


def register(reg: Registry) -> None:
    reg.command("sweep", "knob grid: parallel prebuild, quick check, batched interleaved ab vs base",
                cmd_sweep,
                args=[Arg(("file",), "file with knobs (compiled in as -DK=V; also $KDEV_KNOBS JSON)", type="path"),
                      Arg(("--grid",), "'K=a,b K2=c,d' or cublaslt:SHAPE", required=True),
                      Arg(("--base",), "base file (default: unmodified file)", type="path", default=None),
                      Arg(("--build-parallel",), "parallel builds", type="int", default=8),
                      Arg(("--top",), "top K points", type="int", default=5),
                      Arg(("--quick-check",), "1-seed check before timing", action="store_true"),
                      Arg(("--keep-spills",), "time points whose kernels spill (dropped by default)",
                          action="store_true"),
                      Arg(("--dry-run",), "print the points and job plan only", action="store_true")],
                effects=Effects(gpu="queued", writes=("run_dir", "cache", "ledger")), gpu=True,
                long_running=True, package=P)
