"""CLI: profile, dispatch. Owner: P6."""
from __future__ import annotations

from argparse import Namespace

from kdev.contract import Effects, Result
from kdev.registry import Arg, Context, Registry

P = "P6"
FX = Effects(gpu="queued", writes=("run_dir", "ledger"))


def cmd_profile(args: Namespace, ctx: Context) -> Result:
    from kdev.observe._jobs import file_ref, make_spec, submit
    extra = {"backend": args.backend, "metrics": args.metrics, "nopdl": bool(args.nopdl), "reps": args.reps,
             "kernel": args.kernel, "shape": args.shape}
    spec = make_spec("profile", [file_ref(args.file)], args, ctx, extra)
    return submit(spec, args, ctx)


def cmd_dispatch(args: Namespace, ctx: Context) -> Result:
    from kdev.observe._jobs import file_ref, make_spec, submit
    spec = make_spec("dispatch", [file_ref(args.file)], args, ctx, {}, graph=not args.eager_only)
    return submit(spec, args, ctx)


def register(reg: Registry) -> None:
    reg.command("profile", "top kernels by time (ncu if permitted, else CUPTI; nsys if present)",
                cmd_profile,
                args=[Arg(("file",), "submission", type="path"),
                      Arg(("--backend",), "auto|ncu|nsys|cupti", choices=("auto", "ncu", "nsys", "cupti"),
                          default="auto"),
                      Arg(("--metrics",), "ncu sections (sol,occupancy,warps,mem,... or ncu section ids)",
                          default=None),
                      Arg(("--nopdl",), "disable PDL (KDEV_NOPDL=1/OKC_NOPDL=1) so per-kernel times separate",
                          action="store_true"),
                      Arg(("--reps",), "profiled replays per shape (cupti)", type="int", default=20),
                      Arg(("--kernel",), "ncu kernel-name regex", default=None),
                      Arg(("--shape",), "one shape id", default=None),
                      Arg(("--allow-jit",), "allow compiling inside the GPU lane (opt-out, logged)",
                          action="store_true")],
                effects=FX, gpu=True, long_running=True, package=P)
    reg.command("dispatch", "which kernels launch per scored shape (CUPTI names x count x grid)",
                cmd_dispatch,
                args=[Arg(("file",), "submission", type="path"),
                      Arg(("--eager-only",), "skip the graph-replay probe", action="store_true"),
                      Arg(("--allow-jit",), "allow compiling inside the GPU lane (opt-out, logged)",
                          action="store_true")],
                effects=FX, gpu=True, long_running=True, package=P)
