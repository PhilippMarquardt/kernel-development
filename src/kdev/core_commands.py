"""Core commands: version, doctor, show, runs list. Owner: P1."""
from __future__ import annotations

from argparse import Namespace

from kdev import __version__
from kdev.contract import Effects, Result
from kdev.registry import Arg, Context, Registry


def cmd_version(args: Namespace, ctx: Context) -> Result:
    return Result(tool="version", verdict="ok", summary={"version": __version__}, lines=[__version__])


def cmd_doctor(args: Namespace, ctx: Context) -> Result:
    """Aggregate every package's probe() into PASS/WARN/FAIL lines (kdev.doctor.run_doctor)."""
    from kdev.doctor import run_doctor
    return run_doctor(ctx, problem=args.problem, fix_hints=args.fix_hints)


def cmd_show(args: Namespace, ctx: Context) -> Result:
    """Paged read of a run section (kdev.runs.show)."""
    raise NotImplementedError("show -> kdev.runs.show (P1)")


def cmd_runs_list(args: Namespace, ctx: Context) -> Result:
    raise NotImplementedError("runs list -> kdev.runs.list_runs (P1)")


def register(reg: Registry) -> None:
    reg.command("version", "print the kdev version", cmd_version, package="P1")
    reg.command("doctor", "check GPU, toolchain pin, lock, permissions, backends", cmd_doctor,
                args=[Arg(("--fix-hints",), "print a fix hint per WARN/FAIL", action="store_true")],
                effects=Effects(gpu="none"), package="P1")
    reg.command("show", "paged read of a run's sections", cmd_show, args=[
        Arg(("run_id",), "run id or unique prefix"),
        Arg(("--section",), "summary|meta|log|raw|<files key>", default="summary"),
        Arg(("--offset",), "first line", type="int", default=0),
        Arg(("--limit",), "max lines", type="int", default=60),
    ], package="P1")
    reg.command("runs list", "list recent run directories", cmd_runs_list, args=[
        Arg(("--tool",), "filter by tool", default=None),
        Arg(("--limit",), "max rows", type="int", default=20),
    ], package="P1")
