"""Core commands: version, doctor, show, runs list. Owner: P1."""
from __future__ import annotations

from argparse import Namespace

from kdev import __version__
from kdev.contract import Effects, Result
from kdev.output import table
from kdev.registry import Arg, Context, Registry


def cmd_version(args: Namespace, ctx: Context) -> Result:
    return Result(tool="version", verdict="ok", summary={"version": __version__}, lines=[__version__])


def cmd_doctor(args: Namespace, ctx: Context) -> Result:
    """Aggregate every package's probe() into PASS/WARN/FAIL lines (kdev.doctor.run_doctor)."""
    from kdev.doctor import run_doctor
    return run_doctor(ctx, problem=args.problem, fix_hints=args.fix_hints)


def cmd_show(args: Namespace, ctx: Context) -> Result:
    """Paged read of a run section (kdev.runs.show)."""
    from kdev.runs import show

    page = show(ctx.cfg, args.run_id, section=args.section, offset=args.offset, limit=args.limit)
    end = page.offset + len(page.lines)
    lines = list(page.lines)
    if page.next_offset is not None:
        footer = (f"[lines {page.offset + 1}-{end} of {page.total}]  next: kdev show {page.run_id} "
                  f"--section {page.section} --offset {page.next_offset} --limit {page.limit}")
    else:
        footer = f"[lines {page.offset + 1 if lines else page.offset}-{end} of {page.total}, end]"
    lines.append(footer)
    return Result(tool="show", verdict="ok", run_id=None,
                  summary={"run_id": page.run_id, "section": page.section, "offset": page.offset,
                           "limit": page.limit, "total": page.total, "next_offset": page.next_offset,
                           "path": page.path, "lines": page.lines},
                  files={"section": page.path}, lines=lines)


def cmd_runs_list(args: Namespace, ctx: Context) -> Result:
    from kdev.runs import list_runs

    rows = list_runs(ctx.cfg, tool=args.tool, limit=args.limit)
    lines = table(["run_id", "tool", "status", "verdict"],
                  [[r["run_id"], r["tool"], r["status"], r["verdict"]] for r in rows],
                  more_hint="kdev runs list --json")
    if not rows:
        lines = [f"no runs under {ctx.cfg.runs_dir}"]
    nxt = [f"kdev show {rows[0]['run_id']}"] if rows else []
    return Result(tool="runs list", verdict="ok", summary={"count": len(rows), "runs": rows},
                  lines=lines, next=nxt)


def register(reg: Registry) -> None:
    reg.command("version", "print the kdev version", cmd_version, package="P1")
    reg.command("doctor", "check GPU, toolchain pin, lock, permissions, backends", cmd_doctor,
                args=[Arg(("--fix-hints",), "print a fix hint per WARN/FAIL", action="store_true")],
                effects=Effects(gpu="none"), package="P1")
    reg.command("show", "paged read of a run's sections", cmd_show, args=[
        Arg(("run_id",), "run id or unique prefix"),
        Arg(("--section",), "summary|meta|result|log|raw|<files key>|<file in run dir>", default="summary"),
        Arg(("--offset",), "first line (0-based)", type="int", default=0),
        Arg(("--limit",), "max lines", type="int", default=60),
    ], package="P1")
    reg.command("runs list", "list recent run directories", cmd_runs_list, args=[
        Arg(("--tool",), "filter by tool", default=None),
        Arg(("--limit",), "max rows", type="int", default=20),
    ], package="P1")
