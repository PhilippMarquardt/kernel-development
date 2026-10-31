"""CLI: build. Owner: P5."""
from __future__ import annotations

from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from kdev.build import _runs
from kdev.contract import Effects, ErrorCode, KdevError, Result
from kdev.registry import Arg, Context, Registry

P = "P5"


def coerce(v: str) -> Any:
    """'128' -> 128, '0.5' -> 0.5, 'true' -> True, else the string."""
    s = v.strip()
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    for t in (int, float):
        try:
            return t(s)
        except ValueError:
            pass
    return s


def parse_variants(items: list[str] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for it in items or []:
        for part in it.split():
            k, sep, v = part.partition("=")
            if not sep or not k:
                raise KdevError(ErrorCode.INVALID_REQUEST, f"--variant expects K=V, got {part!r}")
            out[k] = coerce(v)
    return out


def cmd_build(args: Namespace, ctx: Context) -> Result:
    from kdev.build.compile import build_file, build_result, report_rows  # noqa: PLC0415

    knobs = parse_variants(args.variant)
    defines = (args.defines or "").split()
    files = list(args.files)
    home = str(_runs.home(ctx))

    def one(f: str):
        return build_file(f, knobs=knobs, defines=defines, problem=ctx.problem, force=args.force,
                          verbose=args.verbose, home=home)

    with ThreadPoolExecutor(max_workers=max(1, min(args.jobs, len(files)))) as ex:
        reports = list(ex.map(one, files))
    res = build_result(reports, verbose=args.verbose)
    run = _runs.new_run(ctx, "build", Path(files[0]).stem)
    rows = report_rows(reports)
    p = _runs.write_json(run, "build.json", {"schema": "kdev.build/1", **rows, "knobs": knobs, "defines": defines,
                                             "reports": [r.to_dict() for r in reports]})
    res.files["build"] = str(p)
    res.summary["knobs"] = knobs
    return _runs.finish(run, res)


def register(reg: Registry) -> None:
    reg.command("build", "compile-only build with the pinned toolchain; ptxas regs/spills/C751x",
                cmd_build,
                args=[Arg(("files",), "files to build (.py with load_inline/load, or .cu)", type="path", nargs="+"),
                      Arg(("--variant",), "K=V knob, compiled in as -DK=V (repeatable)", action="append",
                          default=None),
                      Arg(("--defines",), "extra -D flags", default=""),
                      Arg(("--jobs",), "parallel builds", type="int", default=4),
                      Arg(("--force",), "ignore cache", action="store_true"),
                      Arg(("--verbose",), "full ptxas output in summary", action="store_true")],
                effects=Effects(gpu="none", writes=("cache", "run_dir", "ledger")), long_running=True,
                package=P)

