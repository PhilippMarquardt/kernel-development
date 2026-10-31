"""`kdev` command-line entry point. Owner: P1.

Dispatch: resolve the command by longest prefix of argv words in the registry, build an argparse parser for
that command only (from its ``Arg`` list + global/GPU args), call the handler, render the Result. Packages add
commands/flags in their own ``commands.py``; this file only changes for global behaviour.
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from typing import Sequence

from kdev import __version__
from kdev.contract import ErrorCode, KdevError, Result, not_implemented
from kdev.registry import GLOBAL_ARGS, GPU_ARGS, Arg, Command, Context, Registry, build_registry

_TYPES = {"str": str, "path": str, "int": int, "float": float}


def _add_arg(p: argparse.ArgumentParser, a: Arg) -> None:
    kw: dict = {"help": a.help}
    if a.action in ("store_true",):
        kw["action"] = a.action
    else:
        if a.action == "append":
            kw["action"] = "append"
        kw["type"] = _TYPES.get(a.type, str)
        if a.choices is not None:
            kw["choices"] = list(a.choices)
        if a.nargs == "...":
            kw["nargs"] = argparse.REMAINDER
        elif a.nargs is not None:
            kw["nargs"] = a.nargs
    if not a.positional:
        kw["default"] = (bool(a.default) if a.action == "store_true" else a.default)
        kw["dest"] = a.dest
        if a.required:
            kw["required"] = True
    elif a.nargs in ("?", "*") and a.default is not None:
        kw["default"] = a.default
    p.add_argument(*a.flags, **kw)


def build_parser(cmd: Command) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=f"kdev {cmd.name}", description=cmd.help)
    for a in cmd.args:
        _add_arg(p, a)
    for a in GLOBAL_ARGS + (GPU_ARGS if cmd.gpu else []):
        if any(f in p._option_string_actions for f in a.flags):  # noqa: SLF001  (command overrides)
            continue
        _add_arg(p, a)
    return p


def top_help(reg: Registry) -> str:
    lines = [f"kdev {__version__} - agent-facing GPU kernel development toolkit", "",
             "usage: kdev <command> [args] [--json]", "", "commands:"]
    for c in reg.all():
        lines.append(f"  {c.name:18s} {c.help}")
    lines += ["", "kdev <command> --help for flags. Docs: docs/PLAN.md"]
    return "\n".join(lines)


def run(argv: Sequence[str] | None = None, reg: Registry | None = None) -> Result:
    """Parse argv and execute; returns the Result (no printing). Used by main() and tests."""
    argv = list(sys.argv[1:] if argv is None else argv)
    reg = reg or build_registry()
    cmd, rest = reg.resolve(argv)
    if cmd is None:
        raise KdevError(ErrorCode.INVALID_REQUEST, f"unknown command: {' '.join(argv[:2])!r}",
                        recovery="run `kdev --help` for the command list")
    args = build_parser(cmd).parse_args(rest)
    ctx = Context(json=getattr(args, "json", False),
                  agent=getattr(args, "agent", None) or os.environ.get("KDEV_AGENT", ""),
                  home=getattr(args, "home", None), problem=getattr(args, "problem", None),
                  quiet=getattr(args, "quiet", False))
    try:
        return cmd.handler(args, ctx)
    except NotImplementedError as e:
        r = not_implemented(cmd.name, str(e) or cmd.name)
        return r
    except KdevError as e:
        return Result.from_error(cmd.name, e)


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    want_json = "--json" in argv
    try:
        reg = build_registry()
        if not argv or argv[0] in ("-h", "--help", "help"):
            print(top_help(reg))
            return 0
        if argv[0] in ("--version", "-V"):
            print(__version__)
            return 0
        result = run(argv, reg)
    except KdevError as e:
        result = Result.from_error(argv[0] if argv else "kdev", e)
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001  - never a bare traceback on stdout
        traceback.print_exc(file=sys.stderr)
        result = Result.from_error(argv[0] if argv else "kdev",
                                   KdevError(ErrorCode.INTERNAL, f"{type(e).__name__}: {e}"))
    if want_json:
        print(result.to_json())
    else:
        from kdev.output import render
        print(render(result))
    return result.exit_code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
