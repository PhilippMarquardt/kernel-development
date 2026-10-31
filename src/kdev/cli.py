"""`kdev` command-line entry point. Owner: P1.

Dispatch: resolve the command by longest prefix of argv words in the registry, build an argparse parser for
that command only (from its ``Arg`` list + global/GPU args), call the handler, render the Result. Packages add
commands/flags in their own ``commands.py``; this file only changes for global behaviour.
"""
from __future__ import annotations

import argparse
import contextlib
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


class _Parser(argparse.ArgumentParser):
    """argparse that raises invalid_request instead of printing usage + exit 2 (so --json stays one object)."""

    def error(self, message: str):  # type: ignore[override]
        raise KdevError(ErrorCode.INVALID_REQUEST, f"{self.prog}: {message}",
                        details={"usage": self.format_usage().strip()},
                        recovery=f"run `{self.prog} --help` for the flags")


def build_parser(cmd: Command) -> argparse.ArgumentParser:
    p = _Parser(prog=f"kdev {cmd.name}", description=cmd.help)
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


def group_help(reg: Registry, prefix: str) -> str:
    lines = [f"usage: kdev {prefix} <subcommand> [args] [--json]", "", "subcommands:"]
    lines += [f"  {c.name:18s} {c.help}" for c in reg.group(prefix)]
    return "\n".join(lines)


def _flags_before_dashdash(argv: Sequence[str]) -> list[str]:
    out = []
    for a in argv:
        if a == "--":
            break
        out.append(a)
    return out


def make_context(args: argparse.Namespace, via: str = "cli") -> Context:
    """Context for a parsed command line (config loaded eagerly so config errors surface as Results)."""
    from kdev.config import load_config

    home = getattr(args, "home", None)
    cfg = load_config(home=home)
    problem = getattr(args, "problem", None) or (cfg.default_problem or None)
    return Context(json=bool(getattr(args, "json", False)),
                   agent=getattr(args, "agent", None) or os.environ.get("KDEV_AGENT", ""),
                   home=home, problem=problem, quiet=bool(getattr(args, "quiet", False)), config=cfg, via=via)


def run(argv: Sequence[str] | None = None, reg: Registry | None = None) -> Result:
    """Parse argv and execute; returns the Result (no printing). Used by main() and tests.

    Unknown commands and argparse errors raise ``KdevError(invalid_request)``; everything raised by a
    handler is converted into an error Result (NotImplementedError -> not_implemented, other exceptions ->
    internal with the traceback on stderr)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    reg = reg or build_registry()
    cmd, rest = reg.resolve(argv)
    if cmd is None:
        sub = reg.group(argv[0]) if argv else []
        if sub:
            raise KdevError(ErrorCode.INVALID_REQUEST, f"`kdev {argv[0]}` needs a subcommand",
                            details={"subcommands": [c.name for c in sub]},
                            recovery=f"use one of: {', '.join(c.name for c in sub)}")
        hint = reg.suggest(argv)
        raise KdevError(ErrorCode.INVALID_REQUEST, f"unknown command: {' '.join(argv[:2])!r}",
                        details={"suggestions": hint},
                        recovery=(f"did you mean: {', '.join(hint)}?" if hint else
                                  "run `kdev --help` for the command list"))
    args = build_parser(cmd).parse_args(rest)
    try:
        ctx = make_context(args)
        result = cmd.handler(args, ctx)
    except NotImplementedError as e:
        return not_implemented(cmd.name, str(e) or cmd.name)
    except KdevError as e:
        return Result.from_error(cmd.name, e)
    except Exception as e:  # noqa: BLE001 - never a bare traceback on stdout
        traceback.print_exc(file=sys.stderr)
        return Result.from_error(cmd.name, KdevError(ErrorCode.INTERNAL, f"{type(e).__name__}: {e}",
                                                     recovery="report the traceback printed on stderr"))
    if not (hasattr(result, "to_dict") and hasattr(result, "exit_code")):  # duck-typed (module reloads)
        return Result.from_error(cmd.name, KdevError(
            ErrorCode.INTERNAL, f"handler for {cmd.name!r} returned {type(result).__name__}, not Result"))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    flags = _flags_before_dashdash(argv)
    want_json = "--json" in flags
    quiet = "--quiet" in flags
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass
    try:
        reg = build_registry()
        if not argv or argv[0] in ("-h", "--help", "help"):
            print(top_help(reg))
            return 0
        if argv[0] in ("--version", "-V"):
            print(__version__)
            return 0
        if reg.resolve(argv)[0] is None and reg.group(argv[0]) and (len(argv) == 1 or argv[1] in ("-h", "--help")):
            print(group_help(reg, argv[0]))
            return 0
        # with --json, stdout carries exactly one JSON object: anything a handler prints goes to stderr
        with contextlib.redirect_stdout(sys.stderr) if want_json else contextlib.nullcontext():
            result = run(argv, reg)
    except KdevError as e:
        result = Result.from_error(argv[0] if argv else "kdev", e)
    except SystemExit as e:  # argparse --help
        return e.code if isinstance(e.code, int) else 0
    except Exception as e:  # noqa: BLE001  - never a bare traceback on stdout
        traceback.print_exc(file=sys.stderr)
        result = Result.from_error(argv[0] if argv else "kdev",
                                   KdevError(ErrorCode.INTERNAL, f"{type(e).__name__}: {e}"))
    if want_json:
        print(result.to_json())
    else:
        from kdev.output import render
        print(render(result, quiet=quiet))
    return result.exit_code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
