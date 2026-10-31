"""Command registry: the single source of truth for the CLI (kdev.cli) and the MCP server (kdev.mcp).

Owner: P1. Each package declares its commands in its own ``commands.py`` via ``register(reg)``; nobody edits
``cli.py`` to add flags. See docs/PLAN.md section 2.1.
"""
from __future__ import annotations

import importlib
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any, Callable

from kdev.contract import Effects, Result

# Packages whose ``commands`` module is imported by build_registry(), in help order.
COMMAND_MODULES = (
    "kdev.core_commands",
    "kdev.build.commands",
    "kdev.checks.commands",
    "kdev.timing.commands",
    "kdev.sweep.commands",
    "kdev.sass.commands",
    "kdev.occupancy.commands",
    "kdev.timeline.commands",
    "kdev.floor.commands",
    "kdev.observe.commands",
    "kdev.service.commands",
    "kdev.ledger.commands",
    "kdev.judge.commands",
    "kdev.collab.commands",
    "kdev.dashboard.commands",
    "kdev.remote.commands",
    "kdev.mcp.commands",
    "kdev.slim.commands",
)


@dataclass
class Arg:
    """One CLI/MCP parameter. ``flags[0]`` without a leading '-' is a positional."""

    flags: tuple[str, ...]
    help: str = ""
    type: str = "str"              # str | int | float | bool | path
    default: Any = None
    choices: tuple[Any, ...] | None = None
    nargs: str | None = None       # None | "?" | "*" | "+" | "..." (REMAINDER)
    required: bool = False
    action: str | None = None      # None | "store_true" | "append"
    mcp: bool = True               # False = not exposed over MCP (safety)

    @property
    def positional(self) -> bool:
        return not self.flags[0].startswith("-")

    @property
    def dest(self) -> str:
        f = self.flags[0] if self.positional else next(x for x in self.flags if x.startswith("--"))
        return f.lstrip("-").replace("-", "_").lower()


@dataclass
class Context:
    """Per-invocation context handed to every handler."""

    json: bool = False
    agent: str = ""
    home: str | None = None
    problem: str | None = None
    quiet: bool = False
    config: Any = None   # kdev.config.Config, loaded lazily by P1 (use ``ctx.cfg``)
    via: str = "cli"     # cli | mcp

    @property
    def cfg(self) -> Any:
        """The loaded :class:`kdev.config.Config` (``--home`` honoured); loaded on first use."""
        if self.config is None:
            from kdev.config import load_config
            self.config = load_config(home=self.home)
        return self.config

    @property
    def problem_name(self) -> str | None:
        """``--problem`` or the config's ``default_problem`` (None if neither)."""
        return self.problem or (self.cfg.default_problem or None)

    def submit(self, spec: Any, *, detach: bool = False, direct: bool = False,
               timeout_s: float | None = None) -> Result:
        """Route a GPU JobSpec: service if up, else direct (flock) mode. See PLAN 2.2.

        Fills ``job_id``, ``submitted_utc``, ``agent`` and ``problem`` when empty, then delegates to
        ``kdev.service.client.submit_and_maybe_wait``. ``KDEV_NO_GPU=1`` + a GPU-lane spec ->
        ``capability_unavailable`` without touching the service (unit tests)."""
        from kdev.config import no_gpu
        from kdev.contract import ErrorCode, KdevError
        from kdev.models import JobSpec
        from kdev.util import new_id, utc_now

        if isinstance(spec, dict):
            spec = JobSpec.from_dict(spec)
        if not isinstance(spec, JobSpec):
            raise KdevError(ErrorCode.INVALID_REQUEST, f"submit() needs a JobSpec, got {type(spec).__name__}")
        spec.job_id = spec.job_id or new_id("j")
        spec.submitted_utc = spec.submitted_utc or utc_now()
        spec.agent = spec.agent or self.agent
        spec.problem = spec.problem or self.problem_name
        if timeout_s is not None:
            spec.timeout_s = int(timeout_s)
        if no_gpu() and spec.resolved_lane() == "gpu":
            raise KdevError(ErrorCode.CAPABILITY_UNAVAILABLE, f"{spec.kind}: GPU disabled (KDEV_NO_GPU=1)",
                            details={"job_id": spec.job_id, "kind": spec.kind},
                            recovery="unset KDEV_NO_GPU and run on the GPU host (or via `kdev remote`)")
        from kdev.service import client
        return client.submit_and_maybe_wait(self.cfg, spec, detach=detach, direct=direct, timeout_s=timeout_s)

    def submit_from_args(self, spec: Any, args: Namespace) -> Result:
        """``submit`` with the standard GPU flags (--detach/--direct/--timeout/--priority/--run-name)."""
        if getattr(args, "priority", None) is not None and hasattr(spec, "priority"):
            spec.priority = int(args.priority)
        if getattr(args, "run_name", None) and hasattr(spec, "run_name") and not spec.run_name:
            spec.run_name = args.run_name
        return self.submit(spec, detach=bool(getattr(args, "detach", False)),
                           direct=bool(getattr(args, "direct", False)), timeout_s=getattr(args, "timeout", None))


Handler = Callable[[Namespace, Context], Result]


@dataclass
class Command:
    name: str                       # "ab", "sass diff", "job wait"
    help: str
    handler: Handler
    args: list[Arg] = field(default_factory=list)
    effects: Effects = field(default_factory=Effects)
    gpu: bool = False               # adds GPU flags (--detach/--direct/--timeout/--priority/--run-name/--shapes)
    long_running: bool = False
    mcp: bool = True                # exposed over MCP at all
    package: str = ""


GPU_ARGS: list[Arg] = [
    Arg(("--detach",), "enqueue and return the job id immediately", action="store_true"),
    Arg(("--direct",), "no daemon: run in the foreground under the exclusive flock", action="store_true"),
    Arg(("--timeout",), "job timeout in seconds", type="int", default=1800),
    Arg(("--priority",), "queue priority (higher first)", type="int", default=0),
    Arg(("--run-name",), "slug for the run directory", default=""),
    Arg(("--shapes",), "substring filter over shape ids / cases", default=None),
]

GLOBAL_ARGS: list[Arg] = [
    Arg(("--json",), "print exactly one JSON Result object", action="store_true", mcp=False),
    Arg(("--problem",), "problem name or directory", default=None),
    Arg(("--agent",), "agent name for attribution (default $KDEV_AGENT)", default=None),
    Arg(("--home",), "KDEV_HOME override", default=None, mcp=False),
    Arg(("--quiet",), "suppress next: hints", action="store_true", mcp=False),
]


class Registry:
    def __init__(self) -> None:
        self._cmds: dict[str, Command] = {}

    def command(self, name: str, help: str, handler: Handler, *, args: list[Arg] | None = None,
                effects: Effects | None = None, gpu: bool = False, long_running: bool = False,
                mcp: bool = True, package: str = "") -> Command:
        if name in self._cmds:
            raise ValueError(f"duplicate command {name!r}")
        cmd = Command(name, help, handler, list(args or []), effects or Effects(), gpu, long_running, mcp,
                      package)
        self._cmds[name] = cmd
        return cmd

    def get(self, name: str) -> Command:
        return self._cmds[name]

    def all(self) -> list[Command]:
        return list(self._cmds.values())

    def names(self) -> list[str]:
        return list(self._cmds)

    def resolve(self, argv: list[str]) -> tuple[Command | None, list[str]]:
        """Longest-prefix match of argv words against command names ("sass diff" beats "sass")."""
        for n in range(min(3, len(argv)), 0, -1):
            name = " ".join(argv[:n])
            if name in self._cmds:
                return self._cmds[name], argv[n:]
        return None, argv

    def group(self, prefix: str) -> list[Command]:
        """Commands under a group word ('job' -> job submit/status/...)."""
        return [c for c in self._cmds.values() if c.name.startswith(prefix + " ")]

    def suggest(self, words: list[str], n: int = 3) -> list[str]:
        """Closest command names for an unknown command (difflib)."""
        import difflib

        names = self.names()
        guess = " ".join(words[:2])
        out = difflib.get_close_matches(guess, names, n=n, cutoff=0.5)
        if not out and words:
            out = difflib.get_close_matches(words[0], names, n=n, cutoff=0.5)
        return out


def build_registry(modules: tuple[str, ...] = COMMAND_MODULES) -> Registry:
    reg = Registry()
    for mod in modules:
        importlib.import_module(mod).register(reg)
    return reg


def todo(target: str) -> Handler:
    """Phase-0 placeholder handler: raises NotImplementedError naming the library function and owner.
    The CLI/MCP turn it into a not_implemented Result (exit 14). Builders replace each todo(...) with a real
    handler ``def cmd_x(args, ctx) -> Result``."""

    def handler(args: Namespace, ctx: Context) -> Result:
        raise NotImplementedError(target)

    handler.__doc__ = f"TODO: {target}"
    return handler
