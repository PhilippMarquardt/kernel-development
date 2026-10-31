"""CLI: sync, remote, remote setup. Owner: P2."""
from __future__ import annotations

from kdev.contract import Effects
from kdev.registry import Arg, Registry, todo

P = "P2"


def register(reg: Registry) -> None:
    reg.command("sync", "md5-verified copy pod<->local (pull|push|both)", todo("remote.sync.sync (P2)"),
                args=[Arg(("direction",), "pull|push|both", choices=("pull", "push", "both"), nargs="?",
                          default="pull"),
                      Arg(("--since",), "only files changed in this window, e.g. 3h", default=None),
                      Arg(("--dry-run",), "plan only", action="store_true")],
                effects=Effects(writes=("files", "remote"), network="ssh"), mcp=False, package=P)
    reg.command("remote", "run a kdev command on the pod over ssh", todo("remote.ssh.run_remote (P2)"),
                args=[Arg(("--host",), "pod host", default=None), Arg(("--port",), "ssh port", type="int",
                                                                    default=None),
                      Arg(("--pull",), "pull the run dir afterwards", action="store_true"),
                      Arg(("argv",), "-- kdev ...", nargs="...")],
                effects=Effects(network="ssh"), mcp=False, package=P)
    reg.command("remote setup", "idempotent pod bootstrap (toolchain pin, venv, self-tests)",
                todo("remote.bootstrap.setup (P2)"),
                args=[Arg(("--force",), "re-run done phases", action="store_true")],
                effects=Effects(writes=("remote",), network="ssh"), mcp=False, package=P)
