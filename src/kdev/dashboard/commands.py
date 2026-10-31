"""CLI: dashboard build|watch. Owner: P8."""
from __future__ import annotations

from kdev.contract import Effects
from kdev.registry import Arg, Registry, todo

P = "P8"


def register(reg: Registry) -> None:
    reg.command("dashboard build", "write the static HTML dashboard (+ dashboard.json)",
                todo("dashboard.collect + dashboard.render.write (P8)"),
                args=[Arg(("--out",), "output dir (default $KDEV_HOME/dashboard)", type="path", default=None),
                      Arg(("--scrape",), "include read-only public board scrape", action="store_true")],
                effects=Effects(writes=("files",), network="okc_readonly"), package=P)
    reg.command("dashboard watch", "rebuild the dashboard every N seconds", todo("dashboard (P8)"),
                args=[Arg(("--every",), "seconds", type="int", default=300),
                      Arg(("--out",), "output dir", type="path", default=None),
                      Arg(("--scrape",), "include board scrape", action="store_true")],
                effects=Effects(writes=("files",), network="okc_readonly"), long_running=True, mcp=False,
                package=P)
