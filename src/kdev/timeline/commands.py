"""CLI: timeline. Owner: P6."""
from __future__ import annotations

from kdev.contract import Effects
from kdev.registry import Arg, Registry, todo

P = "P6"


def register(reg: Registry) -> None:
    reg.command("timeline", "per-CTA phase timeline from %globaltimer marks (+ Perfetto trace)",
                todo("timeline.instrument + timeline.run.run_timeline (P6)"),
                args=[Arg(("file",), "submission file with // @kdev:mark annotations or KD_MARK", type="path"),
                      Arg(("--shape",), "shape id", default=None),
                      Arg(("--marks",), "auto (annotations) | header (explicit KD_MARK)",
                          choices=("auto", "header"), default="auto"),
                      Arg(("--ctas",), "all | sample:N", default="all"),
                      Arg(("--ab",), "two define sets 'A_DEFS' 'B_DEFS'", nargs=2, default=None)],
                effects=Effects(gpu="queued", writes=("run_dir", "ledger")), gpu=True, long_running=True,
                package=P)
