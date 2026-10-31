"""CLI: check, sanitize, comply. Owner: P7."""
from __future__ import annotations

from kdev.contract import Effects
from kdev.registry import Arg, Registry, todo

P = "P7"
FX = Effects(gpu="queued", writes=("run_dir", "ledger"))


def register(reg: Registry) -> None:
    reg.command("check", "correctness: fresh seeds, extra shapes, eager+graph, repeats; classifies failures",
                todo("checks.check.run_check (P7)"),
                args=[Arg(("file",), "submission", type="path"),
                      Arg(("--seeds",), "fresh seeds per case", type="int", default=3),
                      Arg(("--repeat",), "repeats per case (race detection)", type="int", default=3),
                      Arg(("--mode",), "graph|eager|both", choices=("graph", "eager", "both"), default="both"),
                      Arg(("--extra-shapes",), "add held-out shapes", action="store_true"),
                      Arg(("--tests-only",), "only the adapter's tests", action="store_true"),
                      Arg(("--cold",), "measure prepare+compile from an empty cache vs budget",
                          action="store_true")],
                effects=FX, gpu=True, long_running=True, package=P)
    reg.command("sanitize", "compute-sanitizer on one small shape (eager + graph)",
                todo("checks.sanitize (P7)"),
                args=[Arg(("file",), "submission", type="path"),
                      Arg(("--tool",), "sanitizer tool", choices=("memcheck", "racecheck", "synccheck",
                                                                  "initcheck"), default="memcheck"),
                      Arg(("--shape",), "shape id (default: smallest test)", default=None)],
                effects=FX, gpu=True, long_running=True, package=P)
    reg.command("comply", "rules checklist: precision per op, cross-call state, work outside timed call",
                todo("checks.comply.run_comply (P7)"),
                args=[Arg(("file",), "submission", type="path"),
                      Arg(("--static-only",), "no GPU part", action="store_true")],
                effects=FX, gpu=True, long_running=True, package=P)
