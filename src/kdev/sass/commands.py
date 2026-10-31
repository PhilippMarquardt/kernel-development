"""CLI: sass, sass diff. Owner: P5."""
from __future__ import annotations

from kdev.contract import Effects
from kdev.registry import Arg, Registry, todo

P = "P5"
FX = Effects(gpu="none", writes=("run_dir", "ledger"))


def register(reg: Registry) -> None:
    reg.command("sass", "hot-loop opcode mix, per-element histogram, audits (pdl,spills,serialize,repeat,expect)",
                todo("sass.loops.hot_loops + sass.audits.run_audits (P5)"),
                args=[Arg(("target",), "submission .py (built), .so or .cubin", type="path"),
                      Arg(("--kernel",), "kernel name regex", default=None),
                      Arg(("--loops",), "show top hot loops", action="store_true"),
                      Arg(("--top",), "number of hot loops", type="int", default=1),
                      Arg(("--per",), "elements per loop iteration (per-element histogram)", type="float",
                          default=None),
                      Arg(("--audit",), "comma list: pdl,spills,serialize,repeat,expect=HGMMA",
                          default="pdl,spills,serialize,repeat")],
                effects=FX, package=P)
    reg.command("sass diff", "build-vs-build SASS diff", todo("sass.diff.diff_kernels (P5)"),
                args=[Arg(("a",), "build A", type="path"), Arg(("b",), "build B", type="path"),
                      Arg(("--kernel",), "kernel name regex", default=None)],
                effects=FX, package=P)
