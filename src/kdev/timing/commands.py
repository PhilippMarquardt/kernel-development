"""CLI: time, ab, calib (build JobSpecs and route them through ctx.submit). Owner: P3."""
from __future__ import annotations

from kdev.contract import Effects
from kdev.registry import Arg, Registry, todo

P = "P3"

FIDELITY_ARGS = [
    Arg(("--reps",), "paired repetitions per shape", type="int", default=40),
    Arg(("--seeds",), "input seeds per shape (default: adapter variance: 1 or 5)", type="int", default=None),
    Arg(("--flush",), "L2 flush method (non-read = recorded opt-out)", choices=("read", "write", "none"),
        default="read"),
    Arg(("--flush-mb",), "flush buffer MB (default: GPU profile)", type="int", default=None),
    Arg(("--eager",), "time eagerly instead of graph replay (opt-out)", action="store_true"),
    Arg(("--overhead",), "subtraction mode", choices=("judge", "local", "none"), default="judge"),
    Arg(("--allow-jit",), "allow compiling inside the GPU lane (opt-out, logged)", action="store_true"),
]

GPU_FX = Effects(gpu="queued", writes=("run_dir", "ledger"))


def register(reg: Registry) -> None:
    reg.command("time", "judge-faithful timing of one file vs the baseline", todo("timing.engine.run_time (P3)"),
                args=[Arg(("file",), "submission file", type="path"),
                      Arg(("--base",), "baseline|none|PATH", default="baseline")] + FIDELITY_ARGS,
                effects=GPU_FX, gpu=True, long_running=True, package=P)
    reg.command("ab", "interleaved A/B(/C...) timing with paired-ratio CIs and verdicts",
                todo("timing.engine.run_ab (P3)"),
                args=[Arg(("files",), "two or more files", type="path", nargs="+"),
                      Arg(("--order",), "interleave scheme", choices=("abba", "rotate"), default="abba"),
                      Arg(("--set",), "LABEL:K=V knob override (repeatable)", action="append", default=None)]
                + FIDELITY_ARGS,
                effects=GPU_FX, gpu=True, long_running=True, package=P)
    reg.command("calib", "judge-vs-local per-shape calibration (FILE:JUDGE.json ...)",
                todo("timing.calib.run_calib (P3)"),
                args=[Arg(("pairs",), "FILE:JUDGE.json (JUDGE '-' = local only)", nargs="+")] + FIDELITY_ARGS,
                effects=GPU_FX, gpu=True, long_running=True, package=P)
