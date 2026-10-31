"""CLI: floor, floor calibrate. Owner: P6."""
from __future__ import annotations

from kdev.contract import Effects
from kdev.registry import Arg, Registry, todo

P = "P6"


def register(reg: Registry) -> None:
    reg.command("floor", "roofline + latency-skeleton floor (judge-style and achievable)",
                todo("floor.roofline + floor.skeleton (P6)"),
                args=[Arg(("--shape",), "shape id (with --problem)", default=None),
                      Arg(("--bytes",), "bytes moved", type="float", default=None),
                      Arg(("--flops",), "flops", type="float", default=None),
                      Arg(("--dtype",), "bf16|fp8|fp4|fp32", default="bf16"),
                      Arg(("--plan",), "skeleton plan TOML", type="path", default=None),
                      Arg(("--gpu",), "GPU profile name (default: detected)", default=None)],
                effects=Effects(gpu="none"), package=P)
    reg.command("floor calibrate", "run the microbench suite and fill the GPU profile's measured constants",
                todo("floor.microbench.run_calibrate (P6)"),
                args=[Arg(("--gpu",), "profile name", default=None),
                      Arg(("--write",), "write $KDEV_HOME/gpus/<gpu>.toml", action="store_true")],
                effects=Effects(gpu="queued", writes=("run_dir", "files")), gpu=True, long_running=True,
                package=P)
