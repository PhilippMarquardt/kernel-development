"""Standalone .cu with main() adapter (times itself through kdev/bench.cuh; later). Owner: P7.
v1: build + sass/regs/sanitize support only; timing via the program's own output is a stretch goal.

``load`` returns a handle {path, binary}; ``call`` runs the built binary (no graph capture: a separate
process), ``check`` passes when it exits 0 and prints no FAIL line. ``build_hook`` describes the single nvcc
compile unit (P5 builds it, outside the GPU lock). ``kdev sanitize`` runs the binary under compute-sanitizer.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from kdev.adapters.base import CheckResult, ComplyRules, ExtSpec, Shape, TimingProtocol
from kdev.contract import ErrorCode, KdevError
from kdev.util import Runner, run

ARCH = {"h200": "sm_90a", "h100": "sm_90a", "b200": "sm_100a", "a100": "sm_80"}
_MAIN_RE = re.compile(r"\bint\s+main\s*\(")


def has_main(path: str) -> bool:
    try:
        return bool(_MAIN_RE.search(Path(path).read_text(encoding="utf-8", errors="replace")))
    except OSError:
        return False


def binary_path(path: str) -> str:
    """Where the built executable is expected: <dir>/<stem> (``<stem>.exe`` on Windows), or $KDEV_CU_BINARY."""
    env = os.environ.get("KDEV_CU_BINARY")
    if env:
        return env
    p = Path(path)
    exe = p.with_suffix(".exe" if os.name == "nt" else "")
    return str(exe)


class CuMainAdapter:
    name = "cu_main"

    def __init__(self, path: str, gpu: str = "h200", runner: Runner = run) -> None:
        if not path.endswith((".cu", ".cuh", ".cpp")):
            raise KdevError(ErrorCode.INVALID_REQUEST, f"cu_main expects a .cu file, got {path}")
        self.path = os.path.abspath(path)
        self.problem = Path(path).stem
        self.problem_dir = os.path.dirname(self.path)
        self.gpu = gpu
        self.protocol = TimingProtocol(overhead_us=0.0, judge_reps=1, pairing="none", score="ratio")
        self.budget_s = self.protocol.budget_s
        self._runner = runner

    def shapes(self) -> list[Shape]:
        return [Shape("main", {}, weight=1.0, load="main")]

    def tests(self) -> list[Shape]:
        return [Shape("main", {}, scored=False)]

    def extra_shapes(self) -> list[Shape]:
        return []

    def input_variance(self) -> dict[str, bool]:
        return {}

    def judge_seed(self, shape: Shape) -> int | None:
        return None

    def load(self, path: str, tag: str) -> Any:
        return {"path": os.path.abspath(path), "binary": binary_path(path), "tag": tag}

    def baseline_path(self) -> str | None:
        return None

    def prepare(self, module: Any) -> Any:
        b = module["binary"]
        if not os.path.isfile(b):
            raise KdevError(ErrorCode.NEEDS_BUILD, f"no built binary {b}", recovery=f"run `kdev build {module['path']}`")
        return None

    def make_inputs(self, shape: Shape, seed: int) -> Any:
        return {"seed": seed}

    def call(self, module: Any, state: Any, inputs: Any) -> Any:
        env = dict(os.environ, KDEV_SEED=str(inputs.get("seed", 0)))
        return self._runner([module["binary"]], timeout=600, env=env)

    def check(self, inputs: Any, output: Any) -> CheckResult:
        rc = getattr(output, "returncode", 1)
        text = (getattr(output, "stdout", "") or "") + (getattr(output, "stderr", "") or "")
        bad = [ln for ln in text.splitlines() if re.search(r"\bFAIL", ln)]
        ok = rc == 0 and not bad
        return CheckResult(ok, f"exit {rc}" + (f"; {bad[0][:120]}" if bad else ""))

    def score(self, sub_us: dict[str, float], base_us: dict[str, float]) -> float:
        s, b = sub_us.get("main"), base_us.get("main")
        return float(b) / float(s) if s and b else float("nan")

    def rules(self) -> ComplyRules:
        return ComplyRules(forbid_patterns=[], no_cross_call_state=False, graph_capturable=False,
                           lossless_weights=False, notes=["standalone program: only static rules apply"])

    def build_hook(self, path: str) -> list[ExtSpec]:
        arch = ARCH.get(self.gpu, "sm_90a")
        return [ExtSpec(name=Path(path).stem, sources=[os.path.abspath(path)], inline=False,
                        cuda_cflags=["-O3", f"-arch={arch}", "-lineinfo", "-o", binary_path(path)])]

    def judge_cases(self, judge_json: dict[str, Any]) -> list[Shape]:
        return []
