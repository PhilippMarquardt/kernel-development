"""KernelBench-style adapter (Model / ModelNew, get_inputs, get_init_inputs). Owner: P7. Stub in v1.

The problem file defines ``Model``, ``get_inputs()`` and ``get_init_inputs()``; a submission defines
``ModelNew`` (or a ``Model`` replacement). Like KernelBench: both models are built after
``torch.manual_seed(42)`` from ``get_init_inputs()``, inputs come from ``get_inputs()`` after
``torch.manual_seed(seed)``, correctness is ``allclose(atol=rtol=1e-2)`` per output. The baseline is the
problem's own ``Model``. One shape ('default') of weight 1; score = base / sub.
"""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any

from kdev.adapters.base import CheckResult, ComplyRules, ExtSpec, Shape, TimingProtocol
from kdev.adapters.okc import discover_extensions, load_module
from kdev.contract import ErrorCode, KdevError

INIT_SEED = 42
ATOL = RTOL = 1e-2
_DEF_RE = re.compile(r"^class\s+(ModelNew|Model)\b|^def\s+get_inputs\b", re.M)


def looks_like_kernelbench(path: str) -> bool:
    try:
        return bool(_DEF_RE.search(Path(path).read_text(encoding="utf-8", errors="replace")))
    except OSError:
        return False


class KernelBenchAdapter:
    name = "kernelbench"

    def __init__(self, path: str, gpu: str = "h200") -> None:
        self.path = os.path.abspath(path)
        if not os.path.isfile(self.path):
            raise KdevError(ErrorCode.NOT_FOUND, f"kernelbench problem file not found: {path}")
        self.problem = Path(path).stem
        self.problem_dir = os.path.dirname(self.path)
        self.gpu = gpu
        self.protocol = TimingProtocol(score="ratio")
        self.budget_s = self.protocol.budget_s
        self._ref: Any = None
        self._ref_model: Any = None

    # pure part
    def shapes(self) -> list[Shape]:
        return [Shape("default", {}, weight=1.0, load="default")]

    def tests(self) -> list[Shape]:
        return [Shape("default", {}, scored=False)]

    def extra_shapes(self) -> list[Shape]:
        return []

    def input_variance(self) -> dict[str, bool]:
        return {}

    def judge_seed(self, shape: Shape) -> int | None:
        return None

    def baseline_path(self) -> str | None:
        return self.path

    def score(self, sub_us: dict[str, float], base_us: dict[str, float]) -> float:
        s, b = sub_us.get("default"), base_us.get("default")
        return float(b) / float(s) if s and b else float("nan")

    def rules(self) -> ComplyRules:
        return ComplyRules(forbid_patterns=[r"\.item\(\)", r"\.cpu\(\)", r"torch\.cuda\.synchronize"],
                           lossless_weights=False, notes=["KernelBench: ModelNew must match Model within 1e-2"])

    def build_hook(self, path: str) -> list[ExtSpec]:
        return discover_extensions(path)

    def judge_cases(self, judge_json: dict[str, Any]) -> list[Shape]:
        return []

    # worker part
    def reference(self) -> Any:
        if self._ref is None:
            self._ref = load_module(self.path, "_kdev_kb_ref_" + re.sub(r"\W", "_", self.problem))
            for attr in ("Model", "get_inputs", "get_init_inputs"):
                if not hasattr(self._ref, attr):
                    raise KdevError(ErrorCode.INVALID_REQUEST, f"{self.path} lacks {attr}")
        return self._ref

    def _build(self, cls: Any) -> Any:
        import torch
        torch.manual_seed(INIT_SEED)
        init = self.reference().get_init_inputs()
        return cls(*init).cuda().eval()

    def load(self, path: str, tag: str) -> Any:
        h = hashlib.md5(os.path.abspath(path).encode()).hexdigest()[:8]  # noqa: S324
        mod = load_module(path, f"kdev_kb_{re.sub(r'[^0-9A-Za-z_]', '_', tag)}_{h}")
        if not (hasattr(mod, "ModelNew") or hasattr(mod, "Model")):
            raise KdevError(ErrorCode.INVALID_REQUEST, f"{path} defines neither ModelNew nor Model")
        return mod

    def prepare(self, module: Any) -> Any:
        return self._build(getattr(module, "ModelNew", None) or module.Model)

    def make_inputs(self, shape: Shape, seed: int) -> Any:
        import torch
        torch.manual_seed(int(seed))
        return [x.cuda() if isinstance(x, torch.Tensor) else x for x in self.reference().get_inputs()]

    def call(self, module: Any, state: Any, inputs: Any) -> Any:
        import torch
        with torch.no_grad():
            return state(*inputs)

    def check(self, inputs: Any, output: Any) -> CheckResult:
        import torch
        if self._ref_model is None:
            self._ref_model = self._build(self.reference().Model)
        with torch.no_grad():
            ref = self._ref_model(*inputs)
        outs = output if isinstance(output, (tuple, list)) else [output]
        refs = ref if isinstance(ref, (tuple, list)) else [ref]
        if len(outs) != len(refs):
            return CheckResult(False, f"expected {len(refs)} outputs, got {len(outs)}")
        worst = 0.0
        for o, r in zip(outs, refs):
            if not isinstance(o, torch.Tensor) or o.shape != r.shape:
                return CheckResult(False, f"shape mismatch: {getattr(o, 'shape', type(o))} vs {tuple(r.shape)}")
            if not torch.allclose(o.float(), r.float(), atol=ATOL, rtol=RTOL):
                d = float((o.float() - r.float()).abs().max())
                return CheckResult(False, f"max abs diff {d:.3e} > atol {ATOL:g} (rtol {RTOL:g})", tol=ATOL)
            worst = max(worst, float((o.float() - r.float()).norm() / r.float().norm().clamp(min=1e-12)))
        return CheckResult(True, f"allclose; relative L2 error {worst:.3e}", max_rel_err=worst, tol=ATOL)
