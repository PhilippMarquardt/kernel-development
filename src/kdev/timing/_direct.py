"""Minimal OKC problem adapter used by the timing engine when the P7 adapter is unavailable (private).

It does exactly what okc/tools/calib.py does: import reference.py from the problem dir, weights =
generate_weights(WEIGHTS_SEED) (if any), prepare(weights) per file (if any), inputs =
generate_input(**case, seed) (+ case_fields), kernel_fn(data[, state]), check_implementation. Shapes come
from a judge JSON (records[event=benchmark].case) or ``params.extra.cases``.
"""
from __future__ import annotations

import importlib.util
import inspect
import math
import os
import sys
from typing import Any

from kdev.adapters.base import CheckResult, ComplyRules, ExtSpec, Shape, TimingProtocol

DROP_CASE_KEYS = ("load", "repeats", "weight", "router_load", "router_distinct")


def _load_module(path: str, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _n_required_positional(fn: Any) -> int | None:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return None
    n = 0
    for p in sig.parameters.values():
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
            n += 1
        elif p.kind == p.VAR_POSITIONAL:
            return None
    return n


def load_weighted_geomean(shapes: list[Shape], sub_us: dict[str, float], base_us: dict[str, float]) -> float:
    """OKC score: per load, sum(weight * us) for base and sub; geomean over loads of base/sub."""
    loads: dict[str, list[float]] = {}
    for s in shapes:
        if s.shape_id not in sub_us or s.shape_id not in base_us:
            continue
        acc = loads.setdefault(s.load or "", [0.0, 0.0])
        acc[0] += s.weight * base_us[s.shape_id]
        acc[1] += s.weight * sub_us[s.shape_id]
    vals = [b / a for b, a in loads.values() if a > 0 and b > 0]
    if not vals:
        return float("nan")
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


class DirectOkcAdapter:
    name = "okc-direct"

    def __init__(self, problem_dir: str, shapes: list[Shape] | None = None) -> None:
        self.problem_dir = os.path.abspath(problem_dir)
        self.problem = os.path.basename(self.problem_dir.rstrip("/"))
        self.gpu = "h200"
        self.protocol = TimingProtocol()
        self._shapes = list(shapes or [])
        self._ref: Any = None
        self._weights: Any = None
        self._weights_done = False
        self._n = 0

    # --- pure ---------------------------------------------------------------------------------------
    def shapes(self) -> list[Shape]:
        return list(self._shapes)

    def set_shapes(self, shapes: list[Shape]) -> None:
        self._shapes = list(shapes)

    def tests(self) -> list[Shape]:
        return []

    def extra_shapes(self) -> list[Shape]:
        return []

    def input_variance(self) -> dict[str, bool]:
        p = self.problem.lower()
        return {"routing": any(k in p for k in ("moe", "decode_layer", "prefill_layer"))}

    def judge_seed(self, shape: Shape) -> int | None:
        if shape.seed is not None:
            return shape.seed
        s = shape.case.get("seed")
        return int(s) if s is not None else None

    def baseline_path(self) -> str | None:
        p = os.path.join(self.problem_dir, "baseline.py")
        return p if os.path.exists(p) else None

    def score(self, sub_us: dict[str, float], base_us: dict[str, float]) -> float:
        return load_weighted_geomean(self._shapes, sub_us, base_us)

    def rules(self) -> ComplyRules:
        return ComplyRules()

    def build_hook(self, path: str) -> list[ExtSpec]:
        return []

    def judge_cases(self, judge_json: dict[str, Any]) -> list[Shape]:
        from kdev.timing.calib import parse_judge_json

        return [Shape(js.shape_id, dict(js.case), weight=js.weight, load=js.load, seed=js.seed)
                for js in parse_judge_json(judge_json).shapes]

    # --- GPU side (worker only) ---------------------------------------------------------------------
    def _reference(self) -> Any:
        if self._ref is None:
            if self.problem_dir not in sys.path:
                sys.path.insert(0, self.problem_dir)
            self._ref = _load_module(os.path.join(self.problem_dir, "reference.py"), "reference")
            sys.modules.setdefault("reference", self._ref)
        return self._ref

    def weights(self) -> Any:
        if not self._weights_done:
            R = self._reference()  # noqa: N806
            self._weights = R.generate_weights(R.WEIGHTS_SEED) if hasattr(R, "generate_weights") else None
            self._weights_done = True
        return self._weights

    def load(self, path: str, tag: str) -> Any:
        self._reference()
        self._n += 1
        safe = "".join(c if c.isalnum() else "_" for c in tag)
        return _load_module(os.path.abspath(path), f"kdev_sub_{safe}_{self._n}")

    def prepare(self, module: Any) -> Any:
        if not hasattr(module, "prepare"):
            return None
        n = _n_required_positional(module.prepare)
        return module.prepare() if n == 0 else module.prepare(self.weights())

    def make_inputs(self, shape: Shape, seed: int) -> Any:
        R = self._reference()  # noqa: N806
        case = {k: v for k, v in shape.case.items() if k not in DROP_CASE_KEYS}
        case["seed"] = seed
        if hasattr(R, "case_fields"):
            case.update(R.case_fields(case, self.weights()))
        return R.generate_input(**case)

    def call(self, module: Any, state: Any, inputs: Any) -> Any:
        fn = module.kernel_fn
        n = getattr(module, "_kdev_arity", None)
        if n is None:
            n = _n_required_positional(fn)
            n = (2 if state is not None else 1) if n is None else n
            try:
                module._kdev_arity = n
            except Exception:  # noqa: BLE001
                pass
        return fn(inputs, state) if n >= 2 else fn(inputs)

    def check(self, inputs: Any, output: Any) -> CheckResult:
        R = self._reference()  # noqa: N806
        fn = getattr(R, "check_implementation", None)
        if fn is None:
            return CheckResult(True, "no check_implementation in reference.py")
        n = _n_required_positional(fn)
        res = fn(inputs, output, self.weights()) if (n or 0) >= 3 else fn(inputs, output)
        if isinstance(res, tuple):
            return CheckResult(bool(res[0]), str(res[1]) if len(res) > 1 else "")
        return CheckResult(bool(res), "")
