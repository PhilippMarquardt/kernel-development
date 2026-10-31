"""Generic OKC adapter (the five muse_* problems). Owner: P7.

Problem dir layout (e.g. okc/muse_moe_w4a16/muse_moe_w4a16/): TASK.md, reference.py (generate_input,
[generate_weights + WEIGHTS_SEED], check_implementation, [collect], [case_fields], [speed_of_light_ns]),
checks.py, baseline.py, submission.py. Per-problem config: kdev/adapters/okc_problems/<problem>.toml
(scored shapes with ids/weights/loads from TASK.md 'Shapes' table, tests, extra shapes, input_variance,
kernel_fn arity, protocol constants, comply rules). Differences handled here:
- attention: ``kernel_fn(data)``, no prepare/weights; check via ``R.collect(data, out)`` then
  ``R.check_implementation(data, raw)``.
- moe/decode/prefill/bf16: ``state = prepare(R.generate_weights(R.WEIGHTS_SEED))``; ``kernel_fn(data,
  state)``; ``R.check_implementation(data, out, weights)``; optional ``R.case_fields(case, weights)``.
- shapes from a judge JSON: records[event == 'benchmark'].case minus load/repeats/weight/router_* (calib.py).
reference.py is imported from the problem dir with that dir first on sys.path (never from an extracted
archive's own directory).
"""
from __future__ import annotations

from typing import Any

from kdev.adapters.base import CheckResult, ComplyRules, ExtSpec, Shape, TimingProtocol


class OkcAdapter:
    name = "okc"

    def __init__(self, problem_dir: str, config_path: str | None = None) -> None:
        raise NotImplementedError("OkcAdapter.__init__ (P7)")

    problem: str
    problem_dir: str
    gpu: str = "h200"
    protocol: TimingProtocol

    def shapes(self) -> list[Shape]:
        raise NotImplementedError

    def tests(self) -> list[Shape]:
        raise NotImplementedError

    def extra_shapes(self) -> list[Shape]:
        raise NotImplementedError

    def input_variance(self) -> dict[str, bool]:
        raise NotImplementedError

    def judge_seed(self, shape: Shape) -> int | None:
        raise NotImplementedError

    def load(self, path: str, tag: str) -> Any:
        raise NotImplementedError

    def baseline_path(self) -> str | None:
        raise NotImplementedError

    def prepare(self, module: Any) -> Any:
        raise NotImplementedError

    def make_inputs(self, shape: Shape, seed: int) -> Any:
        raise NotImplementedError

    def call(self, module: Any, state: Any, inputs: Any) -> Any:
        raise NotImplementedError

    def check(self, inputs: Any, output: Any) -> CheckResult:
        raise NotImplementedError

    def score(self, sub_us: dict[str, float], base_us: dict[str, float]) -> float:
        raise NotImplementedError

    def rules(self) -> ComplyRules:
        raise NotImplementedError

    def build_hook(self, path: str) -> list[ExtSpec]:
        raise NotImplementedError

    def judge_cases(self, judge_json: dict[str, Any]) -> list[Shape]:
        raise NotImplementedError
