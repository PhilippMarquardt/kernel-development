"""Problem adapter interface (FROZEN after Phase 0; steward P7). See docs/PLAN.md 3.6 and DESIGN 8.

An adapter tells kdev how to load a submission, build inputs, call the timed function, check outputs and
compute the judge's score. Methods that touch the GPU (make_inputs, call, prepare, reference) are only
called inside the worker. ``shapes``/``tests``/``score``/``rules``/``judge_cases`` must be pure and
GPU-free so the CLI and unit tests can use them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class Shape:
    """A scored shape or a correctness case."""

    shape_id: str                 # stable id, e.g. 'd1/sliding' or 'm16/qkv'
    case: dict[str, Any]          # kwargs for reference.generate_input (without seed)
    weight: float = 1.0           # judge weight (times the shape runs per step)
    load: str = ""                # judge load group (geomean over loads)
    scored: bool = True
    seed: int | None = None       # judge seed when known


@dataclass
class CheckResult:
    ok: bool
    message: str = ""
    max_rel_err: float | None = None
    tol: float | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class TimingProtocol:
    """Judge constants for one problem (read from TASK.md/judge spec, never hard-coded in the engine)."""

    overhead_us: float = 4.0
    flush: str = "read"
    flush_bytes: int | None = None
    judge_reps: int = 6
    pairing: str = "order_swapped_pairs"   # judge: median of paired ratios taken two reps at a time
    score: str = "load_weighted_geomean"   # geomean over loads of (base load time / sub load time)
    budget_s: int = 900                    # total incl. compile
    prepare_budget_s: int = 300


@dataclass
class ComplyRules:
    """Rules for `kdev comply` and `judge prepare` static checks."""

    precision: dict[str, str] = field(default_factory=dict)     # op -> required dtype, e.g. {'router': 'fp32'}
    forbid_patterns: list[str] = field(default_factory=list)     # regexes, e.g. r'\.item\(\)', r'synchronize'
    no_cross_call_state: bool = True
    graph_capturable: bool = True
    lossless_weights: bool = True
    notes: list[str] = field(default_factory=list)


@dataclass
class ExtSpec:
    """A compile unit discovered by build_hook (for compile-only builds)."""

    name: str
    sources: list[str] = field(default_factory=list)       # file paths or inline source strings
    inline: bool = False
    cflags: list[str] = field(default_factory=list)
    cuda_cflags: list[str] = field(default_factory=list)
    ldflags: list[str] = field(default_factory=list)


@runtime_checkable
class Adapter(Protocol):
    name: str              # adapter type, e.g. 'okc'
    problem: str           # problem name, e.g. 'muse_moe_w4a16'
    problem_dir: str
    gpu: str               # GPU profile name, e.g. 'h200'
    protocol: TimingProtocol

    def shapes(self) -> list[Shape]: ...
    def tests(self) -> list[Shape]: ...
    def extra_shapes(self) -> list[Shape]: ...
    def input_variance(self) -> dict[str, bool]: ...
    def judge_seed(self, shape: Shape) -> int | None: ...
    def load(self, path: str, tag: str) -> Any: ...
    def baseline_path(self) -> str | None: ...
    def prepare(self, module: Any) -> Any: ...
    def make_inputs(self, shape: Shape, seed: int) -> Any: ...
    def call(self, module: Any, state: Any, inputs: Any) -> Any: ...
    def check(self, inputs: Any, output: Any) -> CheckResult: ...
    def score(self, sub_us: dict[str, float], base_us: dict[str, float]) -> float: ...
    def rules(self) -> ComplyRules: ...
    def build_hook(self, path: str) -> list[ExtSpec]: ...
    def judge_cases(self, judge_json: dict[str, Any]) -> list[Shape]: ...


def load_weighted_geomean(shapes: list[Shape], sub_us: dict[str, float], base_us: dict[str, float]) -> float:
    """OKC score: per load, sum(weight * us) for base and sub; geomean over loads of base/sub. Pure.
    Implemented by P7 (used by P3 for score CIs)."""
    raise NotImplementedError("adapters.base.load_weighted_geomean (P7)")
