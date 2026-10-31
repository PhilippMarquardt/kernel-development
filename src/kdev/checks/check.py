"""`kdev check` (worker kind='check'). Owner: P7.

Per case (tests + scored + extra held-out shapes) and per fresh seed: eager and/or graph-replay output,
repeated R times; adapter.check against the reference with the judge tolerances. Classification:
- wrong: fails on every repeat;
- intermittent: fails k/R repeats (race; e.g. PDL hoist) -> hint `kdev sass --audit pdl`;
- nondeterministic: all pass but outputs differ bitwise between repeats (report max ulp; fp32 atomics);
- pass.
``--cold`` measures prepare + compile time from an empty extension cache against protocol.budget_s.
Always attaches the dispatch summary (kdev.observe.dispatch).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kdev.contract import Result
from kdev.models import JobSpec

CLASSES = ("pass", "nondeterministic", "intermittent", "wrong", "error")


@dataclass
class CaseOutcome:
    shape_id: str
    seed: int
    mode: str                     # eager | graph
    passes: int
    repeats: int
    klass: str
    max_rel_err: float | None = None
    max_ulp: int | None = None
    message: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def classify(passes: list[bool], bitwise_equal: bool) -> str:
    """Pure classification used above."""
    raise NotImplementedError("check.classify (P7)")


def run_check(spec: JobSpec, run_dir: Any) -> Result:
    raise NotImplementedError("check.run_check (P7)")
