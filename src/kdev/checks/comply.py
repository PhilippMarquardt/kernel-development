"""`kdev comply`: rules checklist (M19, LESSONS 5). Owner: P7. Output: PASS/RISK items; RISK needs a human.

Static (no GPU): forbid_patterns (host syncs, .item(), printing tensors, caching dicts keyed on data
pointers), weight precision stored in prepare (lossless rule). Dynamic (worker kind='comply'): per-op
precision probes on intermediate dtypes in eager mode vs adapter rules; cross-call state (call twice with
different inputs in the same buffers, compare against a fresh process); work outside the timed call
(prepare time vs kernel_fn, prepare-time output-shaped buffers); graph capturability.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kdev.adapters.base import ComplyRules
from kdev.contract import Result
from kdev.models import JobSpec


@dataclass
class ComplyItem:
    rule: str
    status: str          # PASS | RISK | FAIL
    detail: str = ""


def static_checks(source: str, rules: ComplyRules) -> list[ComplyItem]:
    raise NotImplementedError("comply.static_checks (P7)")


def run_comply(spec: JobSpec, run_dir: Any) -> Result:
    raise NotImplementedError("comply.run_comply (P7)")
