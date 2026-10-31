"""compute-sanitizer wrapper. Owner: P7.

The worker re-execs itself as ``compute-sanitizer --tool T --print-limit 100 --log-file san.log python -m
kdev.worker job.json --inner`` on one small shape (eager + graph); ``run_sanitize_inner`` is the code that
runs under the sanitizer; ``parse_log`` summarises errors by type with the first 5 distinct errors.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kdev.contract import Result
from kdev.models import JobSpec

TOOLS = ("memcheck", "racecheck", "synccheck", "initcheck")


@dataclass
class SanitizerSummary:
    tool: str
    errors: int
    by_type: dict[str, int] = field(default_factory=dict)
    first: list[str] = field(default_factory=list)


def sanitizer_argv(tool: str, inner_argv: list[str], log_file: str) -> list[str]:
    raise NotImplementedError("sanitize.sanitizer_argv (P7)")


def parse_log(text: str, tool: str) -> SanitizerSummary:
    raise NotImplementedError("sanitize.parse_log (P7)")


def run_sanitize_inner(spec: JobSpec, run_dir: Any) -> Result:
    raise NotImplementedError("sanitize.run_sanitize_inner (P7)")
