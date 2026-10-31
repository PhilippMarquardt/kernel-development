"""`kdev judge prepare FILE`: static validation + duplicate detection; prints the command, never runs it.
Owner: P4."""
from __future__ import annotations

from dataclasses import dataclass, field

from kdev.contract import Result


@dataclass
class PrepareReport:
    path: str
    md5: str
    norm_md5: str
    duplicate_of: list[str] = field(default_factory=list)   # judge ids with the same md5 / norm_md5
    rule_findings: list[str] = field(default_factory=list)  # from adapter.rules() static checks
    command: str = ""                                        # exact command for a human/orchestrator


def prepare(path: str, problem: str | None = None) -> PrepareReport:
    raise NotImplementedError("judge.prepare.prepare (P4)")


def to_result(rep: PrepareReport) -> Result:
    raise NotImplementedError("judge.prepare.to_result (P4)")
