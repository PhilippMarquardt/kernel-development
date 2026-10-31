"""Replay overhead: judge constant (4.0 us) + locally measured empty-graph and 1-kernel-graph costs.
Owner: P3. Mode 'judge' subtracts the constant (default), 'local' the measured 1-kernel graph minus the null
kernel delta, 'none' nothing (opt-out). Both measurements are always reported."""
from __future__ import annotations

from dataclasses import dataclass

JUDGE_CONST_US = 4.0
SENSITIVE_BELOW_US = 10.0


@dataclass
class Overhead:
    judge_const: float = JUDGE_CONST_US
    empty_graph: float | None = None
    one_kernel_graph: float | None = None
    mode: str = "judge"

    @property
    def subtracted(self) -> float:
        """The value actually subtracted per replay under ``mode``."""
        raise NotImplementedError("Overhead.subtracted (P3)")

    def to_dict(self) -> dict[str, float | None]:
        raise NotImplementedError("Overhead.to_dict (P3)")


def measure(flush: object, reps: int = 50) -> Overhead:
    """GPU: median replay time of an empty graph and of a 1-kernel (x.add_(1)) graph after flush."""
    raise NotImplementedError("overhead.measure (P3)")
