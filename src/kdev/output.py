"""Summary rendering and output shaping. Owner: P1. Rules: docs/PLAN.md section 6.

``render(result)`` must return <= 40 lines: verdict line, small table, ``run: <id>  dir: <path>``, <= 2
``next:`` hints. Truncation is always explicit. The Phase-0 ``render`` is a minimal working fallback.
"""
from __future__ import annotations

from typing import Any, Sequence

from kdev.contract import Fidelity, Result

MAX_LINES = 40
MAX_TABLE_ROWS = 15


def fmt_us(x: float | None) -> str:
    """1 decimal >= 10 us, 2 decimals below."""
    if x is None:
        return "-"
    return f"{x:.1f}" if abs(x) >= 10 else f"{x:.2f}"


def fmt_ratio(x: float | None) -> str:
    return "-" if x is None else f"x{x:.3f}"


def table(headers: Sequence[str], rows: Sequence[Sequence[Any]], max_rows: int = MAX_TABLE_ROWS,
          more_hint: str = "") -> list[str]:
    """Fixed-width text table, truncated with an explicit '... N more rows in <hint>' line."""
    raise NotImplementedError("output.table (P1)")


def fidelity_line(f: Fidelity) -> str:
    """'fidelity: nvcc 13.0.88=pin read-flush 256MB graph excl=ok seeds=5 overhead=-4.00us optouts=none';
    opt-outs in CAPS."""
    raise NotImplementedError("output.fidelity_line (P1)")


def render(result: Result) -> str:
    """Human/LLM summary of a Result. Phase-0 fallback: verdict, error, pre-rendered lines, run line, hints."""
    out: list[str] = [f"{(result.verdict or result.status).upper()}  {result.tool}"]
    if result.error is not None:
        out.append(f"error: {result.error.code}: {result.error.message}")
        if result.error.recovery:
            out.append(f"recovery: {result.error.recovery}")
    budget = MAX_LINES - 8
    out.extend(result.lines[:budget])
    if len(result.lines) > budget:
        out.append(f"... {len(result.lines) - budget} more lines (use --json or kdev show)")
    out.extend(f"warning: {w}" for w in result.warnings[:3])
    if result.run_id:
        out.append(f"run: {result.run_id}  dir: {result.files.get('dir', '')}")
    out.extend(f"next: {n}" for n in result.next[:2])
    return "\n".join(out[:MAX_LINES])
