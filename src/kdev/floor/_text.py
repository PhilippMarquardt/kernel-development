"""Small text helpers shared by the P6 commands (fixed-width tables, us formatting). Private to P6."""
from __future__ import annotations

from typing import Any, Sequence

MAX_ROWS = 15


def us(x: float | None) -> str:
    """PLAN 6.5: 1 decimal >= 10 us, 2 decimals below."""
    if x is None:
        return "-"
    return f"{x:.1f}" if abs(x) >= 10 else f"{x:.2f}"


def table(headers: Sequence[str], rows: Sequence[Sequence[Any]], max_rows: int = MAX_ROWS,
          more_hint: str = "") -> list[str]:
    """Fixed-width table, truncated with an explicit '... N more rows' line."""
    shown = [[("" if c is None else str(c)) for c in r] for r in rows[:max_rows]]
    w = [len(h) for h in headers]
    for r in shown:
        for i, c in enumerate(r[: len(w)]):
            w[i] = max(w[i], len(c))
    out = ["  ".join(h.ljust(w[i]) for i, h in enumerate(headers)).rstrip()]
    out += ["  ".join(c.ljust(w[i]) for i, c in enumerate(r[: len(w)])).rstrip() for r in shown]
    if len(rows) > max_rows:
        out.append(f"... {len(rows) - max_rows} more rows" + (f" in {more_hint}" if more_hint else ""))
    return out
