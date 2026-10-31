"""Summary rendering and output shaping. Owner: P1. Rules: docs/PLAN.md section 6.

``render(result)`` returns <= 40 lines: verdict line, small table, ``run: <id>  dir: <path>``, <= 2 ``next:``
hints. Truncation is always explicit. Tools may pre-render their body into ``Result.lines`` (table() and
fidelity_line() help); otherwise a generic body is built from ``summary`` and ``shapes``. The paged tool
``show`` is allowed up to MAX_PAGE_LINES (its page size is the user's ``--limit``). Output is plain ASCII
(no ANSI colours), so it is safe on any terminal or log.
"""
from __future__ import annotations

from typing import Any, Sequence

from kdev.contract import Fidelity, Result

MAX_LINES = 40
MAX_TABLE_ROWS = 15
MAX_PAGE_LINES = 500          # `kdev show` pages (default --limit 60)
PAGED_TOOLS = ("show",)
MAX_WARNINGS = 3
MAX_NEXT = 2
MAX_WIDTH = 160               # line 1 is clipped to this


def fmt_us(x: float | None) -> str:
    """1 decimal >= 10 us, 2 decimals below."""
    if x is None:
        return "-"
    return f"{x:.1f}" if abs(x) >= 10 else f"{x:.2f}"


def fmt_ratio(x: float | None) -> str:
    return "-" if x is None else f"x{x:.3f}"


def fmt_ci(lo: float | None, hi: float | None, ratio: bool = False) -> str:
    """'[lo, hi]' with the same number format as the point value."""
    if lo is None or hi is None:
        return "-"
    f = (lambda v: f"{v:.3f}") if ratio else fmt_us
    return f"[{f(lo)}, {f(hi)}]"


def fmt_cell(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        if v != v:  # NaN
            return "nan"
        return f"{v:.4g}" if abs(v) < 1e5 else f"{v:.3e}"
    if isinstance(v, (list, tuple)):
        return ",".join(fmt_cell(x) for x in v)
    return str(v)


def table(headers: Sequence[str], rows: Sequence[Sequence[Any]], max_rows: int = MAX_TABLE_ROWS,
          more_hint: str = "") -> list[str]:
    """Fixed-width text table, truncated with an explicit '... N more rows in <hint>' line."""
    shown = [[fmt_cell(c) for c in r] for r in rows[:max(0, max_rows)]]
    ncol = max([len(headers)] + [len(r) for r in shown]) if (headers or shown) else 0
    hdr = [str(h) for h in headers] + [""] * (ncol - len(headers))
    for r in shown:
        r.extend([""] * (ncol - len(r)))
    widths = [max([len(hdr[i])] + [len(r[i]) for r in shown]) for i in range(ncol)]
    widths = [min(w, 60) for w in widths]

    def line(cells: list[str]) -> str:
        out = []
        for i, c in enumerate(cells):
            c = c if len(c) <= widths[i] else c[: widths[i] - 3] + "..."
            out.append(c.ljust(widths[i]))
        return "  ".join(out).rstrip()

    lines = [line(hdr)] if any(hdr) else []
    lines += [line(r) for r in shown]
    extra = len(rows) - len(shown)
    if extra > 0:
        lines.append(f"... {extra} more rows" + (f" in {more_hint}" if more_hint else ""))
    return lines


def fidelity_line(f: Fidelity) -> str:
    """'fidelity: nvcc 13.0.88=pin read-flush 256MB graph excl=ok seeds=5 overhead=-4.00us optouts=none';
    opt-outs in CAPS."""
    parts: list[str] = []
    tc = f.toolchain or {}
    nvcc, pin = tc.get("nvcc") or "?", tc.get("pin") or ""
    if not pin:
        parts.append(f"nvcc {nvcc} NOPIN")
    elif tc.get("match", nvcc == pin):
        parts.append(f"nvcc {nvcc}=pin")
    else:
        parts.append(f"NVCC {nvcc}!=PIN {pin}")
    fl = f.flush or {}
    method = str(fl.get("method") or "read")
    size = fl.get("bytes")
    mb = f" {int(size) >> 20}MB" if isinstance(size, (int, float)) and size else ""
    if method == "read":
        parts.append(f"read-flush{mb}")
    elif method == "none":
        parts.append("FLUSH=NONE")
    else:
        parts.append(f"FLUSH={method.upper()}{mb}")
    parts.append("graph" if f.graph else "EAGER")
    ex = f.exclusive or "unknown"
    parts.append(f"excl={ex}" if ex in ("ok", "waited") else f"EXCL={ex.upper()}")
    parts.append(f"seeds={len(f.seeds)}")
    ov = (f.overhead_us or {}).get("subtracted")
    if ov is None:
        parts.append("overhead=?")
    elif ov == 0:
        parts.append("OVERHEAD=NONE")
    else:
        parts.append(f"overhead=-{float(ov):.2f}us")
    parts.append("optouts=" + (",".join(o.upper() for o in f.optouts) if f.optouts else "none"))
    return "fidelity: " + " ".join(parts)


def _key_numbers(summary: dict[str, Any], limit: int = 6) -> str:
    out = []
    for k, v in summary.items():
        if k.startswith("_") or isinstance(v, (dict, list, tuple)) or v is None:
            continue
        s = fmt_cell(v)
        if len(s) > 40:
            continue
        out.append(f"{k}={s}")
        if len(out) >= limit:
            break
    return "  ".join(out)


def shapes_table(shapes: list[dict[str, Any]], more_hint: str = "") -> list[str]:
    """Generic per-shape table of a timing Result (shape, per-file medians, ratio + CI, verdict)."""
    labels: list[str] = []
    for s in shapes:
        for pf in s.get("per_file") or []:
            lab = str(pf.get("label", "?"))
            if lab not in labels:
                labels.append(lab)
    headers = ["shape"] + [f"{lab} us" for lab in labels] + ["ratio", "ci", "verdict"]
    rows = []
    for s in shapes:
        med = {str(pf.get("label")): pf.get("median_us") for pf in s.get("per_file") or []}
        r = s.get("ratio") or {}
        rows.append([s.get("shape_id", "?")] + [fmt_us(med.get(lab)) for lab in labels]
                    + [fmt_ratio(r.get("median")), fmt_ci(r.get("ci_lo"), r.get("ci_hi"), ratio=True),
                       s.get("verdict") or s.get("check_status") or ""])
    return table(headers, rows, more_hint=more_hint)


def render(result: Result, quiet: bool = False) -> str:
    """Human/LLM summary of a Result (<= 40 lines; ``show`` pages may be longer, up to their --limit)."""
    paged = result.tool in PAGED_TOOLS
    max_lines = MAX_PAGE_LINES if paged else MAX_LINES
    head = [f"{(result.verdict or result.status).upper()}  {result.tool}"]
    keys = _key_numbers(result.summary) if not paged else ""
    if keys:
        head[0] = (head[0] + "  " + keys)[:MAX_WIDTH]
    if result.job_id and not result.run_id:
        head.append(f"job: {result.job_id}")
    if result.error is not None:
        head.append(f"error: {result.error.code}: {result.error.message}")
        if result.error.recovery:
            head.append(f"recovery: {result.error.recovery}")

    tail: list[str] = []
    if result.fidelity is not None:
        tail.append(fidelity_line(result.fidelity))
    tail.extend(f"warning: {w}" for w in result.warnings[:MAX_WARNINGS])
    if len(result.warnings) > MAX_WARNINGS:
        tail.append(f"... {len(result.warnings) - MAX_WARNINGS} more warnings (--json)")
    if result.run_id:
        tail.append(f"run: {result.run_id}  dir: {result.files.get('dir', '')}")
    if not quiet:
        tail.extend(f"next: {n}" for n in result.next[:MAX_NEXT])

    where = f"kdev show {result.run_id} --section summary" if result.run_id else "--json"
    if result.lines:
        body = list(result.lines)
    elif result.shapes:
        body = shapes_table(result.shapes, more_hint=where)
    else:
        body = []
    budget = max(1, max_lines - len(head) - len(tail))
    if len(body) > budget:
        body = body[: budget - 1] + [f"... {len(body) - budget + 1} more lines ({where})"]
    out = head + body + tail
    return "\n".join(out[:max_lines])
