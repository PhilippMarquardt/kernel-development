"""Optional read-only public board scrape. Owner: P8.

Source: the public, unauthenticated leaderboard API (HTTP GET only, no token is ever sent)
``https://openkernelchallenge.dev/api/v1/leaderboard/<problem>`` (rows sorted by speedup:
``{job, team, speedup, screened_speedup?, value(=score us), attempts, submitted_at}``) and
``.../leaderboard`` (overall ranks). If the API is unreachable, falls back to the read-only
``kdev.judge.okc_cli.leaderboard`` wrapper (safety rule 8.8). Nothing here can submit or check.

:func:`gaps` diffs the board against our judged bests (LESSONS 4/7: per-problem gap to #1/#10, our rank,
and per shape where the time goes / how much each shape must shrink if the gap were closed uniformly).
"""
from __future__ import annotations

import json
import time
import urllib.request
from typing import Any, Callable
from urllib.parse import quote

from kdev import __version__
from kdev.util import utc_now

BOARD_API = "https://openkernelchallenge.dev/api/v1/leaderboard"
TOP_N = 10

Fetcher = Callable[[str, float], Any]   # (url, timeout_s) -> parsed JSON


def http_get_json(url: str, timeout: float = 20.0) -> Any:
    """GET a JSON document from the public board API (read-only; refuses any other host/prefix)."""
    if not url.startswith(BOARD_API):
        raise ValueError(f"refusing non-board url {url!r}")
    req = urllib.request.Request(url, method="GET", headers={"User-Agent": f"kdev/{__version__}",
                                                             "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed https prefix above
        return json.loads(resp.read().decode("utf-8"))


def _utc(ts: Any) -> str | None:
    try:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(ts)))
    except (TypeError, ValueError):
        return None


def normalize_problem_board(raw: dict[str, Any], problem: str, *, team: str | None = None,
                            top: int = TOP_N) -> dict[str, Any]:
    """API/okc leaderboard JSON -> {problem, n, rows(top), first, tenth, our_row}."""
    rows_in = raw.get("rows") or raw.get("leaderboard") or raw.get("entries") or []
    rows = []
    for i, r in enumerate(rows_in):
        if not isinstance(r, dict):
            continue
        sp = r.get("speedup")
        rows.append({"rank": int(r.get("rank") or i + 1), "team": r.get("team") or r.get("name") or "?",
                     "speedup": float(sp) if isinstance(sp, (int, float)) else None,
                     "screened_speedup": r.get("screened_speedup"), "score_us": r.get("value"),
                     "job": r.get("job"), "attempts": r.get("attempts"),
                     "submitted_utc": _utc(r.get("submitted_at"))})
    rows.sort(key=lambda r: (r["speedup"] is None, -(r["speedup"] or 0.0)))
    for i, r in enumerate(rows):
        r["rank"] = i + 1
    our = next((r for r in rows if team and r["team"].lower() == team.lower()), None)
    return {"problem": raw.get("board") or problem, "n": len(rows), "rows": rows[:top],
            "first": rows[0] if rows else None, "tenth": rows[9] if len(rows) >= 10 else None,
            "our_row": our, "unit": raw.get("sglang_per")}


def scrape(problem: str | None = None, *, problems: list[str] | None = None, team: str | None = None,
           fetch: Fetcher | None = None, okc_fallback: bool = True, timeout: float = 20.0) -> dict[str, Any]:
    """Scrape the board for ``problem`` (or each of ``problems``) plus the overall ranking.

    Returns ``{source, fetched_utc, team, problems: {p: normalized}, overall: {...}|None, errors: {p: msg}}``.
    Never raises for network errors: they land in ``errors`` (missing coverage is unknown, not absence).
    """
    fetch = fetch or http_get_json
    names = [problem] if problem else list(dict.fromkeys(problems or []))
    out: dict[str, Any] = {"source": "api", "url": BOARD_API, "fetched_utc": utc_now(), "team": team,
                           "problems": {}, "overall": None, "errors": {}}
    for p in names:
        try:
            raw = fetch(f"{BOARD_API}/{quote(p)}", timeout)
            out["problems"][p] = normalize_problem_board(raw, p, team=team)
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
            if okc_fallback:
                try:
                    from kdev.judge import okc_cli
                    raw = okc_cli.leaderboard(p)
                    out["problems"][p] = normalize_problem_board(raw, p, team=team)
                    out["problems"][p]["source"] = "okc leaderboard"
                    continue
                except Exception as e2:  # noqa: BLE001
                    err += f"; okc fallback: {type(e2).__name__}: {e2}"
            out["errors"][p] = err
    try:
        raw = fetch(BOARD_API, timeout)
        rows = [r for r in raw.get("rows", []) if isinstance(r, dict)]
        our = next((r for r in rows if team and str(r.get("team", "")).lower() == team.lower()), None)
        out["overall"] = {"n": len(rows), "top": rows[:TOP_N], "our_row": our}
    except Exception as e:  # noqa: BLE001
        out["errors"]["overall"] = f"{type(e).__name__}: {e}"
    return out


def _case_weight(case: Any) -> float | None:
    if isinstance(case, str):
        try:
            case = json.loads(case)
        except ValueError:
            return None
    if isinstance(case, dict) and isinstance(case.get("weight"), (int, float)):
        return float(case["weight"])
    return None


def _pct(a: float | None, b: float | None) -> float | None:
    """(a / b - 1) * 100, i.e. how much faster a is than b."""
    if not a or not b:
        return None
    return round((a / b - 1.0) * 100.0, 2)


def gaps(board: dict[str, Any], ours: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per problem/shape: our best vs board best, gap %.

    ``ours``: dashboard problem dicts with ``problem``, ``best_judged: {speedup, ...}|None`` and
    ``judge_shapes: [{shape_id, sub_us, base_us, speedup, case_json|weight}]``.
    Per problem: board #1/#10 speedups, ``gap_to_1_pct`` (= #1 / ours - 1), ``gap_to_10_pct``, our rank (our
    team row if found, else the rank our judged best would take). Per shape: time share of our weighted
    judged time, and ``cut_to_1_us`` = how much that shape must shrink if the gap to #1 were closed
    uniformly (the board only publishes totals, so this is where the gap *can* be closed, not a measured diff).
    """
    out = []
    bprobs = (board or {}).get("problems", {})
    for p in ours:
        name = p.get("problem")
        b = bprobs.get(name)
        best = (p.get("best_judged") or {}).get("speedup")
        row: dict[str, Any] = {"problem": name, "ours": best, "board_n": None, "board1": None, "board10": None,
                               "board1_team": None, "our_rank": None, "rank_kind": None,
                               "gap_to_1_pct": None, "gap_to_10_pct": None, "shapes": []}
        if b:
            first, tenth = b.get("first") or {}, b.get("tenth") or {}
            row.update(board_n=b.get("n"), board1=first.get("speedup"), board1_team=first.get("team"),
                       board10=tenth.get("speedup"))
            row["gap_to_1_pct"] = _pct(row["board1"], best)
            row["gap_to_10_pct"] = _pct(row["board10"], best)
            if b.get("our_row"):
                row["our_rank"], row["rank_kind"] = b["our_row"]["rank"], "board"
            elif best:
                better = sum(1 for r in b.get("rows", []) if (r.get("speedup") or 0) > best)
                if better >= len(b.get("rows", [])) and (b.get("n") or 0) > len(b.get("rows", [])):
                    row["our_rank"], row["rank_kind"] = None, f">{len(b['rows'])}"
                else:
                    row["our_rank"], row["rank_kind"] = better + 1, "would-be"
        shapes = p.get("judge_shapes") or []
        weighted = []
        for s in shapes:
            w = s.get("weight")
            if w is None:
                w = _case_weight(s.get("case_json"))
            sub = s.get("sub_us")
            weighted.append((s, w if w is not None else 1.0, sub))
        total = sum(w * sub for _, w, sub in weighted if sub)
        need = (1.0 - best / row["board1"]) if (best and row["board1"] and row["board1"] > best) else 0.0
        for s, w, sub in weighted:
            share = (w * sub / total) if (sub and total) else None
            row["shapes"].append({"shape_id": s.get("shape_id"), "weight": w, "sub_us": sub,
                                  "base_us": s.get("base_us"), "speedup": s.get("speedup"),
                                  "time_share": round(share, 4) if share is not None else None,
                                  "cut_to_1_us": round(sub * need, 3) if sub else None})
        row["shapes"].sort(key=lambda r: -(r["time_share"] or 0))
        out.append(row)
    return out
