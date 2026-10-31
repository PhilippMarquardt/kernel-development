"""Private (P4): judge JSON parsing helpers shared by judge.record, judge.pair and the evidence importer.

``parse_judge`` prefers ``kdev.timing.calib.parse_judge_json`` (P3, PLAN 11) and falls back to a local parser
with the same semantics when P3 is not implemented yet: benchmark records only, ``base_us = baseline_ns/1000``,
``speedup`` = the judge's per-shape paired ratio, ``sub_us = base_us / speedup``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

DROP_CASE_KEYS = ("load", "repeats", "weight", "router_load", "router_distinct", "seed")


@dataclass
class JShape:
    shape_id: str
    case: dict[str, Any]
    sub_us: float | None
    base_us: float | None
    speedup: float | None
    weight: float | None = None
    load: str = ""


@dataclass
class JRun:
    judge_id: str
    problem: str | None
    mode: str | None
    status: str
    passed: bool | None
    speedup: float | None
    speedup_ci: list[float] | None
    cluster: str | None
    node: str | None
    content_sha256: str | None
    duplicate_of: str | None
    team: str | None
    created_at: float | None
    finished_at: float | None
    toolchain: str | None
    shapes: list[JShape] = field(default_factory=list)
    limits: dict[str, Any] | None = None
    error: str | None = None


def looks_like_judge(d: Any) -> bool:
    return isinstance(d, dict) and "id" in d and ("result" in d or "content_hash" in d or "board" in d) \
        and not str(d.get("schema", "")).startswith("kdev.")


def shape_id_of(case: dict[str, Any]) -> str:
    """Stable id from a judge case: P3's ``calib.shape_id_for`` when available (same ids as calib), else
    '<load>|k=v,...' over the non-volatile keys (seed/weight/... dropped)."""
    try:
        from kdev.timing.calib import shape_id_for
        return shape_id_for(case)
    except (ImportError, NotImplementedError):
        pass
    load = str(case.get("load", ""))
    rest = ",".join(f"{k}={case[k]}" for k in sorted(case) if k not in DROP_CASE_KEYS)
    return f"{load}|{rest}" if load else rest


_NVCC = re.compile(r"Cuda compilation tools, release [\d.]+, (V[\d.]+)")


def _toolchain(res: dict[str, Any]) -> str | None:
    m = _NVCC.search(str(res.get("log_tail") or ""))
    return m.group(1) if m else None


def _status(d: dict[str, Any], res: dict[str, Any]) -> str:
    if d.get("duplicate_of"):
        return "duplicate"
    st = str(d.get("status") or "")
    if st and st != "done":
        return st                                  # queued | running | failed | timeout ...
    if res.get("status"):
        rs = str(res["status"])
        if rs == "ok" and res.get("passed") is False:
            return "fail"
        return rs
    return st or "unknown"


def _local_parse(d: dict[str, Any]) -> list[JShape]:
    res = d.get("result") if isinstance(d.get("result"), dict) else {}
    out = []
    seen: dict[str, int] = {}
    for r in res.get("records") or []:
        if not isinstance(r, dict) or r.get("event") != "benchmark":
            continue
        case = r.get("case") or {}
        sp = r.get("speedup", r.get("paired_ratio"))
        bns = r.get("baseline_ns")
        base_us = bns / 1000.0 if isinstance(bns, (int, float)) else None
        sub_us = base_us / sp if (base_us is not None and isinstance(sp, (int, float)) and sp) else None
        sid = shape_id_of(case)
        seen[sid] = seen.get(sid, -1) + 1
        if seen[sid]:
            sid = f"{sid}#{seen[sid]}"
        out.append(JShape(sid, case, sub_us, base_us, sp, case.get("weight"),
                          str(case.get("load", ""))))
    return out


def _p3_parse(d: dict[str, Any]) -> list[JShape] | None:
    try:
        from kdev.timing.calib import parse_judge_json
        jr = parse_judge_json(d)
    except (NotImplementedError, ImportError):
        return None
    except Exception:  # noqa: BLE001 - a P3 bug must not block recording; fall back
        return None
    out = []
    for s in getattr(jr, "shapes", []) or []:
        case = dict(getattr(s, "case", {}) or {})
        out.append(JShape(s.shape_id, case, s.sub_us, s.base_us, s.speedup, case.get("weight"),
                          str(case.get("load", ""))))
    return out


def parse_judge(d: dict[str, Any]) -> JRun:
    if not looks_like_judge(d):
        from kdev.contract import ErrorCode, KdevError
        raise KdevError(ErrorCode.INVALID_REQUEST, "not an `okc status/check --json` document",
                        details={"keys": sorted(d)[:12] if isinstance(d, dict) else type(d).__name__},
                        recovery="save `okc status JOB --json` output and pass that file")
    res = d.get("result") if isinstance(d.get("result"), dict) else {}
    machine = res.get("machine") if isinstance(res.get("machine"), dict) else {}
    shapes = _p3_parse(d)
    if shapes is None:
        shapes = _local_parse(d)
    else:
        # keep the judge weights/loads from the raw records when P3 dropped them from the case
        local = {s.shape_id: s for s in _local_parse(d)}
        for s in shapes:
            ls = local.get(s.shape_id)
            if ls is not None:
                s.weight = s.weight if s.weight is not None else ls.weight
                s.load = s.load or ls.load
    ci = res.get("speedup_ci")
    err = res.get("error")
    return JRun(judge_id=str(d["id"]), problem=d.get("board") or d.get("problem"), mode=d.get("mode"),
                status=_status(d, res), passed=res.get("passed"), speedup=res.get("speedup"),
                speedup_ci=list(ci) if isinstance(ci, (list, tuple)) else None, cluster=d.get("cluster"),
                node=machine.get("node"), content_sha256=d.get("content_hash"),
                duplicate_of=d.get("duplicate_of"), team=d.get("team"), created_at=d.get("created_at"),
                finished_at=d.get("finished_at"), toolchain=_toolchain(res), shapes=shapes,
                limits=d.get("limits") if isinstance(d.get("limits"), dict) else None,
                error=(str(err)[:500] if err else None))
