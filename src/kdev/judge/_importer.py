"""Private (P4): ingest existing evidence into the ledger.

- judge JSONs (`okc status/check --json` saved by agents): ``import_judge_file``;
- public leaderboard JSONs, three shapes: the raw API/``okc leaderboard --json`` form ``{board, rows:[{job,
  team, speedup, submitted_at, ...}]}``, the research aggregate ``{problem: {top30: [...], ...}}`` (e.g.
  okc/research/leaderboard_2026-10-31.json), and the overall board ``{board: overall, rows:[{rank, team,
  solved}]}`` (no speedups: skipped): ``import_board_file``;
- an OKC checkout (``import_okc_tree``): ``muse_*/muse_*/{check_*,chk_*}.json`` and everything JSON under
  ``cq_logs/``, ``tester1031/``, ``rn/``, ``r2/`` there, plus ``research/leaderboard_*.json`` and
  ``research/v1_lb_*.json``.

Labels/roles: the submitted file's stem is recovered from the judge's profile/log path
(``profiles/<stem>.<job>.log``) when present, else the JSON's own name; a stem containing ``base`` is
role ``base``, anything else ``cand``. Runs of one directory created within an hour of each other that
contain both roles get a shared ``pair_id`` (``imp-<first job id>``) so ``judge ledger`` can show the
same-hour baseline. The submitted file's md5 is filled only when a local file's sha256 equals the judge's
content hash (never by name alone: pod copies drift).
"""
from __future__ import annotations

import datetime as _dt
import glob
import hashlib
import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any

from kdev.judge._parse import JRun, looks_like_judge, parse_judge
from kdev.judge.record import record_parsed
from kdev.ledger._util import epoch_to_utc, md5_file, norm_md5

PAIR_GAP_S = 3600.0
EVIDENCE_SUBDIRS = ("cq_logs", "tester1031", "rn", "r2")


def _load_json(path: str) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError):
        return None


def submitted_stem(d: dict[str, Any], json_path: str) -> str:
    for p in ((d.get("output") or {}).get("log"), (d.get("profile") or {}).get("json")):
        if isinstance(p, str):
            base = os.path.basename(p)
            m = re.match(r"^(.+)\.([0-9a-f]{16})\.(log|json|txt)$", base)
            if m:
                return m.group(1)
    return Path(json_path).stem


def role_of(stem: str) -> str:
    return "base" if "base" in stem.lower() else "cand"


class ShaIndex:
    """sha256 -> local path, built lazily per directory tree (only *.py)."""

    def __init__(self) -> None:
        self._roots: set[str] = set()
        self._idx: dict[str, str] = {}

    def add_root(self, root: str) -> None:
        root = os.path.abspath(root)
        if root in self._roots or not os.path.isdir(root):
            return
        self._roots.add(root)
        for p in glob.glob(os.path.join(root, "**", "*.py"), recursive=True):
            try:
                h = hashlib.sha256(Path(p).read_bytes()).hexdigest()
            except OSError:
                continue
            self._idx.setdefault(h, p)

    def get(self, sha: str | None) -> str | None:
        return self._idx.get(sha) if sha else None


def import_judge_file(conn: sqlite3.Connection, path: str, *, index: ShaIndex | None = None,
                      search_root: str | None = None) -> tuple[str, JRun] | None:
    """Record one saved judge JSON (None when it is not one)."""
    d = _load_json(path)
    if not looks_like_judge(d):
        return None
    jr = parse_judge(d)
    stem = submitted_stem(d, path)
    local = None
    if index is not None:
        if search_root:
            index.add_root(search_root)
        local = index.get(jr.content_sha256)
    md5 = nm = None
    if local:
        md5 = md5_file(local)
        try:
            nm = norm_md5(local)
        except (OSError, ValueError):
            nm = None
    jid = record_parsed(conn, jr, raw_path=os.path.abspath(path), label=f"{stem}.py", role=role_of(stem),
                        file_md5=md5, file_norm_md5=nm)
    return jid, jr


def assign_pairs(conn: sqlite3.Connection, items: list[tuple[str, str, JRun]], gap_s: float = PAIR_GAP_S) -> int:
    """items: (dir, role, JRun). Same dir + consecutive created_at gaps <= gap_s + both roles -> pair_id."""
    by_dir: dict[str, list[tuple[str, JRun]]] = {}
    for dr, role, jr in items:
        if jr.created_at is not None and jr.status == "ok":
            by_dir.setdefault(dr, []).append((role, jr))
    n = 0
    for lst in by_dir.values():
        lst.sort(key=lambda x: x[1].created_at or 0)
        groups: list[list[tuple[str, JRun]]] = []
        for it in lst:
            if groups and (it[1].created_at or 0) - (groups[-1][-1][1].created_at or 0) <= gap_s:
                groups[-1].append(it)
            else:
                groups.append([it])
        for g in groups:
            roles = {r for r, _ in g}
            if {"base", "cand"} <= roles:
                pid = f"imp-{g[0][1].judge_id}"
                with conn:
                    for _r, jr in g:
                        conn.execute("UPDATE judge_results SET pair_id=? WHERE judge_id=? AND "
                                     "(pair_id IS NULL OR pair_id LIKE 'imp-%')", (pid, jr.judge_id))
                n += 1
    return n


def _mtime_utc(path: str) -> str:
    return _dt.datetime.fromtimestamp(os.path.getmtime(path), _dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def board_rows_from(d: Any, taken_utc: str, source: str) -> list[dict[str, Any]]:
    """Normalise any supported leaderboard JSON into board_rows dicts."""
    out: list[dict[str, Any]] = []

    def add(problem: str, rank: int | None, r: dict[str, Any]) -> None:
        if not isinstance(r, dict) or r.get("speedup") is None or not r.get("team"):
            return
        sub = r.get("submitted_at")
        sub_utc = epoch_to_utc(sub) if isinstance(sub, (int, float)) else (r.get("submitted") or None)
        out.append({"problem": problem, "taken_utc": taken_utc, "rank": r.get("rank", rank), "team": r["team"],
                    "job": r.get("job"), "speedup": r.get("speedup"),
                    "time_us": r.get("time_us", r.get("value")), "screened_speedup": r.get("screened_speedup"),
                    "screening": r.get("screening"), "attempts": r.get("attempts"), "submitted_utc": sub_utc,
                    "source": source})

    if isinstance(d, dict) and isinstance(d.get("rows"), list):
        prob = d.get("board") or d.get("problem")
        if prob and prob != "overall":
            for i, r in enumerate(d["rows"]):
                add(prob, i + 1, r)
    elif isinstance(d, dict):
        for prob, v in d.items():
            if isinstance(v, dict):
                rows = v.get("top30") or v.get("rows") or v.get("top") or []
                for i, r in enumerate(rows if isinstance(rows, list) else []):
                    add(prob, i + 1, r)
    return out


def write_board_rows(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    cols = list(rows[0])
    with conn:
        for r in rows:
            conn.execute(f"INSERT OR REPLACE INTO board_rows ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                         [r[c] for c in cols])
    return len(rows)


def import_board_file(conn: sqlite3.Connection, path: str) -> int:
    d = _load_json(path)
    if d is None:
        return 0
    return write_board_rows(conn, board_rows_from(d, _mtime_utc(path), os.path.basename(path)))


def okc_tree_files(okc_root: str) -> tuple[list[str], list[str]]:
    """(judge JSON candidates, leaderboard JSONs) under an OKC checkout."""
    judge: set[str] = set()
    for pd in glob.glob(os.path.join(okc_root, "muse_*", "muse_*")):
        if not os.path.isdir(pd):
            continue
        for pat in ("check_*.json", "chk_*.json", "check*.json"):
            judge.update(glob.glob(os.path.join(pd, pat)))
        for sub in EVIDENCE_SUBDIRS:
            judge.update(glob.glob(os.path.join(pd, sub, "**", "*.json"), recursive=True))
    boards = sorted(set(glob.glob(os.path.join(okc_root, "research", "leaderboard_*.json"))
                        + glob.glob(os.path.join(okc_root, "research", "v1_lb_*.json"))))
    return sorted(judge), boards


def import_okc_tree(conn: sqlite3.Connection, okc_root: str, *, match_files: bool = True) -> dict[str, Any]:
    """Import every judge JSON + leaderboard JSON of an OKC checkout (idempotent)."""
    judge_files, boards = okc_tree_files(okc_root)
    index = ShaIndex() if match_files else None
    items: list[tuple[str, str, JRun]] = []
    recorded: dict[str, str] = {}
    skipped: list[str] = []
    matched = 0
    for f in judge_files:
        pd = _problem_dir(f)
        res = import_judge_file(conn, f, index=index, search_root=pd)
        if res is None:
            skipped.append(os.path.relpath(f, okc_root))
            continue
        jid, jr = res
        recorded[os.path.relpath(f, okc_root)] = jid
        stem = submitted_stem(_load_json(f) or {}, f)
        items.append((os.path.dirname(os.path.abspath(f)), role_of(stem), jr))
        if index is not None and index.get(jr.content_sha256):
            matched += 1
    pairs = assign_pairs(conn, items)
    nboard = sum(import_board_file(conn, b) for b in boards)
    return {"judge_files": len(recorded), "unique_judge_ids": len(set(recorded.values())), "pairs": pairs,
            "board_rows": nboard, "board_files": [os.path.relpath(b, okc_root) for b in boards],
            "skipped": skipped, "files_matched_by_hash": matched, "recorded": recorded}


def _problem_dir(path: str) -> str | None:
    p = Path(path).resolve()
    for parent in [p.parent, *p.parents]:
        if parent.name.startswith("muse_") and parent.parent.name == parent.name:
            return str(parent)
    return None


def detect_kind(path: str) -> str:
    """result | run_dir | judge | board | unknown."""
    if os.path.isdir(path):
        return "run_dir" if os.path.isfile(os.path.join(path, "result.json")) else "unknown"
    d = _load_json(path)
    if isinstance(d, dict) and str(d.get("schema", "")).startswith("kdev.result/"):
        return "result"
    if looks_like_judge(d):
        return "judge"
    if d is not None and board_rows_from(d, "x", "x"):
        return "board"
    return "unknown"
