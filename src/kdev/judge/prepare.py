"""`kdev judge prepare FILE`: static validation + duplicate detection; prints the command, never runs it.
Owner: P4.

Checks (all static, no GPU, no network):
- line 1 must be ``# okc: problem=<name>`` (TASK.md: it tells okc the problem) and match ``--problem``;
- ``def kernel_fn`` present (both blocking); the adapter's ``rules().forbid_patterns`` (P7) when an adapter
  resolves (non-blocking ``review`` findings: the regexes cannot tell prepare() from kernel_fn);
- duplicates against the ledger: exact md5 / judge content hash -> the judge would return ``duplicate``
  (wasted check); same norm_md5 (comment-only difference) -> informational (a deliberate re-judge).
``--copy`` writes a comment-changed copy ``<stem>__kj<k>.py`` (a ``# kdev judge copy k`` line after the
header): a new judge hash with the same norm_md5, so a same-code re-judge (noise check, ABBA pair) is not
answered with ``duplicate``. Copies are deterministic per k; k is bumped past copies already judged.
"""
from __future__ import annotations

import os
import re
import shlex
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from kdev.contract import ErrorCode, KdevError, Result
from kdev.ledger._util import md5_file, norm_md5, sha256_file

HEADER = re.compile(r"^#\s*okc:\s*problem\s*=\s*(\S+)")
COPY_TAG = "# kdev judge copy {k}: comment-only change so the judge does not answer 'duplicate'"
COPY_NAME = re.compile(r"__kj\d+$")


@dataclass
class PrepareReport:
    path: str
    md5: str
    norm_md5: str
    duplicate_of: list[str] = field(default_factory=list)   # judge ids with the same md5 / judge hash (exact)
    rule_findings: list[str] = field(default_factory=list)  # from adapter.rules() static checks
    command: str = ""                                        # exact command for a human/orchestrator
    sha256: str = ""
    problem: str | None = None
    same_code: list[str] = field(default_factory=list)      # judge ids with the same norm_md5 (comments differ)
    blocking: list[str] = field(default_factory=list)       # findings that make the judge reject the file
    warnings: list[str] = field(default_factory=list)
    copy_of: str | None = None
    record_command: str = ""


def header_problem(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            m = HEADER.match(f.readline())
    except OSError:
        return None
    return m.group(1) if m else None


def _judged(conn: sqlite3.Connection, md5: str, sha: str, nm: str) -> tuple[list[str], list[str]]:
    exact = [r[0] for r in conn.execute(
        "SELECT judge_id FROM judge_results WHERE file_md5=? OR content_sha256=? ORDER BY created_utc",
        (md5, sha))]
    same = [r[0] for r in conn.execute(
        "SELECT judge_id FROM judge_results WHERE norm_md5=? ORDER BY created_utc", (nm,)) if r[0] not in exact]
    return exact, same


def _adapter_rules(path: str, problem: str | None) -> tuple[list[str], list[str]]:
    """(findings, warnings) from the problem adapter's static rules (P7)."""
    try:
        from kdev.adapters.registry import get_adapter
        ad = get_adapter(problem=problem, file=path)
        rules = ad.rules()
    except NotImplementedError:
        return [], ["adapter rules unavailable (P7 not implemented): adapter static checks skipped"]
    except KdevError as e:
        return [], [f"adapter rules unavailable ({e.message}): adapter static checks skipped"]
    except Exception as e:  # noqa: BLE001 - adapter bugs must not block prepare
        return [], [f"adapter rules failed ({type(e).__name__}: {e}): adapter static checks skipped"]
    src = Path(path).read_text(encoding="utf-8", errors="replace")
    out = []
    for pat in getattr(rules, "forbid_patterns", []) or []:
        try:
            m = re.search(pat, src)
        except re.error:
            continue
        if m:
            line = src.count("\n", 0, m.start()) + 1
            out.append(f"forbidden pattern {pat!r} at line {line}")
    return out, []


def copy_path(path: str, k: int) -> str:
    p = Path(path)
    stem = COPY_NAME.sub("", p.stem)
    return str(p.with_name(f"{stem}__kj{k}{p.suffix}"))


def make_copy(path: str, k: int) -> str:
    """Write the k-th comment-changed copy (idempotent: same k -> same bytes). Returns its path."""
    src = Path(path).read_text(encoding="utf-8")
    lines = src.split("\n")
    lines = [ln for ln in lines if not re.match(r"^# kdev judge copy \d+:", ln)]
    at = 1 if lines and (HEADER.match(lines[0]) or lines[0].startswith("#!")) else 0
    lines.insert(at, COPY_TAG.format(k=k))
    out = copy_path(path, k)
    data = "\n".join(lines)
    if not os.path.exists(out) or Path(out).read_text(encoding="utf-8") != data:
        with open(out, "w", encoding="utf-8", newline="") as f:
            f.write(data)
    return out


def fresh_copy(path: str, conn: sqlite3.Connection | None, start: int = 1, avoid: set[str] | None = None) -> str:
    """First comment-changed copy whose hash the ledger has not seen judged (and not in ``avoid``)."""
    avoid = avoid or set()
    for k in range(start, start + 1000):
        cp = make_copy(path, k)
        md5 = md5_file(cp)
        if md5 in avoid:
            continue
        if conn is not None:
            hit = conn.execute("SELECT 1 FROM judge_results WHERE file_md5=? OR content_sha256=?",
                               (md5, sha256_file(cp))).fetchone()
            if hit:
                continue
        return cp
    raise KdevError(ErrorCode.INTERNAL, "could not find a fresh copy index")


def judge_command(path: str) -> str:
    ap = os.path.abspath(path)
    stem = Path(ap).stem
    return (f"cd {shlex.quote(os.path.dirname(ap))} && okc check {shlex.quote(os.path.basename(ap))} --json "
            f"> chk_{stem}.json")


def prepare(path: str, problem: str | None = None, *, conn: sqlite3.Connection | None = None,
            copy: bool = False, check_rules: bool = True) -> PrepareReport:
    """Validate ``path`` for a judge check and compare it with judged hashes in the ledger. ``copy=True``
    first writes a comment-changed copy (see module doc) and validates that instead."""
    if not os.path.isfile(path):
        raise KdevError(ErrorCode.NOT_FOUND, f"no such file: {path}", details={"path": path})
    own = conn is None
    if own:
        from kdev.ledger.db import open_default
        conn = open_default()
    try:
        orig = os.path.abspath(path)
        target = fresh_copy(orig, conn) if copy else orig
        md5, nm, sha = md5_file(target), norm_md5(target), sha256_file(target)
        rep = PrepareReport(path=target, md5=md5, norm_md5=nm, sha256=sha, copy_of=orig if copy else None)
        hp = header_problem(target)
        rep.problem = problem or hp
        if hp is None:
            rep.blocking.append("line 1 is not '# okc: problem=<name>' (okc uses it to pick the problem)")
        elif problem and os.path.basename(problem.rstrip("/\\")) != hp:
            rep.blocking.append(f"header says problem={hp} but --problem is {problem}")
        src = Path(target).read_text(encoding="utf-8", errors="replace")
        if not re.search(r"(?m)^\s*def\s+kernel_fn\s*\(", src):
            rep.blocking.append("no `def kernel_fn(` found")
        if check_rules:
            f, w = _adapter_rules(target, rep.problem)
            rep.rule_findings += f
            rep.warnings += w
        rep.duplicate_of, rep.same_code = _judged(conn, md5, sha, nm)
        if rep.duplicate_of:
            rep.warnings.append(f"identical code already judged ({', '.join(rep.duplicate_of)}): the judge "
                                "returns 'duplicate'; use --copy for a comment-changed copy")
        if rep.same_code:
            rep.warnings.append(f"same code modulo comments judged before ({', '.join(rep.same_code[:5])})")
        rep.command = judge_command(target)
        rep.record_command = (f"kdev judge record chk_{Path(target).stem}.json --file "
                              f"{shlex.quote(os.path.basename(target))}")
        return rep
    finally:
        if own:
            conn.close()


def to_result(rep: PrepareReport) -> Result:
    if rep.blocking:
        verdict = "blocked"
    elif rep.duplicate_of:
        verdict = "duplicate"
    elif rep.rule_findings:      # regex rules are coarse (e.g. .cpu() inside prepare() is legal): review
        verdict = "review"
    else:
        verdict = "ready"
    lines = [f"file: {rep.path}", f"md5 {rep.md5}  norm {rep.norm_md5}  sha256 {rep.sha256[:16]}",
             f"problem: {rep.problem or '?'}"]
    if rep.copy_of:
        lines.append(f"comment-changed copy of {rep.copy_of}")
    lines += [f"BLOCKING: {b}" for b in rep.blocking]
    lines += [f"rule: {r}" for r in rep.rule_findings[:10]]
    lines.append(f"command (not run): {rep.command}")
    lines.append(f"then: {rep.record_command}")
    d: dict[str, Any] = asdict(rep)
    return Result(tool="judge prepare", status="ok", verdict=verdict, summary=d, lines=lines,
                  warnings=list(rep.warnings),
                  next=(["kdev judge prepare FILE --copy"] if rep.duplicate_of else []))
