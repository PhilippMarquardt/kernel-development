"""Idea board: hypotheses with status, evidence and owner (also backs `kdev note`). Owner: P4.

Status machine: open -> claimed (owner set) -> testing -> confirmed|refuted|abandoned; any -> open on
release. claim() of an idea owned by someone else -> KdevError(conflict). kind='limit' (an "at the limit"
claim, LESSONS 1) can only be confirmed with evidence of a `sass` run AND a `sweep` run; kind='dead'
requires ``variant`` (the exact variant tested, LESSONS 6).

Details:
- adding evidence as the owner of a claimed idea moves it to ``testing``;
- a closed idea (confirmed/refuted/abandoned) cannot be claimed (``conflict``); release re-opens only open/
  claimed/testing ideas;
- resolving an idea claimed by someone else is a ``conflict``;
- a sass/sweep run counts when the ledger ``runs`` row has that tool, or (run not ingested yet) when the run
  id has the run-dir form ``<yyyymmdd-hhmmss>-sass-...`` / ``-sweep-...``.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field

from kdev.contract import ErrorCode, KdevError
from kdev.ledger._util import utc_now

STATUSES = ("open", "claimed", "testing", "confirmed", "refuted", "abandoned")
KINDS = ("idea", "dead", "limit", "win")
FINAL = ("confirmed", "refuted", "abandoned")
REF_KINDS = ("run", "judge", "file", "url", "text")
SUPPORTS = ("for", "against", "neutral")


@dataclass
class Evidence:
    ref_kind: str          # run | judge | file | url | text
    ref: str
    note: str = ""
    supports: str = "neutral"
    author: str = ""
    added_utc: str = ""


@dataclass
class Idea:
    idea_id: int | None
    title: str
    problem: str | None = None
    hypothesis: str = ""
    kind: str = "idea"
    status: str = "open"
    owner: str | None = None
    expected_gain_pct: float | None = None
    tags: list[str] = field(default_factory=list)
    variant: str = ""
    created_utc: str = ""
    updated_utc: str = ""
    resolution: str = ""
    evidence: list[Evidence] = field(default_factory=list)

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


def _bad(msg: str, fld: str, recovery: str = "") -> KdevError:
    return KdevError(ErrorCode.INVALID_REQUEST, msg, details={"field": fld}, recovery=recovery)


def _from_row(r: sqlite3.Row) -> Idea:
    return Idea(idea_id=r["idea_id"], title=r["title"], problem=r["problem"], hypothesis=r["hypothesis"] or "",
                kind=r["kind"], status=r["status"], owner=r["owner"], expected_gain_pct=r["expected_gain_pct"],
                tags=[t for t in (r["tags"] or "").split(",") if t], variant=r["variant"] or "",
                created_utc=r["created_utc"], updated_utc=r["updated_utc"], resolution=r["resolution"] or "")


def _evidence(conn: sqlite3.Connection, idea_id: int) -> list[Evidence]:
    cur = conn.execute("SELECT * FROM idea_evidence WHERE idea_id=? ORDER BY rowid", (idea_id,))
    return [Evidence(r["ref_kind"], r["ref"], r["note"] or "", r["supports"] or "neutral", r["author"] or "",
                     r["added_utc"]) for r in cur]


def add(conn: sqlite3.Connection, idea: Idea) -> Idea:
    """Insert a new idea (status forced to open unless a valid status is given; dead needs variant)."""
    if not idea.title.strip():
        raise _bad("idea title is empty", "title")
    if idea.kind not in KINDS:
        raise _bad(f"bad kind {idea.kind!r} (one of {', '.join(KINDS)})", "kind")
    if idea.status not in STATUSES:
        raise _bad(f"bad status {idea.status!r}", "status")
    if idea.kind == "dead" and not idea.variant.strip():
        raise _bad("a 'dead' record needs --variant: the exact variant tested (LESSONS 6)", "variant",
                   "pass --variant 'file/knobs/commit that was measured'")
    now = utc_now()
    with conn:
        cur = conn.execute(
            "INSERT INTO ideas(problem,title,hypothesis,kind,status,owner,expected_gain_pct,tags,variant,"
            "created_utc,updated_utc,resolution) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (idea.problem, idea.title.strip(), idea.hypothesis, idea.kind, idea.status, idea.owner,
             idea.expected_gain_pct, ",".join(t.strip() for t in idea.tags if t.strip()), idea.variant,
             now, now, idea.resolution or None))
        iid = int(cur.lastrowid)
        for ev in idea.evidence:
            _insert_evidence(conn, iid, ev)
    return get(conn, iid)


def get(conn: sqlite3.Connection, idea_id: int) -> Idea:
    r = conn.execute("SELECT * FROM ideas WHERE idea_id=?", (idea_id,)).fetchone()
    if r is None:
        raise KdevError(ErrorCode.NOT_FOUND, f"idea {idea_id} not found", details={"idea_id": idea_id},
                        recovery="kdev idea list")
    idea = _from_row(r)
    idea.evidence = _evidence(conn, idea_id)
    return idea


def list_ideas(conn: sqlite3.Connection, *, problem: str | None = None, status: str | None = None,
               owner: str | None = None, kind: str | None = None, limit: int = 50) -> list[Idea]:
    where, params = [], []
    for col, val in (("problem", problem), ("owner", owner), ("kind", kind)):
        if val:
            where.append(f"{col}=?")
            params.append(val)
    if status:
        sts = [s.strip() for s in status.split(",") if s.strip()]
        where.append(f"status IN ({','.join('?' * len(sts))})")
        params.extend(sts)
    sql = "SELECT * FROM ideas" + (" WHERE " + " AND ".join(where) if where else "")
    sql += " ORDER BY updated_utc DESC, idea_id DESC LIMIT ?"
    return [_from_row(r) for r in conn.execute(sql, (*params, int(limit)))]


def _set(conn: sqlite3.Connection, idea_id: int, **cols: object) -> None:
    cols["updated_utc"] = utc_now()
    sets = ",".join(f"{k}=?" for k in cols)
    conn.execute(f"UPDATE ideas SET {sets} WHERE idea_id=?", (*cols.values(), idea_id))


def claim(conn: sqlite3.Connection, idea_id: int, owner: str) -> Idea:
    if not owner:
        raise _bad("claim needs an owner (--agent or $KDEV_AGENT)", "agent", "pass --agent NAME")
    with conn:
        idea = get(conn, idea_id)
        if idea.status in FINAL:
            raise KdevError(ErrorCode.CONFLICT, f"idea {idea_id} is {idea.status}; cannot claim",
                            details={"status": idea.status}, recovery="add a new idea for a different variant")
        if idea.owner and idea.owner != owner and idea.status in ("claimed", "testing"):
            raise KdevError(ErrorCode.CONFLICT, f"idea {idea_id} is claimed by {idea.owner}",
                            details={"owner": idea.owner}, recovery="pick another idea or ask the owner to release")
        if idea.owner == owner and idea.status in ("claimed", "testing"):
            return idea
        _set(conn, idea_id, owner=owner, status="claimed")
    return get(conn, idea_id)


def release(conn: sqlite3.Connection, idea_id: int, owner: str) -> Idea:
    with conn:
        idea = get(conn, idea_id)
        if idea.owner and owner and idea.owner != owner:
            raise KdevError(ErrorCode.CONFLICT, f"idea {idea_id} is owned by {idea.owner}, not {owner}",
                            details={"owner": idea.owner}, recovery="only the owner can release")
        if idea.status in FINAL:
            raise KdevError(ErrorCode.CONFLICT, f"idea {idea_id} is already {idea.status}",
                            details={"status": idea.status})
        _set(conn, idea_id, owner=None, status="open")
    return get(conn, idea_id)


def _insert_evidence(conn: sqlite3.Connection, idea_id: int, ev: Evidence) -> None:
    if ev.ref_kind not in REF_KINDS:
        raise _bad(f"bad evidence kind {ev.ref_kind!r}", "ref_kind")
    if ev.supports not in SUPPORTS:
        raise _bad(f"bad supports {ev.supports!r}", "supports")
    if not str(ev.ref).strip():
        raise _bad("empty evidence reference", "ref")
    conn.execute("INSERT INTO idea_evidence(idea_id,ref_kind,ref,note,supports,author,added_utc) "
                 "VALUES (?,?,?,?,?,?,?)",
                 (idea_id, ev.ref_kind, str(ev.ref), ev.note or None, ev.supports, ev.author or None,
                  ev.added_utc or utc_now()))


def add_evidence(conn: sqlite3.Connection, idea_id: int, ev: Evidence) -> Idea:
    with conn:
        idea = get(conn, idea_id)
        _insert_evidence(conn, idea_id, ev)
        if idea.status == "claimed" and ev.author and ev.author == idea.owner:
            _set(conn, idea_id, status="testing")
        else:
            _set(conn, idea_id)
    return get(conn, idea_id)


_RUN_TOOL = re.compile(r"^\d{8}-\d{6}-(?P<tool>[a-z_]+)")


def _run_tool(conn: sqlite3.Connection, run_id: str) -> str | None:
    r = conn.execute("SELECT tool FROM runs WHERE run_id=?", (run_id,)).fetchone()
    if r is not None:
        return r["tool"]
    r = conn.execute("SELECT tool FROM runs WHERE run_id LIKE ? ORDER BY run_id LIMIT 2",
                     (run_id + "%",)).fetchall()
    if len(r) == 1:
        return r[0]["tool"]
    m = _RUN_TOOL.match(run_id)
    return m.group("tool") if m else None


def limit_gate(conn: sqlite3.Connection, evidence: list[Evidence]) -> list[str]:
    """Missing requirements for confirming a 'limit' claim (empty = ok): a sass run AND a sweep run."""
    tools = {_run_tool(conn, e.ref) for e in evidence if e.ref_kind == "run" and e.supports != "against"}
    missing = []
    if "sass" not in tools:
        missing.append("a `kdev sass` run (hot-loop SASS read, LESSONS 1a)")
    if "sweep" not in tools:
        missing.append("a `kdev sweep` run (exhaustive knob search, LESSONS 1b)")
    return missing


def resolve(conn: sqlite3.Connection, idea_id: int, status: str, resolution: str, author: str) -> Idea:
    """Validate the gating rules above, then set the final status."""
    if status not in FINAL:
        raise _bad(f"resolve status must be one of {', '.join(FINAL)}", "status")
    with conn:
        idea = get(conn, idea_id)
        if idea.status in FINAL:
            raise KdevError(ErrorCode.CONFLICT, f"idea {idea_id} is already {idea.status}",
                            details={"status": idea.status})
        if idea.owner and author and idea.owner != author and idea.status in ("claimed", "testing"):
            raise KdevError(ErrorCode.CONFLICT, f"idea {idea_id} is claimed by {idea.owner}",
                            details={"owner": idea.owner}, recovery="the owner resolves, or releases first")
        if idea.kind == "limit" and status == "confirmed":
            missing = limit_gate(conn, idea.evidence)
            if missing:
                raise _bad("an 'at the limit' claim needs evidence: " + "; ".join(missing), "evidence",
                           f"run kdev sass + kdev sweep, then kdev idea evidence {idea_id} --run RUN_ID")
        if idea.kind == "dead" and not idea.variant.strip():
            raise _bad("a 'dead' record needs the exact variant tested", "variant")
        _set(conn, idea_id, status=status, resolution=resolution or None)
    return get(conn, idea_id)


def _fts_query(text: str) -> str:
    toks = re.findall(r"\w+", text)
    return " OR ".join('"' + t.replace('"', '') + '"' for t in toks)


def search(conn: sqlite3.Connection, text: str, problem: str | None = None, limit: int = 20) -> list[Idea]:
    """Full-text search over title/hypothesis/variant (fts5 bm25 rank; LIKE fallback)."""
    from kdev.ledger.db import has_fts

    if not text.strip():
        return []
    rows: list[sqlite3.Row] = []
    q = _fts_query(text)
    if has_fts(conn) and q:
        sql = ("SELECT ideas.* FROM ideas_fts JOIN ideas ON ideas.idea_id = ideas_fts.rowid "
               "WHERE ideas_fts MATCH ?" + (" AND ideas.problem = ?" if problem else "")
               + " ORDER BY bm25(ideas_fts) LIMIT ?")
        try:
            rows = conn.execute(sql, (q, problem, limit) if problem else (q, limit)).fetchall()
        except sqlite3.OperationalError:
            rows = []
    if not rows:
        toks = re.findall(r"\w+", text) or [text]
        conds = " OR ".join("(title LIKE ? OR hypothesis LIKE ? OR variant LIKE ?)" for _ in toks)
        params: list[object] = []
        for t in toks:
            params += [f"%{t}%"] * 3
        sql = f"SELECT * FROM ideas WHERE ({conds})" + (" AND problem = ?" if problem else "") + " LIMIT ?"
        rows = conn.execute(sql, (*params, *([problem] if problem else []), limit)).fetchall()
    return [_from_row(r) for r in rows]
