"""Multi-agent message board (``kdev board``). Owner: P9.

Agents talk to each other directly instead of through the coordinator. A post has a kind
(finding|claim|done|ask|answer|warn|decision), an author (``--agent`` / $KDEV_AGENT), an optional problem, an
addressee (``--to AGENT``; default all), typed evidence refs (``--ref``: run ids, ideas, judge jobs, files) and
an optional parent (``--reply-to``; replies share the root's ``thread_id``).

Claims: a ``claim`` post names its scope (``--files`` and ``--shapes``; globs like ``bm_*.py`` allowed; file
refs count too). The scope check and the insert run in one ``BEGIN IMMEDIATE`` transaction, so of two agents
claiming at the same moment the second always sees the first. A claim without ``--files``/``--shapes``
takes the file names in its text (``pp/pp_stk3b.py``) as its scope. Overlap = same problem (or either without a
problem) and any scope item matching (file basenames, fnmatch in both directions; shapes case-insensitive).
Overlaps are reported as conflicts (``--exclusive``: refuse to post). A claim stays open until its owner posts
``done`` (replying to it, or with an overlapping scope, or - no scope, no reply - every open claim of that
agent on the problem), ``kdev board release ID``, or it expires (``--ttl``, default 12 h).

Inbox: posts addressed to me, mentioning ``@me``, claims that conflict with one of mine, or replies to / in a
thread of my posts, by other agents, not yet acked. ``read --unread`` uses a per-agent cursor (advanced by that read).
"""
from __future__ import annotations

import datetime as _dt
import fnmatch
import json
import os
import re
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from kdev.collab.db import has_fts, write_tx
from kdev.contract import ErrorCode, KdevError

KINDS = ("finding", "claim", "done", "ask", "answer", "warn", "decision")
DEFAULT_TTL_S = 12 * 3600
_FILE_TOKEN = re.compile(r"(?<![\w/.-])((?:[\w.*-]+/)*[\w*-][\w.*-]*\.(?:py|cu|cuh|cpp|h|hpp|ptx|sass|toml))(?![\w])")
_MENTION = re.compile(r"(?<![\w@/.])@([A-Za-z0-9][A-Za-z0-9_.-]*[A-Za-z0-9_]|[A-Za-z0-9])")


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc_plus(seconds: float, base: str | None = None) -> str:
    t = (_dt.datetime.strptime(base, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=_dt.timezone.utc) if base
         else _dt.datetime.now(_dt.timezone.utc)) + _dt.timedelta(seconds=seconds)
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_since(since: str | None) -> str | None:
    """'3h' / '30m' / '2d' (relative), 'HH:MM' (today, UTC), ISO date/time -> ISO UTC; None passes."""
    if not since:
        return None
    s = since.strip()
    if re.fullmatch(r"\d{1,2}:\d{2}", s):
        today = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")
        return f"{today}T{int(s.split(':')[0]):02d}:{s.split(':')[1]}:00Z"
    from kdev.ledger._util import since_to_utc
    return since_to_utc(s)


@dataclass
class Post:
    post_id: int
    created_utc: str
    agent: str
    kind: str
    body: str
    problem: str | None = None
    to_agent: str = ""
    reply_to: int | None = None
    thread_id: int | None = None
    refs: list[dict[str, Any]] = field(default_factory=list)
    scope: list[str] = field(default_factory=list)
    claim_state: str = ""
    expires_utc: str | None = None
    closed_utc: str | None = None
    closed_by: int | None = None
    source: str = "cli"
    mentions: list[str] = field(default_factory=list)
    acked: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def claim_open(self) -> bool:
        return self.kind == "claim" and self.claim_state == "open" and not _expired(self.expires_utc)

    def line(self, width: int = 150) -> str:
        """``#12 10-31 16:45 pp_ CLAIM bf16_proj ->bS_ re#9 [files] body``: one line, <= width chars."""
        body = " ".join(self.body.split())
        refs = " ".join(_ref_short(r) for r in self.refs[:3])
        s = f"{self.head()} {body}" + (f" | {refs}" if refs else "")
        return s if len(s) <= width else s[: width - 1] + "~"

    def head(self) -> str:
        """The line prefix without the body."""
        t = self.created_utc[5:16].replace("T", " ")
        bits = [f"#{self.post_id}", t, self.agent, self.kind.upper()]
        if self.problem:
            bits.append(_short_problem(self.problem))
        if self.to_agent:
            bits.append("->" + self.to_agent)
        if self.reply_to:
            bits.append(f"re#{self.reply_to}")
        if self.kind == "claim":
            st = "expired" if self.claim_state == "open" and _expired(self.expires_utc) else self.claim_state
            bits.append(f"[{st}: {','.join(self.scope)[:60]}]" if self.scope else f"[{st}]")
        return " ".join(bits)


def _short_problem(p: str) -> str:
    b = os.path.basename(p.rstrip("/\\")) or p
    return b[5:] if b.startswith("muse_") else b


def _ref_short(r: dict[str, Any]) -> str:
    from kdev.collab.refs import short
    return short(r)


def _expired(expires: str | None) -> bool:
    return bool(expires) and expires < utc_now()  # type: ignore[operator]


def _bad(msg: str, fld: str, recovery: str = "") -> KdevError:
    return KdevError(ErrorCode.INVALID_REQUEST, msg, details={"field": fld}, recovery=recovery)


def _from_row(r: sqlite3.Row, mentions: list[str] | None = None, acked: bool = False) -> Post:
    return Post(post_id=r["post_id"], created_utc=r["created_utc"], agent=r["agent"], kind=r["kind"],
                body=r["body"], problem=r["problem"], to_agent=r["to_agent"] or "", reply_to=r["reply_to"],
                thread_id=r["thread_id"], refs=json.loads(r["refs"] or "[]"), scope=json.loads(r["scope"] or "[]"),
                claim_state=r["claim_state"] or "", expires_utc=r["expires_utc"], closed_utc=r["closed_utc"],
                closed_by=r["closed_by"], source=r["source"] or "cli", mentions=mentions or [], acked=acked)


def _posts(conn: sqlite3.Connection, where: str, params: tuple, order: str = "post_id",
           limit: int | None = None, viewer: str | None = None) -> list[Post]:
    sql = f"SELECT * FROM posts WHERE {where} ORDER BY {order}" + (f" LIMIT {int(limit)}" if limit else "")
    rows = conn.execute(sql, params).fetchall()
    out = []
    for r in rows:
        m = [x[0] for x in conn.execute("SELECT agent FROM post_mentions WHERE post_id=?", (r["post_id"],))]
        acked = bool(viewer) and conn.execute("SELECT 1 FROM acks WHERE post_id=? AND agent=?",
                                              (r["post_id"], viewer)).fetchone() is not None
        out.append(_from_row(r, m, acked))
    return out


def get(conn: sqlite3.Connection, post_id: int, viewer: str | None = None) -> Post:
    ps = _posts(conn, "post_id=?", (int(post_id),), viewer=viewer)
    if not ps:
        raise KdevError(ErrorCode.NOT_FOUND, f"no post #{post_id}", details={"post_id": post_id},
                        recovery="kdev board read to list posts")
    return ps[0]


# ------------------------------------------------------------------------------------------- claims / scope
def scope_items(files: list[str] | None, shapes: list[str] | None, refs: list[dict[str, Any]]) -> list[str]:
    """Normalised claim scope: ``file:<basename or glob>`` and ``shape:<lowercase>`` (order kept, deduped)."""
    out: list[str] = []
    for f in _split(files) + [r["ref"] for r in refs if r.get("kind") == "file"]:
        out.append("file:" + os.path.basename(f.replace("\\", "/").rstrip("/")))
    for s in _split(shapes):
        out.append("shape:" + s.lower())
    return list(dict.fromkeys(x for x in out if x not in ("file:", "shape:")))


def _split(items: list[str] | None) -> list[str]:
    return [p.strip() for it in items or [] for p in str(it).split(",") if p.strip()]


def _item_overlap(a: str, b: str) -> bool:
    ka, va = a.split(":", 1)
    kb, vb = b.split(":", 1)
    if ka != kb:
        return False
    if ka == "file":
        return va == vb or fnmatch.fnmatchcase(va, vb) or fnmatch.fnmatchcase(vb, va)
    return va == vb or fnmatch.fnmatchcase(va, vb) or fnmatch.fnmatchcase(vb, va)


def overlap(a: list[str], b: list[str]) -> list[str]:
    """Items of ``a`` that overlap some item of ``b``."""
    return [x for x in a if any(_item_overlap(x, y) for y in b)]


def _same_problem(p: str | None, q: str | None) -> bool:
    return not p or not q or _short_problem(p) == _short_problem(q)


def open_claims(conn: sqlite3.Connection, problem: str | None = None, agent: str | None = None) -> list[Post]:
    """Open, unexpired claims (problem filter keeps claims without a problem)."""
    ps = _posts(conn, "kind='claim' AND claim_state='open'", ())
    return [p for p in ps if p.claim_open and (problem is None or _same_problem(p.problem, problem))
            and (agent is None or p.agent == agent)]


def conflicts_for(conn: sqlite3.Connection, agent: str, problem: str | None, scope: list[str],
                  exclude: int | None = None) -> list[tuple[Post, list[str]]]:
    """Open claims of *other* agents whose scope overlaps ``scope`` (same problem or unscoped)."""
    if not scope:
        return []
    out = []
    for c in open_claims(conn):
        if c.agent == agent or c.post_id == exclude or not _same_problem(c.problem, problem):
            continue
        ov = overlap(scope, c.scope)
        if ov:
            out.append((c, ov))
    return out


# ------------------------------------------------------------------------------------------------ writing
def post(conn: sqlite3.Connection, *, agent: str, kind: str, body: str, problem: str | None = None,
         to: str | None = None, refs: list[dict[str, Any]] | None = None, reply_to: int | None = None,
         files: list[str] | None = None, shapes: list[str] | None = None, ttl_s: float | None = None,
         exclusive: bool = False, source: str = "cli", created_utc: str | None = None,
         ) -> tuple[Post, list[tuple[Post, list[str]]], list[int]]:
    """Insert a post. Returns (post, conflicts [(claim, overlapping items)], closed claim ids)."""
    kind = (kind or "").strip().lower()
    if kind not in KINDS:
        raise _bad(f"unknown kind {kind!r}", "kind", "use one of " + "|".join(KINDS))
    agent = (agent or "").strip()
    if not agent:
        raise _bad("posting needs an agent name", "agent", "pass --agent NAME or set KDEV_AGENT")
    if not body.strip():
        raise _bad("empty message", "text")
    refs = list(refs or [])
    to = (to or "").strip()
    to = "" if to.lower() in ("", "all", "*") else to.lstrip("@")
    scope = scope_items(files, shapes, refs) if kind in ("claim", "done") else []
    if kind == "claim" and not scope:       # no explicit scope: the files named in the text
        scope = scope_items(_FILE_TOKEN.findall(body), None, [])
    mentions = sorted(set(_MENTION.findall(body)) | ({to} if to else set()))
    now = created_utc or utc_now()
    conflicts: list[tuple[Post, list[str]]] = []
    closed: list[int] = []
    with write_tx(conn):
        thread = None
        parent = None
        if reply_to:
            row = conn.execute("SELECT post_id, thread_id, agent, kind FROM posts WHERE post_id=?",
                               (int(reply_to),)).fetchone()
            if row is None:
                raise KdevError(ErrorCode.NOT_FOUND, f"--reply-to #{reply_to}: no such post",
                                details={"post_id": reply_to}, recovery="kdev board read to find the id")
            parent = row
            thread = row["thread_id"] or row["post_id"]
        if kind == "claim":
            conflicts = conflicts_for(conn, agent, problem, scope)
            if conflicts and exclusive:
                c, ov = conflicts[0]
                raise KdevError(ErrorCode.CONFLICT, f"claim overlaps #{c.post_id} by {c.agent}: {','.join(ov)}",
                                details={"conflicts": [{"post_id": x.post_id, "agent": x.agent, "overlap": o}
                                                       for x, o in conflicts]},
                                recovery=f"coordinate with {c.agent} (kdev board post ask --to {c.agent} ...) "
                                         "or narrow --files/--shapes")
        ttl = DEFAULT_TTL_S if ttl_s is None else ttl_s
        cur = conn.execute(
            "INSERT INTO posts(created_utc, agent, kind, problem, to_agent, reply_to, thread_id, refs, scope, "
            "claim_state, expires_utc, body, source, host) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (now, agent, kind, problem, to, int(reply_to) if reply_to else None, thread, json.dumps(refs),
             json.dumps(scope), "open" if kind == "claim" else "",
             _utc_plus(ttl, now) if kind == "claim" and ttl > 0 else None, body.strip(), source,
             os.environ.get("HOSTNAME", "")))
        pid = int(cur.lastrowid)
        if thread is None:
            conn.execute("UPDATE posts SET thread_id=? WHERE post_id=?", (pid, pid))
        for m in set(mentions) | {c.agent for c, _ in conflicts}:   # conflicting owners get it in their inbox
            conn.execute("INSERT OR IGNORE INTO post_mentions(post_id, agent) VALUES (?,?)", (pid, m))
        if kind == "done":
            closed = _close_claims(conn, agent, problem, scope, parent, pid, "done", now)
    return get(conn, pid), conflicts, closed


def _close_claims(conn: sqlite3.Connection, agent: str, problem: str | None, scope: list[str],
                  parent: sqlite3.Row | None, by: int, state: str, now: str) -> list[int]:
    mine = [c for c in open_claims(conn, agent=agent) if _same_problem(c.problem, problem)]
    if parent is not None and parent["kind"] == "claim" and parent["agent"] == agent:
        target = [c for c in mine if c.post_id == parent["post_id"]]
    elif scope:
        target = [c for c in mine if overlap(scope, c.scope)]
    elif parent is None:
        target = mine
    else:
        target = []
    for c in target:
        conn.execute("UPDATE posts SET claim_state=?, closed_utc=?, closed_by=? WHERE post_id=?",
                     (state, now, by, c.post_id))
    return [c.post_id for c in target]


def release(conn: sqlite3.Connection, post_id: int, agent: str, force: bool = False) -> Post:
    """Close an open claim without a done post (owner only unless ``force``)."""
    p = get(conn, post_id)
    if p.kind != "claim":
        raise _bad(f"#{post_id} is a {p.kind}, not a claim", "post_id")
    if p.agent != agent and not force:
        raise KdevError(ErrorCode.CONFLICT, f"claim #{post_id} belongs to {p.agent}", details={"owner": p.agent},
                        recovery=f"ask {p.agent} to release it, or pass --force (coordinator)")
    with write_tx(conn):
        conn.execute("UPDATE posts SET claim_state='released', closed_utc=? WHERE post_id=? AND claim_state='open'",
                     (utc_now(), int(post_id)))
    return get(conn, post_id)


def ack(conn: sqlite3.Connection, agent: str, post_ids: list[int]) -> int:
    if not agent:
        raise _bad("ack needs an agent name", "agent", "pass --agent NAME or set KDEV_AGENT")
    now = utc_now()
    with write_tx(conn):
        n = 0
        for i in post_ids:
            n += conn.execute("INSERT OR IGNORE INTO acks(post_id, agent, acked_utc) VALUES (?,?,?)",
                              (int(i), agent, now)).rowcount
    return n


# ------------------------------------------------------------------------------------------------ reading
def read(conn: sqlite3.Connection, *, problem: str | None = None, kind: str | None = None,
         agent: str | None = None, since: str | None = None, to: str | None = None, limit: int = 30,
         unread_for: str | None = None, advance: bool = True) -> tuple[list[Post], int]:
    """Newest ``limit`` matching posts in chronological order + total match count. ``unread_for``: only posts
    after that agent's cursor and not by them (the cursor moves to the newest post when ``advance``)."""
    where, params = ["1=1"], []
    if problem:
        where.append("(problem IS NULL OR problem='' OR problem=? OR problem LIKE ?)")
        params += [problem, "%" + _short_problem(problem)]
    if kind:
        ks = [k.strip().lower() for k in kind.split(",") if k.strip()]
        where.append(f"kind IN ({','.join('?' * len(ks))})")
        params += ks
    if agent:
        where.append("agent=?")
        params.append(agent)
    if since:
        where.append("created_utc>=?")
        params.append(parse_since(since))
    if to:
        where.append("(to_agent=? OR to_agent='')")
        params.append(to)
    cursor = 0
    if unread_for:
        row = conn.execute("SELECT last_post_id FROM cursors WHERE agent=?", (unread_for,)).fetchone()
        cursor = int(row[0]) if row else 0
        where.append("post_id>? AND agent<>?")
        params += [cursor, unread_for]
    w = " AND ".join(where)
    total = conn.execute(f"SELECT count(*) FROM posts WHERE {w}", tuple(params)).fetchone()[0]
    ps = _posts(conn, w, tuple(params), order="post_id DESC", limit=max(1, limit), viewer=unread_for)[::-1]
    if unread_for and advance:
        top = conn.execute("SELECT max(post_id) FROM posts").fetchone()[0] or 0
        if top > cursor:
            with write_tx(conn):
                conn.execute("INSERT OR REPLACE INTO cursors(agent, last_post_id, utc) VALUES (?,?,?)",
                             (unread_for, top, utc_now()))
    return ps, int(total)


def inbox(conn: sqlite3.Connection, agent: str, include_acked: bool = False, limit: int = 30) -> list[Post]:
    """Posts by others addressed to ``agent`` (``--to``/``@agent``) or replying to / in a thread started by one
    of their posts; unacked only unless ``include_acked``. Chronological."""
    if not agent:
        raise _bad("inbox needs an agent name", "agent", "pass --agent NAME or set KDEV_AGENT")
    w = ("agent<>? AND (post_id IN (SELECT post_id FROM post_mentions WHERE agent=?) "
         "OR reply_to IN (SELECT post_id FROM posts WHERE agent=?) "
         "OR thread_id IN (SELECT post_id FROM posts WHERE agent=? AND thread_id=post_id))")
    params: tuple = (agent, agent, agent, agent)
    if not include_acked:
        w += " AND post_id NOT IN (SELECT post_id FROM acks WHERE agent=?)"
        params += (agent,)
    return _posts(conn, w, params, order="post_id DESC", limit=limit, viewer=agent)[::-1]


def thread(conn: sqlite3.Connection, post_id: int, viewer: str | None = None) -> list[Post]:
    p = get(conn, post_id)
    root = p.thread_id or p.post_id
    return _posts(conn, "thread_id=? OR post_id=?", (root, root), viewer=viewer)


def search(conn: sqlite3.Connection, query: str, *, problem: str | None = None, kind: str | None = None,
           ref: str | None = None, limit: int = 30) -> list[Post]:
    """Full-text search (fts5; LIKE fallback or for queries fts cannot parse) + filters; chronological."""
    ids: list[int] | None = None
    q = (query or "").strip()
    if q and has_fts(conn):
        try:
            fq = " ".join('"' + t.replace('"', '""') + '"' for t in q.split())
            ids = [r[0] for r in conn.execute("SELECT rowid FROM posts_fts WHERE posts_fts MATCH ? "
                                              "ORDER BY rowid DESC LIMIT 2000", (fq,))]
        except sqlite3.OperationalError:
            ids = None
    where, params = ["1=1"], []
    if q and ids is None:
        for t in q.split():
            where.append("(body LIKE ? OR refs LIKE ? OR agent LIKE ?)")
            params += [f"%{t}%"] * 3
    elif q:
        if not ids:
            return []
        where.append(f"post_id IN ({','.join(str(i) for i in ids)})")
    if problem:
        where.append("(problem=? OR problem LIKE ?)")
        params += [problem, "%" + _short_problem(problem)]
    if kind:
        where.append("kind=?")
        params.append(kind.lower())
    if ref:
        where.append("refs LIKE ?")
        params.append(f"%{ref}%")
    return _posts(conn, " AND ".join(where), tuple(params), order="post_id DESC", limit=limit)[::-1]


# ------------------------------------------------------------------------------------------- legacy import
_LEGACY = re.compile(r"^- (\d{2})-(\d{2}) (\d{2}):(\d{2}) UTC \[([^\]]+)\] ([A-Za-z]+): ?(.*)$")


def import_legacy(conn: sqlite3.Connection, path: str | Path, year: int | None = None,
                  problem: str | None = None) -> tuple[int, int]:
    """Ingest the interim bash board (``- MM-DD HH:MM UTC [agent] KIND: text`` lines; continuation lines are
    appended to the previous post). Idempotent (same time+agent+kind+body skipped). Returns (added, skipped)."""
    year = year or _dt.datetime.now(_dt.timezone.utc).year
    items: list[list[str]] = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        m = _LEGACY.match(line.rstrip())
        if m:
            mo, d, hh, mm, agent, kind, body = m.groups()
            items.append([f"{year}-{mo}-{d}T{hh}:{mm}:00Z", agent.strip(), kind.lower(), body])
        elif items and line.strip():
            items[-1][3] += "\n" + line.rstrip()
    added = skipped = 0
    src = f"import:{Path(path).name}"
    for utc, agent, kind, body in items:
        if kind not in KINDS:
            body, kind = f"[{kind}] {body}", "finding"
        dup = conn.execute("SELECT 1 FROM posts WHERE created_utc=? AND agent=? AND kind=? AND body=?",
                           (utc, agent, kind, body.strip())).fetchone()
        if dup:
            skipped += 1
            continue
        post(conn, agent=agent, kind=kind, body=body, source=src, created_utc=utc, problem=problem)
        added += 1
    return added, skipped
