"""CLI: board *, doc *, collab backup. Owner: P9.

Agent identity is the global ``--agent`` (default $KDEV_AGENT); the problem is the global ``--problem``
(stored as its directory basename). Every command prints <= 40 lines (``--json``: one Result object).
"""
from __future__ import annotations

import os
import re
import time
from argparse import Namespace
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from kdev.contract import Effects, ErrorCode, KdevError, Result
from kdev.registry import Arg, Context, Registry

P = "P9"
W_BOARD = Effects(writes=("board",))
W_DOCS = Effects(writes=("docs",))
LIST_MAX = 30
_DUR = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([smhd]?)\s*$")


@contextmanager
def _conn(ctx: Context) -> Iterator[Any]:
    from kdev.collab.db import open_default
    c = open_default(ctx.home)
    try:
        yield c
    finally:
        c.close()


def _problem(ctx: Context) -> str | None:
    p = ctx.problem
    if not p:
        return None
    return os.path.basename(str(p).replace("\\", "/").rstrip("/")) or p


def _agent(ctx: Context) -> str:
    return (ctx.agent or os.environ.get("KDEV_AGENT") or "").strip()


def _duration_s(text: str | None, default: float) -> float:
    if text is None or str(text).strip() == "":
        return default
    m = _DUR.match(str(text))
    if not m:
        raise KdevError(ErrorCode.INVALID_REQUEST, f"bad duration {text!r}", details={"field": "ttl"},
                        recovery="use e.g. 90m, 12h, 2d (0 = never expires)")
    return float(m.group(1)) * {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}[m.group(2)]


def _clip(lines: list[str], n: int = LIST_MAX, hint: str = "") -> list[str]:
    if len(lines) <= n:
        return lines
    return lines[:n] + [f"... {len(lines) - n} more{(' (' + hint + ')') if hint else ''}"]


def _posts_result(tool: str, posts: list[Any], verdict: str, extra: dict[str, Any] | None = None,
                  empty: str = "no posts", total: int | None = None, hint: str = "") -> Result:
    lines = [p.line() for p in posts] or [empty]
    if total is not None and total > len(posts):
        lines.insert(0, f"(newest {len(posts)} of {total}; {hint or 'use --limit / --since / --kind'})")
    return Result(tool=tool, verdict=verdict, summary={"n": len(posts), **(extra or {}),
                                                       "posts": [p.to_dict() for p in posts]},
                  lines=_clip(lines, LIST_MAX + 1, "use --json"))


# ------------------------------------------------------------------------------------------------ board
def cmd_board_post(args: Namespace, ctx: Context) -> Result:
    from kdev.collab import board, docs
    from kdev.collab.refs import parse_refs
    refs = parse_refs(args.ref)
    agent = _agent(ctx)
    with _conn(ctx) as c:
        p, conflicts, closed = board.post(c, agent=agent, kind=args.kind, body=args.text, problem=_problem(ctx),
                                          to=args.to, refs=refs, reply_to=args.reply_to, files=args.files,
                                          shapes=args.shapes, ttl_s=_duration_s(args.ttl, board.DEFAULT_TTL_S),
                                          exclusive=args.exclusive, source=ctx.via)
        entry = None
        if args.doc:
            name = docs.doc_name(args.doc, _problem(ctx))
            entry = docs.append(c, name, args.section or "Board", f"[{p.kind} #{p.post_id}] {args.text}",
                                author=agent, problem=_problem(ctx), refs=refs)
        unread = len(board.inbox(c, agent)) if agent else 0
    lines = [p.line(200)]
    warnings = []
    for cl, ov in conflicts:
        lines.append(f"CONFLICT with claim #{cl.post_id} by {cl.agent} ({cl.created_utc[5:16].replace('T', ' ')}): {','.join(ov)}"
                     f" -- {' '.join(cl.body.split())[:90]}")
        warnings.append(f"claim overlaps #{cl.post_id} ({cl.agent}): {','.join(ov)}")
    if closed:
        lines.append("closed claims: " + ", ".join(f"#{i}" for i in closed))
    if entry is not None:
        lines.append(f"doc entry {entry.entry_id} -> {entry.doc} / {entry.section}")
    if p.kind == "claim" and not p.scope:
        warnings.append("claim without --files/--shapes: nobody can see what it overlaps")
    nxt = []
    if conflicts:
        nxt.append(f"kdev board post ask \"...\" --to {conflicts[0][0].agent} --reply-to {conflicts[0][0].post_id}")
    if unread:
        nxt.append(f"kdev board inbox  ({unread} unacked for {agent})")
    return Result(tool="board post", verdict="CONFLICT" if conflicts else "posted",
                  summary={"post_id": p.post_id, "kind": p.kind, "conflicts": len(conflicts), "inbox": unread,
                           "post": p.to_dict(), "closed_claims": closed,
                           "conflict_posts": [{"post_id": x.post_id, "agent": x.agent, "overlap": o, "body": x.body}
                                              for x, o in conflicts],
                           "doc_entry": entry.entry_id if entry else None},
                  lines=lines, warnings=warnings, next=nxt)


def cmd_board_read(args: Namespace, ctx: Context) -> Result:
    from kdev.collab import board
    me = _agent(ctx)
    if args.unread and not me:
        raise KdevError(ErrorCode.INVALID_REQUEST, "--unread needs an agent name", details={"field": "agent"},
                        recovery="pass --agent NAME or set KDEV_AGENT")
    with _conn(ctx) as c:
        posts, total = board.read(c, problem=_problem(ctx), kind=args.kind, agent=args.by, since=args.since,
                                  to=args.to, limit=args.limit, unread_for=me if args.unread else None)
        inbox_n = len(board.inbox(c, me)) if me else 0
    r = _posts_result("board read", posts, f"{len(posts)} posts", {"total": total, "inbox": inbox_n},
                      empty="no new posts" if args.unread else "no posts", total=total)
    if inbox_n:
        r.next = [f"kdev board inbox  ({inbox_n} unacked for {me})"]
    return r


def cmd_board_tail(args: Namespace, ctx: Context) -> Result:
    from kdev.collab import board
    if not args.follow:
        args.unread, args.by, args.to, args.since = False, None, None, None
        return cmd_board_read(args, ctx)
    if ctx.json or ctx.via == "mcp":
        raise KdevError(ErrorCode.INVALID_REQUEST, "--follow is interactive (CLI text mode only)")
    with _conn(ctx) as c:
        posts, _ = board.read(c, problem=_problem(ctx), kind=args.kind, limit=args.limit)
        for p in posts:
            print(p.line(200), flush=True)
        last = posts[-1].post_id if posts else 0
        try:
            while True:
                time.sleep(max(1.0, args.every))
                new = [p for p in board.read(c, problem=_problem(ctx), kind=args.kind, limit=200)[0]
                       if p.post_id > last]
                for p in new:
                    print(p.line(200), flush=True)
                    last = p.post_id
        except KeyboardInterrupt:
            pass
    return Result(tool="board tail", verdict="stopped", lines=[])


def cmd_board_inbox(args: Namespace, ctx: Context) -> Result:
    from kdev.collab import board
    me = _agent(ctx)
    with _conn(ctx) as c:
        posts = board.inbox(c, me, include_acked=args.all, limit=args.limit)
        acked = board.ack(c, me, [p.post_id for p in posts]) if args.ack and posts else 0
    r = _posts_result("board inbox", posts, f"{len(posts)} for {me}", {"agent": me, "acked": acked},
                      empty=f"inbox empty for {me}")
    if posts and not args.ack:
        r.next = [f"kdev board ack {' '.join(str(p.post_id) for p in posts[:8])}   (or kdev board inbox --ack)"]
    return r


def cmd_board_ack(args: Namespace, ctx: Context) -> Result:
    from kdev.collab import board
    me = _agent(ctx)
    with _conn(ctx) as c:
        ids = list(args.post_ids or [])
        if args.all:
            ids += [p.post_id for p in board.inbox(c, me, limit=10_000)]
        if not ids:
            raise KdevError(ErrorCode.INVALID_REQUEST, "pass post ids or --all", details={"field": "post_ids"})
        for i in ids:
            board.get(c, i)
        n = board.ack(c, me, ids)
    return Result(tool="board ack", verdict="acked", summary={"acked": n, "ids": ids},
                  lines=[f"{me} acked {n} post(s): " + " ".join(f"#{i}" for i in ids[:20])])


def cmd_board_search(args: Namespace, ctx: Context) -> Result:
    from kdev.collab import board
    with _conn(ctx) as c:
        posts = board.search(c, args.query, problem=_problem(ctx), kind=args.kind, ref=args.ref, limit=args.limit)
    return _posts_result("board search", posts, f"{len(posts)} hits", {"query": args.query}, empty="no match")


def cmd_board_thread(args: Namespace, ctx: Context) -> Result:
    from kdev.collab import board
    from kdev.collab.refs import describe
    with _conn(ctx) as c:
        posts = board.thread(c, args.post_id, viewer=_agent(ctx) or None)
    lines = []
    for p in posts:
        if len(posts) > 6:
            lines.append(p.line(150))
        else:
            lines.append(p.head())
            lines += ["    " + x for x in p.body.splitlines()[:6]]
            if p.refs:
                lines.append("    refs: " + ", ".join(describe(p.refs, ctx.home)))
    return Result(tool="board thread", verdict=f"{len(posts)} posts", summary={"root": posts[0].thread_id,
                  "posts": [p.to_dict() for p in posts]}, lines=_clip(lines, 34, "use --json"))


def cmd_board_claims(args: Namespace, ctx: Context) -> Result:
    from kdev.collab import board
    me = _agent(ctx)
    with _conn(ctx) as c:
        claims = board.open_claims(c, problem=_problem(ctx), agent=me if args.mine else None)
        pairs = []
        for i, a in enumerate(claims):
            for b in claims[i + 1:]:
                if a.agent != b.agent and board._same_problem(a.problem, b.problem):
                    ov = board.overlap(a.scope, b.scope)
                    if ov:
                        pairs.append((a, b, ov))
    lines = [p.line() for p in claims] or ["no open claims"]
    lines += [f"CONFLICT #{a.post_id} ({a.agent}) x #{b.post_id} ({b.agent}): {','.join(ov)}" for a, b, ov in pairs]
    return Result(tool="board claims", verdict="CONFLICT" if pairs else f"{len(claims)} open",
                  summary={"n": len(claims), "conflicts": len(pairs), "claims": [p.to_dict() for p in claims],
                           "conflict_pairs": [[a.post_id, b.post_id, ov] for a, b, ov in pairs]},
                  lines=_clip(lines, 34, "use --json"))


def cmd_board_release(args: Namespace, ctx: Context) -> Result:
    from kdev.collab import board
    with _conn(ctx) as c:
        p = board.release(c, args.post_id, _agent(ctx), force=args.force)
    return Result(tool="board release", verdict=p.claim_state, summary={"post": p.to_dict()}, lines=[p.line()])


def cmd_board_import(args: Namespace, ctx: Context) -> Result:
    from kdev.collab import board
    with _conn(ctx) as c:
        added, skipped = board.import_legacy(c, args.path, year=args.year)
    return Result(tool="board import", verdict="imported", summary={"added": added, "skipped": skipped},
                  lines=[f"{args.path}: {added} posts added, {skipped} already present"])


# -------------------------------------------------------------------------------------------------- doc
def _doc(ctx: Context, name: str) -> str:
    from kdev.collab.docs import doc_name
    return doc_name(name, _problem(ctx))


def cmd_doc_append(args: Namespace, ctx: Context) -> Result:
    from kdev.collab import docs
    from kdev.collab.refs import parse_refs, short
    name = _doc(ctx, args.doc)
    with _conn(ctx) as c:
        e = docs.append(c, name, args.section, args.text, author=_agent(ctx), kind=args.kind, problem=_problem(ctx),
                        refs=parse_refs(args.ref), supersedes=args.supersedes, tags=args.tags)
    lines = [f"entry {e.entry_id} -> {e.doc} / {e.section or '(preamble)'}  by {e.author} {e.created_utc}"]
    if e.refs:
        lines.append("refs: " + ", ".join(short(r) for r in e.refs))
    if args.supersedes:
        lines.append(f"supersedes entry {args.supersedes} (kept in --history)")
    w = [] if e.refs else ["no --ref: findings need evidence (run id, idea, judge job, file)"]
    return Result(tool="doc append", verdict="appended", summary={"entry_id": e.entry_id, "doc": e.doc,
                  "section": e.section, "entry": e.to_dict()}, lines=lines, warnings=w)


def cmd_doc_show(args: Namespace, ctx: Context) -> Result:
    from kdev.collab import docs
    from kdev.collab.refs import describe
    name = _doc(ctx, args.doc)
    with _conn(ctx) as c:
        docs.get_doc(c, name)
        if args.history or args.refs:
            sec = docs.resolve_section(c, name, args.section) if args.section else None
            es = docs.entries(c, name, sec, history=args.history)
            text_lines = []
            for e in es:
                flag = f" SUPERSEDED by {e.superseded_by}" if e.superseded_by else ""
                text_lines.append(f"[{e.entry_id}] {e.section or '(preamble)'} | {e.author} {e.created_utc[:16]} "
                                  f"{e.source}{flag}")
                if args.refs and e.refs:
                    text_lines.append("    refs: " + ", ".join(describe(e.refs, ctx.home)))
                elif args.history:
                    text_lines.append("    " + docs.snippet(e.body, "", 140))
        elif args.section:
            sec = docs.resolve_section(c, name, args.section)
            s = next(x for x in docs.sections(c, name) if x["section"] == sec)
            text_lines = docs.render_section(c, name, s).splitlines()
        else:
            text_lines = docs.render(c, name).splitlines()
    off = max(0, args.offset)
    page = text_lines[off: off + args.limit]
    nxt_off = off + len(page) if off + len(page) < len(text_lines) else None
    lines = page + ([f"... lines {off + 1}-{off + len(page)} of {len(text_lines)}; next: --offset {nxt_off}"]
                    if nxt_off is not None else [])
    return Result(tool="doc show", verdict=name, summary={"doc": name, "offset": off, "lines_total": len(text_lines),
                  "next_offset": nxt_off, "text": "\n".join(page)}, lines=lines)


def cmd_doc_toc(args: Namespace, ctx: Context) -> Result:
    from kdev.collab import docs
    name = _doc(ctx, args.doc)
    with _conn(ctx) as c:
        d = docs.get_doc(c, name)
        rows = []
        for s in docs.sections(c, name):
            es = docs.entries(c, name, s["section"])
            body = docs._combine(es, bool(s["newest_first"]))
            subs = sum(1 for ln in body.splitlines() if ln.startswith("### "))
            last = max((e.created_utc for e in es), default="")
            authors = sorted({e.author for e in es if not e.raw})
            rows.append({"section": s["section"], "entries": len(es), "lines": body.count("\n"), "subsections": subs,
                         "updated_utc": last, "authors": authors, "newest_first": bool(s["newest_first"])})
    lines = [f"{d['title']}  [{d['kind']}]"]
    for r in rows:
        lines.append(f"  {(r['section'] or '(preamble)')[:78]:78s} {r['lines']:4d} lines {r['entries']:2d} entries"
                     + (f" {r['subsections']} sub" if r["subsections"] else "")
                     + (f" by {','.join(r['authors'])[:30]}" if r["authors"] else ""))
    return Result(tool="doc toc", verdict=name, summary={"doc": name, "sections": rows},
                  lines=_clip(lines, 36, "use --json"))


def cmd_doc_list(args: Namespace, ctx: Context) -> Result:
    from kdev.collab import docs
    with _conn(ctx) as c:
        rows = docs.docs_list(c, args.kind)
    lines = [f"{r['doc'][:40]:40s} {r['kind']:8s} {r['sections']:3d} sections {r['entries']:4d} entries "
             f"updated {(r['updated_utc'] or '-')[:16]}" for r in rows] or ["no docs (kdev doc import FILE.md)"]
    return Result(tool="doc list", verdict=f"{len(rows)} docs", summary={"docs": rows}, lines=_clip(lines))


def cmd_doc_search(args: Namespace, ctx: Context) -> Result:
    from kdev.collab import docs
    d = _doc(ctx, args.doc) if args.doc else None
    with _conn(ctx) as c:
        es = docs.search(c, args.query, doc=d, ref=args.ref, limit=args.limit)
    lines = [f"[{e.entry_id}] {e.doc} / {(e.section or '(preamble)')[:40]} ({e.author or '-'} {e.created_utc[:10]}): "
             f"{docs.snippet(e.body, args.query, 90)}" for e in es] or ["no match"]
    return Result(tool="doc search", verdict=f"{len(es)} hits", summary={"query": args.query,
                  "entries": [e.to_dict() for e in es]}, lines=_clip(lines),
                  next=[f"kdev doc show {es[0].doc} --section \"{es[0].section}\""] if es and es[0].section else [])


def cmd_doc_checklist(args: Namespace, ctx: Context) -> Result:
    from kdev.collab import docs
    with _conn(ctx) as c:
        ck = docs.checklist(c, args.doc, rules=args.rules)
    if args.md:
        lines = ["## Brief checklist"] + [f"{i}. {t}" for i, t in enumerate(ck["items"], 1)]
        if ck["rules"]:
            lines += ["", "## Rules"] + [f"- {r}" for r in ck["rules"]]
    else:
        lines = [f"{i}. {t}" for i, t in enumerate(ck["items"], 1)] + [f"rule {r}" for r in ck["rules"]]
    if not ck["items"]:
        lines.insert(0, f"no section with 'checklist' in its heading in {args.doc}")
    return Result(tool="doc checklist", verdict=f"{len(ck['items'])} items", summary=ck,
                  lines=lines[:38] if not args.md else lines)


def cmd_doc_import(args: Namespace, ctx: Context) -> Result:
    from kdev.collab import docs
    reps = []
    with _conn(ctx) as c:
        for p in args.paths:
            reps.append(docs.import_md(c, p, doc=_doc(ctx, args.doc) if args.doc else None, kind=args.kind,
                                       problem=_problem(ctx), author=_agent(ctx) or "import"))
    lines, warns = [], []
    for r in reps:
        lines.append(f"{r.path} -> {r.doc}{' (new)' if r.created else ''}: {len(r.added)} new sections, "
                     f"{len(r.appended)} appended, {len(r.superseded)} replaced, {r.unchanged} unchanged")
        lines += [f"  replaced (old text kept in history): {s[:90]}" for s in r.superseded[:5]]
        if r.kept_missing:
            warns.append(f"{r.doc}: {len(r.kept_missing)} sections not in {Path(r.path).name} were KEPT: "
                         + "; ".join(s[:40] for s in r.kept_missing[:4]))
    return Result(tool="doc import", verdict="imported", summary={"reports": [r.to_dict() for r in reps]},
                  lines=_clip(lines), warnings=warns,
                  next=[f"kdev doc toc {reps[0].doc}"] if reps else [])


def cmd_doc_export(args: Namespace, ctx: Context) -> Result:
    from kdev.collab import docs
    out = []
    with _conn(ctx) as c:
        if args.all:
            if not args.dir:
                raise KdevError(ErrorCode.INVALID_REQUEST, "--all needs --dir DIR", details={"field": "dir"})
            for d in docs.docs_list(c):
                path = Path(args.dir) / (d["doc"].replace("/", "__") + ".md")
                out.append(docs.export_md(c, d["doc"], path, force=args.force))
        else:
            if not args.doc or not args.md:
                raise KdevError(ErrorCode.INVALID_REQUEST, "usage: kdev doc export DOC --md PATH | --all --dir DIR",
                                details={"field": "md"})
            out.append(docs.export_md(c, _doc(ctx, args.doc), args.md, force=args.force))
    lines = [f"{o['status']:9s} {o['doc']} -> {o['path']} ({o['bytes']} B md5 {o['md5'][:8]})" for o in out]
    return Result(tool="doc export", verdict="exported", summary={"exports": out}, lines=_clip(lines),
                  files={o["doc"]: o["path"] for o in out[:50]})


def cmd_collab_backup(args: Namespace, ctx: Context) -> Result:
    from kdev.collab.backup import backup
    rep = backup(ctx.home, args.out)
    lines = [f"{k:28s} {v['bytes']:>10d} B  md5 {v['md5'][:12]}" for k, v in list(rep["files"].items())[:25]]
    lines.insert(0, f"backup -> {rep['dir']} ({rep['posts']} posts, {rep['docs']} docs, {rep['entries']} entries)")
    return Result(tool="collab backup", verdict="backed up", summary=rep, lines=lines, files={"dir": rep["dir"]})


# ------------------------------------------------------------------------------------------------ register
def register(reg: Registry) -> None:
    from kdev.collab.board import KINDS
    from kdev.collab.docs import KINDS as DOC_KINDS
    ref = Arg(("--ref",), "evidence: run id, idea:N/#N, judge:ID, file path, url (repeatable/comma)",
              action="append", default=None)
    limit = Arg(("--limit", "-n"), "max posts", type="int", default=LIST_MAX)
    reg.command("board post", "post to the agent board (claim/finding/done/ask/answer/warn/decision)",
                cmd_board_post, args=[
                    Arg(("kind",), "|".join(KINDS), choices=KINDS),
                    Arg(("text",), "the message (quote it); @agent mentions reach that agent's inbox"),
                    Arg(("--to",), "addressee agent (default all)", default=None),
                    ref,
                    Arg(("--reply-to",), "parent post id (threads)", type="int", default=None),
                    Arg(("--files",), "claim/done scope: files or globs (repeatable/comma)", action="append",
                        default=None),
                    Arg(("--shapes",), "claim/done scope: shapes (repeatable/comma)", action="append", default=None),
                    Arg(("--ttl",), "claim lifetime, e.g. 6h (0 = no expiry; default 12h)", default=None),
                    Arg(("--exclusive",), "refuse the claim if it overlaps another agent's claim",
                        action="store_true"),
                    Arg(("--doc",), "also append it to this doc (e.g. notes)", default=None),
                    Arg(("--section",), "doc section for --doc (default Board)", default=None)],
                effects=W_BOARD, package=P)
    read_args = [Arg(("--kind",), "kinds (comma list)", default=None), Arg(("--by",), "author", default=None),
                 Arg(("--since",), "3h, 30m, HH:MM (UTC today) or ISO", default=None),
                 Arg(("--to",), "addressed to this agent or all", default=None),
                 Arg(("--unread",), "only posts since my last --unread read (moves my cursor)", action="store_true"),
                 limit]
    reg.command("board read", "read the board (filters: problem, kind, author, since, unread)", cmd_board_read,
                args=read_args, effects=W_BOARD, package=P)
    reg.command("board tail", "last posts; --follow streams new ones (CLI)", cmd_board_tail,
                args=[Arg(("--kind",), "kinds (comma list)", default=None), limit,
                      Arg(("--follow", "-f"), "keep printing new posts", action="store_true", mcp=False),
                      Arg(("--every",), "poll interval s for --follow", type="float", default=5.0, mcp=False)],
                package=P)
    reg.command("board inbox", "posts addressed to me, @mentions, replies to my posts (unacked)", cmd_board_inbox,
                args=[Arg(("--all",), "include acked posts", action="store_true"),
                      Arg(("--ack",), "ack everything shown", action="store_true"), limit],
                effects=W_BOARD, package=P)
    reg.command("board ack", "mark posts as read for me", cmd_board_ack,
                args=[Arg(("post_ids",), "post ids", type="int", nargs="*", default=None),
                      Arg(("--all",), "ack my whole inbox", action="store_true")], effects=W_BOARD, package=P)
    reg.command("board search", "full-text search of the board", cmd_board_search,
                args=[Arg(("query",), "words (all must match)"), Arg(("--kind",), "kind", default=None),
                      Arg(("--ref",), "substring of a ref (run id, file, ...)", default=None), limit], package=P)
    reg.command("board thread", "a post with its whole thread", cmd_board_thread,
                args=[Arg(("post_id",), "any post of the thread", type="int")], package=P)
    reg.command("board claims", "open claims and overlaps between them", cmd_board_claims,
                args=[Arg(("--mine",), "only my claims", action="store_true")], package=P)
    reg.command("board release", "close my open claim without a done post", cmd_board_release,
                args=[Arg(("post_id",), "claim id", type="int"),
                      Arg(("--force",), "release someone else's claim (coordinator)", action="store_true")],
                effects=W_BOARD, package=P)
    reg.command("board import", "ingest the interim bash board file (BOARD.md lines)", cmd_board_import,
                args=[Arg(("path",), "BOARD.md", type="path"),
                      Arg(("--year",), "year of the MM-DD stamps (default now)", type="int", default=None)],
                effects=W_BOARD, mcp=False, package=P)

    reg.command("doc append", "append an entry to a doc section (never edit shared .md files)", cmd_doc_append,
                args=[Arg(("doc",), "notes (+--problem) | lessons | tracker | design/X | full name"),
                      Arg(("text",), "entry text (markdown)"),
                      Arg(("--section",), "section heading (created if new; prefix match)", required=True),
                      ref,
                      Arg(("--supersedes",), "entry id this one replaces (old kept in history)", type="int",
                          default=None),
                      Arg(("--kind",), "|".join(DOC_KINDS) + " for a new doc", choices=DOC_KINDS, default=None),
                      Arg(("--tags",), "comma-separated", default="")],
                effects=W_DOCS, package=P)
    reg.command("doc show", "render a doc or one section (paged)", cmd_doc_show,
                args=[Arg(("doc",), "doc name"), Arg(("--section",), "section (prefix match)", default=None),
                      Arg(("--history",), "list entries incl. superseded ones", action="store_true"),
                      Arg(("--refs",), "list entries with their evidence refs", action="store_true"),
                      Arg(("--offset",), "first line", type="int", default=0),
                      Arg(("--limit",), "lines per page", type="int", default=34)], package=P)
    reg.command("doc toc", "sections of a doc with sizes and authors", cmd_doc_toc,
                args=[Arg(("doc",), "doc name")], package=P)
    reg.command("doc list", "all docs", cmd_doc_list,
                args=[Arg(("--kind",), "|".join(DOC_KINDS), default=None)], package=P)
    reg.command("doc search", "full-text search over doc entries", cmd_doc_search,
                args=[Arg(("query",), "words (all must match)"), Arg(("--doc",), "restrict to a doc", default=None),
                      Arg(("--ref",), "substring of a ref (run id, idea, file)", default=None),
                      Arg(("--limit",), "max hits", type="int", default=LIST_MAX)], package=P)
    reg.command("doc checklist", "the lessons' brief checklist (paste into agent briefs)", cmd_doc_checklist,
                args=[Arg(("--doc",), "lessons doc", default="lessons"),
                      Arg(("--rules",), "add one 'Rule:' line per lesson", action="store_true"),
                      Arg(("--md",), "markdown for pasting", action="store_true")], package=P)
    reg.command("doc import", "ingest NOTES/LESSONS/TRACKER.md by headings (idempotent; keeps history)",
                cmd_doc_import,
                args=[Arg(("paths",), "markdown files", type="path", nargs="+"),
                      Arg(("--doc",), "target doc (default from the file name)", default=None),
                      Arg(("--kind",), "|".join(DOC_KINDS), choices=DOC_KINDS, default=None)],
                effects=W_DOCS, mcp=False, package=P)
    reg.command("doc export", "render a doc to a .md file (refuses to overwrite hand edits)", cmd_doc_export,
                args=[Arg(("doc",), "doc name", nargs="?", default=None),
                      Arg(("--md",), "output path", type="path", default=None),
                      Arg(("--all",), "every doc into --dir", action="store_true"),
                      Arg(("--dir",), "output dir for --all", type="path", default=None),
                      Arg(("--force",), "overwrite even if the file has foreign edits", action="store_true")],
                effects=Effects(writes=("files",)), mcp=False, package=P)
    reg.command("collab backup", "consistent snapshot of collab.db + ledger.db + rendered docs + board.jsonl",
                cmd_collab_backup,
                args=[Arg(("--out",), "backup dir (default $KDEV_HOME/backup)", type="path", default=None)],
                effects=Effects(writes=("files",)), mcp=False, package=P)

