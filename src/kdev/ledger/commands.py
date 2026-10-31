"""CLI: ledger *, idea *, note. Owner: P4."""
from __future__ import annotations

import os
from argparse import Namespace
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from kdev.contract import Effects, ErrorCode, KdevError, Result
from kdev.registry import Arg, Context, Registry

P = "P4"
W_LEDGER = Effects(writes=("ledger",))
W_IDEAS = Effects(writes=("ideas",))


@contextmanager
def _conn(ctx: Context) -> Iterator[Any]:
    from kdev.ledger.db import open_default
    c = open_default(ctx.home)
    try:
        yield c
    finally:
        c.close()


def _f(x: Any, nd: int = 4) -> str:
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def _clip(lines: list[str], n: int = 30, hint: str = "") -> list[str]:
    if len(lines) <= n:
        return lines
    return lines[:n] + [f"... {len(lines) - n} more rows{(' (' + hint + ')') if hint else ''}"]


# ---------------------------------------------------------------- ledger
def cmd_query(args: Namespace, ctx: Context) -> Result:
    from kdev.ledger import db, query
    from kdev.ledger._util import md5_file, since_to_utc
    with _conn(ctx) as c:
        if args.sql:
            if ctx.via == "mcp":
                raise KdevError(ErrorCode.QUOTA_FORBIDDEN, "--sql is CLI-only")
            rows = db.readonly_query(c, args.sql, limit=max(1, args.limit))
            lines = [" | ".join(f"{k}={_f(v)}" for k, v in r.items())[:200] for r in rows]
            return Result(tool="ledger query", verdict="ok", summary={"rows": rows[:50], "n": len(rows)},
                          lines=_clip(lines, 30, "use --limit or --json"))
        md5 = args.md5 or (md5_file(args.file) if args.file else None)
        rows = query.runs(c, problem=ctx.problem, md5=md5, tool=args.tool, since_utc=since_to_utc(args.since),
                          limit=args.limit)
    lines = [f"{r['run_id'][:40]:40s} {r['tool']:8s} {r['verdict'] or r['status']:9s} "
             f"{'' if r['comparable'] else 'NOT-COMPARABLE '}{r['files'] or ''}" for r in rows]
    return Result(tool="ledger query", verdict="ok" if rows else "empty", summary={"runs": rows, "n": len(rows)},
                  lines=_clip(lines or ["no runs"], 30, "use --limit/--json"))


def cmd_tried(args: Namespace, ctx: Context) -> Result:
    from kdev.ledger import query
    from kdev.ledger._util import parse_kv
    is_file = os.path.isfile(args.target)
    knobs = parse_kv(args.knobs)
    with _conn(ctx) as c:
        rep = query.tried(c, file=args.target if is_file else None, text=None if is_file else args.target,
                          knobs=knobs or None, problem=ctx.problem)
        if is_file:   # also search ideas mentioning the file name (variants are often named by file)
            more = query.tried(c, text=Path(args.target).stem, problem=ctx.problem).hits
            rep.hits += [h for h in more if h.idea_id not in {x.idea_id for x in rep.hits}]
    lines = [f"{h.match:9s} {(h.run_id or h.judge_id or ('idea ' + str(h.idea_id))):36s} {h.tool:10s} "
             f"{h.verdict:9s} {h.created_utc[:16]} {'' if h.comparable else 'NOT-COMPARABLE '}{h.detail[:60]}"
             for h in rep.hits]
    verdict = "tried" if rep.hits else "new"
    return Result(tool="ledger tried", verdict=verdict,
                  summary={"query": rep.query, "hits": [h.to_dict() for h in rep.hits[:100]]},
                  lines=_clip(lines or ["no match: not tried (md5, norm_md5, knobs, idea text)"], 30))


def cmd_best(args: Namespace, ctx: Context) -> Result:
    from kdev.ledger import query
    with _conn(ctx) as c:
        rows = query.best(c, ctx.problem)
    lines = []
    for r in rows:
        j, lo = r["judged"], r["local"]
        if j:
            b = j.get("baseline")
            lines.append(f"{r['problem']}: judged x{_f(j['speedup'])} {j['judge_id']} {j.get('label') or ''} "
                         f"{j.get('cluster') or ''} {(j.get('created_utc') or '')[:16]}")
            lines.append("    same-hour base: " + (f"x{_f(b['speedup'])} {b['judge_id']} -> vs base "
                                                     f"x{_f(j['vs_baseline'])}" if b else "NONE (judge one, LESSONS 3)"))
            if j.get("board_top"):
                t = j["board_top"]
                lines.append(f"    board #1: {t['team']} x{_f(t['speedup'])} ({t['taken_utc'][:10]})")
        if lo:
            lines.append(f"{r['problem']}: local score {_f(lo['score'])} {lo['run_id']} {lo['label']} "
                         f"{lo.get('path') or ''}")
    return Result(tool="ledger best", verdict="ok" if rows else "empty", summary={"best": rows},
                  lines=_clip(lines or ["ledger has no judged or comparable local results"], 36))


def cmd_history(args: Namespace, ctx: Context) -> Result:
    from kdev.ledger import query
    with _conn(ctx) as c:
        rows = query.shape_history(c, ctx.problem, args.shape, limit=args.limit)
    lines = [f"{r['shape_id'][:40]:40s} {(r['utc'] or '')[:16]} {r['source']:5s} {r['ref'][:20]:20s} "
             f"sub {_f(r['sub_us'], 2)}us x{_f(r['speedup'])} {r['cluster'] or ''} {(r['label'] or '')[:30]}"
             for r in rows]
    return Result(tool="ledger history", verdict="ok" if rows else "empty",
                  summary={"rows": rows[-200:], "n": len(rows)},
                  lines=_clip(lines or ["no rows"], 30, "narrow with --shape / --problem or use --json"))


def cmd_import(args: Namespace, ctx: Context) -> Result:
    from kdev.judge import _importer as imp
    from kdev.ledger.record import ingest_result
    done: dict[str, list[str]] = {"result": [], "judge": [], "board": [], "okc_tree": [], "skipped": []}
    trees: list[dict[str, Any]] = []
    with _conn(ctx) as c:
        for p in args.paths:
            if not os.path.exists(p):
                raise KdevError(ErrorCode.NOT_FOUND, f"no such path: {p}", details={"path": p})
            kind = imp.detect_kind(p)
            if kind in ("result", "run_dir"):
                done["result"].append(ingest_result(p, conn=c))
            elif kind == "judge":
                r = imp.import_judge_file(c, p)
                done["judge"].append(r[0] if r else p)
            elif kind == "board":
                done["board"].append(f"{p}: {imp.import_board_file(c, p)} rows")
            elif os.path.isdir(p) and (any(Path(p).glob("muse_*/muse_*")) or any(Path(p).glob("research"))):
                t = imp.import_okc_tree(c, p)
                trees.append({k: v for k, v in t.items() if k != "recorded"})
                done["okc_tree"].append(p)
            elif os.path.isdir(p):
                found = list(Path(p).rglob("result.json"))
                for f in found:
                    done["result"].append(ingest_result(f, conn=c))
                if not found:
                    done["skipped"].append(p)
            else:
                done["skipped"].append(p)
    lines = [f"results: {len(done['result'])}  judge: {len(done['judge'])}  boards: {len(done['board'])}  "
             f"okc trees: {len(done['okc_tree'])}  skipped: {len(done['skipped'])}"]
    for t in trees:
        lines.append(f"okc tree: {t['judge_files']} judge files ({t['unique_judge_ids']} unique jobs), "
                     f"{t['pairs']} same-hour pairs, {t['board_rows']} board rows, "
                     f"{t['files_matched_by_hash']} files matched by hash, {len(t['skipped'])} non-judge JSONs")
    lines += [f"skipped: {s}" for s in done["skipped"][:5]]
    return Result(tool="ledger import", verdict="ok", summary={**done, "trees": trees}, lines=lines)


def cmd_export(args: Namespace, ctx: Context) -> Result:
    from kdev.ledger import query
    from kdev.ledger.db import default_path
    from kdev.util import utc_now
    tables = tuple(args.table or ("runs",))
    with _conn(ctx) as c:
        text = query.export(c, args.format, tables)
    out = args.out or str(default_path(ctx.home).parent / "exports" /
                          f"ledger_{'_'.join(tables)}_{utc_now().replace(':', '')}.{args.format}")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(text, encoding="utf-8")
    n = text.count("\n")
    return Result(tool="ledger export", verdict="ok", summary={"path": out, "lines": n, "tables": list(tables)},
                  files={"export": out}, lines=[f"wrote {n} lines to {out}"])


# ---------------------------------------------------------------- ideas
def _idea_lines(i: Any) -> list[str]:
    out = [f"#{i.idea_id} [{i.kind}/{i.status}] {i.title}",
           f"  problem={i.problem or '-'} owner={i.owner or '-'} gain={_f(i.expected_gain_pct, 1)}% "
           f"tags={','.join(i.tags) or '-'} updated={i.updated_utc}"]
    if i.hypothesis:
        out.append(f"  hypothesis: {i.hypothesis[:300]}")
    if i.variant:
        out.append(f"  variant: {i.variant[:300]}")
    if i.resolution:
        out.append(f"  resolution: {i.resolution[:300]}")
    for e in i.evidence[-12:]:
        out.append(f"  evidence {e.supports:7s} {e.ref_kind}:{e.ref} {e.note[:80]} ({e.author or '-'} {e.added_utc})")
    return out


def _idea_result(tool: str, i: Any, verdict: str | None = None) -> Result:
    return Result(tool=tool, verdict=verdict or i.status, summary={"idea": i.to_dict()}, lines=_idea_lines(i))


def cmd_idea_add(args: Namespace, ctx: Context) -> Result:
    from kdev.ledger import ideas
    with _conn(ctx) as c:
        i = ideas.add(c, ideas.Idea(None, args.title, problem=ctx.problem, hypothesis=args.hypothesis,
                                    kind=args.kind, expected_gain_pct=args.gain,
                                    tags=[t for t in args.tags.split(",") if t.strip()], variant=args.variant))
        similar = [s for s in ideas.search(c, args.title, problem=ctx.problem, limit=6) if s.idea_id != i.idea_id]
    r = _idea_result("idea add", i, "added")
    r.warnings += [f"similar idea #{s.idea_id} [{s.kind}/{s.status}] {s.title[:80]}" for s in similar[:3]]
    return r


def cmd_idea_list(args: Namespace, ctx: Context) -> Result:
    from kdev.ledger import ideas
    with _conn(ctx) as c:
        lst = ideas.list_ideas(c, problem=ctx.problem, status=args.status, owner=args.owner, kind=args.kind,
                               limit=args.limit)
    lines = [f"#{i.idea_id:<4d} {i.kind:5s} {i.status:9s} {(i.owner or '-')[:12]:12s} {i.title[:80]}" for i in lst]
    return Result(tool="idea list", verdict="ok" if lst else "empty",
                  summary={"ideas": [i.to_dict() for i in lst]}, lines=_clip(lines or ["no ideas"], 30))


def cmd_idea_show(args: Namespace, ctx: Context) -> Result:
    from kdev.ledger import ideas
    with _conn(ctx) as c:
        return _idea_result("idea show", ideas.get(c, args.idea_id))


def cmd_idea_claim(args: Namespace, ctx: Context) -> Result:
    from kdev.ledger import ideas
    with _conn(ctx) as c:
        return _idea_result("idea claim", ideas.claim(c, args.idea_id, ctx.agent))


def cmd_idea_release(args: Namespace, ctx: Context) -> Result:
    from kdev.ledger import ideas
    with _conn(ctx) as c:
        return _idea_result("idea release", ideas.release(c, args.idea_id, ctx.agent))


def cmd_idea_evidence(args: Namespace, ctx: Context) -> Result:
    from kdev.ledger import ideas
    refs = [(k, v) for k, v in (("run", args.run), ("judge", args.judge), ("file", args.file), ("url", args.url),
                                ("text", args.text)) if v]
    if not refs:
        raise KdevError(ErrorCode.INVALID_REQUEST, "pass one of --run/--judge/--file/--url/--text",
                        details={"field": "ref"})
    with _conn(ctx) as c:
        i = None
        for kind, ref in refs:
            note = ""
            if kind == "file":
                from kdev.ledger._util import md5_file
                if not os.path.isfile(ref):
                    raise KdevError(ErrorCode.NOT_FOUND, f"no such file: {ref}", details={"path": ref})
                note = f"md5 {md5_file(ref)}"
                ref = os.path.abspath(ref)
            i = ideas.add_evidence(c, args.idea_id, ideas.Evidence(kind, ref, note=note, supports=args.supports,
                                                                    author=ctx.agent))
    return _idea_result("idea evidence", i)


def cmd_idea_resolve(args: Namespace, ctx: Context) -> Result:
    from kdev.ledger import ideas
    with _conn(ctx) as c:
        return _idea_result("idea resolve", ideas.resolve(c, args.idea_id, args.status, args.resolution, ctx.agent))


def cmd_note(args: Namespace, ctx: Context) -> Result:
    """`note CLAIM --kind dead|limit|win --evidence RUN,RUN --variant V`: a closed record in one step."""
    from kdev.ledger import ideas
    runs = [r.strip() for r in args.evidence.split(",") if r.strip()]
    ev = [ideas.Evidence("run", r, supports="for", author=ctx.agent) for r in runs]
    status = {"win": "confirmed", "dead": "refuted", "limit": "confirmed"}[args.kind]
    with _conn(ctx) as c:
        if args.kind == "limit":
            missing = ideas.limit_gate(c, ev)
            if missing:
                raise KdevError(ErrorCode.INVALID_REQUEST, "an 'at the limit' note needs evidence: "
                                + "; ".join(missing), details={"field": "evidence"},
                                recovery="kdev note CLAIM --kind limit --evidence SASS_RUN,SWEEP_RUN")
        if args.kind == "dead" and not args.variant.strip():
            raise KdevError(ErrorCode.INVALID_REQUEST, "a 'dead' note needs --variant (exact variant tested)",
                            details={"field": "variant"})
        if args.kind in ("win",) and not runs:
            raise KdevError(ErrorCode.INVALID_REQUEST, "a 'win' note needs --evidence RUN_ID[,...]",
                            details={"field": "evidence"})
        i = ideas.add(c, ideas.Idea(None, args.claim, problem=ctx.problem, kind=args.kind, variant=args.variant,
                                    owner=ctx.agent or None, evidence=ev))
        i = ideas.resolve(c, i.idea_id, status, args.claim, ctx.agent)
    return _idea_result("note", i, "recorded")


def register(reg: Registry) -> None:
    reg.command("ledger query", "query recorded runs", cmd_query,
                args=[Arg(("--file",), "file path (md5 lookup)", type="path", default=None),
                      Arg(("--md5",), "md5", default=None), Arg(("--tool",), "tool", default=None),
                      Arg(("--since",), "e.g. 6h or 2026-10-31", default=None),
                      Arg(("--limit",), "rows", type="int", default=30),
                      Arg(("--sql",), "read-only SELECT", default=None, mcp=False)], package=P)
    reg.command("ledger tried", "has this file / knob set / idea been tried?", cmd_tried,
                args=[Arg(("target",), "file path or free text"),
                      Arg(("--knobs",), "K=V (repeatable)", action="append", default=None)], package=P)
    reg.command("ledger best", "best judged (+ same-hour base, board #1) and best local per problem", cmd_best,
                package=P)
    reg.command("ledger history", "per-shape history: judged and local times over time", cmd_history,
                args=[Arg(("--shape",), "substring of the shape id", default=None),
                      Arg(("--limit",), "rows", type="int", default=200)], package=P)
    reg.command("ledger import", "ingest result.json / run dirs / judge JSONs / leaderboard JSONs / an OKC tree",
                cmd_import,
                args=[Arg(("paths",), "result.json, run dirs, okc status JSONs, board JSONs, okc root",
                          type="path", nargs="+")], effects=W_LEDGER, package=P)
    reg.command("ledger export", "export tables to a file", cmd_export,
                args=[Arg(("--format",), "csv|jsonl", choices=("csv", "jsonl"), default="jsonl"),
                      Arg(("--table",), "table (repeatable)", action="append", default=None),
                      Arg(("--out",), "output file (default $KDEV_HOME/exports/...)", type="path", default=None)],
                effects=Effects(writes=("files",)), package=P)
    reg.command("idea add", "add a hypothesis to the board", cmd_idea_add,
                args=[Arg(("title",), "short title"), Arg(("--hypothesis",), "what and why", default=""),
                      Arg(("--kind",), "idea|dead|limit|win", choices=("idea", "dead", "limit", "win"),
                          default="idea"),
                      Arg(("--gain",), "expected gain %", type="float", default=None),
                      Arg(("--tags",), "comma-separated", default=""),
                      Arg(("--variant",), "exact variant (required for dead)", default="")],
                effects=W_IDEAS, package=P)
    reg.command("idea list", "list ideas", cmd_idea_list,
                args=[Arg(("--status",), "status (comma list)", default=None),
                      Arg(("--owner",), "owner", default=None),
                      Arg(("--kind",), "kind", default=None), Arg(("--limit",), "rows", type="int", default=30)],
                package=P)
    reg.command("idea show", "one idea with evidence", cmd_idea_show,
                args=[Arg(("idea_id",), "id", type="int")], package=P)
    reg.command("idea claim", "claim an idea (owner = --agent)", cmd_idea_claim,
                args=[Arg(("idea_id",), "id", type="int")], effects=W_IDEAS, package=P)
    reg.command("idea release", "release a claimed idea", cmd_idea_release,
                args=[Arg(("idea_id",), "id", type="int")], effects=W_IDEAS, package=P)
    reg.command("idea evidence", "attach evidence (run/judge/file/url/text)", cmd_idea_evidence,
                args=[Arg(("idea_id",), "id", type="int"), Arg(("--run",), "run id", default=None),
                      Arg(("--judge",), "judge job id", default=None),
                      Arg(("--file",), "file", type="path", default=None), Arg(("--url",), "url", default=None),
                      Arg(("--text",), "text", default=None),
                      Arg(("--supports",), "for|against|neutral", choices=("for", "against", "neutral"),
                          default="neutral")],
                effects=W_IDEAS, package=P)
    reg.command("idea resolve", "close an idea (gated for limit/dead)", cmd_idea_resolve,
                args=[Arg(("idea_id",), "id", type="int"),
                      Arg(("--status",), "confirmed|refuted|abandoned",
                          choices=("confirmed", "refuted", "abandoned"), required=True),
                      Arg(("--resolution",), "one line", default="")],
                effects=W_IDEAS, package=P)
    reg.command("note", "record a claim (alias of idea add --kind; limit needs sass+sweep evidence)", cmd_note,
                args=[Arg(("claim",), "the claim"),
                      Arg(("--kind",), "dead|limit|win", choices=("dead", "limit", "win"), default="win"),
                      Arg(("--evidence",), "comma-separated run ids", default=""),
                      Arg(("--variant",), "exact variant", default="")],
                effects=W_IDEAS, package=P)

