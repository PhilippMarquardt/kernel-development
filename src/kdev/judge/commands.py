"""CLI: judge prepare|record|ledger|pair|import, okc status|leaderboard (read-only). Owner: P4."""
from __future__ import annotations

import json
import os
from argparse import Namespace
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from kdev.contract import Effects, ErrorCode, KdevError, Result
from kdev.registry import Arg, Context, Registry

P = "P4"


@contextmanager
def _conn(ctx: Context) -> Iterator[Any]:
    from kdev.ledger.db import open_default
    c = open_default(ctx.home)
    try:
        yield c
    finally:
        c.close()


def _f(x: Any) -> str:
    return "-" if x is None else f"{x:.4f}"


def cmd_prepare(args: Namespace, ctx: Context) -> Result:
    from kdev.judge.prepare import prepare, to_result
    with _conn(ctx) as c:
        return to_result(prepare(args.file, ctx.problem, conn=c, copy=args.copy))


def cmd_record(args: Namespace, ctx: Context) -> Result:
    from kdev.judge._parse import parse_judge
    from kdev.judge.record import record_parsed
    p = Path(args.job_json)
    if not p.is_file():
        raise KdevError(ErrorCode.NOT_FOUND, f"no such file: {p}", details={"path": str(p)})
    try:
        d = json.loads(p.read_text(encoding="utf-8", errors="replace"))
    except ValueError as e:
        raise KdevError(ErrorCode.INVALID_REQUEST, f"{p}: not JSON ({e})") from e
    jr = parse_judge(d)
    if args.file and not os.path.isfile(args.file):
        raise KdevError(ErrorCode.NOT_FOUND, f"no such file: {args.file}", details={"path": args.file})
    label = args.label or (os.path.basename(args.file) if args.file else None)
    with _conn(ctx) as c:
        jid = record_parsed(c, jr, file_path=args.file, raw_path=str(p.resolve()), label=label, role=args.role)
    lines = [f"recorded {jid} {jr.problem} {jr.mode} {jr.status} x{_f(jr.speedup)} "
             f"ci={jr.speedup_ci} cluster={jr.cluster} nvcc={jr.toolchain or '?'} shapes={len(jr.shapes)}"]
    warnings = []
    if not args.file:
        warnings.append("no --file: the exact submitted file is not linked (LESSONS 6: record the variant)")
    if jr.duplicate_of:
        warnings.append(f"judge answered duplicate of {jr.duplicate_of}")
    return Result(tool="judge record", verdict=jr.status,
                  summary={"judge_id": jid, "problem": jr.problem, "speedup": jr.speedup, "status": jr.status,
                           "cluster": jr.cluster, "toolchain": jr.toolchain, "n_shapes": len(jr.shapes)},
                  lines=lines, warnings=warnings)


def cmd_ledger(args: Namespace, ctx: Context) -> Result:
    from kdev.judge.record import ledger
    with _conn(ctx) as c:
        rows = ledger(c, ctx.problem)
    lines = []
    for r in rows:
        b = r["baseline"]
        lines.append(f"{r['problem']:24s} best x{_f(r['speedup'])} {r['judge_id']} {(r['label'] or '')[:28]:28s} "
                     f"{r['cluster'] or ''} n={r['n_judged']}")
        lines.append("    same-hour base " + (f"x{_f(b['speedup'])} {b['judge_id']} ({(b.get('label') or '')[:24]}) "
                                              f"-> x{_f(r['vs_baseline'])}"
                                              + (f" CLUSTER {b.get('cluster')}!={r['cluster']} (1-2% offset)"
                                                 if r.get("cluster_mismatch") else "") if b else "NONE"))
        if r.get("board_top"):
            t = r["board_top"]
            lines.append(f"    board #1 {t['team']} x{_f(t['speedup'])} ({t['taken_utc'][:10]})")
    return Result(tool="judge ledger", verdict="ok" if rows else "empty", summary={"problems": rows},
                  lines=lines or ["no judged runs; `kdev ledger import OKC_ROOT` or `kdev judge record`"])


def cmd_pair(args: Namespace, ctx: Context) -> Result:
    from kdev.judge.pair import run_pair
    if args.allow_check and ctx.via != "cli":
        raise KdevError(ErrorCode.QUOTA_FORBIDDEN, "`judge pair --allow-check` is CLI-only",
                        recovery="a human/orchestrator runs it from a shell")
    with _conn(ctx) as c:
        return run_pair(args.candidate, args.base, ctx.problem, allow_check=bool(args.allow_check),
                        rounds=args.rounds, conn=c)


def cmd_import(args: Namespace, ctx: Context) -> Result:
    from kdev.judge._importer import import_okc_tree
    root = args.okc_root
    if not root:
        try:
            from kdev.config import load_config
            root = getattr(load_config(home=ctx.home), "okc_root", "") or ""
        except Exception:  # noqa: BLE001
            root = ""
    if not root or not os.path.isdir(root):
        raise KdevError(ErrorCode.INVALID_REQUEST, f"OKC root not found: {root!r}", details={"field": "okc_root"},
                        recovery="kdev judge import /path/to/okc")
    with _conn(ctx) as c:
        s = import_okc_tree(c, root, match_files=not args.no_match)
    lines = [f"{s['judge_files']} judge JSONs ({s['unique_judge_ids']} unique jobs), {s['pairs']} same-hour "
             f"pairs, {s['board_rows']} board rows from {len(s['board_files'])} files, "
             f"{s['files_matched_by_hash']} submitted files matched by hash"]
    lines += [f"  {k} -> {v}" for k, v in list(s["recorded"].items())[:20]]
    return Result(tool="judge import", verdict="ok", summary={k: v for k, v in s.items()}, lines=lines)


def cmd_okc_status(args: Namespace, ctx: Context) -> Result:
    from kdev.judge import okc_cli
    d = okc_cli.status(args.job_id)
    lines = [f"{d.get('id', args.job_id)} {d.get('board', '')} {d.get('status', '')} "
             f"x{_f((d.get('result') or {}).get('speedup'))} {d.get('cluster', '')}"]
    files: dict[str, str] = {}
    if args.save or args.record:
        out = Path(args.save or f"chk_{args.job_id}.json")
        out.write_text(json.dumps(d, indent=1), encoding="utf-8")
        files["json"] = str(out.resolve())
        lines.append(f"saved {out}")
    if args.record:
        from kdev.judge.record import record
        with _conn(ctx) as c:
            record(c, d, raw_path=files["json"])
        lines.append("recorded in the ledger")
    small = {k: v for k, v in d.items() if k != "result"}
    small["speedup"] = (d.get("result") or {}).get("speedup")
    return Result(tool="okc status", verdict=str(d.get("status") or "ok"), summary=small, lines=lines, files=files)


def cmd_okc_leaderboard(args: Namespace, ctx: Context) -> Result:
    from kdev.judge import okc_cli
    from kdev.judge._importer import board_rows_from, write_board_rows
    from kdev.util import utc_now
    d = okc_cli.leaderboard(ctx.problem)
    rows = board_rows_from(d, utc_now(), "okc leaderboard")
    lines = [f"{(r['rank'] or 0):>4} {r['team'][:24]:24s} x{_f(r['speedup'])} {r.get('screening') or ''}"
             for r in rows[:15]]
    if len(rows) > 15:
        lines.append(f"... {len(rows) - 15} more rows (--json)")
    if args.record:
        with _conn(ctx) as c:
            n = write_board_rows(c, rows)
        lines.append(f"recorded {n} board rows")
    return Result(tool="okc leaderboard", verdict="ok", summary={"rows": rows[:100], "n": len(rows)},
                  lines=lines or [json.dumps(d)[:300]])


def register(reg: Registry) -> None:
    reg.command("judge prepare", "validate a file for judging, warn on duplicates, print the command",
                cmd_prepare, args=[Arg(("file",), "file", type="path"),
                                   Arg(("--copy",), "write + validate a comment-changed copy (new judge hash)",
                                       action="store_true")],
                effects=Effects(writes=("files",)), package=P)
    reg.command("judge record", "ingest an `okc status --json` result into the ledger", cmd_record,
                args=[Arg(("job_json",), "okc status/check JSON", type="path"),
                      Arg(("--file",), "the exact submitted file (for md5)", type="path", default=None),
                      Arg(("--label",), "label (default: file name)", default=None),
                      Arg(("--role",), "base|cand|other", choices=("base", "cand", "other"), default=None)],
                effects=Effects(writes=("ledger",)), package=P)
    reg.command("judge ledger", "judged bests and same-hour baselines", cmd_ledger, package=P)
    reg.command("judge pair", "same-hour judge pair candidate+base (prints commands unless --allow-check)",
                cmd_pair,
                args=[Arg(("candidate",), "candidate file", type="path"),
                      Arg(("--base",), "unmodified base file", type="path", default=None),
                      Arg(("--rounds",), "1 = base,cand; 2 = ABBA (default; 4 checks)", type="int", default=2),
                      Arg(("--allow-check",), "ACTUALLY run `okc check` (2 per round; consumes quota)",
                          action="store_true", mcp=False)],
                effects=Effects(network="okc_check", quota="explicit_flag", writes=("ledger",)), mcp=False,
                package=P)
    reg.command("judge import", "import saved judge/leaderboard JSONs of an OKC checkout into the ledger",
                cmd_import,
                args=[Arg(("okc_root",), "OKC checkout (default: config okc_root)", type="path", nargs="?",
                          default=None),
                      Arg(("--no-match",), "skip matching submitted files by hash", action="store_true")],
                effects=Effects(writes=("ledger",)), package=P)
    reg.command("okc status", "read-only `okc status JOB --json`", cmd_okc_status,
                args=[Arg(("job_id",), "judge job id"),
                      Arg(("--save",), "write the JSON to this file", type="path", default=None),
                      Arg(("--record",), "also record it in the ledger", action="store_true")],
                effects=Effects(network="okc_readonly", writes=("ledger",)), package=P)
    reg.command("okc leaderboard", "read-only public board (--problem P)", cmd_okc_leaderboard,
                args=[Arg(("--record",), "store a board snapshot in the ledger", action="store_true")],
                effects=Effects(network="okc_readonly", writes=("ledger",)), package=P)
