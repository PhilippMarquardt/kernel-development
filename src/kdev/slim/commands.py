"""CLI: kdev slim STACK.py OUT.py --trace RUN|dispatch.json. CPU only (never GPU, never okc)."""
from __future__ import annotations

import os
from argparse import Namespace

from kdev.contract import Effects, ErrorCode, KdevError, Result
from kdev.registry import Arg, Context, Registry

P = "KDEV"


def _is_okc(path: str, problem: str | None) -> bool:
    try:
        from kdev.timing.commands import is_okc

        return is_okc([path], problem)
    except Exception:  # noqa: BLE001
        return "/okc/" in path.replace("\\", "/")


def cmd_slim(args: Namespace, ctx: Context) -> Result:
    from kdev.slim import core

    src_p = os.path.abspath(os.path.expanduser(args.stack))
    out_p = os.path.abspath(os.path.expanduser(args.out))
    if not os.path.isfile(src_p):
        raise KdevError(ErrorCode.NOT_FOUND, f"no such file {src_p}")
    if os.path.normcase(src_p) == os.path.normcase(out_p):
        raise KdevError(ErrorCode.INVALID_REQUEST, "OUT must differ from STACK (slim never edits in place)",
                        recovery="kdev slim STACK.py STACK_slim.py --trace RUN")
    if os.path.exists(out_p) and not args.force:
        raise KdevError(ErrorCode.INVALID_REQUEST, f"{out_p} exists", recovery="pick another OUT or pass --force")
    if not args.trace:
        raise KdevError(ErrorCode.INVALID_REQUEST, "slim needs a dispatch trace of STACK (which kernels launch on "
                        "the scored shapes) to know which routes are dead",
                        recovery=f"kdev dispatch {args.stack}   then   kdev slim {args.stack} {args.out} --trace RUN_ID")
    with open(src_p, "rb") as f:
        raw = f.read()
    try:
        trace, tpath, tmd5 = core.load_trace(args.trace, ctx.cfg)
    except core.SlimRefused as e:
        raise KdevError(ErrorCode.NOT_FOUND, str(e), recovery=f"kdev dispatch {args.stack}") from e
    in_md5 = core.md5s(raw)
    if tmd5 and tmd5 != in_md5 and not args.any_trace:
        raise KdevError(ErrorCode.INVALID_REQUEST, f"the trace is of another file version (md5 {tmd5[:8]} != "
                        f"{in_md5[:8]})", recovery=f"kdev dispatch {args.stack} (or --any-trace if only comments changed)")
    okc = _is_okc(src_p, ctx.problem)
    limit = args.max_bytes if args.max_bytes is not None else (core.OKC_LIMIT if okc else None)
    hook = core.load_hook(os.path.abspath(args.hook)) if args.hook else None
    try:
        text, rep = core.slim(raw.decode("utf-8"), trace, drop=not args.no_drop,
                              force=tuple(args.drop or ()), keep=tuple(args.keep or ()), hook=hook,
                              limit=limit or None, allow_grow=args.allow_grow,
                              all_dead=bool(getattr(args, "all_dead", False)))
    except core.SlimRefused as e:
        return Result(tool="slim", status="fail", verdict="refused",
                      summary={"stack": src_p, "trace": tpath, "reason": str(e)},
                      lines=[f"REFUSED  slim {os.path.basename(src_p)}: {e}", "nothing written"])
    data = text.encode("utf-8")
    tmp = out_p + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, out_p)
    out_md5 = core.md5s(data)
    lines = [f"OK  slim {os.path.basename(src_p)} -> {os.path.basename(out_p)}",
             f"in  {len(raw):>9,} B md5 {in_md5}",
             f"out {len(data):>9,} B md5 {out_md5}  ({(len(data) - len(raw)) / max(1, len(raw)):+.1%})"
             + (f"  limit {limit:,}: {'UNDER' if len(data) < limit else 'OVER'}" if limit else ""),
             f"decoded {rep.decoded} blob(s); {len(rep.dropped)} dead route(s) dropped, {rep.live_builders} live, "
             f"{len(rep.kept_dead)} dead but kept (unsafe)"]
    for d in rep.dropped[:8]:
        lines.append(f"  dropped {d['where']}:{d['name']} ({d.get('bytes', 0):,} B) kernels {', '.join(d['kernels'][:3])}")
    for d in rep.kept_dead[:8]:
        lines.append(f"  kept    {d['where']}:{d['name']}: {d['reason']}")
    if rep.leftovers:
        lines.append("leftovers: " + ", ".join(f"{k} x{v}" for k, v in rep.leftovers.items()))
    return Result(tool="slim", status="ok", verdict="ok",
                  summary={"stack": src_p, "out": out_p, "trace": tpath, "bytes_in": len(raw), "bytes_out": len(data),
                           "md5_in": in_md5, "md5_out": out_md5, "limit": limit, **rep.to_dict()},
                  files={"out": out_p}, lines=lines, warnings=rep.warnings[:10],
                  next=[f"kdev build {args.out}", f"kdev check {args.out}", f"kdev ab {args.stack} {args.out}"])


def register(reg: Registry) -> None:
    reg.command("slim", "plain-source copy of a stack file: decode zlib/base64 blobs, drop routes that never launch "
                "in a dispatch trace (safe cases only), size/md5 report, OKC 1,000,000-byte rule; CPU only", cmd_slim,
                args=[Arg(("stack",), "submission stack file", type="path"),
                      Arg(("out",), "output file (never the input)", type="path"),
                      Arg(("--trace",), "dispatch trace of STACK: run id / run dir / dispatch.json", default=None),
                      Arg(("--drop",), "force-drop this dead builder even if referenced outside try (repeatable)",
                          action="append", default=None),
                      Arg(("--all-dead",), "drop every dead builder not dispatched by name, even if called outside "
                          "try (a prepare() that calls one unconditionally then raises: kdev check OUT)",
                          action="store_true"),
                      Arg(("--keep",), "never drop this builder (repeatable)", action="append", default=None),
                      Arg(("--no-drop",), "decode blobs only", action="store_true"),
                      Arg(("--hook",), "per-problem rewrite module with slim(src, ctx) -> src", type="path",
                          default=None),
                      Arg(("--max-bytes",), "size limit (default 1,000,000 on OKC problems)", type="int", default=None),
                      Arg(("--allow-grow",), "write even if the output is larger than the input", action="store_true"),
                      Arg(("--any-trace",), "accept a trace made for another md5 of STACK", action="store_true"),
                      Arg(("--force",), "overwrite OUT", action="store_true")],
                effects=Effects(gpu="none", writes=("files",)), package=P)
