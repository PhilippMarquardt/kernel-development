"""CLI: timeline. Owner: P6.

Flow: instrument each file (static, here) into $KDEV_HOME/instrumented/<stem>-<md5>/ (.py: the original text +
a footer that rewrites every load_inline CUDA source at build time, so import-time source patches are seen;
.cu: rewritten in place) -> prebuild the copies and the unmarked originals through ``kdev build`` (CPU, plus the
short shared-lock discovery rounds for prepare-time extensions; never compiles under the exclusive lock;
``--no-build`` skips) -> one 'timeline' GPU job: per shape x seed fresh inputs + graphs, ``--replays`` read-flush
replays interleaved with the unmarked originals (mark overhead), decode to per-mark / per-phase / per-CTA /
per-role tables (+ JSON, Perfetto trace). ``--diff B.py`` (or ``--diff A.py B.py``) times both and prints B - A.
"""
from __future__ import annotations

import hashlib
import json
import os
import shlex
from argparse import Namespace
from pathlib import Path
from typing import Any

from kdev.contract import Effects, ErrorCode, KdevError, Result
from kdev.registry import Arg, Context, Registry

P = "P6"
HINTS_FILE = "kdev_timeline.json"


def _home(ctx: Context) -> Path:
    cfg = getattr(ctx, "config", None)
    h = getattr(cfg, "home", None) or ctx.home or os.environ.get("KDEV_HOME") or "~/.kdev"
    return Path(h).expanduser()


def instrument_into_home(src: str, ctx: Context, *, styles: str | None = "auto", layout: str = "auto"):
    from kdev.timeline.instrument import instrument_file
    data = Path(src).read_bytes()
    key = hashlib.md5(data + f"|{styles}|{layout}".encode()).hexdigest()[:10]  # noqa: S324 - identity
    out = _home(ctx) / "instrumented" / f"{Path(src).stem}-{key}"
    return instrument_file(src, str(out), styles=styles, layout=layout)


def _hints_file(src: str) -> dict[str, Any]:
    """Decode hints next to the file (or one/two directories up): kdev_timeline.json {names, roles, t0}."""
    p = Path(src).resolve().parent
    for d in (p, p.parent, p.parent.parent):
        f = d / HINTS_FILE
        if f.is_file():
            try:
                return dict(json.loads(f.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                return {}
    return {}


def _decode_opts(args: Namespace, inst: Any, src: str) -> dict[str, Any]:
    from kdev.timeline.instrument import parse_kv, parse_roles
    hf = _hints_file(src)
    names: dict[str, str] = dict(hf.get("names") or {})
    names.update(inst.hints.get("names") or {})
    names.update(parse_kv(args.names or ""))
    roles_txt: dict[str, str] = {str(k): v for k, v in (hf.get("roles") or {}).items()}
    roles_txt.update(inst.hints.get("roles") or {})
    roles_txt.update(parse_kv(args.roles or ""))
    roles = parse_roles(roles_txt)
    t0 = args.t0 or inst.hints.get("t0") or hf.get("t0")
    t0 = names.get(t0, t0) if t0 else None
    return {"names": names, "roles": {str(k): v for k, v in roles.items()}, "t0": t0}


def _build(paths: list[str], defines: str, ctx: Context) -> Result | None:
    from kdev.build.commands import cmd_build
    ns = Namespace(files=paths, variant=None, defines=defines, jobs=8, force=False, verbose=False, no_discover=False,
                   clean_pending=False, discover_rounds=6, discover_timeout=None)
    res = cmd_build(ns, ctx)
    return res


def cmd_timeline(args: Namespace, ctx: Context) -> Result:
    from kdev.observe._jobs import file_ref, make_spec, submit
    srcs = ([args.file] if args.file else []) + list(args.diff or [])
    if not srcs:
        raise KdevError(ErrorCode.INVALID_REQUEST, "give a file (kdev timeline FILE) or --diff A.py B.py")
    if args.diff and len(srcs) != 2:
        raise KdevError(ErrorCode.INVALID_REQUEST, f"--diff compares two files, got {len(srcs)}",
                        recovery="kdev timeline A.py --diff B.py  (or --diff A.py B.py)")
    if args.ab and args.diff:
        raise KdevError(ErrorCode.INVALID_REQUEST, "--ab (two define sets of one file) and --diff are exclusive")
    srcs = [os.path.abspath(os.path.expanduser(s)) for s in srcs]
    for s in srcs:
        if not os.path.isfile(s):
            raise KdevError(ErrorCode.NOT_FOUND, f"file not found: {s}")
    ctas = args.ctas or "all"
    if ctas != "all" and not (ctas.startswith("sample:") and ctas[7:].isdigit()):
        raise KdevError(ErrorCode.INVALID_REQUEST, f"--ctas must be all|sample:N, got {ctas!r}")
    insts = [instrument_into_home(s, ctx, styles=args.styles, layout=args.layout) for s in srcs]
    for s, inst in zip(srcs, insts):
        if not inst.marks:
            raise KdevError(ErrorCode.INVALID_REQUEST, f"no marks in {s}",
                            recovery="add `// @kdev:mark NAME` lines inside the kernel, KD_MARK(n) (--marks header), "
                                     "or keep an agent mark macro (DBG(k)/MWT(c, info) storing gtimer()) with calls")
    inst = insts[0]
    if args.marks == "header" and len(inst.marks) == len([n for n in inst.marks.values() if n.startswith("m")]):
        inst.warnings.append("explicit KD_MARK ids without names: add `KD_MARK(3); // NAME` comments")
    user_defs = shlex.split(args.defines or "")
    files = []
    warnings: list[str] = []
    if args.ab:
        sets = [user_defs + shlex.split(s) for s in args.ab]
        files += [file_ref(inst.path, lab, "sub" if i == 0 else "variant", d)
                  for i, (lab, d) in enumerate(zip(["A", "B"], sets))]
        files += [file_ref(srcs[0], f"{lab}.plain", "base", d) for lab, d in zip(["A", "B"], sets)]
    else:
        labels = ["A", "B"][: len(insts)] if len(insts) > 1 else ["marked"]
        files += [file_ref(i.path, lab, "sub" if k == 0 else "variant", user_defs)
                  for k, (lab, i) in enumerate(zip(labels, insts))]
        if not args.no_plain:
            files += [file_ref(s, f"{lab}.plain", "base", user_defs) for lab, s in zip(labels, srcs)]
    for i in insts:
        warnings += [f"{Path(i.source).name}: {w}" for w in i.warnings]
        fams = ", ".join(f"{f['name']}({f['kind']}, ids {f['base']}..{f['base'] + f['range'] - 1})"
                         for f in i.families)
        if fams:
            warnings.append(f"{Path(i.source).name}: agent mark macros rewritten to kdev marks: {fams}; "
                            f"layout {i.layout}")
    builds: list[str] = []
    if not args.no_build:
        groups: dict[str, list[str]] = {}
        for f in files:
            groups.setdefault(" ".join(f.defines), [])
            if f.path not in groups[" ".join(f.defines)]:
                groups[" ".join(f.defines)].append(f.path)
        for defs, paths in groups.items():
            try:
                br = _build(paths, defs, ctx)
            except NotImplementedError:
                warnings.append("prebuild unavailable: run kdev build on the instrumented copy first")
                break
            if br is not None:
                builds.append(br.run_id or "")
                if br.status != "ok" or br.error is not None:
                    br.warnings = list(br.warnings) + [f"timeline: build of {', '.join(paths)} failed; the "
                                                        "instrumented copy is in $KDEV_HOME/instrumented/"]
                    return br
                warnings += [w for w in br.warnings if "pending" in w]
    opts = _decode_opts(args, inst, srcs[0])
    marks_by = {f.label: {str(k): v for k, v in i.marks.items()} for f, i in
                zip([f for f in files if f.role != "base"], insts if not args.ab else [inst, inst])}
    extra = {"marks": {str(k): v for k, v in inst.marks.items()}, "marks_by": marks_by, "shape": args.shape,
             "ctas": ctas, "reps": args.replays or args.reps, "seeds": args.seeds,
             "seed_list": [int(x) for x in str(args.seed_list or "").replace(",", " ").split()], "names": opts["names"],
             "roles": opts["roles"], "t0": opts["t0"], "capacity_ctas": args.capacity_ctas,
             "ab": [" ".join(s) for s in args.ab] if args.ab else None, "copies": [i.path for i in insts],
             "sources": srcs, "layout": inst.layout, "families": inst.families, "builds": builds}
    spec = make_spec("timeline", files, args, ctx, extra)
    res = submit(spec, args, ctx)
    res.warnings = warnings + list(res.warnings)
    return res


def register(reg: Registry) -> None:
    reg.command("timeline", "per-CTA phase timeline from globaltimer marks (min/med/max, CTA spread, slowest CTA, "
                "role waits, mark cost); --diff A B",
                cmd_timeline,
                args=[Arg(("file",), "file with @kdev:mark / KD_MARK / DBG-MWT mark macros", type="path", nargs="?"),
                      Arg(("--diff",), "B.py (or A.py B.py): B - A", type="path", nargs="+", default=None),
                      Arg(("--shape",), "one shape id", default=None),
                      Arg(("--seeds",), "inputs per shape", type="int", default=3),
                      Arg(("--seed-list",), "explicit seeds, e.g. 9132,9169", default=None, mcp=False),
                      Arg(("--replays",), "replays per seed", type="int", default=None),
                      Arg(("--reps",), "alias of --replays", type="int", default=5, mcp=False),
                      Arg(("--names",), "DBG6=gathered,...", default=None),
                      Arg(("--roles",), "0-1=consumer,2=producer", default=None),
                      Arg(("--t0",), "time origin mark", default=None),
                      Arg(("--styles",), "auto|none|DBG,MWT", default="auto", mcp=False),
                      Arg(("--layout",), "mark buffer", choices=("auto", "slots", "ctalog", "log"), default="auto", mcp=False),
                      Arg(("--defines",), "extra -D flags", default="", mcp=False),
                      Arg(("--marks",), "auto|header", choices=("auto", "header"), default="auto", mcp=False),
                      Arg(("--ctas",), "all|sample:N", default="all", mcp=False),
                      Arg(("--ab",), "two define sets", nargs=2, default=None, mcp=False),
                      Arg(("--capacity-ctas",), "buffer CTAs", type="int", default=1024, mcp=False),
                      Arg(("--no-plain",), "no overhead reference", action="store_true", mcp=False),
                      Arg(("--no-build",), "skip the prebuild", action="store_true"),
                      Arg(("--allow-jit",), "allow JIT (opt-out)", action="store_true", mcp=False)],
                effects=Effects(gpu="queued", writes=("run_dir", "ledger", "files", "cache")), gpu=True,
                long_running=True, package=P)
