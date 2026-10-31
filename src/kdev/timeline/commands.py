"""CLI: timeline. Owner: P6.

Flow: instrument the file (static, here) into $KDEV_HOME/instrumented/<stem>-<md5>/ -> prebuild the copy
compile-only through kdev.build (never under the GPU lock; skipped with --no-build) -> submit a 'timeline'
GPU job with the instrumented copy (one per --ab define set) plus the unmarked original (mark overhead).
"""
from __future__ import annotations

import hashlib
import os
import shlex
from argparse import Namespace
from pathlib import Path

from kdev.contract import Effects, ErrorCode, KdevError, Result
from kdev.registry import Arg, Context, Registry

P = "P6"


def _home(ctx: Context) -> Path:
    cfg = getattr(ctx, "config", None)
    h = getattr(cfg, "home", None) or ctx.home or os.environ.get("KDEV_HOME") or "~/.kdev"
    return Path(h).expanduser()


def instrument_into_home(src: str, ctx: Context):  # -> Instrumented
    from kdev.timeline.instrument import instrument_file
    data = Path(src).read_bytes()
    key = hashlib.md5(data).hexdigest()[:10]  # noqa: S324 - identity
    out = _home(ctx) / "instrumented" / f"{Path(src).stem}-{key}"
    return instrument_file(src, str(out))


def cmd_timeline(args: Namespace, ctx: Context) -> Result:
    from kdev.observe._jobs import file_ref, make_spec, submit
    src = os.path.abspath(os.path.expanduser(args.file))
    if not os.path.isfile(src):
        raise KdevError(ErrorCode.NOT_FOUND, f"file not found: {args.file}")
    ctas = args.ctas or "all"
    if ctas != "all" and not (ctas.startswith("sample:") and ctas[7:].isdigit()):
        raise KdevError(ErrorCode.INVALID_REQUEST, f"--ctas must be all|sample:N, got {ctas!r}")
    inst = instrument_into_home(src, ctx)
    if not inst.marks:
        raise KdevError(ErrorCode.INVALID_REQUEST, f"no marks in {args.file}",
                        recovery="add `// @kdev:mark NAME` lines inside the kernel (or KD_MARK(n) with --marks header)")
    if args.marks == "header" and len(inst.marks) == len([n for n in inst.marks.values() if n.startswith("m")]):
        inst.warnings.append("explicit KD_MARK ids without names: add `KD_MARK(3); // NAME` comments")
    sets = [shlex.split(s) for s in args.ab] if args.ab else [[]]
    labels = ["A", "B"] if args.ab else ["marked"]
    files = [file_ref(inst.path, lab, "sub" if i == 0 else "variant", ["-DKDEV_MARKS"] + s)
             for i, (lab, s) in enumerate(zip(labels, sets))]
    files.append(file_ref(src, "plain", "base", sets[0]))
    warnings = list(inst.warnings)
    if not args.no_build:
        try:
            from kdev.build.compile import build_file
            for f in files:                   # marked copies + the plain original (overhead reference)
                rep = build_file(f.path, defines=f.defines, problem=ctx.problem)
                if rep.errors:
                    raise KdevError(ErrorCode.BUILD_FAILED, f"build of {f.label} failed: {rep.errors[0][:300]}",
                                    details={"copy": inst.path, "file": f.path},
                                    recovery=f"kdev build {f.path} --defines \"{' '.join(f.defines)}\" --verbose")
        except NotImplementedError:
            warnings.append(f"prebuild unavailable: build {inst.path} first (kdev build ... --defines -DKDEV_MARKS)")
    extra = {"marks": {str(k): v for k, v in inst.marks.items()}, "shape": args.shape, "ctas": ctas,
             "reps": args.reps, "ab": [" ".join(s) for s in sets] if args.ab else None, "copy": inst.path,
             "source": src}
    spec = make_spec("timeline", files, args, ctx, extra)
    res = submit(spec, args, ctx)
    res.warnings = warnings + list(res.warnings)
    return res


def register(reg: Registry) -> None:
    reg.command("timeline", "per-CTA phase timeline from %globaltimer marks (+ Perfetto trace)",
                cmd_timeline,
                args=[Arg(("file",), "submission file with // @kdev:mark annotations or KD_MARK", type="path"),
                      Arg(("--shape",), "shape id (default: first scored shape)", default=None),
                      Arg(("--marks",), "auto (annotations) | header (explicit KD_MARK)",
                          choices=("auto", "header"), default="auto"),
                      Arg(("--ctas",), "all | sample:N", default="all"),
                      Arg(("--ab",), "two define sets 'A_DEFS' 'B_DEFS'", nargs=2, default=None),
                      Arg(("--reps",), "timed replays per variant", type="int", default=15),
                      Arg(("--no-build",), "skip the compile-only prebuild of the instrumented copy",
                          action="store_true"),
                      Arg(("--allow-jit",), "allow compiling inside the GPU lane (opt-out, logged)",
                          action="store_true")],
                effects=Effects(gpu="queued", writes=("run_dir", "ledger", "files")), gpu=True, long_running=True,
                package=P)
