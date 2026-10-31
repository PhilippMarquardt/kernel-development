"""CLI: check, sanitize, comply. Owner: P7.

Each GPU command builds a JobSpec (kind check|sanitize|comply) and routes it through ``ctx.submit`` (service or
direct flock mode, PLAN 2.2); the worker runs ``checks.check.run_check`` / ``checks.sanitize.run_sanitize_inner`` /
``checks.comply.run_comply``. ``kdev comply --static-only`` runs in the CLI process (no GPU, no torch).
"""
from __future__ import annotations

import os
from argparse import Namespace
from typing import Any

from kdev.contract import Effects, ErrorCode, KdevError, Result
from kdev.models import FileRef, JobSpec
from kdev.registry import Arg, Context, Registry
from kdev.util import md5_file, utc_now

P = "P7"
FX = Effects(gpu="queued", writes=("run_dir", "ledger"))


def _file_ref(path: str) -> FileRef:
    p = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(p):
        raise KdevError(ErrorCode.NOT_FOUND, f"file not found: {path}")
    return FileRef(path=p, label="A", md5=md5_file(p), role="sub")


def _problem(args: Namespace, ctx: Context) -> str | None:
    return getattr(args, "problem", None) or ctx.problem


def build_spec(kind: str, args: Namespace, ctx: Context, params: dict[str, Any]) -> JobSpec:
    """JobSpec for a P7 GPU command (pure; unit-tested)."""
    return JobSpec(kind=kind, files=[_file_ref(args.file)], params=params, problem=_problem(args, ctx),
                   adapter=None, agent=ctx.agent or os.environ.get("KDEV_AGENT", ""),
                   priority=int(getattr(args, "priority", 0) or 0), timeout_s=int(getattr(args, "timeout", 1800) or 1800),
                   run_name=getattr(args, "run_name", "") or "", submitted_utc=utc_now(), requires_prebuilt=True)


def _submit(ctx: Context, spec: JobSpec, args: Namespace) -> Result:
    return ctx.submit(spec, detach=bool(getattr(args, "detach", False)), direct=bool(getattr(args, "direct", False)),
                      timeout_s=float(getattr(args, "timeout", 0) or 0) or None)


def check_params(args: Namespace) -> dict[str, Any]:
    return {"seeds": args.seeds, "repeat": args.repeat, "mode": args.mode, "shapes": getattr(args, "shapes", None),
            "extra": {"extra_shapes": bool(args.extra_shapes), "tests_only": bool(args.tests_only),
                      "cold": bool(args.cold)}}


def cmd_check(args: Namespace, ctx: Context) -> Result:
    return _submit(ctx, build_spec("check", args, ctx, check_params(args)), args)


def cmd_sanitize(args: Namespace, ctx: Context) -> Result:
    params = {"mode": "both", "extra": {"tool": args.tool, "shape": args.shape}}
    spec = build_spec("sanitize", args, ctx, params)
    spec.timeout_s = max(spec.timeout_s, 3600)   # everything runs ~10-100x slower under the sanitizer
    return _submit(ctx, spec, args)


def cmd_comply(args: Namespace, ctx: Context) -> Result:
    if args.static_only:
        from kdev.adapters.registry import get_adapter
        from kdev.checks.comply import make_result, run_static, unverified_precision_items
        ref = _file_ref(args.file)
        adapter = get_adapter(problem=_problem(args, ctx), file=ref.path)
        items = run_static(adapter, ref.path) + unverified_precision_items(adapter.rules())
        return make_result(items, problem=adapter.problem, file=ref.path, mode="static")
    params = {"shapes": getattr(args, "shapes", None), "extra": {"static_only": False,
                                                                 "no_baseline": bool(args.no_baseline)}}
    return _submit(ctx, build_spec("comply", args, ctx, params), args)


def register(reg: Registry) -> None:
    reg.command("check", "correctness: fresh seeds, extra shapes, eager+graph, repeats; classifies failures",
                cmd_check,
                args=[Arg(("file",), "submission", type="path"),
                      Arg(("--seeds",), "fresh seeds per case", type="int", default=3),
                      Arg(("--repeat",), "repeats per case (race detection)", type="int", default=3),
                      Arg(("--mode",), "graph|eager|both", choices=("graph", "eager", "both"), default="both"),
                      Arg(("--extra-shapes",), "add held-out shapes", action="store_true"),
                      Arg(("--tests-only",), "only the adapter's tests", action="store_true"),
                      Arg(("--cold",), "measure prepare+compile from an empty cache vs budget",
                          action="store_true")],
                effects=FX, gpu=True, long_running=True, package=P)
    reg.command("sanitize", "compute-sanitizer on one small shape (eager + graph)",
                cmd_sanitize,
                args=[Arg(("file",), "submission", type="path"),
                      Arg(("--tool",), "sanitizer tool", choices=("memcheck", "racecheck", "synccheck",
                                                                  "initcheck"), default="memcheck"),
                      Arg(("--shape",), "shape id (default: smallest test)", default=None)],
                effects=FX, gpu=True, long_running=True, package=P)
    reg.command("comply", "rules checklist: precision per op, cross-call state, work outside timed call",
                cmd_comply,
                args=[Arg(("file",), "submission", type="path"),
                      Arg(("--static-only",), "no GPU part", action="store_true"),
                      Arg(("--no-baseline",), "skip the precision-vs-SGLang-baseline probe", action="store_true")],
                effects=FX, gpu=True, long_running=True, package=P)
