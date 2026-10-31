"""CLI: time, ab, calib (build JobSpecs and route them through ctx.submit). Owner: P3.

Handlers never touch the GPU or import torch: they validate arguments, build a ``kdev.job/1`` JobSpec and
hand it to ``ctx.submit`` (service queue, or direct flock mode; PLAN 2.2).
"""
from __future__ import annotations

import os
from argparse import Namespace
from typing import Any

from kdev.contract import Effects, ErrorCode, KdevError, Result
from kdev.models import FileRef, JobSpec, TimingParams
from kdev.registry import Arg, Context, Registry

P = "P3"

FIDELITY_ARGS = [
    Arg(("--reps",), "paired repetitions per shape (per seed)", type="int", default=40),
    Arg(("--seeds",), "input seeds per shape (default: adapter variance: 1 or 5)", type="int", default=None),
    Arg(("--flush",), "L2 flush method (non-read = recorded opt-out)", choices=("read", "write", "none"),
        default="read"),
    Arg(("--flush-mb",), "flush buffer MB (default: GPU profile)", type="int", default=None),
    Arg(("--eager",), "time eagerly instead of graph replay (opt-out)", action="store_true"),
    Arg(("--overhead",), "subtraction: judge = 4.0 us const, local = measured 2*one-two graph cost (opt-out)",
        choices=("judge", "local", "none"), default="judge"),
    Arg(("--allow-jit",), "allow compiling inside the GPU lane (opt-out, logged)", action="store_true"),
]

GPU_FX = Effects(gpu="queued", writes=("run_dir", "ledger"))
LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _md5(path: str) -> str:
    from kdev.util import md5_file

    try:
        return md5_file(path)
    except OSError:
        return ""


def _file(path: str, label: str, role: str = "sub", judge_json: str | None = None) -> FileRef:
    p = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(p):
        raise KdevError(ErrorCode.NOT_FOUND, f"file not found: {path}", recovery="check the path")
    return FileRef(path=p, label=label, md5=_md5(p), role=role, judge_json=judge_json)


def _get(args: Namespace, name: str, default: Any = None) -> Any:
    return getattr(args, name, default)


def timing_params(args: Namespace, **over: Any) -> dict[str, Any]:
    """TimingParams dict from the shared fidelity flags (+ overrides)."""
    flush_mb = _get(args, "flush_mb")
    reps = _get(args, "reps", 40)
    if reps is not None and reps < 2:
        raise KdevError(ErrorCode.INVALID_REQUEST, "--reps must be >= 2 (pairs of order-swapped reps)",
                        recovery="use --reps 40 (default)")
    p = TimingParams(reps=reps if reps is not None else 40, seeds=_get(args, "seeds"),
                     shapes=_get(args, "shapes"), order=_get(args, "order", "abba") or "abba",
                     flush=_get(args, "flush", "read") or "read",
                     flush_bytes=(int(flush_mb) << 20) if flush_mb else None,
                     graph=not _get(args, "eager", False), overhead=_get(args, "overhead", "judge") or "judge",
                     allow_jit=bool(_get(args, "allow_jit", False)))
    d = p.to_dict()
    d.update(over)
    return d


def _spec(kind: str, files: list[FileRef], params: dict[str, Any], args: Namespace, ctx: Context) -> JobSpec:
    from kdev.util import new_id, utc_now

    return JobSpec(kind=kind, files=files, params=params, problem=ctx.problem or _get(args, "problem"),
                   job_id=new_id("j"), lane="gpu", agent=ctx.agent or os.environ.get("KDEV_AGENT", ""),
                   priority=int(_get(args, "priority", 0) or 0), timeout_s=int(_get(args, "timeout", 1800) or 1800),
                   run_name=_get(args, "run_name", "") or "", submitted_utc=utc_now(),
                   requires_prebuilt=not bool(_get(args, "allow_jit", False)))


def _submit(spec: JobSpec, args: Namespace, ctx: Context) -> Result:
    return ctx.submit(spec, detach=bool(_get(args, "detach", False)), direct=bool(_get(args, "direct", False)),
                      timeout_s=_get(args, "timeout"))


def build_time_spec(args: Namespace, ctx: Context) -> JobSpec:
    base = _get(args, "base", "baseline") or "baseline"
    if base not in ("baseline", "none"):
        if not os.path.isfile(os.path.expanduser(base)):
            raise KdevError(ErrorCode.NOT_FOUND, f"--base file not found: {base}",
                            recovery="--base baseline|none|PATH")
        base = os.path.abspath(os.path.expanduser(base))
    return _spec("time", [_file(args.file, "A")], timing_params(args, base=base), args, ctx)


def parse_sets(sets: list[str] | None, labels: list[str]) -> dict[str, dict[str, Any]]:
    """--set LABEL:K=V (repeatable) -> {label: {K: V}} (V parsed as int/float/bool when possible)."""
    out: dict[str, dict[str, Any]] = {}
    for s in sets or []:
        lb, sep, kv = s.partition(":")
        k, eq, v = kv.partition("=")
        if not sep or not eq or not k or lb not in labels:
            raise KdevError(ErrorCode.INVALID_REQUEST, f"bad --set {s!r} (want LABEL:K=V, LABEL in {labels})",
                            recovery="e.g. --set B:TILE=128")
        out.setdefault(lb, {})[k] = _parse_value(v)
    return out


def _parse_value(v: str) -> Any:
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    for t in (int, float):
        try:
            return t(v)
        except ValueError:
            pass
    return v


def build_ab_spec(args: Namespace, ctx: Context) -> JobSpec:
    paths = list(args.files or [])
    if len(paths) < 2:
        raise KdevError(ErrorCode.INVALID_REQUEST, "ab needs two or more files", recovery="kdev ab A.py B.py")
    if len(paths) > len(LABELS):
        raise KdevError(ErrorCode.INVALID_REQUEST, f"at most {len(LABELS)} files", recovery="split the batch")
    files = [_file(p, LABELS[i], "sub" if i == 0 else "variant") for i, p in enumerate(paths)]
    knobs = parse_sets(_get(args, "set"), [f.label for f in files])
    for f in files:
        f.knobs = knobs.get(f.label, {})
    return _spec("ab", files, timing_params(args, base="none"), args, ctx)


def parse_pair(s: str) -> tuple[str, str | None]:
    """'FILE.py:JUDGE.json' / 'FILE.py:-' / 'FILE.py' -> (file, judge or None). The separator is the colon
    right after the '.py' suffix, so Windows drive letters survive."""
    i = s.find(".py:")
    if i < 0:
        return s, None
    head, tail = s[:i + 3], s[i + 4:]
    return head, (None if tail in ("-", "") else tail)


def build_calib_spec(args: Namespace, ctx: Context) -> JobSpec:
    pairs = [parse_pair(s) for s in (args.pairs or [])]
    if not pairs:
        raise KdevError(ErrorCode.INVALID_REQUEST, "no FILE:JUDGE.json pairs", recovery="kdev calib A.py:A.json")
    files = []
    for i, (p, j) in enumerate(pairs):
        if j is not None and not os.path.isfile(os.path.expanduser(j)):
            raise KdevError(ErrorCode.NOT_FOUND, f"judge JSON not found: {j}", recovery="okc status JOB --json > J")
        files.append(_file(p, LABELS[i], "sub", os.path.abspath(os.path.expanduser(j)) if j else None))
    if not any(f.judge_json for f in files):
        raise KdevError(ErrorCode.INVALID_REQUEST, "calib needs at least one judge JSON (the shape list)",
                        recovery="kdev calib FILE.py:JUDGE.json")
    return _spec("calib", files, timing_params(args, base="baseline"), args, ctx)


def cmd_time(args: Namespace, ctx: Context) -> Result:
    return _submit(build_time_spec(args, ctx), args, ctx)


def cmd_ab(args: Namespace, ctx: Context) -> Result:
    return _submit(build_ab_spec(args, ctx), args, ctx)


def cmd_calib(args: Namespace, ctx: Context) -> Result:
    return _submit(build_calib_spec(args, ctx), args, ctx)


def register(reg: Registry) -> None:
    reg.command("time", "judge-faithful timing of one file vs the baseline", cmd_time,
                args=[Arg(("file",), "submission file", type="path"),
                      Arg(("--base",), "baseline|none|PATH", default="baseline")] + FIDELITY_ARGS,
                effects=GPU_FX, gpu=True, long_running=True, package=P)
    reg.command("ab", "interleaved A/B(/C...) timing with paired-ratio CIs and verdicts", cmd_ab,
                args=[Arg(("files",), "two or more files", type="path", nargs="+"),
                      Arg(("--order",), "interleave scheme", choices=("abba", "rotate"), default="abba"),
                      Arg(("--set",), "LABEL:K=V knob override (repeatable)", action="append", default=None)]
                + FIDELITY_ARGS,
                effects=GPU_FX, gpu=True, long_running=True, package=P)
    reg.command("calib", "judge-vs-local per-shape calibration (FILE:JUDGE.json ...)", cmd_calib,
                args=[Arg(("pairs",), "FILE:JUDGE.json (JUDGE '-' = local only)", nargs="+")] + FIDELITY_ARGS,
                effects=GPU_FX, gpu=True, long_running=True, package=P)
