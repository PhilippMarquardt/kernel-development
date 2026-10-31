"""CLI: build. Owner: P5."""
from __future__ import annotations

from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from kdev.build import _runs
from kdev.contract import Effects, ErrorCode, KdevError, Result
from kdev.registry import Arg, Context, Registry

P = "P5"


def coerce(v: str) -> Any:
    """'128' -> 128, '0.5' -> 0.5, 'true' -> True, else the string."""
    s = v.strip()
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    for t in (int, float):
        try:
            return t(s)
        except ValueError:
            pass
    return s


def parse_variants(items: list[str] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for it in items or []:
        for part in it.split():
            k, sep, v = part.partition("=")
            if not sep or not k:
                raise KdevError(ErrorCode.INVALID_REQUEST, f"--variant expects K=V, got {part!r}")
            out[k] = coerce(v)
    return out


def cmd_build(args: Namespace, ctx: Context) -> Result:
    from kdev.build.compile import build_file, build_result, report_rows  # noqa: PLC0415

    knobs = parse_variants(args.variant)
    defines = (args.defines or "").split()
    files = list(args.files)
    home = str(_runs.home(ctx))

    def one(f: str):
        return build_file(f, knobs=knobs, defines=defines, problem=ctx.problem, force=args.force,
                          verbose=args.verbose, home=home)

    with ThreadPoolExecutor(max_workers=max(1, min(args.jobs, len(files)))) as ex:
        reports = list(ex.map(one, files))
    rounds: list[dict[str, Any]] = []
    disc_warn: list[str] = []
    if not getattr(args, "no_discover", False):
        for i, f in enumerate(files):
            if not f.endswith(".py") or not reports[i].ok:
                continue
            rep, rr, w = _discover_loop(f, reports[i], one, knobs, defines, ctx,
                                        max_rounds=int(getattr(args, "discover_rounds", 6) or 6))
            reports[i] = rep
            rounds += rr
            disc_warn += w
    res = build_result(reports, verbose=args.verbose)
    if rounds:
        res.summary["discover_rounds"] = rounds
        last = {}
        for r in rounds:
            last[r["file"]] = r
        for fpath, r in last.items():
            res.lines.append(f"discover {Path(fpath).name}: {r['round']} round(s), last pass "
                             f"{r['verdict']} ({r['pending']} pending), run {r.get('run_id') or '-'}")
            if r["verdict"] != "complete":
                res.warnings.append(f"{Path(fpath).name}: prepare-time extensions still pending after "
                                    f"{r['round']} discovery round(s); GPU jobs will report needs_build")
    res.warnings += disc_warn
    run = _runs.new_run(ctx, "build", Path(files[0]).stem)
    rows = report_rows(reports)
    p = _runs.write_json(run, "build.json", {"schema": "kdev.build/1", **rows, "knobs": knobs, "defines": defines,
                                             "reports": [r.to_dict() for r in reports]})
    res.files["build"] = str(p)
    res.summary["knobs"] = knobs
    return _runs.finish(run, res)


def _discover_loop(path: str, report: Any, rebuild: Any, knobs: dict[str, Any], defines: list[str], ctx: Context,
                   *, max_rounds: int = 6) -> tuple[Any, list[dict[str, Any]], list[str]]:
    """Build -> GPU discovery job (prepare + one call per shape, hook in discover mode: nothing compiles under
    the lock) -> rebuild the recorded calls, until a discovery pass finds nothing pending. Returns
    (final report, rounds, warnings). Skipped (with a warning) when GPU jobs are unavailable."""
    import os

    from kdev.models import FileRef, JobSpec
    from kdev.util import md5_file, new_id, utc_now

    rounds: list[dict[str, Any]] = []
    for rnd in range(1, max_rounds + 1):
        spec = JobSpec(kind="discover", lane="gpu", files=[FileRef(path=os.path.abspath(path), label="A",
                                                                   md5=md5_file(path), knobs=dict(knobs),
                                                                   defines=list(defines))],
                       params={"extra": {"budget_s": 600}}, problem=ctx.problem, job_id=new_id("j"),
                       agent=ctx.agent or os.environ.get("KDEV_AGENT", ""), timeout_s=900,
                       run_name=f"discover-{Path(path).stem}", submitted_utc=utc_now(), requires_prebuilt=False)
        try:
            r = ctx.submit(spec)
        except KdevError as e:
            return report, rounds, [f"discovery skipped ({e.code.value}): prepare-time extensions are built when a "
                                    "GPU job records them (needs_build -> kdev build again)"]
        if r.status == "error" or r.error is not None:
            code = r.error.code if r.error else "error"
            msg = r.error.message if r.error else ""
            return report, rounds, [f"discovery round {rnd} failed ({code}): {msg[:160]}"]
        n = int((r.summary or {}).get("pending", 0))
        rounds.append({"file": path, "round": rnd, "verdict": r.verdict, "pending": n, "run_id": r.run_id,
                       "names": (r.summary or {}).get("pending_names", [])[:10]})
        if n == 0:
            break
        report = rebuild(path)              # compiles the recorded calls (CPU only, no lock)
        if not report.ok:
            break
    return report, rounds, []


def register(reg: Registry) -> None:
    reg.command("build", "compile-only build with the pinned toolchain; ptxas regs/spills/C751x",
                cmd_build,
                args=[Arg(("files",), "files to build (.py with load_inline/load, or .cu)", type="path", nargs="+"),
                      Arg(("--variant",), "K=V knob, compiled in as -DK=V (repeatable)", action="append",
                          default=None),
                      Arg(("--defines",), "extra -D flags", default=""),
                      Arg(("--jobs",), "parallel builds", type="int", default=4),
                      Arg(("--force",), "ignore cache", action="store_true"),
                      Arg(("--verbose",), "full ptxas output in summary", action="store_true"),
                      Arg(("--no-discover",), "skip the GPU discovery rounds for prepare-time extensions (short "
                          "GPU jobs under the lock; nothing compiles there)", action="store_true"),
                      Arg(("--discover-rounds",), "max build+discover rounds", type="int", default=6)],
                effects=Effects(gpu="none", writes=("cache", "run_dir", "ledger")), long_running=True,
                package=P)

