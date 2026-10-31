"""Discovery job (worker kind ``discover``, GPU lane, P5): find the extensions a submission builds at prepare
time or on its first call, without compiling anything under the GPU lock.

Real OKC submissions build most extensions inside ``prepare()`` (self-tests and route choices on GPU
results, weight-dependent instance lists) or lazily on the first ``kernel_fn``. The CPU-only build lane
cannot reproduce that, so ``kdev build`` runs this short job: the load_inline hook is installed in
*discover* mode, so a call already in the kdev cache imports the prebuilt library, and a missing call is
written to ``<ext_root>/pending/<key>.json`` and gets a stub module back. Then ``load -> prepare ->
kernel_fn`` runs once per shape (eager, one seed, outputs are not checked). Stubs can change later route
choices, so ``kdev build`` repeats build + discover until a pass reports nothing pending. Typically that
takes 2-3 rounds of ~30 s GPU time each, and nothing compiles while the lock is held.
"""
from __future__ import annotations

import os
import time
from typing import Any

from kdev.contract import ErrorCode, KdevError, Result
from kdev.models import JobSpec


def run_discover(spec: JobSpec, run_dir: Any) -> Result:
    """Worker entry for kind='discover' (imports torch)."""
    import torch

    from kdev.adapters.registry import get_adapter
    from kdev.build.cache import install_load_inline_hook
    from kdev.build.toolchain import pin_for

    if not spec.files:
        raise KdevError(ErrorCode.INVALID_REQUEST, "discover needs one file")
    f = spec.files[0]
    t0 = time.monotonic()
    hook = install_load_inline_hook(f, pin_for(spec.problem, f.path), os.environ.get("KDEV_HOME"))
    hook.discover = True
    steps: list[str] = []
    errors: list[str] = []

    def step(name: str, fn: Any) -> Any:
        try:
            r = fn()
            torch.cuda.synchronize()
            steps.append(name)
            return r
        except Exception as e:  # noqa: BLE001 - stubs make later steps fail; keep discovering
            errors.append(f"{name}: {type(e).__name__}: {str(e)[:160]}")
            return None

    adapter = get_adapter(problem=spec.problem, file=f.path, adapter=spec.adapter)
    module = step("load", lambda: adapter.load(f.path, f.label or "A"))
    if module is not None:
        for k, v in (f.knobs or {}).items():
            setattr(module, k, v)
        state = step("prepare", lambda: adapter.prepare(module))
        budget = float(((spec.params or {}).get("extra") or {}).get("budget_s", 600))
        for sh in adapter.shapes():
            if time.monotonic() - t0 > budget:
                errors.append(f"budget {budget:.0f}s exhausted before {sh.shape_id}")
                break
            seed = getattr(sh, "seed", None)

            def call(sh: Any = sh, seed: Any = seed) -> Any:
                return adapter.call(module, state, adapter.make_inputs(sh, int(seed) if seed is not None else 1))

            step(f"call {sh.shape_id}", call)
    recs = list(hook.records)
    pending = sorted({r["name"] for r in recs if r.get("status") == "pending"})
    hits = sorted({r["name"] for r in recs if r.get("cache_hit")})
    verdict = "pending" if pending else "complete"
    lines = [f"{verdict.upper()}  discover  {len(pending)} extension(s) pending, {len(hits)} prebuilt, "
             f"{len(steps)} step(s) ok, {len(errors)} failed"]
    lines += [f"  pending: {n}" for n in pending[:15]]
    if len(pending) > 15:
        lines.append(f"  ... {len(pending) - 15} more")
    lines += [f"  step failed: {e}" for e in errors[:5]]
    return Result(tool="discover", status="ok", verdict=verdict, lines=lines,
                  summary={"pending": len(pending), "pending_names": pending[:40], "prebuilt": len(hits),
                           "steps_ok": len(steps), "steps_failed": len(errors), "file": f.path,
                           "elapsed_s": round(time.monotonic() - t0, 1)},
                  warnings=[e for e in errors[:3]] if pending else [])


def discover_loop(path: str, report: Any, rebuild: Any, knobs: dict[str, Any], defines: list[str], ctx: Any,
                   *, max_rounds: int = 6) -> tuple[Any, list[dict[str, Any]], list[str]]:
    """Build -> GPU discovery job (prepare + one call per shape, hook in discover mode: nothing compiles under
    the lock) -> rebuild the recorded calls, until a discovery pass finds nothing pending. Returns
    (final report, rounds, warnings). Skipped (with a warning) when GPU jobs are unavailable."""
    from pathlib import Path

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
