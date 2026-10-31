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
