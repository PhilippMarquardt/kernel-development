"""GPU worker process: ``python -m kdev.worker JOB.json``. Owner: P3. See docs/PLAN.md section 2.3.

Runs in the adapter's pinned interpreter; the ONLY entry point that may import torch. Loads the JobSpec,
creates/opens the run dir, resolves the adapter, dispatches on ``spec.kind``, writes result.json/raw.jsonl/
meta.json and exits with ``Result.exit_code``. Refuses to compile (needs_build) unless params.allow_jit.

Dispatch table (callee owners in brackets):
  time|ab          -> kdev.timing.engine.run_time / run_ab                      [P3]
  calib            -> kdev.timing.calib.run_calib                               [P3]
  check            -> kdev.checks.check.run_check                               [P7]
  comply           -> kdev.checks.comply.run_comply                             [P7]
  sanitize         -> kdev.checks.sanitize.run_sanitize_inner (under compute-sanitizer) [P7]
  timeline         -> kdev.timeline.run.run_timeline                            [P6]
  profile          -> kdev.observe.cupti.run_profile (or ncu/nsys)              [P6]
  dispatch         -> kdev.observe.dispatch.run_dispatch                        [P6]
  floor_calibrate  -> kdev.floor.microbench.run_calibrate                       [P6]
  exec             -> subprocess argv (deny-list checked by the service)        [P2]
"""
from __future__ import annotations

import sys
from typing import Callable

from kdev.contract import Result
from kdev.models import JobSpec

KIND_TABLE: dict[str, str] = {
    "time": "kdev.timing.engine:run_time",
    "ab": "kdev.timing.engine:run_ab",
    "calib": "kdev.timing.calib:run_calib",
    "check": "kdev.checks.check:run_check",
    "comply": "kdev.checks.comply:run_comply",
    "sanitize": "kdev.checks.sanitize:run_sanitize_inner",
    "timeline": "kdev.timeline.run:run_timeline",
    "profile": "kdev.observe.cupti:run_profile",
    "dispatch": "kdev.observe.dispatch:run_dispatch",
    "floor_calibrate": "kdev.floor.microbench:run_calibrate",
}

# Every function in KIND_TABLE has this signature:
JobFn = Callable[[JobSpec, "object"], Result]   # (spec, run_dir: kdev.runs.RunDir) -> Result


def resolve(kind: str) -> JobFn:
    """Import the function for a job kind (lazy, so torch is imported only inside the worker)."""
    raise NotImplementedError("worker.resolve (P3)")


def execute(spec: JobSpec, run_dir_path: str | None = None) -> Result:
    """Run one job in this process and write its artifacts; never raises (errors become Results)."""
    raise NotImplementedError("worker.execute (P3)")


def main(argv: list[str] | None = None) -> int:
    """CLI: ``python -m kdev.worker JOB.json [--run-dir DIR]``."""
    raise NotImplementedError("worker.main (P3)")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
