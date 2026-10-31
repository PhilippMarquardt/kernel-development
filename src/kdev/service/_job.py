"""Private build-lane entry: ``python -m kdev.service._job JOB.json --run-dir DIR``. Owner: P2.

Runs with CUDA_VISIBLE_DEVICES="" (set by the service) and calls ``kdev.build.compile.build_job`` (P5).
Writes result.json and exits with Result.exit_code. Never takes the GPU lock.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

from kdev.contract import ErrorCode, KdevError, Result, not_implemented
from kdev.models import JobSpec
from kdev.service._common import write_json


def _run_dir(path: Path):
    try:
        from kdev.runs import RunDir
        return RunDir(path.name, path)
    except Exception:  # noqa: BLE001
        from kdev.service._common import _RunDirFallback
        return _RunDirFallback(path.name, path)


def _write_result(d: Path, res: Result) -> None:
    try:
        from kdev.runs import write_result_files
        write_result_files(d, res)
    except ImportError:          # P1 runs unavailable
        write_json(d / "result.json", res.to_dict())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="kdev.service._job")
    ap.add_argument("job")
    ap.add_argument("--run-dir", required=True)
    a = ap.parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""          # belt and braces: no CUDA context in the build lane
    spec = JobSpec.from_dict(json.loads(Path(a.job).read_text(encoding="utf-8")))
    rd = _run_dir(Path(a.run_dir))
    try:
        if spec.kind != "build":
            raise KdevError(ErrorCode.INVALID_REQUEST, f"build lane got kind {spec.kind!r}")
        from kdev.build.compile import build_job
        res = build_job(spec, rd)
    except NotImplementedError as e:
        res = not_implemented(spec.kind, str(e))
    except KdevError as e:
        res = Result.from_error(spec.kind, e)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        res = Result.from_error(spec.kind, KdevError(ErrorCode.INTERNAL, f"{type(e).__name__}: {e}"))
    res.job_id = spec.job_id
    res.run_id = res.run_id or rd.run_id
    _write_result(Path(a.run_dir), res)
    return res.exit_code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
