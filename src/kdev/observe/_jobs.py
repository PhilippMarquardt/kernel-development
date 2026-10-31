"""JobSpec helpers for P6's GPU commands (timeline, profile, dispatch). Torch-free. Private to P6."""
from __future__ import annotations

import os
from argparse import Namespace
from typing import Any

from kdev.contract import ErrorCode, KdevError, Result
from kdev.models import FileRef, JobSpec
from kdev.registry import Context


def file_ref(path: str, label: str = "A", role: str = "sub", defines: list[str] | None = None) -> FileRef:
    from kdev.util import md5_file
    p = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(p):
        raise KdevError(ErrorCode.NOT_FOUND, f"file not found: {path}", recovery="check the path")
    return FileRef(path=p, label=label, md5=md5_file(p), role=role, defines=list(defines or []))


def make_spec(kind: str, files: list[FileRef], args: Namespace, ctx: Context, extra: dict[str, Any],
              **params: Any) -> JobSpec:
    from kdev.util import new_id, utc_now
    p: dict[str, Any] = {"shapes": getattr(args, "shapes", None), "flush": "read", "graph": True,
                         "allow_jit": bool(getattr(args, "allow_jit", False)), "extra": extra}
    p.update(params)
    return JobSpec(kind=kind, files=files, params=p, problem=ctx.problem or getattr(args, "problem", None),
                   job_id=new_id("j"), lane="gpu", agent=ctx.agent or os.environ.get("KDEV_AGENT", ""),
                   priority=int(getattr(args, "priority", 0) or 0),
                   timeout_s=int(getattr(args, "timeout", 1800) or 1800),
                   run_name=getattr(args, "run_name", "") or "", submitted_utc=utc_now(),
                   requires_prebuilt=not bool(getattr(args, "allow_jit", False)))


def submit(spec: JobSpec, args: Namespace, ctx: Context) -> Result:
    return ctx.submit(spec, detach=bool(getattr(args, "detach", False)), direct=bool(getattr(args, "direct", False)),
                      timeout_s=getattr(args, "timeout", None))
