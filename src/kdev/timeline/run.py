"""Worker entry for kind='timeline' (GPU). Owner: P6.

Builds (via the prebuilt instrumented copy), allocates the mark buffer, runs one shape (graph replay after
read flush, like timing), decodes, writes trace.json/timeline.csv, and measures mark overhead as marked vs
unmarked replay delta. ``params.extra['ab']`` = two define sets -> interleaved timelines.
"""
from __future__ import annotations

from typing import Any

from kdev.contract import Result
from kdev.models import JobSpec


def run_timeline(spec: JobSpec, run_dir: Any) -> Result:
    raise NotImplementedError("timeline.run.run_timeline (P6)")
