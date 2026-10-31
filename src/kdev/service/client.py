"""Client side of the service: submit/wait/status/cancel, with the direct (no-daemon) fallback. Owner: P2.

``kdev.registry.Context.submit`` (P1) calls :func:`submit_and_maybe_wait`.
"""
from __future__ import annotations

from kdev.config import Config
from kdev.contract import Result
from kdev.models import JobSpec, JobStatus


def submit(cfg: Config, spec: JobSpec) -> JobStatus:
    """Enqueue (service must be alive) and return the status."""
    raise NotImplementedError("client.submit (P2)")


def wait(cfg: Config, job_id: str, timeout_s: float | None = None, poll_s: float = 1.0) -> Result:
    """Block until done; return the job's Result (from result.json). On timeout return a Result with status
    'ok' and verdict 'pending' carrying job_id (the job keeps running)."""
    raise NotImplementedError("client.wait (P2)")


def run_direct(cfg: Config, spec: JobSpec, timeout_s: float | None = None) -> Result:
    """No daemon: build-lane jobs run in-process; GPU jobs run the worker subprocess under lock.exclusive()."""
    raise NotImplementedError("client.run_direct (P2)")


def submit_and_maybe_wait(cfg: Config, spec: JobSpec, *, detach: bool = False, direct: bool = False,
                          timeout_s: float | None = None) -> Result:
    """Routing rule of PLAN 2.2: daemon alive -> enqueue (+wait unless detach); else direct; KDEV_NO_GPU=1
    and a GPU lane -> capability_unavailable."""
    raise NotImplementedError("client.submit_and_maybe_wait (P2)")
