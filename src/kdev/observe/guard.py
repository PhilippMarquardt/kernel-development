"""Profiler taint guard. Owner: P6. Torch-free.

Finding (coordinator, b5_ measurement): once torch.profiler has run, CUPTI stays enabled in the process and
adds 2-5 us to every later CUDA-event timing. So:
- every in-process profiler session (``kdev.observe.cupti.profile_events``) calls :func:`mark_profiled`;
- timing code calls :func:`check_clean` before timing: it refuses (``KdevError not_exclusive``-style
  ``invalid_request``) or, with ``refuse=False``, returns the opt-out string ``profiler_in_process`` to put in
  ``Fidelity.optouts`` (which makes the row ``comparable=0`` in the ledger);
- dispatch info for `time`/`check` Results must come from a separate process
  (:func:`kdev.observe.dispatch.probe_isolated`), never from a profiler session in the timing worker.
The flag is per process (also exported as ``KDEV_PROFILED=1`` so a re-exec'd helper of the same job sees it;
fresh worker processes started by the service get a clean environment from the daemon).
"""
from __future__ import annotations

import os

from kdev.contract import ErrorCode, KdevError

OPTOUT = "profiler_in_process"
ENV = "KDEV_PROFILED"
_state = {"profiled": False, "by": ""}


def mark_profiled(by: str = "torch.profiler") -> None:
    _state["profiled"] = True
    _state["by"] = by
    os.environ[ENV] = "1"


def profiled() -> bool:
    return _state["profiled"] or os.environ.get(ENV) == "1"


def check_clean(tool: str, refuse: bool = True) -> str | None:
    """None if no profiler ran in this process; else raise (refuse) or return the opt-out tag."""
    if not profiled():
        return None
    if refuse:
        raise KdevError(ErrorCode.INVALID_REQUEST,
                        f"{tool}: a profiler session ({_state['by'] or 'CUPTI'}) already ran in this process; "
                        "CUPTI stays enabled and adds 2-5 us to every later timing",
                        recovery="run profile/dispatch in a separate job (kdev dispatch FILE) or "
                                 "kdev.observe.dispatch.probe_isolated()")
    return OPTOUT


def _reset_for_tests() -> None:
    _state.update(profiled=False, by="")
    os.environ.pop(ENV, None)
