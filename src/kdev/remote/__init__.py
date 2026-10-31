"""Pod transport: ssh runner, md5-verified sync, pod bootstrap. Owner: P2."""
from __future__ import annotations

from kdev.models import Check


def probe(ctx: object, problem: str | None = None) -> list[Check]:
    """Doctor checks: KDEV_HOME not on a network/mfs path (e.g. /workspace), free disk, ssh key readable
    when remote is configured. Never raises."""
    raise NotImplementedError("remote.probe (P2)")
