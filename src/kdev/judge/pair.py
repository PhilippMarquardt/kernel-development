"""`kdev judge pair A [--base B] [--allow-check]`: same-hour judge pair of a candidate and its unmodified
base (LESSONS 3). Owner: P4.

Without --allow-check: prepare both, print the two exact `okc check` commands, exit 0 (no network).
With --allow-check (CLI only, never MCP): refuse duplicates, run both checks back to back via
okc_cli._okc_check(_allow=_ALLOW_CHECK), record both, report the judged A/B ratio per shape.
"""
from __future__ import annotations

from kdev.contract import Result

_ALLOW_CHECK = object()   # identity token passed to okc_cli._okc_check


def run_pair(candidate: str, base: str | None, problem: str | None, *, allow_check: bool = False) -> Result:
    raise NotImplementedError("judge.pair.run_pair (P4)")
