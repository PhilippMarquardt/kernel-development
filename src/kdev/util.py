"""Small shared helpers. Owner: P1.

Subprocess-wrapping code everywhere takes ``runner: Runner = run`` so tests can inject canned output.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import secrets
import subprocess
import time
from dataclasses import dataclass
from typing import Callable, Sequence


@dataclass
class Proc:
    """Completed subprocess (what a Runner returns)."""

    argv: list[str]
    returncode: int
    stdout: str
    stderr: str
    seconds: float = 0.0


Runner = Callable[..., Proc]


def run(argv: Sequence[str], *, timeout: float | None = None, env: dict[str, str] | None = None,
        cwd: str | None = None) -> Proc:
    """Run a command (never through a shell) and capture text output."""
    t0 = time.monotonic()
    cp = subprocess.run(list(argv), capture_output=True, text=True, timeout=timeout, env=env, cwd=cwd,
                        check=False)
    return Proc(list(argv), cp.returncode, cp.stdout, cp.stderr, time.monotonic() - t0)


def md5_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()  # noqa: S324 - identity hash, not security
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def norm_md5(path: str) -> str:
    """md5 of the source with comments and whitespace removed (Python '#', C '//' and '/* */'), so a
    comment-only change (the judge 'duplicate' workaround) maps to the same identity."""
    raise NotImplementedError("util.norm_md5 (P1)")


def utc_now() -> str:
    """ISO-8601 UTC timestamp with 'Z', second resolution."""
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_id(prefix: str) -> str:
    """Sortable id: <prefix><yyyymmdd-hhmmss>-<4 hex>, e.g. j20261031-143012-7f3a."""
    return f"{prefix}{_dt.datetime.now(_dt.timezone.utc).strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}"
