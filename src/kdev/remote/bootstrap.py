"""`kdev remote setup`: idempotent pod bootstrap with phase markers. Owner: P2.

Port of arcagi3/scripts/okc_bootstrap.sh + okc_pod_setup.sh: venv pins, judge toolchain (CUDA 13.0.88 pip
wheels + wrapper dir), TORCH_EXTENSIONS_DIR per toolchain, read-flush self-test, `kdev doctor`. Each phase
writes ``$KDEV_HOME/bootstrap/<phase>.done`` and is skipped when present.
"""
from __future__ import annotations

from kdev.config import Config

PHASES = ("venv", "toolchain", "kdev_install", "flush_selftest", "doctor")


def setup(cfg: Config, phases: tuple[str, ...] = PHASES, force: bool = False) -> dict[str, str]:
    """Run phases on the remote host; returns {phase: done|skipped|failed}."""
    raise NotImplementedError("bootstrap.setup (P2)")
