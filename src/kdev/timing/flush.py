"""L2 flush before every replay. Owner: P3. Default = READ flush (M1: the judge reads; a write flush leaves
dirty lines written back during the timed kernel and inflates small kernels by 1-2.5 us)."""
from __future__ import annotations

from typing import Any

DEFAULT_FLUSH_BYTES = 256 << 20     # >= 4x L2 on H200; GPU profile flush_bytes overrides


class Flusher:
    """Allocates the flush buffer once; ``__call__`` enqueues the flush on the current stream.

    method: 'read' (``buf.sum()`` into a persistent sink, the judge method), 'write' (``buf.zero_()``,
    opt-out, recorded), 'none' (opt-out)."""

    def __init__(self, method: str = "read", nbytes: int = DEFAULT_FLUSH_BYTES, device: str = "cuda") -> None:
        self.method = method
        self.nbytes = nbytes
        self.device = device
        self._buf: Any = None

    def __call__(self) -> None:
        raise NotImplementedError("Flusher.__call__ (P3)")

    def describe(self) -> dict[str, Any]:
        return {"method": self.method, "bytes": self.nbytes}


def selftest_read_vs_write(nbytes: int = DEFAULT_FLUSH_BYTES) -> dict[str, float]:
    """Doctor self-test: tiny-kernel time after read vs write flush (us); a large delta means dirty-line
    cost. GPU only."""
    raise NotImplementedError("flush.selftest_read_vs_write (P3)")
