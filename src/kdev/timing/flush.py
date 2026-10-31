"""L2 flush before every replay. Owner: P3. Default = READ flush (M1: the judge reads; a write flush leaves
dirty lines written back during the timed kernel and inflates small kernels by 1-2.5 us).

The read flush is ``buf.sum()`` over a uint8 buffer (exactly calib.py's flush, which produced
okc/calibration/all_calib.md); the result is kept in ``self._sink`` so the reduction cannot be elided and no
extra copy kernel is enqueued. An optional gate (``gate_cycles`` of ``torch.cuda._sleep``) after the flush
mimics the judge's ~300 us gate in front of each replay, so the replay launch is queued behind GPU work.
"""
from __future__ import annotations

from typing import Any

DEFAULT_FLUSH_BYTES = 256 << 20     # >= 4x L2 on H200; GPU profile flush_bytes overrides
METHODS = ("read", "write", "none")


class Flusher:
    """Allocates the flush buffer once; ``__call__`` enqueues the flush on the current stream.

    method: 'read' (``buf.sum()`` into a persistent sink, the judge method), 'write' (``buf.zero_()``,
    opt-out, recorded), 'none' (opt-out)."""

    def __init__(self, method: str = "read", nbytes: int = DEFAULT_FLUSH_BYTES, device: str = "cuda",
                 gate_cycles: int = 0) -> None:
        if method not in METHODS:
            raise ValueError(f"unknown flush method {method!r} (expected one of {METHODS})")
        self.method = method
        self.nbytes = int(nbytes)
        self.device = device
        self.gate_cycles = int(gate_cycles or 0)
        self._buf: Any = None
        self._sink: Any = None

    def _ensure(self) -> None:
        if self._buf is None and self.method != "none":
            import torch

            self._buf = torch.empty(self.nbytes, dtype=torch.uint8, device=self.device)
            if self.method == "write":
                self._buf.zero_()

    def __call__(self) -> None:
        if self.method == "none":
            pass
        else:
            self._ensure()
            if self.method == "read":
                self._sink = self._buf.sum()
            else:
                self._buf.zero_()
        if self.gate_cycles:
            import torch

            torch.cuda._sleep(self.gate_cycles)

    def describe(self) -> dict[str, Any]:
        d: dict[str, Any] = {"method": self.method, "bytes": self.nbytes}
        if self.gate_cycles:
            d["gate_cycles"] = self.gate_cycles
        return d

    def release(self) -> None:
        self._buf = None
        self._sink = None


def selftest_read_vs_write(nbytes: int = DEFAULT_FLUSH_BYTES, reps: int = 50) -> dict[str, float]:
    """Doctor self-test: tiny-kernel (graph of x.add_(1) over 1 MB) replay time after a read vs a write
    flush (us, medians); a large ``delta_us`` means dirty-line write-back cost lands in the timed kernel.
    GPU only."""
    import torch

    from kdev.timing import graph as G
    from kdev.timing.stats import median

    x = torch.zeros(1 << 18, dtype=torch.float32, device="cuda")
    g = G.capture(lambda: x.add_(1))
    fr, fw = Flusher("read", nbytes), Flusher("write", nbytes)
    G.time_replay(g, fr)
    G.time_replay(g, fw)
    r, w = [], []
    for i in range(reps):
        if i % 2 == 0:
            r.append(G.time_replay(g, fr))
            w.append(G.time_replay(g, fw))
        else:
            w.append(G.time_replay(g, fw))
            r.append(G.time_replay(g, fr))
    fr.release()
    fw.release()
    mr, mw = median(r), median(w)
    return {"read_us": mr, "write_us": mw, "delta_us": mw - mr}
