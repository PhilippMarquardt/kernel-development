"""Worker-side runtime helpers shared by P6's GPU jobs (timeline, profile, dispatch, floor calibrate).

torch is imported lazily inside functions only (never at module import). The timing recipe follows the judge
(PLAN 3.7): read flush (``buf.sum()`` into a sink) before every replay, CUDA-graph replay, CUDA events. The
P3 implementations (kdev.timing.flush / kdev.timing.graph) are used when present; these fallbacks keep P6's
jobs usable while P3 is in flight and follow the same recipe.
"""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
from typing import Any, Callable

from kdev.contract import ErrorCode, KdevError
from kdev.models import JobSpec

DEFAULT_FLUSH_BYTES = 256 << 20


def run_path(run_dir: Any) -> Path:
    p = Path(getattr(run_dir, "path", run_dir) or ".")
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: Path, obj: Any) -> str:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, path)
    return str(path)


class ReadFlush:
    """Judge read flush: ``sink = buf.sum()`` over >= 4x L2 (default 256 MB)."""

    def __init__(self, nbytes: int = DEFAULT_FLUSH_BYTES, method: str = "read") -> None:
        import torch
        self.method = method
        self.nbytes = nbytes
        self.buf = torch.ones(max(1, nbytes // 4), dtype=torch.float32, device="cuda") if method != "none" else None
        self.sink = torch.zeros((), dtype=torch.float32, device="cuda")

    def __call__(self) -> None:
        if self.buf is None:
            return
        if self.method == "write":
            self.buf.zero_()
        else:
            self.sink.copy_(self.buf.sum())


def make_flush(nbytes: int | None = None, method: str = "read") -> Callable[[], None]:
    nb = nbytes or DEFAULT_FLUSH_BYTES
    try:
        from kdev.timing.flush import Flusher
        f = Flusher(method=method, nbytes=nb)
        f()          # P3 implementation present?
        return f
    except NotImplementedError:
        return ReadFlush(nb, method)


def capture(fn: Callable[[], Any]) -> Any:
    """Judge capture recipe: call, side-stream warm-up, capture."""
    try:
        from kdev.timing.graph import capture as p3_capture
        return p3_capture(fn)
    except NotImplementedError:
        pass
    import torch
    fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return g


def time_replay_us(g: Any, flush: Callable[[], None] | None) -> float:
    import torch
    if flush is not None:
        flush()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    g.replay()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1000.0


def time_eager_us(fn: Callable[[], Any], flush: Callable[[], None] | None) -> float:
    import torch
    if flush is not None:
        flush()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1000.0


def loaded_libraries(maps_path: str = "/proc/self/maps") -> list[str]:
    """Paths of shared objects mapped into this process (Linux), in load order."""
    out: list[str] = []
    try:
        with open(maps_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.split(None, 5)
                if len(parts) == 6:
                    p = parts[5].strip()
                    if p.endswith(".so") or ".so." in p:
                        if p not in out:
                            out.append(p)
    except OSError:
        pass
    return out


def marks_libraries() -> list[tuple[str, Any]]:
    """[(path, kd_marks_init)] for every loaded extension exporting the timeline host hook."""
    found = []
    for p in loaded_libraries():
        if any(s in p for s in ("/libcuda", "/libc.", "/libtorch", "/libnv", "/libcu", "/python")):
            continue
        try:
            lib = ctypes.CDLL(p)
            fn = lib.kd_marks_init
        except (OSError, AttributeError):
            continue
        fn.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        fn.restype = ctypes.c_int
        found.append((p, fn))
    return found


def extension_paths(obj: Any, depth: int = 2) -> list[str]:
    """.so paths of extension modules reachable from a loaded submission module (globals, and dicts /
    lists / tuples / objects one level down), e.g. ``M = load_inline(...)`` or ``_EXTS = {key: ext}``."""
    import types
    out: list[str] = []
    seen: set[int] = set()

    def visit(o: Any, d: int) -> None:
        if id(o) in seen or d < 0:
            return
        seen.add(id(o))
        if isinstance(o, types.ModuleType):
            f = getattr(o, "__file__", "") or ""
            if f.endswith(".so") or ".so." in f:
                if f not in out:
                    out.append(os.path.realpath(f))
                return
            if d == depth:                           # the submission module itself: walk its globals
                for v in list(vars(o).values()):
                    visit(v, d - 1)
            return
        if isinstance(o, dict):
            for v in list(o.values())[:256]:
                visit(v, d - 1)
        elif isinstance(o, (list, tuple, set)):
            for v in list(o)[:256]:
                visit(v, d - 1)
        elif hasattr(o, "__dict__") and not isinstance(o, type) and d < depth:
            for v in list(vars(o).values())[:64]:
                visit(v, d - 1)

    visit(obj, depth)
    return out


def get_adapter_for(spec: JobSpec) -> Any:
    from kdev.adapters.registry import get_adapter
    f0 = spec.files[0].path if spec.files else None
    return get_adapter(problem=spec.problem, file=f0, adapter=spec.adapter)


def select_shapes(adapter: Any, params: dict[str, Any], one: str | None = None) -> list[Any]:
    shapes = list(adapter.shapes())
    flt = params.get("shapes")
    if flt:
        fs = [f.strip() for f in str(flt).split(",") if f.strip()]      # comma = any of (as timing)
        shapes = [s for s in shapes if any(f in s.shape_id or f in json.dumps(s.case, sort_keys=True) for f in fs)]
    if one:
        exact = [s for s in shapes if s.shape_id == one]
        shapes = exact or [s for s in shapes if one in s.shape_id]
    if not shapes:
        raise KdevError(ErrorCode.INVALID_REQUEST, f"no scored shape matches {one or flt!r}",
                        recovery="list shapes with `kdev time FILE --shapes ''` or check kdev_problem.toml")
    return shapes


def seed_for(adapter: Any, shape: Any, fallback: int = 11) -> int:
    try:
        s = adapter.judge_seed(shape)
    except Exception:  # noqa: BLE001 - optional adapter hook
        s = None
    if s is None:
        s = getattr(shape, "seed", None)
    return int(s) if s is not None else fallback


def use_prebuilt(spec: JobSpec, f: Any) -> Any:
    """Route the next load_inline/load calls to the prebuilt cache entry of FileRef ``f`` (its knobs and
    defines; P5 ``build.cache.install_load_inline_hook``). If a hook is already installed (worker-level), its
    knobs/defines are switched to ``f`` so several files of one job (timeline --ab, marked + plain) each load
    their own build. Returns the hook, or None when the build package is unavailable or
    ``KDEV_NO_BUILD_HOOK=1`` (tests that build their own extensions)."""
    if os.environ.get("KDEV_NO_BUILD_HOOK") == "1":
        return None
    try:
        from kdev.build.cache import install_load_inline_hook, select_variant
        from kdev.build.toolchain import pin_for
    except Exception:  # noqa: BLE001 - optional integration
        return None
    hook = select_variant(f)
    if hook is not None:
        return hook
    allow = bool((spec.params or {}).get("allow_jit")) or not spec.requires_prebuilt
    try:
        return install_load_inline_hook(f, pin_for(spec.problem, f.path), os.environ.get("KDEV_HOME"),
                                        allow_jit=allow)
    except (NotImplementedError, ImportError):    # no torch in this interpreter (unit tests)
        return None


def median(xs: list[float]) -> float:
    s = sorted(xs)
    if not s:
        return float("nan")
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
