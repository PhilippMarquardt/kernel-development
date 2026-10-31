"""CPU-only prepare pass for the build lane (P5, private, opt-in: ``KDEV_CPU_PREPARE=1``). Imports torch.

Off by default: prepares that self-test kernels or pick routes from GPU results (all OKC bests) take
different paths on the CPU, so the GPU discovery job (``kdev.build.discover``, run by ``kdev build``) is the
default way to find prepare-time extensions. Useful for submissions whose prepare is data-independent.

Real OKC submissions build most of their extensions inside ``prepare()`` or at the first ``kernel_fn`` call
(the kernel set depends on weights / shapes), so importing the module and calling its ``_build*`` entry
points finds only a few. This pass runs the adapter's ``load -> prepare -> kernel_fn(shape)`` for every
shape with **no CUDA context** (the build lane has ``CUDA_VISIBLE_DEVICES=""``): ``torch.cuda`` is faked and
every ``device="cuda"`` tensor is created on the CPU, while the build hook (build mode) compiles each
intercepted ``load_inline``/``load`` call into the content-keyed kdev cache and returns a stub module.

Best effort by design: a step that needs real GPU results (a stub extension returning None, a CUDA-only
op, data-dependent asserts) fails, is logged, and the pass moves on to the next shape. Whatever is still
missing is recorded by the GPU worker as a pending call and built by the next ``kdev build`` (see
``_shim.Hook._record_pending``). Never launches a kernel; never touches the GPU driver.
"""
from __future__ import annotations

import os
import time
import traceback
from typing import Any


class _FakeProps:
    def __init__(self, cap: tuple[int, int]) -> None:
        self.name = "NVIDIA H200 (kdev cpu prepare pass)"
        self.major, self.minor = cap
        self.multi_processor_count = 132
        self.total_memory = 141 << 30
        self.L2_cache_size = 50 << 20
        self.max_threads_per_multi_processor = 2048
        self.regs_per_multiprocessor = 65536
        self.shared_memory_per_block_optin = 232448
        self.warp_size = 32
        self.uuid = "kdev-fake"
        self.gcnArchName = ""


class _Nop:
    """Stand-in for torch.cuda.Stream / Event / CUDAGraph: every method is a no-op."""

    def __init__(self, *a: Any, **k: Any) -> None:
        pass

    def __getattr__(self, item: str) -> Any:
        if item.startswith("__"):
            raise AttributeError(item)
        return lambda *a, **k: 0.0

    def __enter__(self) -> "_Nop":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False


def _is_cuda(dev: Any) -> bool:
    s = str(dev) if dev is not None else ""
    return s == "cuda" or s.startswith("cuda:") or (isinstance(dev, int) and not isinstance(dev, bool))


def install_fake_cuda(cap: tuple[int, int] = (9, 0)) -> Any:
    """Patch torch so CUDA code paths run on the CPU. Returns the active TorchFunctionMode (kept for the
    rest of the process: the build lane is a throwaway process)."""
    import torch
    from torch.overrides import TorchFunctionMode

    cpu = torch.device("cpu")
    tc = torch.cuda
    tc.is_available = lambda: True
    tc.device_count = lambda: 1
    tc.current_device = lambda: 0
    tc.set_device = lambda *a, **k: None
    tc.synchronize = lambda *a, **k: None
    tc.get_device_capability = lambda *a, **k: cap
    tc.get_device_name = lambda *a, **k: "NVIDIA H200"
    tc.get_device_properties = lambda *a, **k: _FakeProps(cap)
    tc.manual_seed = lambda s: torch.manual_seed(s)
    tc.manual_seed_all = lambda s: torch.manual_seed(s)
    tc.empty_cache = lambda: None
    tc.is_initialized = lambda: True
    tc.init = lambda: None
    for name in ("Stream", "Event", "CUDAGraph", "ExternalStream"):
        setattr(tc, name, _Nop)
    tc.current_stream = lambda *a, **k: _Nop()
    tc.default_stream = lambda *a, **k: _Nop()
    tc.stream = lambda *a, **k: _Nop()
    tc.graph = lambda *a, **k: _Nop()
    tc.memory_allocated = lambda *a, **k: 0
    tc.max_memory_allocated = lambda *a, **k: 0
    tc.mem_get_info = lambda *a, **k: (100 << 30, 141 << 30)
    torch.Tensor.cuda = lambda self, *a, **k: self
    torch.Tensor.is_cuda = property(lambda self: self.device.type == "cpu")  # type: ignore[assignment]
    real_gen = torch.Generator

    class _GenMeta(type):            # torch.Generator(device="cuda") -> a CPU generator; isinstance still works
        def __call__(cls, device: Any = "cpu", *a: Any, **k: Any) -> Any:
            return real_gen(device="cpu")

        def __instancecheck__(cls, obj: Any) -> bool:
            return isinstance(obj, real_gen)

    class _Gen(metaclass=_GenMeta):
        pass

    torch.Generator = _Gen  # type: ignore[misc]

    class _CudaToCpu(TorchFunctionMode):
        def __torch_function__(self, func: Any, types: Any, args: tuple = (), kwargs: dict | None = None) -> Any:
            kwargs = dict(kwargs or {})
            if _is_cuda(kwargs.get("device")):
                kwargs["device"] = cpu
            name = getattr(func, "__name__", "")
            if name == "to" and args:
                args = tuple(cpu if (i > 0 and _is_cuda(a)) else a for i, a in enumerate(args))
            if kwargs.get("pin_memory"):
                kwargs["pin_memory"] = False
            return func(*args, **kwargs)

    mode = _CudaToCpu()
    mode.__enter__()
    return mode


def run(file: str, *, cap: tuple[int, int] = (9, 0), max_shapes: int = 64, budget_s: float = 900,
        log: Any = None) -> dict[str, Any]:
    """Adapter load -> prepare -> kernel_fn per scored shape on the CPU. Returns {steps, errors, seconds}."""
    t0 = time.monotonic()
    out: dict[str, Any] = {"steps": [], "errors": [], "seconds": 0.0, "adapter": ""}

    def note(step: str, err: BaseException | None = None) -> None:
        if err is None:
            out["steps"].append(step)
        else:
            out["errors"].append(f"{step}: {type(err).__name__}: {str(err)[:200]}")
            if log:
                log(f"{step}: " + traceback.format_exc()[-1200:])

    try:
        install_fake_cuda(cap)
        from kdev.adapters.registry import get_adapter
        ad = get_adapter(file=file)
        out["adapter"] = type(ad).__name__
    except Exception as e:  # noqa: BLE001
        note("setup", e)
        out["seconds"] = round(time.monotonic() - t0, 1)
        return out
    try:
        mod = ad.load(file, "kdev_cpuprep")
        note("load")
    except Exception as e:  # noqa: BLE001
        note("load", e)
        out["seconds"] = round(time.monotonic() - t0, 1)
        return out
    state = None
    try:
        state = ad.prepare(mod)
        note("prepare")
    except Exception as e:  # noqa: BLE001
        note("prepare", e)
    try:
        shapes = list(ad.shapes())[:max_shapes]
    except Exception as e:  # noqa: BLE001
        note("shapes", e)
        shapes = []
    for sh in shapes:
        if time.monotonic() - t0 > budget_s:
            out["errors"].append(f"budget {budget_s:.0f}s exhausted before {sh.shape_id}")
            break
        try:
            seed = getattr(sh, "seed", None)
            inputs = ad.make_inputs(sh, int(seed) if seed is not None else 1)
            ad.call(mod, state, inputs)
            note(f"call {sh.shape_id}")
        except Exception as e:  # noqa: BLE001
            note(f"call {sh.shape_id}", e)
    out["seconds"] = round(time.monotonic() - t0, 1)
    return out


if __name__ == "__main__":  # pragma: no cover - manual probe
    import json
    import sys
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    print(json.dumps(run(sys.argv[1]), indent=1))
