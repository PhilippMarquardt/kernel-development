"""P6 GPU integration (pod, KDEV_GPU_TESTS=1): marks.cuh end to end, run_timeline with a fake adapter, CUPTI
kernel table, dispatch probe, microbench subset. Run under the GPU lock (gpu_run.sh / kdev lock run)."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.gpu
os.environ.setdefault("KDEV_NO_BUILD_HOOK", "1")   # test modules build their own extensions

MODULE = r'''
import hashlib, os
import torch
from torch.utils.cpp_extension import load_inline
CPP = "void tl_run(torch::Tensor x, int64_t spin);"
CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
__global__ void tl_k(float* x, int spin) {
  // @kdev:mark sm
  // @kdev:mark entry
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float v = x[i];
  // @kdev:mark loaded
  for (int k = 0; k < spin; ++k) {
    v = v * 1.0001f + 0.5f;
    // @kdev:mark iter
  }
  // @kdev:mark computed
  x[i] = v;
  // @kdev:mark stored
}
void tl_run(torch::Tensor x, int64_t spin) {
  int n = (int)x.numel();
  tl_k<<<n / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(x.data_ptr<float>(), (int)spin);
}
"""
FLAGS = os.environ.get("TL_FLAGS", "").split()
NAME = "tlx_" + hashlib.md5((CUDA + " ".join(FLAGS)).encode()).hexdigest()[:10]
BUILD = os.path.join(os.environ.get("TL_BUILD", "/tmp"), NAME)
os.makedirs(BUILD, exist_ok=True)
M = load_inline(name=NAME, cpp_sources=CPP, cuda_sources=CUDA, functions=["tl_run"],
                extra_cuda_cflags=["-O3", "-gencode=arch=compute_90,code=sm_90"] + FLAGS, build_directory=BUILD)
def run(x, spin=64):
    M.tl_run(x, spin)
'''


def _import(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _setup(d: Path) -> SimpleNamespace:
    from kdev.timeline import instrument
    os.environ["TL_BUILD"] = str(d / "build")
    src = d / "src" / "tlmod.py"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(MODULE, encoding="utf-8")
    inst = instrument.instrument_file(str(src), str(d / "inst"))
    return SimpleNamespace(dir=d, src=src, inst=inst)


def prebuild(d: Path) -> None:
    """Compile-only (CUDA_VISIBLE_DEVICES="", TORCH_CUDA_ARCH_LIST=9.0): every extension the tests load."""
    s = _setup(d)
    _import(Path(s.inst.path), "pb_marked")
    _import(s.src, "pb_plain")
    os.environ["TL_FLAGS"] = "-DKD_MARKS_LOG"
    _import(Path(s.inst.path), "pb_log")
    os.environ.pop("TL_FLAGS")
    from kdev.floor import microbench
    microbench.load_ext("90")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    torch = pytest.importorskip("torch")
    d = Path(os.environ["P6_GPU_DIR"]) if os.environ.get("P6_GPU_DIR") else tmp_path_factory.mktemp("p6gpu")
    s = _setup(d)
    s.torch = torch
    return s


def _buf(torch, cap):
    return torch.zeros(16 + 16 * cap, dtype=torch.uint8, device="cuda")


def test_marks_slot_layout_end_to_end(built):
    torch = built.torch
    from kdev.observe import _rt
    from kdev.timeline import decode
    assert built.inst.marks == {0: "entry", 1: "loaded", 2: "iter", 3: "computed", 4: "stored"}
    mod = _import(Path(built.inst.path), "tl_marked")
    x = torch.ones(264 * 256, device="cuda")
    mod.run(x)                                    # pointer NULL: marks are no-ops
    torch.cuda.synchronize()
    libs = [(p, f) for p, f in _rt.marks_libraries() if mod.NAME in p]
    assert libs, _rt.loaded_libraries()
    cap = 264 * 8 * 33
    buf = _buf(torch, cap)
    flags = libs[0][1](buf.data_ptr(), cap)
    assert flags >= 0 and flags & 1 == 0 and (flags >> 8) & 0xFF == 32
    mod.run(x)
    torch.cuda.synchronize()
    recs, dropped = decode.parse_buffer(buf.cpu().numpy().tobytes())
    assert dropped == 0
    s = decode.summarize(recs, built.inst.marks)
    assert s.n_ctas == 264 and s.n_tracks == 264 * 2 and s.n_sms > 100
    order = [m.name for m in s.marks]
    assert order.index("entry") < order.index("loaded") < order.index("computed") < order.index("stored")
    assert s.resolution_ns is not None and s.resolution_ns <= 1024
    print(f"\nslot: tick {s.resolution_ns} ns, sm {s.sm_mhz} MHz, span {s.span_ns:.0f} ns, "
          f"start spread {s.cta_start_spread_ns:.0f} ns, durations {s.duration_source}")
    for p in s.phases:
        print(f"  {p.name:20s} p50 {p.p50_ns:8.1f} ns  n {p.n}")
    # graph replay after the read flush records too
    flush = _rt.make_flush()
    g = _rt.capture(lambda: mod.run(x))
    buf.zero_()
    libs[0][1](buf.data_ptr(), cap)
    _rt.time_replay_us(g, flush)
    recs2, _ = decode.parse_buffer(buf.cpu().numpy().tobytes())
    assert len([r for r in recs2 if r.mark != decode.SM_MARK]) == 264 * 2 * 5


def test_marks_log_layout(built):
    torch = built.torch
    from kdev.observe import _rt
    from kdev.timeline import decode
    os.environ["TL_FLAGS"] = "-DKD_MARKS_LOG"
    try:
        mod = _import(Path(built.inst.path), "tl_marked_log")
    finally:
        os.environ.pop("TL_FLAGS", None)
    x = torch.ones(132 * 256, device="cuda")
    mod.run(x, 4)
    torch.cuda.synchronize()
    lib = [f for p, f in _rt.marks_libraries() if mod.NAME in p][0]
    cap = 100000
    buf = _buf(torch, cap)
    assert lib(buf.data_ptr(), cap) & 1 == 1
    mod.run(x, 4)
    torch.cuda.synchronize()
    recs, dropped = decode.parse_buffer(buf.cpu().numpy().tobytes())
    it = [r for r in recs if r.mark == 2]
    assert dropped == 0 and len(it) == 132 * 2 * 4            # every loop iteration kept
    small = _buf(torch, 50)
    lib(small.data_ptr(), 50)
    mod.run(x, 4)
    torch.cuda.synchronize()
    recs, dropped = decode.parse_buffer(small.cpu().numpy().tobytes())
    assert len(recs) == 50 and dropped > 0


def test_run_timeline_with_fake_adapter(built, monkeypatch, tmp_path):
    torch = built.torch
    from kdev.adapters.base import Shape
    from kdev.models import FileRef, JobSpec
    from kdev.observe import _rt
    from kdev.timeline import run as tlrun
    n = [0]

    class FakeAdapter:
        def shapes(self):
            return [Shape("s264", {"ctas": 264})]

        def judge_seed(self, shape):
            return 11

        def load(self, path, tag):
            n[0] += 1
            return _import(Path(path), f"tl_fake_{tag}_{n[0]}")

        def prepare(self, module):
            return None

        def make_inputs(self, shape, seed):
            return torch.ones(shape.case["ctas"] * 256, device="cuda")

        def call(self, module, state, inputs):
            module.run(inputs)

    monkeypatch.setattr(_rt, "get_adapter_for", lambda spec: FakeAdapter())
    monkeypatch.setattr(tlrun, "_CLAIMED", set())
    spec = JobSpec(kind="timeline", files=[FileRef(built.inst.path, "marked", role="sub"),
                                           FileRef(str(built.src), "plain", role="base")],
                   params={"extra": {"marks": {str(k): v for k, v in built.inst.marks.items()}, "reps": 6,
                                     "capacity": 264 * 8 * 33}})
    res = tlrun.run_timeline(spec, tmp_path)
    assert res.status == "ok", res
    for k in ("trace", "csv", "chart", "timeline"):
        assert Path(res.files[k]).is_file()
    pl = res.summary["per_label"]["marked"]
    assert pl["ctas"] == 264 and pl["mark_overhead_us"] is not None
    print("\n" + "\n".join(res.lines))


def test_microbench_subset(tmp_path):
    pytest.importorskip("torch")
    from kdev.floor import microbench
    from kdev.models import JobSpec
    spec = JobSpec(kind="floor_calibrate", params={"extra": {"reps": 10, "benches": [
        "null_kernel", "dram_chase_1cta", "membar_gpu", "atom_acqrel", "globaltimer_tick", "graph_node"]}})
    res = microbench.run_calibrate(spec, tmp_path)
    print("\n" + "\n".join(res.lines))
    import json
    m = json.loads(Path(res.files["measured"]).read_text())["measured"]
    assert 0.1 < m["dram_trip_1cta_us"]["value"] < 1.5
    assert -2.0 < m["null_kernel_rel_us"]["value"] < 1.0
    assert m["globaltimer_tick_ns"]["value"] in (32, 64, 128, 256, 512, 1000, 1024)


# runs LAST: a profiler session taints every later timing in this process (CUPTI stays on)
def test_cupti_kernel_table_and_dispatch_probe():
    torch = pytest.importorskip("torch")
    from kdev.observe import _rt, cupti, dispatch
    x = torch.zeros(1 << 20, device="cuda")
    t = cupti.kernel_table(cupti.profile_events(lambda: x.add_(1)))
    assert t and t[0].count == 1 and t[0].grid
    g = _rt.capture(lambda: x.mul_(2))
    tg = cupti.kernel_table(cupti.profile_events(lambda: g.replay()))
    assert tg, "graph-replayed kernels must be visible to CUPTI"
    ad = SimpleNamespace(make_inputs=lambda s, seed: x, call=lambda m, st, inp: inp.add_(1))
    ks = dispatch.probe_shape(ad, None, None, None, 0)
    assert ks and ks[0]["count"] == 1
    from kdev.observe import guard
    assert guard.check_clean("t", refuse=False) == guard.OPTOUT   # CUPTI taint recorded




if __name__ == "__main__":  # pragma: no cover
    if len(sys.argv) == 3 and sys.argv[1] == "--prebuild":
        prebuild(Path(sys.argv[2]))
        print("prebuilt")
        sys.exit(0)
    sys.exit(pytest.main([__file__, "-s", "-q"]))
