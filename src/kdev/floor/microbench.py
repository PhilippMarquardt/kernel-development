"""Microbenchmark suite for `kdev floor calibrate` (GPU, worker kind='floor_calibrate'). Owner: P6.

Ports the ideas of okc fb.py/fl_src.py/tlb.cu/ic.py: null kernel, empty/1-kernel graph, dependent DRAM
chase (1 CTA, all SMs), L2 hit, kernel boundary (with/without PDL), extra graph node, cooperative launch,
MEMBAR.GPU, cold atom.acq_rel, barrier.cluster, 200 MB stream. All after a read flush, graph replay, median.

Method (identical for every bench): each variant is captured as a CUDA graph; variants of one bench are
replayed interleaved (ABBA) ``reps`` times, each replay after the 256 MB read flush, timed with CUDA events;
the median is taken. Absolute values are stored in judge units (raw us - 4.0); costs of one primitive are the
delta of two graphs that differ only in that primitive (``*_us`` keys of the GPU profile). The extension is
compiled outside the GPU lock by ``python -m kdev.floor.microbench --build`` (CUDA_VISIBLE_DEVICES="");
the GPU job then loads it from the cache.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Any, Callable

from kdev.contract import Result
from kdev.models import JobSpec

BENCHES = ("null_kernel", "empty_graph", "one_kernel_graph", "dram_chase_1cta", "dram_chase_all",
           "l2_hit", "kernel_boundary", "kernel_boundary_pdl", "graph_node", "coop_launch", "membar_gpu",
           "atom_acqrel", "barrier_cluster", "stream_200mb", "gather_round", "merge_lastcta", "globaltimer_tick")

JUDGE_FIXED_US = 4.0

CPP = r"""
#include <torch/extension.h>
void mb_null(int64_t G, int64_t T, int64_t coop, int64_t pdl);
void mb_chase(torch::Tensor buf, int64_t nslots, int64_t n, int64_t G, torch::Tensor out);
void mb_stream(torch::Tensor a, int64_t chunks, int64_t pdl, torch::Tensor sink);
void mb_sync(torch::Tensor line, int64_t mode);
void mb_cluster(int64_t G, int64_t cs, int64_t mode);
void mb_gather(torch::Tensor kv, int64_t nrows, int64_t G, int64_t R, torch::Tensor sink);
void mb_merge(torch::Tensor ws, torch::Tensor cnt, torch::Tensor out, int64_t G, int64_t nf, int64_t mode);
void mb_tick(torch::Tensor out, int64_t n);
"""

CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <stdint.h>
typedef unsigned long long u64;
typedef unsigned int u32;
static cudaStream_t mb_stream_() { return at::cuda::getCurrentCUDAStream(); }
__device__ __forceinline__ u64 mb_gt() { u64 t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) :: "memory"); return t; }
__device__ __forceinline__ u64 mb_ldcg(const u64* p) { u64 v; asm volatile("ld.global.cg.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ u32 mb_smem(const void* p) { return (u32)__cvta_generic_to_shared(p); }

static void mb_launch(const void* fn, dim3 g, dim3 b, size_t smem, int cluster, int coop, int pdl, void** args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = g; cfg.blockDim = b; cfg.dynamicSmemBytes = smem; cfg.stream = mb_stream_();
  cudaLaunchAttribute at[3]; int na = 0;
  if (cluster > 1) {
    cudaFuncSetAttribute(fn, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    at[na].id = cudaLaunchAttributeClusterDimension; at[na].val.clusterDim.x = cluster;
    at[na].val.clusterDim.y = 1; at[na].val.clusterDim.z = 1; ++na;
  }
  if (coop) { at[na].id = cudaLaunchAttributeCooperative; at[na].val.cooperative = 1; ++na; }
  if (pdl) { at[na].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[na].val.programmaticStreamSerializationAllowed = 1; ++na; }
  cfg.attrs = at; cfg.numAttrs = na;
  cudaError_t e = cudaLaunchKernelExC(&cfg, fn, args);
  TORCH_CHECK(e == cudaSuccess, "kdev microbench launch: ", cudaGetErrorString(e));
}

__global__ void k_null() {}
__global__ void k_null_pdl() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
void mb_null(int64_t G, int64_t T, int64_t coop, int64_t pdl) {
  void* a[] = {nullptr};
  mb_launch(pdl ? (const void*)k_null_pdl : (const void*)k_null, dim3(G), dim3(T), 0, 1, (int)coop, (int)pdl, a);
}

// dependent chase: one thread per CTA follows the permutation cycle n times (128-B slots, 16 u64 each)
__global__ void k_chase(const u64* buf, u64 nslots, int n, u64* out) {
  if (threadIdx.x) return;
  u64 p = ((u64)blockIdx.x * 7919ull % nslots) * 16ull;
  for (int i = 0; i < n; ++i) p = mb_ldcg(buf + p);
  out[blockIdx.x] = p;
}
void mb_chase(torch::Tensor buf, int64_t nslots, int64_t n, int64_t G, torch::Tensor out) {
  const u64* b = (const u64*)buf.data_ptr(); u64 ns = (u64)nslots; int nn = (int)n; u64* o = (u64*)out.data_ptr();
  void* a[] = {&b, &ns, &nn, &o};
  mb_launch((const void*)k_chase, dim3(G), dim3(32), 0, 1, 0, 0, a);
}

// streaming read of a contiguous chunk by all CTAs (grid-stride float4), optional PDL chain
__global__ void k_stream(const float4* a, long long n4, float* sink, int pdl) {
  if (pdl) { asm volatile("griddepcontrol.wait;" ::: "memory"); asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
  float s = 0.f;
  for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n4; i += (long long)gridDim.x * blockDim.x) {
    float4 v = __ldcs(a + i); s += v.x + v.y + v.z + v.w;
  }
  if (s == 1.2345f) sink[0] = s;
}
void mb_stream(torch::Tensor a, int64_t chunks, int64_t pdl, torch::Tensor sink) {
  long long n4 = a.numel() / 4 / chunks; float* sk = (float*)sink.data_ptr(); int p = (int)pdl;
  int sms = 0; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  for (int c = 0; c < chunks; ++c) {
    const float4* base = (const float4*)a.data_ptr() + (size_t)c * n4;
    void* args[] = {&base, &n4, &sk, &p};
    mb_launch((const void*)k_stream, dim3(sms * 4), dim3(512), 0, 1, 0, c > 0 ? p : 0, args);
  }
}

// cold store + (0) nothing (1) fence.acq_rel.gpu (2) atom.acq_rel.gpu (3) fence.cta
__global__ void k_sync(u32* line, int mode) {
  if (threadIdx.x) return;
  line[0] = 1u;
  if (mode == 1) asm volatile("fence.acq_rel.gpu;" ::: "memory");
  if (mode == 2) { u32 old; asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;" : "=r"(old) : "l"(line + 32) : "memory"); if (old == 0xffffffffu) line[1] = old; }
  if (mode == 3) asm volatile("fence.acq_rel.cta;" ::: "memory");
}
void mb_sync(torch::Tensor line, int64_t mode) {
  u32* l = (u32*)line.data_ptr(); int m = (int)mode; void* a[] = {&l, &m};
  mb_launch((const void*)k_sync, dim3(1), dim3(32), 0, 1, 0, 0, a);
}

__global__ void k_cluster(int mode) {
  if (mode) asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
void mb_cluster(int64_t G, int64_t cs, int64_t mode) {
  int m = (int)mode; void* a[] = {&m};
  mb_launch((const void*)k_cluster, dim3(G), dim3(128), 0, (int)cs, 0, 0, a);
}

// random 128-B row gather: R rows per CTA, 8 lanes x 16 B per row, rolled issue loop, one cp.async round
__global__ void k_gather(const uint8_t* kv, u32 nrows, int R, float* sink) {
  extern __shared__ __align__(128) uint8_t sm[];
  const int tid = threadIdx.x, ch = tid & 7;
  for (int r = tid >> 3; r < R; r += blockDim.x >> 3) {
    u32 row = (u32)(((u32)(blockIdx.x * R + r) * 2654435761u) % nrows);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(mb_smem(sm + r * 128 + ch * 16)), "l"(kv + (size_t)row * 128 + ch * 16));
  }
  asm volatile("cp.async.wait_all;" ::: "memory");
  __syncthreads();
  float s = ((const float*)sm)[tid];
  if (s == 1.2345f) sink[blockIdx.x] = s;
}
void mb_gather(torch::Tensor kv, int64_t nrows, int64_t G, int64_t R, torch::Tensor sink) {
  const uint8_t* k = (const uint8_t*)kv.data_ptr(); u32 nr = (u32)nrows; int r = (int)R; float* s = (float*)sink.data_ptr();
  void* a[] = {&k, &nr, &r, &s};
  size_t smem = (size_t)R * 128;
  cudaFuncSetAttribute((const void*)k_gather, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  mb_launch((const void*)k_gather, dim3(G), dim3(256), smem, 1, 0, 0, a);
}

// last-CTA merge: G CTAs red.add.v4 a nf-float partial, acq_rel ticket, last CTA reads + writes the sum
__global__ void k_merge(float* ws, u32* cnt, float* out, int nf, int mode) {
  if (!mode) return;
  __shared__ int last;
  for (int i = threadIdx.x * 4; i < nf; i += blockDim.x * 4)
    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(ws + i), "f"(1.f), "f"(1.f), "f"(1.f), "f"(1.f) : "memory");
  __syncthreads();
  if (threadIdx.x == 0) {
    u32 old; asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;" : "=r"(old) : "l"(cnt) : "memory");
    last = old == gridDim.x - 1;
  }
  __syncthreads();
  if (!last) return;
  for (int i = threadIdx.x * 4; i < nf; i += blockDim.x * 4) {
    float4 v = __ldcg((const float4*)(ws + i));
    *(float4*)(out + i) = v;
  }
  if (threadIdx.x == 0) cnt[0] = 0u;
}
void mb_merge(torch::Tensor ws, torch::Tensor cnt, torch::Tensor out, int64_t G, int64_t nf, int64_t mode) {
  float* w = (float*)ws.data_ptr(); u32* c = (u32*)cnt.data_ptr(); float* o = (float*)out.data_ptr(); int n = (int)nf, m = (int)mode;
  void* a[] = {&w, &c, &o, &n, &m};
  mb_launch((const void*)k_merge, dim3(G), dim3(256), 0, 1, 0, 0, a);
}

// globaltimer tick: thread 0 spins and records the first n distinct timer values
__global__ void k_tick(u64* out, int n) {
  if (threadIdx.x) return;
  u64 prev = mb_gt();
  for (int i = 0; i < n; ) { u64 t = mb_gt(); if (t != prev) { out[i++] = t - prev; prev = t; } }
}
void mb_tick(torch::Tensor out, int64_t n) {
  u64* o = (u64*)out.data_ptr(); int nn = (int)n; void* a[] = {&o, &nn};
  mb_launch((const void*)k_tick, dim3(1), dim3(32), 0, 1, 0, 0, a);
}
"""

FUNCTIONS = ["mb_null", "mb_chase", "mb_stream", "mb_sync", "mb_cluster", "mb_gather", "mb_merge", "mb_tick"]


def _src_hash() -> str:
    return hashlib.sha256((CPP + CUDA).encode()).hexdigest()[:12]


def build_dir(home: str | None = None) -> str:
    """$KDEV_MICROBENCH_DIR or $KDEV_HOME/cache/microbench, keyed by the source hash."""
    root = os.environ.get("KDEV_MICROBENCH_DIR")
    h = Path(root).expanduser() if root else Path(home or os.environ.get("KDEV_HOME", "~/.kdev")).expanduser() / "cache" / "microbench"
    d = h / _src_hash()
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


def load_ext(arch: str = "90", verbose: bool = False) -> Any:
    """Compile (or load from the cache) the microbench extension."""
    from torch.utils.cpp_extension import load_inline
    return load_inline(name=f"kdev_mb_{_src_hash()}", cpp_sources=CPP, cuda_sources=CUDA, functions=FUNCTIONS,
                       extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}"],
                       build_directory=build_dir(), verbose=verbose)


def build_argv(python: str | None = None, arch: str = "90") -> list[str]:
    return [python or sys.executable, "-m", "kdev.floor.microbench", "--build", "--arch", arch]


class _Bench:
    def __init__(self, reps: int, flush_bytes: int | None) -> None:
        from kdev.observe import _rt
        self.rt = _rt
        self.reps = reps
        self.flush = _rt.make_flush(flush_bytes)

    def graphs(self, fns: list[Callable[[], Any]]) -> list[Any]:
        return [self.rt.capture(f) for f in fns]

    def time(self, fns: list[Callable[[], Any]], flush: bool = True) -> list[float]:
        """Interleaved (ABBA) graph replays; raw median us per variant."""
        gs = self.graphs(fns)
        ts: list[list[float]] = [[] for _ in gs]
        for r in range(self.reps + 1):
            idx = list(range(len(gs))) if r % 2 == 0 else list(reversed(range(len(gs))))
            for k in idx:
                t = self.rt.time_replay_us(gs[k], self.flush if flush else None)
                if r:
                    ts[k].append(t)
        return [self.rt.median(t) for t in ts]


def _measure(names: set[str], reps: int, flush_bytes: int | None, log: Callable[[str], None]) -> dict[str, Any]:
    import torch
    from kdev.floor.gpus import Measured
    cc = torch.cuda.get_device_capability()
    M = load_ext(arch=f"{cc[0]}{cc[1]}")
    B = _Bench(reps, flush_bytes)
    out: dict[str, Measured] = {}
    raw: dict[str, Any] = {}
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    how = f"graph replay after read flush, median of {reps} interleaved reps"

    def put(k: str, v: float, unit: str = "us", note: str = "") -> None:
        out[k] = Measured(value=round(float(v), 4), unit=unit, source="kdev floor calibrate", how=how + note)
        log(f"{k:24s} {v:9.3f} {unit}")

    x = torch.zeros(1, device="cuda")
    t_add, t_null = B.time([lambda: x.add_(1), lambda: M.mb_null(sms, 256, 0, 0)])
    raw.update(add_graph_raw=t_add, null_raw=t_null)
    if names & {"one_kernel_graph", "null_kernel"}:
        put("one_kernel_graph_us", t_add, note=" (raw, x.add_(1))")
        put("null_kernel_us", t_null - JUDGE_FIXED_US, note=" (minus 4.0)")
        put("null_kernel_rel_us", t_null - t_add, note=" (vs 1-kernel graph)")
    if "empty_graph" in names:
        try:
            put("empty_graph_us", B.time([lambda: None])[0] - JUDGE_FIXED_US, note=" (minus 4.0)")
        except Exception as e:  # noqa: BLE001 - empty capture is not supported everywhere
            raw["empty_graph_error"] = str(e)[:200]
    if names & {"dram_chase_1cta", "dram_chase_all", "l2_hit"}:
        nslots = (512 << 20) // 128
        perm = torch.randperm(nslots, device="cuda")
        buf = torch.zeros(nslots * 16, dtype=torch.int64, device="cuda")
        buf[perm * 16] = torch.roll(perm, -1) * 16
        o = torch.zeros(max(sms, 1), dtype=torch.int64, device="cuda")
        n = 16
        if "dram_chase_1cta" in names:
            a, b = B.time([lambda: M.mb_chase(buf, nslots, 0, 1, o), lambda: M.mb_chase(buf, nslots, n, 1, o)])
            put("dram_trip_1cta_us", (b - a) / n, note=f" ({n}-trip chase delta, 512 MB cycle)")
        if "dram_chase_all" in names:
            a, b = B.time([lambda: M.mb_chase(buf, nslots, 0, sms, o), lambda: M.mb_chase(buf, nslots, n, sms, o)])
            put("dram_trip_132cta_us", (b - a) / n, note=f" ({sms} CTAs)")
        if "l2_hit" in names:
            small = (8 << 20) // 128
            p2 = torch.randperm(small, device="cuda")
            b2 = torch.zeros(small * 16, dtype=torch.int64, device="cuda")
            b2[p2 * 16] = torch.roll(p2, -1) * 16
            a, b = B.time([lambda: M.mb_chase(b2, small, 0, 1, o), lambda: M.mb_chase(b2, small, n, 1, o)], flush=False)
            put("l2_hit_us", (b - a) / n, note=" (8 MB cycle, no flush)")
            del b2, p2
        del buf, perm
        torch.cuda.empty_cache()
    if names & {"stream_200mb", "kernel_boundary", "kernel_boundary_pdl"}:
        a = torch.ones((200 << 20) // 4, dtype=torch.float32, device="cuda")
        sink = torch.zeros(1, device="cuda")
        t1, t8, t8p = B.time([lambda: M.mb_stream(a, 1, 0, sink), lambda: M.mb_stream(a, 8, 0, sink),
                              lambda: M.mb_stream(a, 8, 1, sink)])
        raw.update(stream1_raw=t1, stream8_raw=t8, stream8_pdl_raw=t8p)
        if "stream_200mb" in names:
            put("stream_tbps", (200 << 20) / ((t1 - JUDGE_FIXED_US) * 1e-6) / 1e12, unit="TB/s",
                note=" (200 MiB, 1 kernel, minus 4.0)")
        if "kernel_boundary" in names:
            put("kernel_boundary_us", (t8 - t1) / 7, note=" (200 MiB as 8 serial kernels vs 1)")
        if "kernel_boundary_pdl" in names:
            put("pdl_boundary_us", (t8p - t1) / 7, note=" (8 PDL kernels, wait at entry)")
        del a
        torch.cuda.empty_cache()
    if "graph_node" in names:
        a1, a2 = B.time([lambda: M.mb_null(sms, 256, 0, 0), lambda: (M.mb_null(sms, 256, 0, 0), M.mb_null(sms, 256, 0, 0))])
        put("extra_graph_node_us", a2 - a1, note=" (second null kernel node)")
    if "coop_launch" in names:
        a1, a2 = B.time([lambda: M.mb_null(sms, 256, 0, 0), lambda: M.mb_null(sms, 256, 1, 0)])
        put("coop_launch_us", a2 - a1, note=" (cooperative attribute)")
    if names & {"membar_gpu", "atom_acqrel"}:
        line = torch.zeros(64, dtype=torch.int32, device="cuda")
        t0, t1_, t2, t3 = B.time([lambda m=m: M.mb_sync(line, m) for m in range(4)])
        if "membar_gpu" in names:
            put("membar_gpu_us", t1_ - t0, note=" (cold store + fence.acq_rel.gpu)")
            put("fence_cta_us", t3 - t0, note=" (fence.acq_rel.cta)")
        if "atom_acqrel" in names:
            put("atom_acqrel_cold_us", t2 - t0, note=" (atom.acq_rel.gpu on a cold line)")
    if "barrier_cluster" in names:
        g = (sms // 2) * 2
        a1, a2 = B.time([lambda: M.mb_cluster(g, 2, 0), lambda: M.mb_cluster(g, 2, 1)])
        put("barrier_cluster_us", a2 - a1, note=" (cluster 2, one barrier.cluster)")
    if "gather_round" in names:
        nrows = (704 << 20) // 128
        kv = torch.zeros(nrows * 128, dtype=torch.uint8, device="cuda")
        sink = torch.zeros(sms, device="cuda")
        a1, a2 = B.time([lambda: M.mb_null(sms, 256, 0, 0), lambda: M.mb_gather(kv, nrows, sms, 64, sink)])
        put("gather_round_8k_us", a2 - a1, note=f" ({sms} CTAs x 64 random 128-B rows, 704 MB pool)")
        del kv
        torch.cuda.empty_cache()
    if "merge_lastcta" in names:
        nf = 1024
        ws = torch.zeros(nf, device="cuda")
        cnt = torch.zeros(1, dtype=torch.int32, device="cuda")
        outb = torch.zeros(nf, device="cuda")
        a1, a2 = B.time([lambda: M.mb_merge(ws, cnt, outb, 33, nf, 0), lambda: M.mb_merge(ws, cnt, outb, 33, nf, 1)])
        put("merge_lastcta_us", a2 - a1, note=" (33 CTAs x 4 KB red.add.v4 + acq_rel ticket + last-CTA read)")
    if "globaltimer_tick" in names:
        ticks = torch.zeros(256, dtype=torch.int64, device="cuda")
        M.mb_tick(ticks, 256)
        torch.cuda.synchronize()
        v = sorted(int(t) for t in ticks.cpu().tolist() if t > 0)
        if v:
            put("globaltimer_tick_ns", v[len(v) // 2], unit="ns", note=" (median distinct-step of %globaltimer)")
    return {"measured": out, "raw": raw}


def run_calibrate(spec: JobSpec, run_dir: Any) -> Result:
    import torch
    from kdev.floor import gpus
    from kdev.floor._text import table
    from kdev.observe import _rt
    from kdev.observe.guard import check_clean
    check_clean("floor calibrate")              # CUPTI left on by a profiler would add 2-5 us per replay
    rd = _rt.run_path(run_dir)
    params = spec.params or {}
    extra = params.get("extra") or {}
    reps = int(extra.get("reps", params.get("reps", 30)) or 30)
    want = set(extra.get("benches") or BENCHES)
    unknown = want - set(BENCHES)
    name = extra.get("gpu") or gpus.detect_name(torch.cuda.get_device_name(0))
    log_lines: list[str] = []
    res = _measure(want - unknown, reps, params.get("flush_bytes"), log_lines.append)
    measured = res["measured"]
    try:
        prof = gpus.load_profile(name)
    except Exception:  # noqa: BLE001 - a new GPU has no profile yet
        prof = None
    rows = []
    for k in sorted(measured):
        old = prof.const(k) if prof else None
        m = measured[k]
        rows.append([k, f"{m.value:.3f}", m.unit, "-" if old is None else f"{old:.3f}",
                     "-" if old is None else f"{m.value - old:+.3f}"])
    files = {"measured": _rt.write_json(rd / "measured.json", {"gpu": name, "device": torch.cuda.get_device_name(0),
                                                                "measured": {k: vars(v) for k, v in measured.items()},
                                                                "raw": res["raw"]})}
    (rd / "calibrate.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    files["log_calibrate"] = str(rd / "calibrate.log")
    warnings = [f"unknown benches ignored: {', '.join(sorted(unknown))}"] if unknown else []
    if extra.get("write"):
        rid = getattr(run_dir, "run_id", "") or rd.name
        for m in measured.values():
            m.run = rid
        files["profile"] = gpus.write_measured(name, measured, os.environ.get("KDEV_HOME", str(Path.home() / ".kdev")))
    lines = [f"gpu {name} ({torch.cuda.get_device_name(0)}), {len(measured)} constants, reps {reps}"]
    lines += table(["constant", "value", "unit", "profile", "delta"], rows, max_rows=25, more_hint=files["measured"])
    return Result(tool="floor calibrate", verdict="ok", summary={"gpu": name, "n": len(measured),
                                                                 "written": bool(extra.get("write"))},
                  files=files, warnings=warnings, lines=lines,
                  next=[] if extra.get("write") else ["kdev floor calibrate --write  (store as $KDEV_HOME/gpus/<gpu>.toml)"])


def _main(argv: list[str]) -> int:
    if "--build" in argv:
        arch = argv[argv.index("--arch") + 1] if "--arch" in argv else "90"
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{arch[0]}.{arch[1:]}")
        load_ext(arch=arch, verbose="-v" in argv)
        print(f"built kdev microbench ({build_dir()})")
        return 0
    print("usage: python -m kdev.floor.microbench --build [--arch 90] [-v]", file=sys.stderr)
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_main(sys.argv[1:]))
