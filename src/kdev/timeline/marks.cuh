// kdev timeline marks (device side). Owner: P6.
//
// Usage in a kernel:
//   KD_MARK(3);       // lane 0 of each warpgroup (linear tid % 128 == 0) records mark 3
//   KD_MARK_CTA(3);   // only thread 0 of the CTA records
//   KD_MARK_T(3);     // the calling thread records (caller guarantees one thread per warpgroup, e.g. a producer)
//   KD_MARK_SM();     // records the SM id of this CTA (mark id 255, SM id in the clk field)
//   KD_MARK_I(3, v) / KD_MARK_CTA_I / KD_MARK_T_I   // timed mark carrying a 32-bit info word instead of clock_lo
//   KD_VAL_T(9, v) / KD_VAL_CTA(9, v)                // value record (not a time mark): e.g. accumulated wait cycles
//   KD_ACC_DECL(w); KD_ACC_BEGIN(w); KD_ACC_END(w); KD_ACC_FLUSH(w, 9);  // per-thread clock accumulator -> KD_VAL_T
// Ids are 0..KD_MAX_MARKS-1 (`kdev timeline` assigns them from `// @kdev:mark NAME` annotations).
// Zero cost when KDEV_MARKS is undefined: every macro expands to nothing.
//
// Host side: the instrumented TU exports  extern "C" int kd_marks_init(void* buf, unsigned capacity_records)
// (found with ctypes in the loaded extension .so); it writes the header and sets the device pointer. It must be
// called before the kernels run (a NULL pointer makes every mark a no-op, so warm-up launches are safe).
// kd_marks_init(NULL, 0) disables the marks and returns the flags (the host sizes the buffer from them).
//
// Buffer = header {uint32 count; uint32 capacity; uint32 aux; uint32 flags} + capacity records.
// Record (16 bytes, little endian, one vector store):
//   uint64 globaltimer_ns;  uint32 clock_lo (or info / value);  uint16 cta_linear;  uint8 wg | kind;  uint8 mark_id
//   kind bits in the wg byte: 0x40 = info record (timed, clk field = info word), 0x80 = value record (not timed).
// Layouts (flags bits 0-1):
//   0 = per-CTA slots (default): record index = (cta * KD_MAX_WG + wg) * (KD_MAX_MARKS + 1) + id; no atomics,
//       one record per (CTA, warpgroup, mark) -- a mark inside a loop keeps its last iteration; `count` counts
//       dropped marks (CTA/warpgroup/id outside the slot range).
//   1 = append log (-DKD_MARKS_LOG): atomicAdd on `count` per mark (keeps every loop iteration; costs one
//       atomic round trip per mark); records beyond `capacity` are dropped (count - capacity).
//   2 = per-(CTA, warpgroup) log (-DKD_MARKS_CTALOG): region (cta * KD_MAX_WG + wg) of KD_CTA_RECS records,
//       record 0 = counter. Every iteration is kept; the single writer of a region (lane 0 / thread 0) bumps the
//       counter with a plain load/store (no atomic round trip); KD_MARK_T* use an atomicAdd (several threads of
//       one warpgroup may record). aux = KD_CTA_RECS; overflow is counted in `count`.
// flags = 0x80000000 | KD_MAX_WG << 16 | KD_MAX_MARKS << 8 | log2(KD_CTA_RECS) << 2 | layout.
// Timer reads use asm volatile with a "memory" clobber so the compiler does not move loads/stores across a
// mark (define KD_MARKS_LIGHT to drop the clobber). globaltimer ticks are 32 ns or 256 ns on H200 depending on
// the kernel; the decoder detects the tick and uses clock_lo deltas within a CTA for finer phase durations.
//
// Wait helpers (always defined, also without KDEV_MARKS): kd_ld_relaxed / kd_ld_acquire / kd_wait_geq.
// LESSON (mtn_, board #179): a weak load such as ld.global.cg inside an asm volatile spin loop is HOISTED out
// of the loop by ptxas -> the loop spins forever on a stale value. Poll with ld.relaxed.gpu / ld.acquire.gpu
// (or ld.volatile) and bound every spin with a globaltimer timeout that traps, never an unbounded while.
// No backslashes in this file: kdev inlines it into Python string literals of load_inline sources.
#pragma once
#ifndef KDEV_MARKS_CUH
#define KDEV_MARKS_CUH

#ifndef KD_MAX_MARKS
#define KD_MAX_MARKS 32
#endif
#ifndef KD_MAX_WG
#define KD_MAX_WG 8
#endif
#ifndef KD_CTA_RECS_LOG2
#define KD_CTA_RECS_LOG2 8
#endif
#define KD_CTA_RECS (1u << (KD_CTA_RECS_LOG2))
#define KD_MARK_SM_ID 255
#define KD_KIND_INFO 0x40u
#define KD_KIND_VALUE 0x80u

__device__ __forceinline__ int kd_ld_relaxed(const int* a) {
  int v; asm volatile("ld.relaxed.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(a) : "memory"); return v;
}
__device__ __forceinline__ int kd_ld_acquire(const int* a) {
  int v; asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(a) : "memory"); return v;
}
__device__ __forceinline__ unsigned long long kd_gtimer_raw() {
  unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) :: "memory"); return t;
}
// Spin until *a >= target (acquire on success). Traps after timeout_ns (default 2 s: never in a correct run).
__device__ __forceinline__ int kd_wait_geq(const int* a, int target, unsigned long long timeout_ns = 2000000000ull) {
  int v = kd_ld_acquire(a);
  if (v >= target) return v;
  const unsigned long long t0 = kd_gtimer_raw();
  while (true) {
    __nanosleep(32);
    v = kd_ld_relaxed(a);
    if (v >= target) break;
    if (kd_gtimer_raw() - t0 > timeout_ns) __trap();
  }
  asm volatile("fence.acq_rel.gpu;" ::: "memory");
  return v;
}

#ifdef KDEV_MARKS
#if defined(KD_MARKS_CTALOG)
#define KD_MARKS_LAYOUT 2u
#elif defined(KD_MARKS_LOG)
#define KD_MARKS_LAYOUT 1u
#else
#define KD_MARKS_LAYOUT 0u
#endif
#define KD_MARKS_FLAGS (0x80000000u | ((unsigned int)(KD_MAX_WG) << 16) | ((unsigned int)(KD_MAX_MARKS) << 8) | ((unsigned int)(KD_CTA_RECS_LOG2) << 2) | KD_MARKS_LAYOUT)

struct kd_mark_rec { unsigned long long t_ns; unsigned int clk_lo; unsigned short cta; unsigned char wg; unsigned char id; };
struct kd_mark_hdr { unsigned int count; unsigned int capacity; unsigned int aux; unsigned int flags; };

#ifndef KD_MARKS_NO_DEFINE
__constant__ kd_mark_hdr* kd_marks_buf = 0;
__constant__ unsigned int kd_marks_cap = 0;

extern "C" __attribute__((visibility("default"))) int kd_marks_init(void* buf, unsigned int capacity) {
  kd_mark_hdr h;
  h.count = 0; h.capacity = capacity; h.aux = KD_CTA_RECS; h.flags = KD_MARKS_FLAGS;
  if (buf != 0 && cudaMemcpy(buf, &h, sizeof(h), cudaMemcpyHostToDevice) != cudaSuccess) return -1;
  kd_mark_hdr* p = (kd_mark_hdr*)buf;
  unsigned int cap = buf != 0 ? capacity : 0u;
  if (cudaMemcpyToSymbol(kd_marks_buf, &p, sizeof(p)) != cudaSuccess) return -2;
  if (cudaMemcpyToSymbol(kd_marks_cap, &cap, sizeof(cap)) != cudaSuccess) return -3;
  if (cudaDeviceSynchronize() != cudaSuccess) return -4;
  return (int)(KD_MARKS_FLAGS & 0x7fffffffu);
}
#else
extern __constant__ kd_mark_hdr* kd_marks_buf;
extern __constant__ unsigned int kd_marks_cap;
#endif

#ifdef KD_MARKS_LIGHT
__device__ __forceinline__ unsigned long long kd_globaltimer() {
  unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}
__device__ __forceinline__ unsigned int kd_clock_lo() {
  unsigned int c; asm volatile("mov.u32 %0, %%clock;" : "=r"(c)); return c;
}
#else
__device__ __forceinline__ unsigned long long kd_globaltimer() {
  unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) :: "memory"); return t;
}
__device__ __forceinline__ unsigned int kd_clock_lo() {
  unsigned int c; asm volatile("mov.u32 %0, %%clock;" : "=r"(c) :: "memory"); return c;
}
#endif
__device__ __forceinline__ unsigned int kd_smid() {
  unsigned int s; asm volatile("mov.u32 %0, %%smid;" : "=r"(s)); return s;
}
__device__ __forceinline__ unsigned int kd_cta_linear() {
  return blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
}
__device__ __forceinline__ unsigned int kd_tid_linear() {
  return threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
}

// kind: 0 = timed mark (clk = clock_lo), KD_KIND_INFO = timed + info word, KD_KIND_VALUE = value record.
// shared_writer: several threads of one warpgroup may call (KD_MARK_T* / KD_VAL_T): the region log uses an atomic.
__device__ __forceinline__ void kd_mark_write2(unsigned int id, unsigned int word, unsigned int kind, bool shared_writer) {
  unsigned long long t = kd_globaltimer();
  unsigned int clk = kind ? word : kd_clock_lo();
  kd_mark_hdr* h = kd_marks_buf;
  if (h == 0) return;
  const unsigned int cap = kd_marks_cap;
  const unsigned int cta = kd_cta_linear();
  const unsigned int wg = kd_tid_linear() >> 7;
  if (id == KD_MARK_SM_ID) clk = kd_smid();
  unsigned long long slot;
#if KD_MARKS_LAYOUT == 1u
  (void)shared_writer;
  slot = atomicAdd(&h->count, 1u);
  if (slot >= cap) return;
#elif KD_MARKS_LAYOUT == 2u
  const unsigned long long region = ((unsigned long long)cta * KD_MAX_WG + wg) * (unsigned long long)KD_CTA_RECS;
  if (wg >= KD_MAX_WG || region + KD_CTA_RECS > cap) { atomicAdd(&h->count, 1u); return; }
  unsigned int* ctr = reinterpret_cast<unsigned int*>(reinterpret_cast<ulonglong2*>(h + 1) + region);
  unsigned int n;
  if (shared_writer) { n = atomicAdd(ctr, 1u); } else { n = *ctr; *ctr = n + 1u; }
  if (n + 1u >= KD_CTA_RECS) { atomicAdd(&h->count, 1u); return; }
  slot = region + 1u + n;
#else
  (void)shared_writer;
  const unsigned int mid = id == KD_MARK_SM_ID ? (unsigned int)(KD_MAX_MARKS) : id;
  slot = ((unsigned long long)cta * KD_MAX_WG + wg) * (unsigned long long)(KD_MAX_MARKS + 1) + mid;
  if (wg >= KD_MAX_WG || mid > (unsigned int)(KD_MAX_MARKS) || slot >= cap) { atomicAdd(&h->count, 1u); return; }
#endif
  const unsigned long long packed = (unsigned long long)clk | ((unsigned long long)(cta & 0xffffu) << 32)
                                  | ((unsigned long long)(((wg & 0x3fu) | kind) & 0xffu) << 48)
                                  | ((unsigned long long)(id & 0xffu) << 56);
  ulonglong2* r = reinterpret_cast<ulonglong2*>(h + 1) + slot;
  *r = make_ulonglong2(t, packed);
}
__device__ __forceinline__ void kd_mark_write(unsigned int id) { kd_mark_write2(id, 0u, 0u, false); }

#define KD_MARK(id) do { if ((kd_tid_linear() & 127u) == 0u) kd_mark_write((unsigned int)(id)); } while (0)
#define KD_MARK_CTA(id) do { if (kd_tid_linear() == 0u) kd_mark_write((unsigned int)(id)); } while (0)
#define KD_MARK_T(id) do { kd_mark_write2((unsigned int)(id), 0u, 0u, true); } while (0)
#define KD_MARK_SM() do { if ((kd_tid_linear() & 127u) == 0u) kd_mark_write(KD_MARK_SM_ID); } while (0)
#define KD_MARK_I(id, v) do { if ((kd_tid_linear() & 127u) == 0u) kd_mark_write2((unsigned int)(id), (unsigned int)(v), KD_KIND_INFO, false); } while (0)
#define KD_MARK_CTA_I(id, v) do { if (kd_tid_linear() == 0u) kd_mark_write2((unsigned int)(id), (unsigned int)(v), KD_KIND_INFO, false); } while (0)
#define KD_MARK_T_I(id, v) do { kd_mark_write2((unsigned int)(id), (unsigned int)(v), KD_KIND_INFO, true); } while (0)
#define KD_VAL_T(id, v) do { kd_mark_write2((unsigned int)(id), (unsigned int)(v), KD_KIND_VALUE, true); } while (0)
#define KD_VAL_CTA(id, v) do { if (kd_tid_linear() == 0u) kd_mark_write2((unsigned int)(id), (unsigned int)(v), KD_KIND_VALUE, false); } while (0)
#define KD_ACC_DECL(n) unsigned int kd_acc_##n = 0u, kd_acc0_##n = 0u
#define KD_ACC_BEGIN(n) do { kd_acc0_##n = kd_clock_lo(); } while (0)
#define KD_ACC_END(n) do { kd_acc_##n += kd_clock_lo() - kd_acc0_##n; } while (0)
#define KD_ACC_FLUSH(n, id) do { KD_VAL_T(id, kd_acc_##n); } while (0)
#else
#define KD_MARK(id) do { } while (0)
#define KD_MARK_CTA(id) do { } while (0)
#define KD_MARK_T(id) do { } while (0)
#define KD_MARK_SM() do { } while (0)
#define KD_MARK_I(id, v) do { } while (0)
#define KD_MARK_CTA_I(id, v) do { } while (0)
#define KD_MARK_T_I(id, v) do { } while (0)
#define KD_VAL_T(id, v) do { } while (0)
#define KD_VAL_CTA(id, v) do { } while (0)
#define KD_ACC_DECL(n) do { } while (0)
#define KD_ACC_BEGIN(n) do { } while (0)
#define KD_ACC_END(n) do { } while (0)
#define KD_ACC_FLUSH(n, id) do { } while (0)
#endif

#endif
