// kdev timeline marks (device side). Owner: P6.
//
// Usage in a kernel:
//   KD_MARK(3);       // lane 0 of each warpgroup (linear tid % 128 == 0) records mark 3
//   KD_MARK_CTA(3);   // only thread 0 of the CTA records
//   KD_MARK_T(3);     // the calling thread records (caller guarantees one thread per warpgroup, e.g. a producer)
//   KD_MARK_SM();     // records the SM id of this CTA (mark id 255, SM id in the clk field)
// Ids are 0..KD_MAX_MARKS-1 (`kdev timeline` assigns them from `// @kdev:mark NAME` annotations).
// Zero cost when KDEV_MARKS is undefined: every macro expands to nothing.
//
// Host side: the instrumented TU exports  extern "C" int kd_marks_init(void* buf, unsigned capacity_records)
// (found with ctypes in the loaded extension .so); it writes the header and sets the device pointer. It must be
// called before the kernels run (a NULL pointer makes every mark a no-op, so warm-up launches are safe).
//
// Buffer = header {uint32 count; uint32 capacity; uint32 smid_bits; uint32 flags} + capacity records.
// Record (16 bytes, little endian, one vector store):
//   uint64 globaltimer_ns;  uint32 clock_lo;  uint16 cta_linear;  uint8 warpgroup;  uint8 mark_id
// Layouts (flags bit 0):
//   0 = per-CTA slots (default): record index = (cta * KD_MAX_WG + wg) * (KD_MAX_MARKS + 1) + id; no atomics,
//       one record per (CTA, warpgroup, mark) -- a mark inside a loop keeps its last iteration; `count` counts
//       dropped marks (CTA/warpgroup/id outside the slot range).
//   1 = append log (-DKD_MARKS_LOG): atomicAdd on `count` per mark (keeps every loop iteration; costs one
//       atomic round trip per mark); records beyond `capacity` are dropped (count - capacity).
// flags = 0x80000000 | KD_MAX_WG << 16 | KD_MAX_MARKS << 8 | layout. smid_bits is reserved (0).
// Timer reads use asm volatile with a "memory" clobber so the compiler does not move loads/stores across a
// mark (define KD_MARKS_LIGHT to drop the clobber). globaltimer ticks are 32 ns or 256 ns on H200 depending on
// the kernel; the decoder detects the tick and uses clock_lo deltas within a CTA for finer phase durations.
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
#define KD_MARK_SM_ID 255

#ifdef KDEV_MARKS
#ifdef KD_MARKS_LOG
#define KD_MARKS_LAYOUT 1u
#else
#define KD_MARKS_LAYOUT 0u
#endif
#define KD_MARKS_FLAGS (0x80000000u | ((unsigned int)(KD_MAX_WG) << 16) | ((unsigned int)(KD_MAX_MARKS) << 8) | KD_MARKS_LAYOUT)

struct kd_mark_rec { unsigned long long t_ns; unsigned int clk_lo; unsigned short cta; unsigned char wg; unsigned char id; };
struct kd_mark_hdr { unsigned int count; unsigned int capacity; unsigned int smid_bits; unsigned int flags; };

#ifndef KD_MARKS_NO_DEFINE
__constant__ kd_mark_hdr* kd_marks_buf = 0;
__constant__ unsigned int kd_marks_cap = 0;

extern "C" __attribute__((visibility("default"))) int kd_marks_init(void* buf, unsigned int capacity) {
  kd_mark_hdr h;
  h.count = 0; h.capacity = capacity; h.smid_bits = 0; h.flags = KD_MARKS_FLAGS;
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

__device__ __forceinline__ void kd_mark_write(unsigned int id) {
  unsigned long long t = kd_globaltimer();
  unsigned int clk = kd_clock_lo();
  kd_mark_hdr* h = kd_marks_buf;
  if (h == 0) return;
  const unsigned int cap = kd_marks_cap;
  const unsigned int cta = kd_cta_linear();
  const unsigned int wg = kd_tid_linear() >> 7;
  if (id == KD_MARK_SM_ID) clk = kd_smid();
  unsigned long long slot;
#ifdef KD_MARKS_LOG
  slot = atomicAdd(&h->count, 1u);
  if (slot >= cap) return;
#else
  const unsigned int mid = id == KD_MARK_SM_ID ? (unsigned int)(KD_MAX_MARKS) : id;
  slot = ((unsigned long long)cta * KD_MAX_WG + wg) * (unsigned long long)(KD_MAX_MARKS + 1) + mid;
  if (wg >= KD_MAX_WG || mid > (unsigned int)(KD_MAX_MARKS) || slot >= cap) { atomicAdd(&h->count, 1u); return; }
#endif
  const unsigned long long packed = (unsigned long long)clk | ((unsigned long long)(cta & 0xffffu) << 32)
                                  | ((unsigned long long)(wg & 0xffu) << 48) | ((unsigned long long)(id & 0xffu) << 56);
  ulonglong2* r = reinterpret_cast<ulonglong2*>(h + 1) + slot;
  *r = make_ulonglong2(t, packed);
}

#define KD_MARK(id) do { if ((kd_tid_linear() & 127u) == 0u) kd_mark_write((unsigned int)(id)); } while (0)
#define KD_MARK_CTA(id) do { if (kd_tid_linear() == 0u) kd_mark_write((unsigned int)(id)); } while (0)
#define KD_MARK_T(id) do { kd_mark_write((unsigned int)(id)); } while (0)
#define KD_MARK_SM() do { if ((kd_tid_linear() & 127u) == 0u) kd_mark_write(KD_MARK_SM_ID); } while (0)
#else
#define KD_MARK(id) do { } while (0)
#define KD_MARK_CTA(id) do { } while (0)
#define KD_MARK_T(id) do { } while (0)
#define KD_MARK_SM() do { } while (0)
#endif

#endif
