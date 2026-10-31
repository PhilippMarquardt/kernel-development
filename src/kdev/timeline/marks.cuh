// kdev timeline marks (device side). Owner: P6. Phase-0 contract; P6 implements the bodies.
//
// Usage in a kernel:   KD_MARK(3);        // phase id 0..KD_MAX_MARKS-1, recorded by one lane per CTA/warpgroup
// Host side:           kd_marks_init(ptr, capacity_records)  (exposed by the instrumented module)
// Zero cost when KDEV_MARKS is undefined: KD_MARK expands to nothing.
//
// Record layout (16 bytes, little endian), one per mark event:
//   uint64 globaltimer_ns;  uint32 clock64_lo;  uint16 cta_linear;  uint8 warpgroup;  uint8 mark_id
// Buffer: header {uint32 count; uint32 capacity; uint32 smid_bits; uint32 pad} followed by records; the
// write index is an atomicAdd on count; overflow drops records and is reported by the decoder.
#pragma once
#include <stdint.h>

#ifndef KD_MAX_MARKS
#define KD_MAX_MARKS 64
#endif

#ifdef KDEV_MARKS
struct kd_mark_rec { unsigned long long t_ns; unsigned int clk_lo; unsigned short cta; unsigned char wg; unsigned char id; };
struct kd_mark_hdr { unsigned int count; unsigned int capacity; unsigned int smid_bits; unsigned int pad; };
extern __device__ kd_mark_hdr* kd_marks_buf;   // defined once in the instrumented TU (P6)

__device__ __forceinline__ unsigned long long kd_globaltimer() {
  unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}
// P6: implement KD_MARK(id): lane 0 of each warpgroup (threadIdx.x % 128 == 0) reads globaltimer + clock,
// atomicAdd on kd_marks_buf->count, writes a kd_mark_rec if below capacity.
#define KD_MARK(id) do { /* P6 */ } while (0)
#else
#define KD_MARK(id) do { } while (0)
#endif
