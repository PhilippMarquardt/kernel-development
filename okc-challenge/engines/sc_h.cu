// sc_h.cu (pm2/hard): pm2/d1/sc.cu 4c86fd12 + ss word re-read after count == G. Build with -DSAFE=1. -DTOC=1 test build reports spin timeouts in dbg. No smem source for cp.async here.
// sc.cu: d1-only chain (M = 1, N = 1 on every route). The S-C red path failed its falsifier (logs/f3.log), so this is the task's named alternative:
// same order as S-R (A then B) with B staged ahead so issue never stops, and 72 KB B jobs. No cluster: every CTA owns 1/128 of each expert and K is split among its warps.
// Layout per expert per CTA: P1 job = 64 rows x 4096 K = 4 stages of 36 KB (warp w: row set w&3, K half w>>2); P2 job = 32 rows x 4096 K = 2 stages (warp w: row set w&1, K quarter w>>1).
// Order: P1(e0) .. P1(eE-1) P2(e0) .. P2(eE-1). Weight producer only ever waits for ring slots.
// x (8 KB) resident before the first weight stage. Inter is stored bf16 in global, row copied by TMA once per expert, sentinel scanned. P1 counter per expert: relaxed.
// y of a CTA's 32 rows stays in its smem (each CTA owns its rows, so no cross-CTA reduction of y). ss + count: one red.v4 word, polled. No MEMBAR after the mbarrier init.
// Cleaning happens at call start (sentinels, counters), nothing at the end. All code ours; stage() and dq4 come from pm2/sr/sr.cu.
typedef unsigned int u32;
typedef unsigned long long u64;
typedef unsigned short u16;

#ifndef NS
#define NS 3
#endif
#define NSM 2
#ifndef SAFE
#define SAFE 0    // 1: no reliance on timing: two banks of counters/ss zeroed by CTA 0 at the end of the previous call, writer read-back instead of the sentinel scan
#endif
#ifndef HOLD
#define HOLD 0    // 1: first weight stage of the last P2 waits until its inter row is issued, so the row is not queued behind it
#endif
#ifndef SCAN32
#define SCAN32 0  // 1: scan one word per 16 B store (a st.v4 is one transaction)
#endif
#ifndef EARLY
#define EARLY 0   // row copy issued when this many CTAs are still missing; the lanes fetch the holes
#endif
#define NXB 4
#define NCW 8
#define PW NCW
#define XW (NCW + 1)
#define EW (NCW + 2)
#define NTHR ((NCW + 3) * 32)
#define HID 4096
#define STAGE 36864
#define WSL 4608
#define HT13 (4 * STAGE)
#define HT2 (2 * STAGE)
#define W13E (128 * HT13)
#define W2E (128 * HT2)
#define MS 0xFFFFFFFFu

struct Params {
  const int* ids;
  const float* tw;
  const u16* x;
  u16* out;
  const unsigned char* w13;  // [32][128][4][8][4608]
  const unsigned char* w2;   // [32][128][2][8][4608]
  const float* pw1;
  u16* inter;   // [8][4096]
  float* ssw;   // [3 words][4], [0..2] ss, [3] count
  int* cnt;     // [e * 32]
  u64* dbg;
  float gs13[32], gs2[32];
  int M, G;
};

struct Tab {
  u64 full[NS], empty[NS], xfirst, xfull[NXB], xready[NXB], rfull[NSM];
  int edone, E, rowi, bank;
  int lst[8];
  float twe[8], ssl[9], sse[9];
  __align__(16) u16 ib[32];
  __align__(16) float P[NSM][8][8][2];
  float y[8][32];
};

#define OFF_X (NS * STAGE)
#define OFF_I (OFF_X + 8192)
#define OFF_PW (OFF_I + NXB * 8192)
#define OFF_TAB (OFF_PW + 128)
#define SMEM_BYTES (OFF_TAB + (int)sizeof(Tab))
#define SPIN 400000
#ifndef SSRR
#define SSRR 1         // 1: ss word read once more after its count == G was seen
#endif
#ifndef TOC
#define TOC 0          // test build: every spin timeout adds 1 to int slot of p.dbg (site id); slot 30 barrier waits, 31 copies redone, 29 read-back given up
#endif
#ifndef TOCPC
#define TOCPC 0        // test: ss poll waits for G + TOCPC, positive control of the TOC counter
#endif
#if TOC
#define TOC_(id_) red_rlx(reinterpret_cast<int*>(p.dbg) + (id_), 1)
#else
#define TOC_(id_) ((void)0)
#endif
#define SPIN_OK(it_, id_) ((it_) < SPIN || (TOC_(id_), false))

__device__ __forceinline__ u32 smem_u32(const void* p) { u32 a; asm volatile("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }" : "=r"(a) : "l"(p)); return a; }
__device__ __forceinline__ void mb_init(u64* b, int n) { asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(b)), "r"(n)); }
__device__ __forceinline__ void mb_expect(u64* b, u32 tx) { asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(b)), "r"(tx) : "memory"); }
__device__ __forceinline__ void mb_arrive(u64* b) { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(b)) : "memory"); }
#if TOC
// %0 = polls left, 0 only after a timeout
__device__ __forceinline__ u32 mb_wait_r(u32 a, u32 par, u32 ns) {
  u32 c_;
  asm volatile(
      "{ .reg .pred p; mov.u32 %0, 400000; W_%=: mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; @p bra D_%=; nanosleep.u32 %3; sub.u32 %0, %0, 1; setp.ne.u32 p, %0, 0; @p bra W_%=; D_%=: }" : "=&r"(c_) : "r"(a), "r"(par), "r"(ns) : "memory");
  return c_;
}
#define mb_wait(b_, par_) do { if (mb_wait_r(smem_u32(b_), (par_), 1u) == 0u) TOC_(30); } while (0)
#define mb_wait_s(b_, par_, ns_) do { if (mb_wait_r(smem_u32(b_), (par_), (ns_)) == 0u) TOC_(30); } while (0)
#define mb_wait_a(a_, par_) do { if (mb_wait_r((a_), (par_), 1u) == 0u) TOC_(30); } while (0)
#endif
#if !TOC
__device__ __forceinline__ void mb_wait(u64* b, u32 par) {
  u32 a = smem_u32(b);
  asm volatile(
      "{ .reg .pred p; .reg .u32 c; mov.u32 c, 400000; W_%=: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1; @p bra D_%=; sub.u32 c, c, 1; setp.ne.u32 p, c, 0; @p bra W_%=; D_%=: }" ::"r"(a), "r"(par)
      : "memory");
}
#endif
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, u32 bytes, u64* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ uint4 lds128(u32 a) { uint4 v; asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(a)); return v; }
// lane g == 0 only: only the n = 0 column of B is non-zero at N = 1, no branch in the MMA block
__device__ __forceinline__ void lds128p(u32 a, u32 g, uint4& v) {
  asm volatile("{ .reg .pred p; setp.eq.u32 p, %5, 0; @p ld.shared.v4.b32 {%0,%1,%2,%3}, [%4]; }" : "+r"(v.x), "+r"(v.y), "+r"(v.z), "+r"(v.w) : "r"(a), "r"(g));
}
__device__ __forceinline__ void sts128(u32 a, u32 x, u32 y, u32 z, u32 w) { asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(a), "r"(x), "r"(y), "r"(z), "r"(w) : "memory"); }
__device__ __forceinline__ uint4 ldg4(const void* p) { uint4 v; asm volatile("ld.relaxed.gpu.global.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory"); return v; }
// any 16-bit half equal to 0xFFFF: not yet visible
__device__ __forceinline__ bool hole16(const uint4& v) {
  const u32 t0 = ~v.x, t1 = ~v.y, t2 = ~v.z, t3 = ~v.w;
  return (((t0 - 0x00010001u) & ~t0) | ((t1 - 0x00010001u) & ~t1) | ((t2 - 0x00010001u) & ~t2) | ((t3 - 0x00010001u) & ~t3)) & 0x80008000u;
}
__device__ __forceinline__ void stg_v4(void* p, u32 x, u32 y, u32 z, u32 w) { asm volatile("st.global.v4.b32 [%0], {%1,%2,%3,%4};" ::"l"(p), "r"(x), "r"(y), "r"(z), "r"(w) : "memory"); }
__device__ __forceinline__ int ld_sm_rlx(const int* p) { int v; asm volatile("ld.relaxed.cta.shared::cta.b32 %0, [%1];" : "=r"(v) : "r"(smem_u32(p)) : "memory"); return v; }
__device__ __forceinline__ int ld_rlx(const int* p) { int v; asm volatile("ld.relaxed.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void red_rlx(int* p, int v) { asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;" ::"l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ void red4f(float* p, float a, float b, float c, float d) { asm volatile("red.relaxed.gpu.global.add.v4.f32 [%0], {%1,%2,%3,%4};" ::"l"(p), "f"(a), "f"(b), "f"(c), "f"(d) : "memory"); }
__device__ __forceinline__ float4 ld4f(const float* p) { float4 v; asm volatile("ld.relaxed.gpu.global.v4.b32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ u32 bmul2_lo(u32 a, u32 b) { u32 d; asm("{ .reg .b16 l, h; .reg .b32 t; mov.b32 {l, h}, %2; mov.b32 t, {l, l}; mul.rn.bf16x2 %0, %1, t; }" : "=r"(d) : "r"(a), "r"(b)); return d; }
__device__ __forceinline__ u32 bmul2_hi(u32 a, u32 b) { u32 d; asm("{ .reg .b16 l, h; .reg .b32 t; mov.b32 {l, h}, %2; mov.b32 t, {h, h}; mul.rn.bf16x2 %0, %1, t; }" : "=r"(d) : "r"(a), "r"(b)); return d; }
__device__ __forceinline__ u32 prmt(u32 a, u32 b, u32 s) { u32 d; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s)); return d; }
__device__ __forceinline__ void mma16816(float* c, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ u16 f2bf(float f) { u16 h; asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(h) : "f"(f)); return h; }
__device__ __forceinline__ u64 gtime() { u64 t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }

#ifdef SC_TL
#define TLH 32
#define TLX(i) if (p.dbg) p.dbg[blockIdx.x * TLH + (i)] = gtime()
#else
#define TLX(i)
#endif

__device__ __forceinline__ void dq4(u32 q, u32& r0, u32& r1, u32& r2, u32& r3) {
  const u32 M = 0x81C081C0u;
  const u32 s1 = q << 2, s2 = q << 6, s3 = q >> 6, s4 = __funnelshift_l(q, q, 12);
  r0 = q & M;
  r1 = (s1 ^ s2) & M;
  r2 = (s1 ^ s3) & M;
  r3 = (s1 ^ s4) & M;
}

// One warp, one stage: 16 rows x 512 K. Permuted K: lane t stays in scale block t for 4 steps. B only on lanes g == 0.
__device__ __forceinline__ void stage(u32 cb, u32 sb, u32 xb, float (&acc)[2][4], u32 k16, u32 g) {
  const uint4 sv4 = lds128(sb);
  uint4 u0 = make_uint4(0u, 0u, 0u, 0u), u1 = u0;
#pragma unroll
  for (int q = 0; q < 8; ++q) {
    const uint4 wq = lds128(cb + q * 512);
    lds128p(xb + q * 128, g, u0);
    lds128p(xb + q * 128 + 16, g, u1);
    const u32 ww[4] = {wq.x, wq.y, wq.z, wq.w};
    const u32 bb[8] = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};
    const u32 sw = (q >> 1) == 0 ? sv4.x : (q >> 1) == 1 ? sv4.y : (q >> 1) == 2 ? sv4.z : sv4.w;
    const u32 sv = prmt(sw, 0u, (q & 1) ? 0x4342u : 0x4140u) * k16 + 0x77007700u;
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) {
      u32 r0, r1, r2, r3;
      dq4(ww[jj], r0, r1, r2, r3);
      r0 = bmul2_lo(r0, sv);
      r2 = bmul2_lo(r2, sv);
      r1 = bmul2_hi(r1, sv);
      r3 = bmul2_hi(r3, sv);
      mma16816(acc[jj & 1], r0, r1, r2, r3, bb[jj * 2], bb[jj * 2 + 1]);
    }
  }
}

extern "C" __global__ void __launch_bounds__(NTHR, 1) sc_moe(Params p) {
  extern __shared__ __align__(128) unsigned char sm[];
  Tab& T = *reinterpret_cast<Tab*>(sm + OFF_TAB);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, c = blockIdx.x;
  TLX(0);
  if (tid == 0) {
    for (int s = 0; s < NS; ++s) { mb_init(&T.full[s], 1); mb_init(&T.empty[s], NCW); }
    mb_init(&T.xfirst, 1);
    for (int s = 0; s < NXB; ++s) { mb_init(&T.xfull[s], 1); mb_init(&T.xready[s], 1); }
    for (int s = 0; s < NSM; ++s) mb_init(&T.rfull[s], NCW);
    T.edone = 0;
    T.rowi = 0;
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  // ---- x first, then the routing read; both issued at once
  if (warp == XW) {
    if (lane == 0) {
      mb_expect(&T.xfirst, 8192 + 128);
      bulk_g2s(sm + OFF_X, p.x, 8192, &T.xfirst);
      bulk_g2s(sm + OFF_PW, p.pw1 + 32 * c, 128, &T.xfirst);
      const int4 a = __ldg(reinterpret_cast<const int4*>(p.ids));
      const int4 b = __ldg(reinterpret_cast<const int4*>(p.ids + 4));
      const float4 ta = __ldg(reinterpret_cast<const float4*>(p.tw));
      const float4 tb = __ldg(reinterpret_cast<const float4*>(p.tw + 4));
      const int v8[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
      const float t8[8] = {ta.x, ta.y, ta.z, ta.w, tb.x, tb.y, tb.z, tb.w};
      int n = 0;
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        if ((u32)v8[k] < 32u) { T.lst[n] = v8[k]; T.twe[n] = t8[k]; ++n; }
      }
      T.E = n;
#if SAFE
      T.bank = ld_rlx(p.cnt + 1000);
#endif
    }
  }
  __syncthreads();
  TLX(1);
  const int E = T.E;
#if SAFE
  const int bank = T.bank;
#else
  const int bank = 0;
#endif
  int* const cnt_ = p.cnt + 256 * bank;
  float* const ssw_ = p.ssw + 12 * bank;
  if (warp == PW) {
    // ---------------------------------------------------------------- weight producer: never waits on anything but ring slots
    if (lane == 0) {
      const int tot = 6 * E;
      int slot = 0;
      u32 par = 0;
      int lap0 = 1;
      for (int i = 0; i < tot; ++i) {
        const unsigned char* src;
        if (i < 4 * E) src = p.w13 + (size_t)T.lst[i >> 2] * W13E + (size_t)c * HT13 + (size_t)(i & 3) * STAGE;
        else { const int u = i - 4 * E; src = p.w2 + (size_t)T.lst[u >> 1] * W2E + (size_t)c * HT2 + (size_t)(u & 1) * STAGE; }
#if HOLD
        if (i == tot - 2) { for (int it = 0; SPIN_OK(it, 1); ++it) if (ld_sm_rlx(&T.rowi) >= E) break; }
#endif
        if (!lap0) mb_wait(&T.empty[slot], (par + 1) & 1);
        mb_expect(&T.full[slot], STAGE);
        bulk_g2s(sm + slot * STAGE, src, STAGE, &T.full[slot]);
        if (++slot == NS) { slot = 0; par ^= 1; lap0 = 0; }
      }
    }
    return;
  }
  if (warp == XW) {
    // ---------------------------------------------------------------- call-start cleaning, then inter rows
#if !SAFE
    if (lane < 4) {
      for (int ei = 0; ei < E; ++ei) stg_v4(p.inter + (size_t)ei * HID + 32 * c + 8 * lane, MS, MS, MS, MS);
    }
    if (c == 0 && lane == 0) {
      for (int i = 0; i < 3; ++i) stg_v4(p.ssw + 4 * i, 0u, 0u, 0u, 0u);
      for (int i = 0; i < E; ++i) p.cnt[i * 32] = 0;
    }
#endif
    const u32 x0 = smem_u32(sm + OFF_I);
    for (int ei = 0; ei < E; ++ei) {
      const int b = ei & (NXB - 1);
      if (lane == 0) {
        // own P1 job first: no global poll before that, and by then CTA 0's call-start zero of the counter has long landed (a stale count from the last call can't be read)
        for (int it = 0; SPIN_OK(it, 2); ++it) if (ld_sm_rlx(&T.edone) > ei) break;
        // E > NXB: the row slot is reused, wait until P2 of the expert that holds it is through the epilogue
        if (ei >= NXB) { for (int it = 0; SPIN_OK(it, 3); ++it) if (ld_sm_rlx(&T.edone) > E + ei - NXB) break; }
        for (int it = 0; SPIN_OK(it, 4); ++it) if (ld_rlx(cnt_ + ei * 32) >= p.G - EARLY) break;
        TLX(6);
      }
      // one row copy; a hole (store still on its way) is fetched by that lane alone, no re-copy
      if (lane == 0) {
        asm volatile("fence.proxy.async.global;" ::: "memory");
        mb_expect(&T.xfull[b], 8192);
        bulk_g2s(sm + OFF_I + b * 8192, p.inter + (size_t)ei * HID, 8192, &T.xfull[b]);
        *const_cast<volatile int*>(&T.rowi) = ei + 1;
        mb_wait(&T.xfull[b], (u32)(ei >> 2) & 1);
      }
      __syncwarp();
#if !SAFE
      for (int j = lane; j < 512; j += 32) {
        const u32 ad = x0 + b * 8192 + j * 16;
#if SCAN32
        u32 w0_;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(w0_) : "r"(ad));
        if (((~w0_ - 0x00010001u) & w0_ & 0x80008000u) == 0u) continue;
        uint4 v = lds128(ad);
#else
        uint4 v = lds128(ad);
#endif
        if (hole16(v)) {
#ifdef SC_TL
          if (p.dbg) { const u64 one_ = 1; asm volatile("red.relaxed.gpu.global.add.u64 [%0], %1;" ::"l"(p.dbg + c * TLH + 20), "l"(one_) : "memory"); }
#endif
          for (int it = 0; SPIN_OK(it, 5) && hole16(v); ++it) v = ldg4(p.inter + (size_t)ei * HID + j * 8);
          sts128(ad, v.x, v.y, v.z, v.w);
        }
      }
      __syncwarp();
#endif
      if (lane == 0) { mb_arrive(&T.xready[b]); TLX(7); }
    }
    return;
  }
  if (warp == EW) {
    // ---------------------------------------------------------------- epilogue warp
    const u32 Pb = smem_u32(&T.P[0][0][0][0]);
    for (int j = 0; j < 2 * E; ++j) {
      const int ph2 = j >= E, ei = ph2 ? j - E : j, e = T.lst[ei], sl = j & (NSM - 1);
      mb_wait(&T.rfull[sl], (u32)(j >> 1) & 1);
      if (!ph2) {
        const int rs = lane >> 3, gg = lane & 7;
        const float gs = p.gs13[e];
        const float2 a = *reinterpret_cast<const float2*>(&T.P[sl][rs][gg][0]), b = *reinterpret_cast<const float2*>(&T.P[sl][rs + 4][gg][0]);
        const float g0 = (a.x + b.x) * gs, u0 = (a.y + b.y) * gs;
        T.ib[lane] = f2bf(g0 / (1.f + expf(-g0)) * u0);
        __syncwarp();
        if (lane < 4) {
          const uint4 v = lds128(smem_u32(&T.ib[0]) + lane * 16);
          u16* ad_ = p.inter + (size_t)ei * HID + 32 * c + 8 * lane;
          stg_v4(ad_, v.x, v.y, v.z, v.w);
#if SAFE
          // read back: an equal old value can only come from an identical input, so either value is right
          for (int it = 0; SPIN_OK(it, 6); ++it) { const uint4 r_ = ldg4(ad_); if (r_.x == v.x && r_.y == v.y && r_.z == v.z && r_.w == v.w) break; }
#endif
        }
#if SAFE
        __syncwarp();
#endif
        if (lane == 0) { red_rlx(cnt_ + ei * 32, 1); if (ei == E - 1) TLX(5); }
      } else {
        const int rs = lane >> 4, m = lane & 15, gg = m & 7, h = m >> 3;
        float v = 0.f;
#pragma unroll
        for (int k = 0; k < 4; ++k) v += T.P[sl][2 * k + rs][gg][h];
        const float yv = v * p.gs2[e] * reinterpret_cast<const float*>(sm + OFF_PW)[lane];
        T.y[ei][lane] = yv;
        float q = yv * yv;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) q += __shfl_xor_sync(0xffffffffu, q, o);
        if (lane == 0) T.ssl[ei] = q;
      }
      __syncwarp();
      if (lane == 0) *const_cast<volatile int*>(&T.edone) = j + 1;
    }
    TLX(11);
    (void)Pb;
    // ---- ss + count travel in one red.v4 (3 experts per word), no fence: y is local
    const int nw = (E + 2) / 3;
    if (lane < nw) {
      const int wd = lane;
      const float s0 = wd * 3 + 0 < E ? T.ssl[wd * 3] : 0.f, s1 = wd * 3 + 1 < E ? T.ssl[wd * 3 + 1] : 0.f, s2 = wd * 3 + 2 < E ? T.ssl[wd * 3 + 2] : 0.f;
      red4f(ssw_ + 4 * wd, s0, s1, s2, 1.f);
      float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
      for (int it = 0; SPIN_OK(it, 7); ++it) { v = ld4f(ssw_ + 4 * wd); if (v.w >= (float)(p.G + TOCPC) - 0.5f) break; }
#if SSRR
      // hardening, not a proof: a red.v4 is atomic per element only, the count may land before the ss elements of the same word
      v = ld4f(ssw_ + 4 * wd);
#endif
      T.sse[wd * 3] = v.x; T.sse[wd * 3 + 1] = v.y; T.sse[wd * 3 + 2] = v.z;
    }
    __syncwarp();
    TLX(12);
    float o = 0.f;
    for (int ei = 0; ei < E; ++ei) o += T.y[ei][lane] * T.twe[ei] * rsqrtf(T.sse[ei] * (1.f / HID) + 1e-8f);
    T.ib[lane] = f2bf(o);
    __syncwarp();
    if (lane < 4) {
      const uint4 v = lds128(smem_u32(&T.ib[0]) + lane * 16);
      stg_v4(p.out + 32 * c + 8 * lane, v.x, v.y, v.z, v.w);
    }
#if SAFE
    if (c == 0) {
      const int ob = bank ^ 1;
      if (lane < 3) stg_v4(p.ssw + 12 * ob + 4 * lane, 0u, 0u, 0u, 0u);
      if (lane < 8) p.cnt[256 * ob + lane * 32] = 0;
      if (lane == 0) *const_cast<volatile int*>(p.cnt + 1000) = ob;
    }
#endif
    TLX(13);
    return;
  }
  // ---------------------------------------------------------------- math warps
  const int w = warp, g = lane >> 2, t4 = lane & 3;
  const u32 k16 = 16u + (u32)(p.M >> 30);
  const u32 sm0 = smem_u32(sm);
  const u32 cb0 = sm0 + w * WSL + lane * 16;
  const u32 sb0 = sm0 + w * WSL + 4096 + lane * 16;
  const u32 xa = sm0 + OFF_X + t4 * 32 + 4096 * (w >> 2);
  const u32 ia = sm0 + OFF_I + t4 * 32 + 2048 * (w >> 1);
  mb_wait(&T.xfirst, 0);
  if (tid == 0) TLX(2);
  int slot = 0, j = 0;
  u32 par = 0;
  const int rs4 = 0; (void)rs4;
  for (int ei = 0; ei < E; ++ei, ++j) {
    float acc[2][4];
#pragma unroll
    for (int a = 0; a < 2; ++a)
#pragma unroll
      for (int k = 0; k < 4; ++k) acc[a][k] = 0.f;
#pragma unroll 1
    for (int s = 0; s < 4; ++s) {
      mb_wait(&T.full[slot], par);
#ifdef SC_TL
      if (tid == 0 && j == 0 && s == 0) TLX(3);
#endif
      stage(cb0 + (u32)slot * STAGE, sb0 + (u32)slot * STAGE, xa + (u32)s * 1024, acc, k16, (u32)g);
      __syncwarp();
      if (lane == 0) mb_arrive(&T.empty[slot]);
      if (++slot == NS) { slot = 0; par ^= 1; }
    }
    if (j >= NSM) { for (int it = 0; SPIN_OK(it, 8); ++it) if (ld_sm_rlx(&T.edone) > j - NSM) break; }
    if (t4 == 0) {
      T.P[j & (NSM - 1)][w][g][0] = acc[0][0] + acc[1][0];
      T.P[j & (NSM - 1)][w][g][1] = acc[0][2] + acc[1][2];
    }
    __syncwarp();
    if (lane == 0) mb_arrive(&T.rfull[j & (NSM - 1)]);
  }
#ifdef SC_TL
  if (tid == 0) TLX(4);
#endif
  for (int ei = 0; ei < E; ++ei, ++j) {
    const int b = ei & (NXB - 1);
    mb_wait(&T.xready[b], (u32)(ei >> 2) & 1);
#ifdef SC_TL
    if (tid == 0 && ei == E - 1) TLX(8);
#endif
    float acc[2][4];
#pragma unroll
    for (int a = 0; a < 2; ++a)
#pragma unroll
      for (int k = 0; k < 4; ++k) acc[a][k] = 0.f;
#pragma unroll 1
    for (int s = 0; s < 2; ++s) {
      mb_wait(&T.full[slot], par);
#ifdef SC_TL
      if (tid == 0 && ei == E - 1 && s == 0) TLX(9);
#endif
      stage(cb0 + (u32)slot * STAGE, sb0 + (u32)slot * STAGE, ia + (u32)b * 8192 + (u32)s * 1024, acc, k16, (u32)g);
      __syncwarp();
      if (lane == 0) mb_arrive(&T.empty[slot]);
      if (++slot == NS) { slot = 0; par ^= 1; }
    }
    if (j >= NSM) { for (int it = 0; SPIN_OK(it, 9); ++it) if (ld_sm_rlx(&T.edone) > j - NSM) break; }
    if (t4 == 0) {
      T.P[j & (NSM - 1)][w][g][0] = acc[0][0] + acc[1][0];
      T.P[j & (NSM - 1)][w][g][1] = acc[0][2] + acc[1][2];
    }
    __syncwarp();
    if (lane == 0) mb_arrive(&T.rfull[j & (NSM - 1)]);
  }
#ifdef SC_TL
  if (tid == 0) TLX(10);
#endif
}

extern "C" __global__ void sc_probe(int* o) { o[0] = SMEM_BYTES; }
