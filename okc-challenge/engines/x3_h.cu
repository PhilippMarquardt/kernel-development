// x3_h.cu (pm2/hard): pm2/v1x/x3.cu 0bcdabfb + ss word re-read after count == G. -DTOC=1 test build reports spin timeouts in dbg. No smem source for cp.async here.
// x3.cu (pm2/v1x): engine U3 of pm2/up/up3.cu for M <= 8, N <= 8, any routing. Own code; stage / dq4 / helpers from pm2/sr/sr.cu; 1/128 of each expert per CTA, epilogue warp, y-local, ss word, x predicate from pm2/d1/sc.cu (same project).
// G = 128 CTAs, no clusters. Per CTA: A(e0) .. A(eE-1), B(e0) .. B(eE-1). A job: 64 rows x 4096 K, 4 stages, warp w: row group w>>1, K half w&1. B job: 32 rows x 4096 K, 2 stages: warp w: row group w>>2, K quarter w&3.
// Lane (g, t) of a math warp holds rows (g, g+8) x token columns (2t, 2t+1) in registers; at job end it stores 4 partials in smem and goes on; the epilogue warp sums K parts, does silu / y.
// A-to-B: one counter per expert. The writer spins on a read-back of its own inter stores before the counter red, so count == G means every store is visible; the row bulk runs after that.
// ss + count: one red.v4 word per 3 pairs, polled (FORMALSS=1: ss by atom with return, count in its own word). No MEMBAR after the mbarrier init.
// State across calls: two banks of counters and ss words, chosen by a bank word read with the routing. CTA 0 zeroes the other bank and flips the word at the end of the call, so the next call
// starts clean by kernel completion alone; nothing depends on how long ago a store was issued (as pm2/d1/sc.cu SAFE=1).
// Limits: M <= 8 (launcher refuses more). Then N <= 8, E <= 32, P <= 64 always. P > YMAX: 2 slots + 2 x 8 rows and y in the space that leaves. Scales: SCF 5 takes any byte (escape records); SCF 8 only positive normal bytes, x_prep.plan chooses.
typedef unsigned int u32;
typedef unsigned long long u64;
typedef unsigned short u16;

#ifndef NS
#define NS 4            // ring slots in smem
#endif
#ifndef NSLO
#define NSLO 3          // ring slots in use below EHI experts
#endif
#ifndef EHI
#define EHI 99          // from this many experts: all NS slots + self-paced issue (when PACE)
#endif
#ifndef PACE
#define PACE 0          // self-paced issue (pm2/gap): stage k not before g0 + (k - PLD) * STAGE * G / RATE, late SMs never held. Reads %globaltimer; constant tuned on this H200 (near-rule flag)
#endif
#ifndef RATE
#define RATE 4620       // B/ns
#endif
#ifndef PLD
#define PLD 1           // stages of lead
#endif
#ifndef XROWS
#define XROWS 8         // at most this many tokens per expert
#endif
#ifndef XP
#define XP 1            // 1: x loaded only on lanes g < N, B = 0 on the others; 0: all lanes
#endif
#ifndef XREUSE
#define XREUSE 1        // M=1: x row read once per CTA, every A job uses it
#endif
#ifndef XEARLY
#define XEARLY 1        // M=1: the x row bulk goes out right after the first syncthreads, before routing is known
#endif
#ifndef TWB
#define TWB 1           // route weights by TMA with gs + pw, not lane loads in the routing read
#endif
#ifndef XSPLIT
#define XSPLIT 1        // 1: math waits for the x row only, gs + pw + tw on their own barrier for the epilogue warp
#endif
#ifndef TESTHOLE
#define TESTHOLE 0      // test: no dependency poll, rows read too early, the NaN path has to fix it
#endif
#ifndef MW
#define MW 8            // math warps: 8 or 16 (16: 18 KB per warp pair of the same stage size, N <= 4 only so far)
#endif
#define PW MW
#define XW (MW + 1)
#define EW (MW + 2)
#define NTHR ((MW + 3) * 32)
#define NSM 2
#ifndef HOLD
#define HOLD 0          // 1: first stage of the last B waits until its row is issued, so the row is not queued behind it
#endif
#define HID 4096
#define KST (MW == 16 ? 256 : 512)
#ifndef SCF
#define SCF 8           // scale bytes per lane per 512 K: 8 = bytes (16 B), 5 = 5-bit + escape (10 B), 6 = 6-bit (12 B)
#endif
#if SCF == 5
#define WSL 4416
#elif SCF == 6
#define WSL 4480
#else
#define WSL 4608
#endif
#define SCB2 256
#define SCL1 (SCF == 8 ? QPS * 2 : 8)
#define SCL2 (SCF == 5 ? 2 : 4)
#define STAGE (WSL * 8)
#define CODEB (KST * 8)
#define QPS (KST / 64)
#define JOB (4 * STAGE)
#define JOBB (2 * STAGE)
#define HW (MW / 4)     // A: warps per row group
#define HWB (MW / 2)    // B: warps per row group
#define SUBW 4          // A: sub-blocks per warp per job
#define SUBB 2          // B
#define NST 4
#define NSTB 2
#define XS 8208
#define MAXM 32
#define MAXP 64         // pairs: M * 8
#define YMAX 48         // pairs with y in Tab
#define NWD 22          // ss words: (MAXP + 2) / 3
#define CNTB 528        // ints per counter bank: [33][16], 32 experts + the FORMALSS count
#ifndef FORMALSS
#define FORMALSS 0      // 1: ss by atom with return per pair, count in its own word, instead of ss + count in one red.v4
#endif
#define NJA 128         // A jobs per expert
#define NJB 128         // B units per expert

struct Params {
  const int* ids;
  const float* tw;
  const u16* x;
  u16* out;
  const unsigned char* w13;  // [32][128][JOB]
  const unsigned char* w2q;  // [32][128][JOBB]
  const float* gs;           // [64]: gs13[32], gs2[32], 256 B
  const float* pw1;
  u16* inter;   // [pairs][4096]
  float* ssw;   // [2 banks][NWD words][4]: ss, ss, ss, count
  int* cnt;     // [2 banks][CNTB], [16 e]: A warps done
  int* bank;    // [0]: current bank
  u64* dbg;
  const uint4* side;  // SCF 5: escape records, 32 B each
  u32 svbias;         // SCF 5, 6: bf16 of the window base byte, both halves
  int M, G;
};

struct Tab {
  u64 full[NS], empty[NS], xfull[4], xready[4], xempty[2], rfull[NSM], xfirst, xrow0, fin;
  u32 mask[32][2];
  int rt[MAXM * 8], nr[MAXM];   // token -> its local routes, (e << 8) | pair is filled at plan end
#if TWB
  __align__(16) float twb[MAXM * 8];
#else
  float rtw[MAXM * 8];
#endif
  int ne[32], pb[32], lst[32];
  int ne_, npairs, xgo, ns, pc, edone, rowi, xbuf, xtb, bank;
  __align__(16) float gsa[64];
  __align__(16) float pw[32];
  __align__(16) float P[NSM][MW][32][4];   // job partials
  float ssl[MAXP];
  float sse[MAXP + 3];
  __align__(16) u16 ib[MW + 1][32];
  float y[YMAX][32];
};

#ifndef X13
#define X13 1           // 1 (SCF 5 only, no room otherwise): some N > 4 and every two neighbouring jobs need <= XRW rows: 3 slots + XRW rows, even jobs from the bottom, odd jobs from the top. Else 2 slots + 2 x 8 rows
#endif
#if SCF != 5
#undef X13
#define X13 0
#endif
#define XRW 13
#define YOFF (2 * STAGE + 16 * XS)         // P > YMAX: y [pairs][32] behind 2 slots + 2 x 8 rows
#define OFF_YEND (YOFF + MAXP * 128)
#if X13
#define OFF_TAB0 (3 * STAGE + XRW * XS)
#else
#define OFF_TAB0 (NS * STAGE + 2 * 4 * XS) // N <= 4: NS or NSLO slots + 2 x 4 rows. Some N > 4: 2 slots + 2 x 8 rows
#endif
#define OFF_TAB (OFF_TAB0 > OFF_YEND ? OFF_TAB0 : OFF_YEND)
#define SMEM_BYTES (OFF_TAB + (int)sizeof(Tab))
#define OFFM1 (NS * STAGE)
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
static_assert(OFF_TAB % 16 == 0 && SMEM_BYTES <= 232448, "smem limit");

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
// only lanes with on != 0 load, no branch in the MMA block
__device__ __forceinline__ void lds128p(u32 a, u32 on, uint4& v) {
  asm volatile("{ .reg .pred p; setp.ne.u32 p, %5, 0; @p ld.shared.v4.b32 {%0,%1,%2,%3}, [%4]; }" : "+r"(v.x), "+r"(v.y), "+r"(v.z), "+r"(v.w) : "r"(a), "r"(on));
}
__device__ __forceinline__ void sts128(u32 a, u32 x, u32 y, u32 z, u32 w) { asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(a), "r"(x), "r"(y), "r"(z), "r"(w) : "memory"); }
__device__ __forceinline__ uint4 ldg4(const void* a) { uint4 v; asm volatile("ld.relaxed.gpu.global.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(a) : "memory"); return v; }
__device__ __forceinline__ void stg_v4(void* p, u32 x, u32 y, u32 z, u32 w) { asm volatile("st.global.v4.b32 [%0], {%1,%2,%3,%4};" ::"l"(p), "r"(x), "r"(y), "r"(z), "r"(w) : "memory"); }
__device__ __forceinline__ int ld_sm_rlx(const int* p) { int v; asm volatile("ld.relaxed.cta.shared::cta.b32 %0, [%1];" : "=r"(v) : "r"(smem_u32(p)) : "memory"); return v; }
__device__ __forceinline__ int ld_rlx(const int* p) { int v; asm volatile("ld.relaxed.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
__device__ __forceinline__ void red_rlx(int* p, int v) { asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;" ::"l"(p), "r"(v) : "memory"); }
__device__ __forceinline__ void red4f(float* p, float a, float b, float c, float d) { asm volatile("red.relaxed.gpu.global.add.v4.f32 [%0], {%1,%2,%3,%4};" ::"l"(p), "f"(a), "f"(b), "f"(c), "f"(d) : "memory"); }
__device__ __forceinline__ float atomf(float* p, float a) { float o; asm volatile("atom.relaxed.gpu.global.add.f32 %0, [%1], %2;" : "=f"(o) : "l"(p), "f"(a) : "memory"); return o; }
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

#ifdef UP_TL
#define TL(i) if (p.dbg) p.dbg[blockIdx.x * 16 + (i)] = gtime()
#else
#define TL(i)
#endif

__device__ __forceinline__ void dq4(u32 q, u32& r0, u32& r1, u32& r2, u32& r3) {
  const u32 M = 0x81C081C0u;
  const u32 s1 = q << 2, s2 = q << 6, s3 = q >> 6, s4 = __funnelshift_l(q, q, 12);
  r0 = q & M;
  r1 = (s1 ^ s2) & M;
  r2 = (s1 ^ s3) & M;
  r3 = (s1 ^ s4) & M;
}

// ---- packed scales (pm2/sc5): SCF 8 bytes, 5 = 5-bit + escape record, 6 = 6-bit. sv8 = 8 bf16x2 scale pairs of one lane and one 512 K block.
#if SCF == 5
__device__ __forceinline__ void sv_unp(u32 w0, u32 w1, u32 s2, u32 bias, u32* sv8) {
  const u32 m = 0x01F001F0u;
  const u32 w2 = prmt(s2, 0u, 0x4140u) | ((w0 >> 7) & 0x01000100u) | ((w1 >> 6) & 0x02000200u);
  sv8[0] = ((w0 << 4) & m) + bias; sv8[1] = ((w0 >> 1) & m) + bias; sv8[2] = ((w0 >> 6) & m) + bias;
  sv8[3] = ((w1 << 4) & m) + bias; sv8[4] = ((w1 >> 1) & m) + bias; sv8[5] = ((w1 >> 6) & m) + bias;
  sv8[6] = ((w2 << 4) & m) + bias; sv8[7] = ((w2 >> 1) & m) + bias;
}
#elif SCF == 6
__device__ __forceinline__ void sv_unp(u32 w0, u32 w1, u32 w2, u32 bias, u32* sv8) {
  const u32 m = 0x03F003F0u;
  const u32 w3 = ((w0 >> 12) & 0x000F000Fu) | ((w1 >> 8) & 0x00F000F0u) | ((w2 >> 4) & 0x0F000F00u);
  sv8[0] = ((w0 << 4) & m) + bias; sv8[1] = ((w0 >> 2) & m) + bias;
  sv8[2] = ((w1 << 4) & m) + bias; sv8[3] = ((w1 >> 2) & m) + bias;
  sv8[4] = ((w2 << 4) & m) + bias; sv8[5] = ((w2 >> 2) & m) + bias;
  sv8[6] = ((w3 << 4) & m) + bias; sv8[7] = ((w3 >> 2) & m) + bias;
}
#endif
#if SCF == 5
// escape: per-lane read of 8 ready pairs. sv by reference, not a stack array: the array form put 3 STL per lane per stage in the fast path
#define SV_LOAD(sb, sb2, bias, side, s0, s1, s2_, s3, s4, s5, s6, s7)                                                    u32 s0, s1, s2_, s3, s4, s5, s6, s7;                                                                                    {                                                                                                                         u32 w0_, w1_, u_;                                                                                                       asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];" : "=r"(w0_), "=r"(w1_) : "r"(sb));                                       asm volatile("ld.shared.u16 %0, [%1];" : "=r"(u_) : "r"(sb2));                                                          if (__builtin_expect((w0_ & 31u) == 31u, 0)) {                                                                            const uint4 a_ = __ldg(side + w1_ * 2), b_ = __ldg(side + w1_ * 2 + 1);                                                 s0 = a_.x; s1 = a_.y; s2_ = a_.z; s3 = a_.w; s4 = b_.x; s5 = b_.y; s6 = b_.z; s7 = b_.w;                              } else {                                                                                                                  const u32 m_ = 0x01F001F0u, w2_ = prmt(u_, 0u, 0x4140u) | ((w0_ >> 7) & 0x01000100u) | ((w1_ >> 6) & 0x02000200u);       s0 = ((w0_ << 4) & m_) + bias; s1 = ((w0_ >> 1) & m_) + bias; s2_ = ((w0_ >> 6) & m_) + bias;                           s3 = ((w1_ << 4) & m_) + bias; s4 = ((w1_ >> 1) & m_) + bias; s5 = ((w1_ >> 6) & m_) + bias;                            s6 = ((w2_ << 4) & m_) + bias; s7 = ((w2_ >> 1) & m_) + bias;                                                         }                                                                                                                     }
#elif SCF == 6
#define SV_LOAD(sb, sb2, bias, side, s0, s1, s2_, s3, s4, s5, s6, s7)                                                    u32 s0, s1, s2_, s3, s4, s5, s6, s7;                                                                                    {                                                                                                                         u32 w0_, w1_, w2_;                                                                                                      asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];" : "=r"(w0_), "=r"(w1_) : "r"(sb));                                       asm volatile("ld.shared.b32 %0, [%1];" : "=r"(w2_) : "r"(sb2));                                                         const u32 m_ = 0x03F003F0u, w3_ = ((w0_ >> 12) & 0x000F000Fu) | ((w1_ >> 8) & 0x00F000F0u) | ((w2_ >> 4) & 0x0F000F00u);     s0 = ((w0_ << 4) & m_) + bias; s1 = ((w0_ >> 2) & m_) + bias; s2_ = ((w1_ << 4) & m_) + bias; s3 = ((w1_ >> 2) & m_) + bias;     s4 = ((w2_ << 4) & m_) + bias; s5 = ((w2_ >> 2) & m_) + bias; s6 = ((w3_ << 4) & m_) + bias; s7 = ((w3_ >> 2) & m_) + bias;   }
#endif

// One warp, one 512 K sub-block: 16 rows x 512 K
__device__ __forceinline__ void stage(u32 cb, u32 sb, u32 sb2, u32 xb, float (&acc)[2][4], u32 k16, u32 bias, const uint4* side, u32 xon) {
#if SCF != 8
  SV_LOAD(sb, sb2, bias, side, sv_0, sv_1, sv_2, sv_3, sv_4, sv_5, sv_6, sv_7)
#elif MW == 16
  uint4 sv4 = make_uint4(0u, 0u, 0u, 0u);
  asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];" : "=r"(sv4.x), "=r"(sv4.y) : "r"(sb));
#else
  const uint4 sv4 = lds128(sb);
#endif
  uint4 u0 = make_uint4(0u, 0u, 0u, 0u), u1 = u0;
#pragma unroll
  for (int q = 0; q < QPS; ++q) {
    const uint4 wq = lds128(cb + q * 512);
#if XP
    lds128p(xb + q * 128, xon, u0);
    lds128p(xb + q * 128 + 16, xon, u1);
#else
    u0 = lds128(xb + q * 128);
    u1 = lds128(xb + q * 128 + 16);
#endif
    const u32 ww[4] = {wq.x, wq.y, wq.z, wq.w};
    const u32 bb[8] = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};
#if SCF != 8
    const u32 sv = q == 0 ? sv_0 : q == 1 ? sv_1 : q == 2 ? sv_2 : q == 3 ? sv_3 : q == 4 ? sv_4 : q == 5 ? sv_5 : q == 6 ? sv_6 : sv_7;
#else
    const u32 sw = (q >> 1) == 0 ? sv4.x : (q >> 1) == 1 ? sv4.y : (q >> 1) == 2 ? sv4.z : sv4.w;
    const u32 sv = prmt(sw, 0u, (q & 1) ? 0x4342u : 0x4140u) * k16 + 0x77007700u;
#endif
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

extern "C" __global__ void __launch_bounds__(NTHR, 1) up_moe(Params p) {
  extern __shared__ __align__(128) unsigned char sm[];
  Tab& T = *reinterpret_cast<Tab*>(sm + OFF_TAB);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int M = p.M, G = p.G, c = blockIdx.x;
  TL(0);
  if (tid == 0) {
    for (int s = 0; s < NS; ++s) { mb_init(&T.full[s], 1); mb_init(&T.empty[s], MW); }
    for (int s = 0; s < 4; ++s) { mb_init(&T.xfull[s], 1); mb_init(&T.xready[s], 1); }
    for (int s = 0; s < 2; ++s) mb_init(&T.xempty[s], MW);
    for (int s = 0; s < NSM; ++s) mb_init(&T.rfull[s], MW);
    mb_init(&T.xfirst, 1);
    mb_init(&T.xrow0, 1);
    mb_init(&T.fin, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    T.xgo = 0; T.edone = 0; T.rowi = 0;
  }
  if (tid < 64) T.mask[tid >> 1][tid & 1] = 0u;
  __syncthreads();
  // x (M=1), gs and this CTA's pw go together with the routing read.
  if (warp == XW && lane == 0) {
    T.bank = ld_rlx(p.bank);
#if XSPLIT
    if (M == 1) { mb_expect(&T.xrow0, 8192u); bulk_g2s(sm + OFFM1, p.x, 8192u, &T.xrow0); *const_cast<volatile int*>(&T.xgo) = 1; }
    mb_expect(&T.xfirst, 256u + 128u + (TWB ? (u32)M * 32u : 0u));
#else
    mb_expect(&T.xfirst, (M == 1 ? 8192u : 0u) + 256u + 128u + (TWB ? (u32)M * 32u : 0u));
    if (M == 1) bulk_g2s(sm + OFFM1, p.x, 8192u, &T.xfirst);
    if (M == 1) *const_cast<volatile int*>(&T.xgo) = 1;
#endif
#if TWB
    bulk_g2s(T.twb, p.tw, (u32)M * 32u, &T.xfirst);
#endif
    bulk_g2s(T.gsa, p.gs, 256u, &T.xfirst);
    bulk_g2s(T.pw, p.pw1 + 32 * c, 128u, &T.xfirst);
  }
  if (tid < M) {
    const int4 a = __ldg(reinterpret_cast<const int4*>(p.ids + tid * 8));
    const int4 b = __ldg(reinterpret_cast<const int4*>(p.ids + tid * 8 + 4));
#if !TWB
    const float4 ta = __ldg(reinterpret_cast<const float4*>(p.tw + tid * 8));
    const float4 tb = __ldg(reinterpret_cast<const float4*>(p.tw + tid * 8 + 4));
    const float t8[8] = {ta.x, ta.y, ta.z, ta.w, tb.x, tb.y, tb.z, tb.w};
#endif
    const int v8[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
    int nr_ = 0;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      if ((u32)v8[k] < 32u) {
        T.rt[tid * 8 + nr_] = (v8[k] << 8) | k;
#if !TWB
        T.rtw[tid * 8 + nr_] = t8[k];
#endif
        ++nr_;
        atomicOr(&T.mask[v8[k]][tid >> 5], 1u << (tid & 31));
      }
    }
    T.nr[tid] = nr_;
  }
  __syncthreads();
  if (warp == 0) {
    const int n = __popc(T.mask[lane][0]) + __popc(T.mask[lane][1]), pres = n > 0 ? 1 : 0;
    T.ne[lane] = n;
    int pk = n | (pres << 16);
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int v = __shfl_up_sync(0xffffffffu, pk, o);
      if (lane >= o) pk += v;
    }
    const int inc = pk & 0xffff, incp = pk >> 16;
    T.pb[lane] = inc - n;
    if (pres) T.lst[incp - pres] = lane;
    const u32 big_ = __ballot_sync(0xffffffffu, n > 4);
    if (lane == 31) { T.npairs = inc; T.ne_ = incp; T.ns = (big_ || inc > YMAX) ? 2 : incp >= EHI ? NS : NSLO; T.xbuf = (big_ || inc > YMAX ? 8 : 4) * XS; T.pc = PACE && incp >= EHI && !big_; T.xtb = 0; }
#if X13
    __syncwarp();
    if (lane == 31 && big_ && T.npairs <= YMAX) {
      int ok_ = 1;
      const int E_ = T.ne_;
      for (int v = 0; v < E_; ++v) if (T.ne[T.lst[v]] + T.ne[T.lst[v + 1 < E_ ? v + 1 : 0]] > XRW) ok_ = 0;
      if (ok_) { T.ns = 3; T.xtb = 1; }
    }
#endif
  }
  __syncthreads();
  TL(1);
  const int ns = T.ns, E = T.ne_, nj = 2 * E;
  int* const cnt = p.cnt + CNTB * T.bank;
  float* const ssw = p.ssw + 4 * NWD * T.bank;

  if (warp == PW) {
    // ---------------------------------------------------------------- weight producer: never waits on anything but ring slots (and the pace)
    if (lane == 0) {
      for (int it = 0; SPIN_OK(it, 1); ++it) if (*const_cast<volatile int*>(&T.xgo) || E == 0) break;
      int slot = 0, lap0 = 1;
      u32 par = 0;
#if PACE
      const int pc_ = T.pc;
      u32 g0l_; asm volatile("mov.u32 %0, %%globaltimer_lo;" : "=r"(g0l_));
      const u32 tsn_ = (u32)STAGE * (u32)G / RATE;
      int ks_ = 0;
#endif
      for (int j = 0; j < nj; ++j) {
        const int a_ = j < E, ei = a_ ? j : j - E, e = T.lst[ei];
        const unsigned char* src = a_ ? p.w13 + ((size_t)e * NJA + c) * JOB : p.w2q + ((size_t)e * NJB + c) * JOBB;
        const int nst_ = a_ ? NST : NSTB;
#if HOLD
        if (j == nj - 1) { for (int it = 0; SPIN_OK(it, 2); ++it) if (ld_sm_rlx(&T.rowi) >= E) break; }
#endif
        for (int s = 0; s < nst_; ++s) {
          if (!lap0) mb_wait(&T.empty[slot], (par + 1) & 1);
#if PACE
          if (pc_ && ks_ >= PLD) {
            const u32 tgt_ = (u32)(ks_ - PLD) * tsn_;
            for (int it = 0; it < 8192; ++it) { u32 t_; asm volatile("mov.u32 %0, %%globaltimer_lo;" : "=r"(t_)); if (t_ - g0l_ >= tgt_) break; }
          }
          ++ks_;
#endif
          mb_expect(&T.full[slot], STAGE);
          bulk_g2s(sm + slot * STAGE, src + s * STAGE, STAGE, &T.full[slot]);
          if (++slot == ns) { slot = 0; par ^= 1; lap0 = 0; }
        }
      }
    }
    return;
  }
  if (warp == XW) {
    // ---------------------------------------------------------------- x producer: token rows, inter rows. Lane 0.
    if (lane != 0) return;
    if (M == 1) {
      // A jobs all use the x row (xfirst). B rows: row slots 1..4 of the x area
      for (int ei = 0; ei < E; ++ei) {
        const int b = ei & 3;
        // own A job first: by then CTA 0's call-start zero of the counter has long landed, a stale count can't be read
        for (int it = 0; SPIN_OK(it, 3); ++it) if (ld_sm_rlx(&T.edone) > ei) break;
        if (ei >= 4) { for (int it = 0; SPIN_OK(it, 4); ++it) if (ld_sm_rlx(&T.edone) > E + ei - 4) break; }
#if !TESTHOLE
        for (int it = 0; SPIN_OK(it, 5); ++it) if (ld_rlx(cnt + 16 * ei) >= NJA) break;
#endif
        if (ei == E - 1) { TL(5); }
        asm volatile("fence.proxy.async.global;" ::: "memory");
        mb_expect(&T.xfull[b], 8192u);
        bulk_g2s(sm + OFFM1 + (1 + b) * XS, p.inter + (size_t)ei * HID, 8192u, &T.xfull[b]);
        *const_cast<volatile int*>(&T.rowi) = ei + 1;
        mb_wait(&T.xfull[b], (u32)(ei >> 2) & 1u);
        if (ei == E - 1) { TL(6); }
        mb_arrive(&T.xready[b]);
      }
      return;
    }
    for (int j = 0; j < nj; ++j) {
      const int a_ = j < E, ei = a_ ? j : j - E, e = T.lst[ei], rows = T.ne[e], pb = T.pb[e], b = j & 1;
      unsigned char* xb = T.xtb ? sm + ns * STAGE + (b ? (XRW - rows) * XS : 0) : sm + ns * STAGE + b * T.xbuf;
      if (j >= 2) mb_wait(&T.xempty[b], (u32)((j >> 1) - 1) & 1u);
      if (!a_) {
        for (int it = 0; SPIN_OK(it, 6); ++it) if (ld_sm_rlx(&T.edone) > ei) break;
#if !TESTHOLE
        for (int it = 0; SPIN_OK(it, 7); ++it) if (ld_rlx(cnt + 16 * ei) >= NJA) break;
#endif
        if (ei == E - 1) { TL(5); }
      }
      asm volatile("fence.proxy.async.global;" ::: "memory");
      mb_expect(&T.xfull[b], (u32)(rows * 8192));
      u32 m0_ = T.mask[e][0], m1_ = T.mask[e][1];
      for (int n = 0; n < rows; ++n) {
        int tk;
        if (m0_) { tk = __ffs(m0_) - 1; m0_ &= m0_ - 1; } else { tk = 32 + __ffs(m1_) - 1; m1_ &= m1_ - 1; }
        const u16* src = a_ ? p.x + (size_t)tk * HID : p.inter + (size_t)(pb + n) * HID;
        bulk_g2s(xb + n * XS, src, 8192u, &T.xfull[b]);
      }
      if (j == 0) *const_cast<volatile int*>(&T.xgo) = 1;
      if (!a_) *const_cast<volatile int*>(&T.rowi) = ei + 1;
      mb_wait(&T.xfull[b], (u32)(j >> 1) & 1u);
      if (j == 0) { TL(2); }
      if (j == nj - 1) { TL(6); }
      mb_arrive(&T.xready[b]);
    }
    return;
  }
  const u32 sm0 = smem_u32(sm);
  const u32 k16 = 16u + (u32)(p.M >> 30);
  const u32 svb_ = p.svbias;
  const uint4* side_ = p.side;
  const int g = lane >> 2, t4 = lane & 3;
  if (warp == EW) {
    // ---------------------------------------------------------------- epilogue warp: lane = value index of the job. Math never waits for it unless it is 2 jobs behind.
    const u32 ibs = smem_u32(&T.ib[0][0]);
    float* const yb = T.npairs > YMAX ? reinterpret_cast<float*>(sm + YOFF) : &T.y[0][0];
#if XSPLIT
    mb_wait(&T.xfirst, 0u);
#endif
    for (int j = 0; j < nj; ++j) {
      const int a_ = j < E, ei = a_ ? j : j - E, e = T.lst[ei], N = T.ne[e], pb = T.pb[e], sl = j & (NSM - 1);
      mb_wait(&T.rfull[sl], (u32)(j >> 1) & 1u);
      if (a_) {
        const int rs = lane >> 3, gg = lane & 7;
        const float gs = T.gsa[e];
        for (int n = 0; n < N; ++n) {
          const int lm = (gg << 2) | (n >> 1), od = n & 1;
          float g0 = 0.f, u0 = 0.f;
#pragma unroll
          for (int k = 0; k < HW; ++k) { g0 += T.P[sl][HW * rs + k][lm][od]; u0 += T.P[sl][HW * rs + k][lm][2 + od]; }
          g0 *= gs; u0 *= gs;
          T.ib[n][lane] = f2bf(g0 / (1.f + expf(-g0)) * u0);
        }
        __syncwarp();
        if (lane < 4 * N) {
          const int n = lane >> 2, i = lane & 3;
          const uint4 v = lds128(ibs + (u32)n * 64 + i * 16);
          u16* ad = p.inter + (size_t)(pb + n) * HID + 32 * c + 8 * i;
          stg_v4(ad, v.x, v.y, v.z, v.w);
          // read back: an equal old value can only come from an identical input, so either value is right
          for (int it = 0; SPIN_OK(it, 8); ++it) { const uint4 r = ldg4(ad); if (r.x == v.x && r.y == v.y && r.z == v.z && r.w == v.w) break; }
        }
        __syncwarp();
        if (lane == 0) { red_rlx(cnt + 16 * ei, 1); if (ei == E - 1) { TL(4); } }
      } else {
        const int rs = lane >> 4, m = lane & 15, gg = m & 7, hr = m >> 3;
        const float mul = T.gsa[32 + e] * T.pw[lane];
        for (int n = 0; n < N; ++n) {
          const int lm = (gg << 2) | (n >> 1), vi = (n & 1) + 2 * hr;
          float v = 0.f;
#pragma unroll
          for (int k = 0; k < HWB; ++k) v += T.P[sl][HWB * rs + k][lm][vi];
          const float yv = v * mul;
          yb[(pb + n) * 32 + lane] = yv;
          float q = yv * yv;
#pragma unroll
          for (int o = 1; o < 32; o <<= 1) q += __shfl_xor_sync(0xffffffffu, q, o);
          if (lane == 0) T.ssl[pb + n] = q;
        }
      }
      __syncwarp();
      if (lane == 0) *const_cast<volatile int*>(&T.edone) = j + 1;
    }
    TL(9);
    const int npr = T.npairs;
#if FORMALSS
    // formal order: ss landed (atom return) before the count word, count alone polled, then ss read
    for (int pr = lane; pr < npr; pr += 32) { const float o_ = atomf(ssw + pr, T.ssl[pr]); (void)o_; }
    __syncwarp();
    if (lane == 0) {
      red_rlx(cnt + 16 * 32, 1);
      for (int it = 0; SPIN_OK(it, 9); ++it) if (ld_rlx(cnt + 16 * 32) >= G) break;
    }
    __syncwarp();
    for (int pr = lane; pr < npr; pr += 32) T.sse[pr] = __ldg(ssw + pr);
#else
    // ss + count travel in one red.v4 (3 pairs per word); y is local
    const int nw = (npr + 2) / 3;
    if (lane < nw) {
      const int wd = lane;
      const float s0 = wd * 3 + 0 < npr ? T.ssl[wd * 3] : 0.f, s1 = wd * 3 + 1 < npr ? T.ssl[wd * 3 + 1] : 0.f, s2 = wd * 3 + 2 < npr ? T.ssl[wd * 3 + 2] : 0.f;
      red4f(ssw + 4 * wd, s0, s1, s2, 1.f);
      float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
      for (int it = 0; SPIN_OK(it, 10); ++it) { v = ld4f(ssw + 4 * wd); if (v.w >= (float)(G + TOCPC) - 0.5f) break; }
#if SSRR
      // hardening, not a proof: a red.v4 is atomic per element only, the count may land before the ss elements of the same word
      v = ld4f(ssw + 4 * wd);
#endif
      T.sse[wd * 3] = v.x; T.sse[wd * 3 + 1] = v.y; T.sse[wd * 3 + 2] = v.z;
    }
#endif
    __syncwarp();
    TL(11);
    if (lane == 0) mb_arrive(&T.fin);
    // ---- next call's bank: zeroed here, flipped here, visible when this kernel is complete
    if (c == 0) {
      const int ob = T.bank ^ 1;
      if (lane < NWD) stg_v4(p.ssw + 4 * NWD * ob + 4 * lane, 0u, 0u, 0u, 0u);
      p.cnt[CNTB * ob + 16 * lane] = 0;
      if (lane == 0) { p.cnt[CNTB * ob + 16 * 32] = 0; *const_cast<volatile int*>(p.bank) = ob; }
    }
    return;
  }
  // ---------------------------------------------------------------- math warps
  const int w = warp;
  const int n0 = 2 * t4;
  const int xb_ = T.xbuf;
  const int xtb_ = T.xtb;
  const u32 xa0 = sm0 + (M == 1 ? OFFM1 : ns * STAGE) + (xb_ == 8 * XS ? g : g & 3) * XS + t4 * 32;
  const u32 cb0 = sm0 + w * WSL + lane * 16;
  const u32 pma = smem_u32(&T.P[0][w][lane][0]);
  int slot = 0;
  u32 par = 0;
#if XSPLIT
  if (M == 1) mb_wait(&T.xrow0, 0u);
#else
  mb_wait(&T.xfirst, 0u);
#endif
  for (int j = 0; j < nj; ++j) {
    const int a_ = j < E, ei = a_ ? j : j - E, e = T.lst[ei], N = T.ne[e];
    u32 xoff;
    if (M == 1) {
      xoff = a_ ? 0u : (u32)(1 + (ei & 3)) * XS;
      if (!a_) mb_wait(&T.xready[ei & 3], (u32)(ei >> 2) & 1u);
    } else {
      xoff = xtb_ ? ((j & 1) ? (u32)(XRW - N) * XS : 0u) : (u32)(j & 1) * (u32)xb_;
      mb_wait(&T.xready[j & 1], (u32)(j >> 1) & 1u);
    }
    const int h = a_ ? w & (HW - 1) : w & (HWB - 1), subw = a_ ? SUBW : SUBB, nst = a_ ? NST : NSTB;
    const u32 xh = xa0 + xoff + (u32)(h * subw) * (KST * 2);
    const u32 xon = g < N ? 1u : 0u;
    float acc[2][4];
#pragma unroll
    for (int a = 0; a < 2; ++a)
#pragma unroll
      for (int c_ = 0; c_ < 4; ++c_) acc[a][c_] = 0.f;
#pragma unroll 1
    for (int s = 0; s < nst; ++s) {
      mb_wait(&T.full[slot], par);
      if (tid == 0) { if (j == 0 && s == 0) { TL(3); } if (j == E && s == 0) { TL(7); } if (j == nj - 1 && s == 0) { TL(8); } }
      const u32 cb = cb0 + (u32)slot * STAGE;
      stage(cb, cb + CODEB + lane * SCL1 - lane * 16, cb + CODEB + SCB2 + lane * SCL2 - lane * 16, xh + (u32)s * (KST * 2), acc, k16, svb_, side_, xon);
      __syncwarp();
      if (lane == 0) mb_arrive(&T.empty[slot]);
      if (++slot == ns) { slot = 0; par ^= 1; }
    }
    if (j >= NSM) { for (int it = 0; SPIN_OK(it, 11); ++it) if (ld_sm_rlx(&T.edone) > j - NSM) break; }
    if (n0 < N) sts128(pma + (u32)(j & (NSM - 1)) * (MW * 32 * 16), __float_as_uint(acc[0][0] + acc[1][0]), __float_as_uint(acc[0][1] + acc[1][1]), __float_as_uint(acc[0][2] + acc[1][2]), __float_as_uint(acc[0][3] + acc[1][3]));
    __syncwarp();
    if (M != 1 && lane == 0) mb_arrive(&T.xempty[j & 1]);
    if (lane == 0) mb_arrive(&T.rfull[j & (NSM - 1)]);
  }
  if (tid == 0) { TL(10); }
  // ---------------------------------------------------------------- final: this warp's tokens, rows 32c.. of them from the CTA's own y
  if (w >= M) return;
  mb_wait(&T.fin, 0u);
  const u32 ibw = smem_u32(&T.ib[w][0]);
  const float* const yb = T.npairs > YMAX ? reinterpret_cast<const float*>(sm + YOFF) : &T.y[0][0];
  for (int t = w; t < M; t += MW) {
    float o = 0.f;
    const int nrt = T.nr[t];
    for (int i = 0; i < nrt; ++i) {
      const int e = T.rt[t * 8 + i] >> 8;
      const u32 below = t < 32 ? __popc(T.mask[e][0] & ((1u << t) - 1u)) : __popc(T.mask[e][0]) + __popc(T.mask[e][1] & ((1u << (t - 32)) - 1u));
      const int pr = T.pb[e] + (int)below;
#if TWB
      o += yb[pr * 32 + lane] * T.twb[t * 8 + (T.rt[t * 8 + i] & 255)] * rsqrtf(T.sse[pr] * (1.f / HID) + 1e-8f);
#else
      o += yb[pr * 32 + lane] * T.rtw[t * 8 + i] * rsqrtf(T.sse[pr] * (1.f / HID) + 1e-8f);
#endif
    }
    T.ib[w][lane] = f2bf(o);
    __syncwarp();
    if (lane < 4) {
      const uint4 v = lds128(ibw + lane * 16);
      stg_v4(p.out + (size_t)t * HID + 32 * c + 8 * lane, v.x, v.y, v.z, v.w);
    }
    __syncwarp();
  }
  if (tid == 0) { TL(12); }
}
