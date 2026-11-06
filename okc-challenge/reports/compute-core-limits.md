# CORE_REPORT: fastest y = W x per N, the clock, and the k3 gap (H200, 2026-11-04)

Measured on an H200 (132 SMs) with exclusive GPU access. Built with nvcc through `load_inline` (`-O3`, sm_90a, no forced C++ standard). SASS comes from the cubin inside the extension that ran (`ck.py`).

Unless marked **inferred**, a number is measured. "Core" numbers are self-contained loops in smem with 1 CTA/SM and no TMA, so they are ceilings for a kernel, not kernel rates. The reference was used only as the black-box number in `runs_local/gate2/current.log`.

Units:
- One step = 16 rows x 16 K = 144 B of weights.
- Memory budget = 18.7 B/clk/SM at 1.80 GHz (4.45 TB/s).
- Prefer B/clk/SM when comparing variants. TB/s moves with the clock, and the clock has history (section 3).

## 1. Answers

| N | fastest core measured | TB/s-eq | B/clk/SM | instr/step | limit | vs stream |
|---|---|---|---|---|---|---|
| 1, 2, 4 | same as N=8 (padding changes nothing, section 2b) | 6.55-6.64 with the rotate decode; 6.91 with XOR (**inferred**, XOR not run padded) | 27.6-27.9 (rotate) | 15.2-15.8 | issue, ALU + FMA pipes | above |
| 8 | mma.sync, B in registers, XOR decode with plain shifts, 16 warps | **6.91** | 29.10 | 14.2 | issue, ALU + FMA pipes (IPC 2.88); tensor about 20% of nominal | above |
| 16 | wgmma RS m64n16k16, 16 warps, commit 8, wait 1, XOR | **5.72** | 23.91 | 16.1 | decode issue; IPC falls as N rises (not explained) | above |
| 32 | wgmma RS m64n32k16, 16 warps, commit 4, wait 2, XOR | **5.09** | 21.30 | 16.0 | decode issue; tensor 59% busy | above, by 14% only |
| 64 | wgmma RS m64n64k16, 16 warps, commit 4, wait 1 | **3.55** | 15.74 | 16.3 | tensor (87% of nominal) | **below** |
| 128 | wgmma RS, 8 warps | **2.00** | 9.00 | 16.3 | tensor, 100% of nominal | below |
| 256 | wgmma RS, 8 warps | **0.84-1.02** (clock 1.47-1.72 GHz) | 4.50 | 15.5 | tensor, 100% of nominal | below |

- No core reached 7 TB/s at N <= 8; the best is 6.91.
- N = 32 keeps up with the stream in isolation. Ring + TMA took 25% off the N=8 stage at 18 KB stages in PM_REPORT, so N = 32 probably will not keep up inside a kernel (**inferred**).
- N >= 64 cannot keep up on any core: 2048 MAC/clk/SM gives 18 B/clk/SM at N=64, below the budget.

## 2. Tables

### 2a. Rate vs N, by core (B/clk/SM | TB/s). Rotate decode unless noted; scale pair per 4 steps

| N | mma B regs | mma ldmatrix | mma LDS.128 | wgmma RS decode (best W) | wgmma tensor only | acc in smem |
|---|---|---|---|---|---|---|
| 8 | 27.9 / 6.63 (W32, plain shift) | 26.1 / 6.19 (W32) | 25.1 / 5.96 (W16) | 25.8 / 6.17 (W32) | 44.3 / 10.5 | |
| 16 | 22.0 / 5.25 (W16, plain shift) | 20.5 / 4.85 (W16) | 20.0 / 4.74 (W16) | 23.3 / 5.52 (W16, GS8) | 44.3 / 10.5 | |
| 32 | 14.9 / 3.55 (W16) | 14.5 / 3.42 (W16) | 14.2 / 3.36 (W16) | 20.8 / 4.96 (W16, GS4 WT2) | 36.0 / 7.8 | |
| 64 | 9.1 / 2.17 (W8) | 8.0 / 1.88 (W16) | 7.8 / 1.85 (W16) | 15.8 / 3.56 (W16, GS4) | 18.0 / 3.87 | 1.76 / 0.42 |
| 128 | no registers | 4.2 / 0.99 (W12) | 4.0 / 0.95 (W8) | 9.0 / 2.00 (W8) | 9.0 / 1.94 | 0.89 / 0.21 |
| 256 | no registers | 1.7 / 0.41 (W8, spills) | 2.1 / 0.49 (W8) | 4.5 / 0.97 (W8) | 4.5 / 0.97 | 0.44 / 0.11 |

Warps per SM, mma N=8 B in registers, plain shifts, rotate decode: W8 6.32, W12 6.45, W16 6.55, W24 6.57, W32 6.62. wgmma N=32 (GS2): W4 3.12, W8 4.04, W16 4.61, W32 4.54 (C7512, serialized).

Files: `r_mr.log`, `r_ml.log`, `r_ms.log`, `r_wg.log`, `r_wc.log`, `r_t4*.log`, `r_t6.log`, `r_t7.log`.

### 2b. What limits each (B/clk/SM)

| N, W | full | no scale | no scale, no decode | tensor only | nominal 2048 MAC/clk |
|---|---|---|---|---|---|
| 8, W32 | 25.8 | 28.9 | 41.0 (C7512) | 44.3 | 144 |
| 16, W32 | 21.5 (C7512) | 24.8 (C7512) | 37.0 (C7512) | 44.3 | 72 |
| 32, W16 | 19.8 | 22.2 | 34.4 | 36.0 | 36 |
| 64, W16 | 15.8 | 17.2 | 18.0 | 18.0 | 18 |
| 128, W8 | 9.0 | 9.0 | 9.0 | 9.0 | 9 |
| 256, W8 | 4.5 | 4.5 | 4.5 | 4.5 | 4.5 |

Measured:
- **Tensor pipe:** wgmma reaches exactly 2048 MAC/clk/SM from N=32 up. N=128 and N=256 are at 100% with full decode.
- **Small-N wgmma:** at N=8 and 16 it stops at 44.3 B/clk/SM, about 13 clk per HGMMA per SMSP against 4 nominal.
- **Smem ports:** not binding on these loops. N=256 hits nominal with W4 and W8 alike.
- **Issue:** at N <= 32, removing decode raises the rate far more than removing anything else.
- **ldmatrix:** 0.25/clk/SM caps mma at about 9 B/clk/SM at N=64 and about 4.5 at N=128 (ml64 8.0, ml128 4.2).
- **Registers:**
  - N=256 needs W <= 8.
  - C7512 serialization at N=64 W24, N=32 W32, N=16 W32 with GS4.
  - Spills in the hot loop at mr4 W24, ml64 W24, ml256 W8 and all of ma128/ma256 above W8.
- **Accumulators in smem:** 4x slower than registers. Dead.

Not explained: IPC falls with N while the tensor pipe has room (wgmma 3.16 at N=8, 2.39 at N=32 with tensor 55% busy). A deeper queue (WT 2 or 3) moved it only 5%.

Wgmma knobs, per N:
- **Commit group:** GS4 is best or close to it at most N; GS1 is 4-14% slower than GS4. GS8 wins at N=16 and loses at N=32 W8.
- **Wait:** `wait_group 0` costs 3-21% against wait 1. Exception: N=16 W32, where GS4 with wait 1 serializes (C7512) and wait 0 does not.
- **Descriptor in R instead of UR:** 20.2 vs 17.8 instr/step, 8-17% slower at N <= 32, 5% by B/clk at N=64. Keep UR.
- **Fence per group:** no cost (5.21 vs 5.21 at N=8 W8; at N=64 W16 the B/clk is unchanged).

Layouts:
- Tokens as M with mma (A = x, B = W, N=16): 21.75 vs 22.0 with tokens as N, same instruction mix. No gain.
- wgmma SR (A descriptor, B in registers) is rejected by ptxas (`Arguments mismatch`), so wgmma tokens as M would need decoded weights in smem. Not tested; earlier docs call that dead.
- Row-major B on wgmma: not tested, only K-major SW128.
- N <= 8 padding: pad1 / 2 / 4 / 8 at W16 read 27.67 / 27.68 / 27.58 / 27.58. Mostly-zero x neither helps nor hurts.

Dense peak: wgmma runs at nominal per clock; the shortfall from 989 TFLOPS is the clock.
- N=256 equivalents: 0.84 TB/s = 765 TF (20 us burst, 1.47 GHz), 0.97 = 883 TF (3 ms), 1.02 = 930 TF (best run, 1.72 GHz).
- 795 TF is nominal at 1.47 GHz, the clock seen in 20 us bursts (**inferred**, the 795 run was not repeated).

### 2c. Decode and scale alternatives, in situ (TB/s; B regs N=8)

| variant | W8 | W16 | W32 | matters? |
|---|---|---|---|---|
| 9-op rotate, IMAD by register multiplier (k3) | 6.03 | 6.37 | 6.35 | baseline |
| 9-op rotate, plain shifts | 6.32 | 6.55 | 6.62 | yes: +3-5% |
| fk1 8-op XOR as written (opaque multiply, umulhi) | 5.56 | 5.87 | 5.98 | worse, -6 to -8% |
| **8-op XOR, plain shifts** | 6.53 | **6.91** | 6.78 | **best: +7-9% over k3** |
| XOR, plain-shift scale pair too | 6.41 | 6.55 | 6.65 | no gain (28.7 vs 29.1 B/clk at W16; clock history also lowers it) |
| scale pair every step (nk) | 5.58 | 5.82 | 5.91 | -7 to -9%: K permutation matters |
| FFMA scale on accumulator, x biased 2^119 | 5.88 | 5.71 | 5.66 | no: worse than HMUL2, needs biased x |
| FFMA + fk1 XOR as written | 5.52 | 5.44 | 5.41 | no |
| no scale at all (upper bound) | 7.53 | 7.47 | 7.53 | scale work costs 17-25% |
| no decode at all (upper bound) | 10.27 | 11.06 | 11.18 | |

- XOR plain at other N: mma N=16 W16 5.32 (vs 5.25), wgmma N=32 W16 4.97 (vs 4.72, GS4 WT1).
- In the real k3 kernel the decode matters much less: 7.23 -> 7.18 (plain) -> 7.08 us/expert (XOR).
- XOR needs fk1's bit layout in prep. Correctness comes from fk1's own PASS. These probes use random codes and did not re-check it.

**Subnormal A is exact.** HMMA m16n8k16 and HGMMA m64n8k16 RS, 3 trials, 512 values each: A = E2M1*2^-126 (raw decode) with x*2^119, result * 128, equals A = E2M1 with x, 0 differences, bit for bit (`r_t5.log`). Also no slow path: no-scale runs with biased x and with f32-subnormal results both read 7.53 TB/s. fk1's 9.7 vs 6.6 us on constants did not reproduce. The FFMA route is possible but pointless on the numbers above. Overflow of x*2^119 was excluded in the test by construction (x exponent checked), not handled.

## 3. Clock

**1.815 GHz is real, and wgmma does not escape it.** ALU rate tracks %clock64 (W8, SHF per ns per SM, loop SASS read):

| kernel | clock64 GHz | SHF/ns | SHF/clk |
|---|---|---|---|
| ALU only; HMUL2 sparse | 1.98 | 3.90 | 1.97 |
| HMMA in a dead branch | 1.98 | 3.81 | 1.93 |
| HGMMA in a dead branch | 1.98 | 3.84 | 1.94 |
| 1 HMMA per 1024 SHF per warp | 1.80 | 3.58 | 1.99 |
| 1 HGMMA N=64 per 1024 SHF per warp | 1.73-1.75 | 3.41-3.45 | 1.97 |
| 1 HGMMA N=8 same | 1.70-1.74 | 3.37-3.45 | 1.98 |
| 1 HGMMA N=256 same | 1.80 | 3.47 | 1.93 |
| 1 HMMA or HGMMA per 16384 SHF | 1.96 | SHF rate invalid: I-cache | |

- **PM_REPORT correction:** the drop is not static. A tensor op that is never executed keeps 1.98, and so does a very sparse one. It is dynamic and duty-dependent. Any real core is far above the threshold (it needs about one tensor op per 7.75 clk).
- **Levels seen:**
  - N <= 16: usually 1.80; range 1.70-1.815.
  - N=32: 1.65-1.81.
  - N=64: 1.62-1.81.
  - N >= 128: 1.47-1.72.
  - The spread within a row is mostly run order (see History).
- **Bursts are not faster:** mma N=8 at 20 us 1.742, 100 us 1.788, 3 ms 1.800. N=256 at 20 us 1.47, 100 us 1.51, 10 ms 1.60.
- **Idle gap:** 0 to 1 s before a 100 us burst changes nothing (N=8 1.77-1.79, N=32 1.73-1.75, N=256 1.68-1.69).
- **History:** the clock depends on what ran before, on a slower timescale. The same kernel read 1.80 and then 1.545 right after 10 ms dense bursts. This is why B/clk is the safe comparison. The timescale was not measured.
- **Duty threshold:** somewhere between 1 tensor op per 1024 and per 16384 SHF per warp. Not resolved, because the fine sweep ran after dense kernels and is contaminated by history.

## 4. k3 vs reference: the 11%

Reference (`gate2/current.log`, crafted routing, least squares): 6.46 us/expert, intercept 12.8. Crafted routing 1, 2, 4, 8, 16, 40 reps; k3a = `pm/k3.cu` + switches (`mk_k3a.py`). Same session; HMMA count checked in every build (`hcount.py`).

| build | us/expert | TB/s | intercept | step |
|---|---|---|---|---|
| base k3 (correct output) | **7.23** | 3.92 | 15.1 | |
| no global epilogue work (stores, counters, P1 wait) | 6.98 | 4.05 | 10.9 | -0.25 |
| epilogue only releases the slot (no partial reads, gs/pw1 loads) | 6.87 | 4.12 | 10.8 | -0.11 |
| no handshake (no rfull/rfree), consumers still store | 6.51 | 4.35 | 10.7 | **-0.36** |
| accumulators read per job, no smem store | 6.53 | 4.33 | 10.5 | +0.02 |
| math only + x: no per-job accumulator read | 6.49 | 4.37 | 10.8 | -0.04 |
| same, no per-key x reload | 6.43 | 4.40 | 10.9 | -0.06 |
| no math, ring only | 6.26 | 4.52 | 9.0 | -0.17 |

Other builds:
- no math, stores + handshake on: 6.30. So the handshake is free without math and costs 0.36 with it. The mechanism is not established (warp skew at the stage end is my guess).
- no math, reduction and epilogue on: 6.81.
- x only off in base: 7.06 (-0.17).
- plain shifts in base: 7.18. XOR in base: 7.08. `PM_NOSYNC`: 7.17.
- no final pass: 7.12.
- one accumulator chain: 7.40 (worse).
- with math + x only, plain shifts 6.39, XOR 6.37.

**Answer: the 11% is the per-job reduction path, not the math.** Of the 0.77 us/expert gap to 6.46, the hand-off (0.36) and the epilogue (0.36) make up 0.72. Math + x with no reduction runs at 6.49, about the reference's rate, so the math is not what separates k3 from the reference at N=1. The smem stores themselves cost about 0: 6.29 vs 6.26 with no math, 6.51 vs 6.53 with math. The reference produces correct output at 6.46, so a cheap reduction path exists; how it does that is unknown.

The epilogue also carries about 4 us of fixed cost (15.1 -> 10.9), and math about 1.8 us (10.8 -> 9.0).

Discarded runs:
- My first ablations (`nored`, `noxred`, `shlall`, `r_nost`, `r_arr`) read 6.25-6.32, but ptxas removed the HMMAs because the accumulators were unused (`HMMA 0` in SASS). Their slopes are no-math numbers and were not used above.
- `-DNRB=4` does not fit in smem (invalid argument at launch).
- `-DNEPI=4` hung until the 235 s timeout, holding the lock that long; not investigated.

## 5. Which design per N (inferred from the above; no kernel was built)

| N | core | why |
|---|---|---|
| <= 8 | mma.sync, XOR plain shifts, K-permuted scale pair, 16 warps; ldmatrix with row split (28.67 B/clk) or B in registers with K split (29.10) | ldmatrix costs 1.5% here. A row split needs no cross-warp hand-off, and the hand-off is the largest loss found in k3. Worth a test. |
| 9-16 | wgmma m64n16k16 W16 (23.9), or mma B regs W16 (22.46) | Both clear the stream. mma is simpler, wgmma has 6% more margin. |
| 17-32 | wgmma m64n32k16, W16 | mma is below the stream (14.9). wgmma at W8 is at the stream (18.3) and needs W16 for margin. |
| 33-64 | wgmma m64n64k16, W16 | Tensor-bound; one pass. |
| 65-256 | wgmma m64nNk16, W8 | Tensor-bound at nominal. Decode no longer matters. |

Estimated core-only excess against today's measured hot cost (**inferred**; rates above, stream 6.4 us/expert):

| case | core-only excess | measured on the reference |
|---|---|---|
| E16 N16 | 0 | +10 us |
| E8 N32 | 0 | +18.5 us |
| E4 N64 | about +6 us | +33 us |
| E2 N128 | about +16 us | +44 us |
| E1 N256 | about +22 us | +49 us |

**Main caveat:** this leaves out x delivery, which no probe here measured. At N=256 a CTA holding whole K pulls 2 MB of x against about 214 KB of weights, so x traffic may be the real N >= 64 limit. Measure that before trusting this table.

## 6. Not tested

Core next to live TMA and a ring (the real overhead per N), x delivery at high N, K-split across CTAs at high N, 2 CTAs per SM, wgmma with row-major B, XOR decode correctness on these probes, clock under real memory traffic.

## 7. Files (`pm2/core/`)

- `core.h`, `ck.py`: probe templates and harness.
- `t1.py` table 1; `t2.py` decode; `t3.py`, `t5.py` clock (`t5.py` also exactness); `t4.py`, `t7.py` wgmma knobs; `t6.py` padding, limits, tokens as M, idle gap; `t8.py`, `t9.py` XOR.
- `mk_k3a.py`, `k3a.cu`, `abl.sh`, `fit.py`, `hcount.py`, `k3sass.py`: k3 ablations; logs in `abl/`.
- `r_*.log` run logs, `b_*.log` build logs with ptxas output, `*.json` per-kernel rows.
- Note: `r_t3.log`, `r_t4*` and `r_t6.log` were built before `core.h` gained `DEC` 4 and 5; existing `DEC` values are unchanged.
