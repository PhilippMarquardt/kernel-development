# MEM_REPORT: true weight-streaming limit and start-up cost on the H200 (2026-11-04)

The measurement scripts are not included.

## 1. Answer

| question | answer | status |
|---|---|---|
| Stream model | **T = 1.7 us + bytes / 4.62 TB/s**, over an empty kernel; raw = 6.3 + bytes / 4.62 | measured (24 sizes, 10 MB to 2 GB; slope 4.58-4.64 over every range) |
| Steady rate between stalls | **4.69-4.72 TB/s** | measured |
| Periodic stall | about every 100 us, about 5 us wide, costs about 1.5 us of stream each | measured |
| Above 4.7 on real weights | **no pattern found** | measured |
| 4.45 "limit" | wrong by 3.8% | measured |
| "66 MiB reads only reach 3.6-3.8" | wrong: 69 MB at 4.35-4.41 after subtracting the empty kernel | measured |
| Limiter | HBM side, not the SM-to-L2 network (L2-resident 8.8-9.4 TB/s) | measured for the network; HBM vs L2 fill path is inferred |
| True fixed start cost | 1.4-1.9 us above an empty kernel | measured |

Accounting check: 4.69-4.72 steady, less 1.5 us per 100 us, gives 4.62-4.65, which matches the fit.

**Best pattern (measured).** Plain `cp.async.bulk` to smem with mbarriers, no L2 hint, 132 CTAs, contiguous per CTA, unit 36-98 KB, at least 144 KB in flight per SM, source aligned to at least 128 B.
- 701 MB: 152.0-153.5 us, 4.57-4.61 TB/s end to end.
- 2 GB: 434 us, 4.61 TB/s.
- All timed runs use a 512 MB read flush before each replay.

## 2. Method and trust

- Each config is captured as a CUDA graph. Per rep: read flush, event, replay, event. A 0-byte twin of the same kernel is interleaved. Medians of 25-301 reps.
- Clocks agree (measured): 3000 replays took 0.503 s on the host, globaltimer delta 0.503 s, event sum 0.475 s (replay gaps account for the rest). Globaltimer ticks at 32 ns and is synced across SMs.
- Flush works (measured): 123 us per 512 MB flush. 14.6 MB reads 4.45 us with flush, 2.34 us without. A torch.sum flush matches mine within 0.2 us.
- Bytes checked (measured): the source holds each word's offset. The check build compares first and last word of each TMA unit, or every word for LDG, plus unit counts. No mismatches on any verified config. Compressed data: kernel XOR through TMA equals the CPU XOR.
- **Unit of "over-empty":** raw minus an empty kernel. That empty raw moved between 4.6 and 4.9 us across processes, so over-empty values carry about +-0.3 us. Raw values repeat within about 0.2 us.
- **Not calibrated to the judge.** rf_ measured an empty kernel at 3.94 raw; I get 4.6-4.9 with the same style of flush. I don't know why (it could be the pod). I did not run kdev or okc. Treat over-empty as the judge-equivalent stream time: inferred, not checked.

## 3. Rate vs configuration (701 MB unless noted; all measured)

### Unit size x in flight per SM, one issuer (TB/s, 701 MB)

| unit | 16K | 32-36K | 48K | 64-72K | 96K | 128-144K | 192-216K |
|---|---|---|---|---|---|---|---|
| 8 KB | 2.77 | 3.98 | 4.27 | 4.29 | 4.33 | 4.34 | 4.32 |
| 16 KB | 2.69 | 4.08 | 4.50 | 4.57 | 4.53 | 4.55 | 4.54 |
| 18 KB | | 4.21 | | 4.55 | 4.56 | 4.58 | 4.57 |
| 36 KB | | 4.08 | | 4.55 | | 4.60 | 4.61 |
| 64 KB | | | | 4.54 | | 4.60 | 4.60 |
| 96 KB | | | | | 4.56 | | 4.61 |

At 73 MB the best cells are 4.31-4.35 at 72-144 KB in flight. Below 64 KB in flight the rate collapses at both sizes.

### Issue-rate limit
One issuer sustains one `cp.async.bulk` per **248 ns**, about 490 clk (B=4 KB, T=1: 2.18 TB/s; T=2: 4.34; T=4: 4.55).
- A unit under about 8.2 KB per issuer is issue-limited. This is the "500 clk per stage" seen on 11-03.
- More issuers remove the limit: 4 KB x T=8 gives 4.56; 16 KB x T=4 gives 4.61.
- Self-wait vs release-only consumer: no difference (B36K: 4.57 vs 4.57-4.61).

### Grid and active SMs (36 KB S4; 9 MB per CTA up to G=66)

| CTAs | 1 | 8 | 16 | 24 | 33 | 44 | 66 | 88 | 110 | 120 | 130 | 132 | 264 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TB/s | 0.146 | 1.15 | 2.14 | 2.93 | 3.42 | 4.32 | 4.39 | 4.22 | 4.40 | 4.50 | 4.61 | 4.61 | 4.60 |
| GB/s per SM | 146 | 143 | 134 | 122 | 104 | 98 | 67 | 48 | 40 | 38 | 35 | 35 | |

- Per-SM cap 146 GB/s (74 B/clk). The total saturates by about 44 SMs. Fewer SMs never beat 132.
- G=88-120 lie below 132; unexplained. CTA placement is my guess.

### L2 policy hints (TMA, 15 reps, verified)

| hint | 36 KB | 18 KB |
|---|---|---|
| none | 4.596 | 4.552 |
| evict_first | 4.592 | 4.541 |
| evict_last | 4.567 | 4.551 |
| evict_unchanged | 4.581 | 4.553 |
| **evict_normal** | **3.827** | **3.813** |

- Avoid explicit `evict_normal` (-17%); reason unknown. Nothing beats no hint.
- `no_allocate`: ptxas 13.0 rejects it as both primary and secondary priority of `createpolicy.fractional`, so untested.
- Prefetch-ahead (`cp.async.bulk.prefetch.L2`, distance 1-64 units): 4.04 down to 2.53 TB/s, always worse. At 73 MB: 2.88-3.57 vs 4.29.

### Plain loads (verified)
Best LDG 4.537 (K=2 v4 loads per row, 1024 threads, 264 CTAs); others 4.45-4.49, down to 1.25 for few threads.
- `L2::128B` and `L2::256B` both -2%.
- LDG has a long tail: at 73 MB the last thread finishes at 23.7 us vs 16.8 for TMA, though its first load lands 0.8 us sooner.
- TMA wins by 1% at 701 MB.

### Address pattern and allocation (measured; ranges include repeats)

| variation | TB/s at 701 MB |
|---|---|
| control, repeated through the run | 4.558-4.585 |
| RR chunks 36 KB-2.3 MB, B=36 KB | 4.557-4.583 |
| RR chunks 64 KB-2 MB, B=64 KB | 4.574-4.600 |
| RR chunks 16 KB-4 MB, B=16 KB | 4.502-4.555 |
| CTA map: identity, reverse, golden, bit-reversed, 2 stripes, random | 4.557-4.585 |
| Offset 128 B to 2 GB, including 4 KB / 64 KB / 2 MB alignment | 4.560-4.596 |
| **Offset 16 B** | **3.38-3.48** |
| **Offset 64 B** | **3.25-3.35** |
| torch, cudaMalloc (800 MB to 8 GB), cudaMallocAsync, cuMemCreate | 4.559-4.578 |

- **Trap: a source below 128 B alignment costs 26%** (73 MB: 18.7-21.0 vs 16.9 us). Offsets 32 and 48 were not tested.
- Driver granularity is 2 MiB for both compression types. I did not manage to change page size; allocation size classes made no difference.
- Nothing else moves the rate by more than noise.

### Limiter (measured unless marked)

| read | TB/s |
|---|---|
| L2-resident, 132 CTAs, 19-39 MB | 8.7-8.9 end to end, 9.44 median per 2 us bin |
| L2-resident, 66 CTAs | 6.6 |
| L2-resident, 33 CTAs | 3.4 |
| Compressible all-zero, no data from HBM (inferred) | 9.06 |
| HBM, 132 CTAs, steady | 4.69-4.72 |

- The network delivers twice the HBM rate, so it is not the limit.
- Inferred: HBM. 4.70 is 98% of the 4.8 nominal and 96% of a 4.915 pin rate (6 stacks x 1024 bit x 6.402 Gb/s; stack count from the public spec). I can't separate DRAM from the L2 fill path directly.

### 100 us stall (measured; 85 dips in 60 runs)
- 0.94 per 100 us, 5 us wide, costs about 7 MB (1.49 us of stream). Absent from L2-resident reads.
- Gaps between dips in a long run are exactly 100 us. Phase vs globaltimer mod 100 us is not clearly fixed.
- A 73 MB run hits one about 17% of the time (inferred), so its median is mostly clean. d32 pays about 1, v32 about 1.5.
- Cause unknown; inferred as periodic DRAM maintenance. No software lever found.

### Imbalance (measured)
- 24 CTAs in 12 adjacent pairs finish about 25% early (113 vs 153 us at 701 MB).
- The set is **not stable per blockIdx**: (2,3) and (6,7) swap between replays, and other pairs differ across processes. Reversing the address map keeps it on the same blockIdx, so it follows the SM, not the address.
- Static rebalance by measured speed made things worse (4.57 down to 4.35).
- Dynamic atomic claim evens out every CTA (last spread 1 us) but gains nothing: raw delta +0.00 to +0.5 at 73 MB and -0.13 to +2.4 at larger sizes.
- An early -0.7 us at d1 was an artifact of the reset node in the twin.
- So the imbalance is harmless and 4.62 is not caused by it.

## 4. Ramp (exact stamps from s7/s8; us)

Kernel: 1-thread CTA, B=36 KB, S=4. Stamps run from the earliest CTA entry. Entry spread is at most 64 ns.

| requested | actual MB | raw | over-empty | excess over bytes/4.62 | first unit done min/med/max | last byte min/med/max |
|---|---|---|---|---|---|---|
| 19 | 14.6 | 9.22 | 4.61 | 1.45 | 1.86 / 2.34 / 2.98 | 3.68 / 4.06 / 4.26 |
| 28 | 24.3 | 11.39 | 6.78 | 1.52 | 1.82 / 2.37 / 2.98 | 4.99 / 6.08 / 6.30 |
| 73 | 73.0 | 21.89 | 17.28 | 1.48 | 1.73 / 2.37 / 3.30 | 12.80 / 16.42 / 16.77 |
| 238 | 233.6 | 57.06 | 52.45 | 1.89 | 1.86 / 2.40 / 3.07 | 38.50 / 51.87 / 52.38 |
| 449 | 447.7 | 103.26 | 98.66 | 1.76 | 1.70 / 2.37 / 3.04 | 72.61 / 97.50 / 98.05 |
| 701 | 700.7 | 158.11 | 153.50 | 1.83 | 1.76 / 2.37 / 3.71 | 112.22 / 152.90 / 153.54 |

Units are whole 36 KB per CTA, so 19 and 28 round down. Whole-region units are worse (19.0 MB in one 144 KB unit per CTA: raw 10.75).

### Where the fixed cost goes
Measured pieces:
- Entry to mbarriers ready 0.32, to first issue 0.51 (S=1-4). Same warm or flushed, so it isn't cold-cache cost.
- A 32-thread kernel with `__syncthreads` takes 0.96 from entry to first issue and adds 0.5 us to raw.
- First 4 KB, 1 slot per SM, from issue to complete, medians (the stamps have 32 ns resolution):

| CTAs | cold | L2 hit |
|---|---|---|
| 1 | 0.39 | 0.22 |
| 132 | 0.54 | 0.23 |

- Full burst, 4 issuers with 4 slots each: first 4 KB done at 2.21 med / 3.10 max cold vs 2.11 / 3.04 L2 hit. Burst first-byte queueing is therefore **not** HBM.
- Last-byte spread (max vs median): about 0.35 at 73 MB.

Inferred sum: 0.51 prologue + 0.54 cold latency + 0.35 tail = 1.4, leaving about 0.1 for extra kernel cost. That covers the 1.48 at 73 MB.

**True floor for the prologue (inferred):** at most 0.2-0.3 us is left to remove (first issue itself, mbarrier init). The other roughly 1.2 us is physics.

### What doesn't help (measured, delta vs control, raw us)

| experiment | 73 MB | 238 MB | 701 MB |
|---|---|---|---|
| TLB touch, 1 / 8 / 32 / 64 lines in prologue | +0.35 / +0.58 / +0.58 / +1.22 | +0.19 / +0.54 / +0.67 / +1.18 | +0.61 / +0.64 / +0.80 / +1.47 |
| Stagger 32-256 ns x cta%6, %12, %33, %132 | +0.19 to +0.64 | -0.16 to +0.58 | not run |
| 1 slot of 36 KB instead of 2-4 | +2.1 | | |

Stagger at 28 MB: +0.10 to +0.80. Slots at 73 MB: S=1 24.0, S=2 22.1, S=4 22.0, S=6 22.0 raw.

## 5. Small input read at start (topk_ids, 8 B x M, cold; raw delta vs control; measured)

| mode | M=1 | 8 | 16 | 32 | 64 | 256 | 1024 |
|---|---|---|---|---|---|---|---|
| all CTAs read before the stream, 73 MB | +0.51 | +0.51 | +0.42 | +0.45 | +0.86 | +0.61 | +1.09 |
| same, 238 MB | +0.38 | +0.16 | +0.45 | +0.35 | +0.77 | +0.61 | +1.12 |
| one CTA reads, others don't wait, 73 MB | +0.16 | +0.16 | +0.13 | +0.16 | +0.19 | +0.32 | +0.19 |
| own warp, concurrent, delta at 73 MB | +0.06 | -0.06 | +0.13 | +0.06 | +0.13 | +1.73 | +19.8 |

Concurrent read, entry to done, 73 MB (med): 0.93 / 1.02 / 1.06 / 2.56 / 8.10 / 18.66 / 36.06 for the same M.

- Budget **+0.45 us** for a blocking read at M up to 32, about +1.1 at M=1024. Reading ids before the stream is the only form that gives addresses in time; the concurrent form cannot, since addresses depend on ids (inferred).
- A single-thread sequential read during a saturated stream is very slow (M=64: 8 us). Read in parallel lanes and before the stream.

## 6. Hardware compression (`cuMemCreate`, `CU_MEM_ALLOCATION_COMP_GENERIC`)

**Bottom line: no real gain on NVFP4 weights.** At most -1.0% on a v32-sized stream, with a plane merge to decode.

### Compressor rules (measured, 380 MB, cold, kernel checksum and read-back correct)

| content of a 128 B block | TB/s |
|---|---|
| all zero or all ones | 9.06 (701 MB) / 8.96 (380 MB) |
| at most 24 B of information (nonzero prefix), constant byte, 8-16 B repeated word, u32 with up to 4 low random bits | 7.3-7.4 |
| 32-56 B of information, u32 with 8-12 low bits, u16 with 4 low bits, u32 ramp with up to 13 noise bits | 6.2-6.4 |
| 60 B or more, or 3+ random bits in every byte, n distinct values per byte (even 2), 32 B repeated word | 4.53 |

- **Block size 128 B** (inferred from a measured pattern): 64 B zero blocks with p=0.5 read 5.62, exactly the rate of 128 B blocks with p=0.25, which is the chance both neighbours are zero.
- Blocks are independent: 112 of 256 B nonzero gives 6.34; 192 of 256 gives 4.54.
- 3 bits per byte at a global or per-block base: 4.53. Delta and ramp only compress at word level (up to 13 of 32 noise bits).
- Mixed blocks average, for example 50% zero 128 B blocks: 6.42.
- Write path doesn't matter: copy engine, compute writes and memset compress the same.
- Plain allocations don't compress: all zero reads 4.57.
- Compressible type on random data: 4.53-4.59, no penalty.
- Cold check: own flush, torch.sum flush and first replay all show the gain (701 MB zeros: 9.06, 9.06, 80.35 vs 77.3 us).

### Real data

| stream (same logical size unless noted) | TB/s | decode needed |
|---|---|---|
| Codes, uniform | 4.53 | |
| Raw scales, row / tile order | 4.55 / 4.51 | none |
| Scale stats: 3.71 bits entropy, 35 distinct in 400 MB, range 92-126, top bit 0 | | |
| Rank LUT, sorted / by frequency | 4.54 / 4.54 | LUT per scale |
| Per-block min subtract, 64 B to 4 KB, raw or rank | 4.52-4.55 | add base |
| Byte delta / xor vs neighbour, rank delta, delta vs previous row | 4.53-4.55 | prefix sum / xor (serial) |
| Hi nibble / lo nibble, raw | 4.52 / 4.48 | nibble merge |
| Rank hi nibble (0.52 bits per nibble) | 5.31 | nibble merge |
| Rank lo nibble | 4.44 | nibble merge |
| Raw bit planes 0-4 | 4.33-4.41 | bit interleave |
| Raw planes 5, 6, 7 (near constant) | 8.1 / 8.1 / 7.5 | bit interleave |
| Scale widened to u16, 2x logical | 6.32 = **3.16 payload** | truncate |
| Scale widened to u32, 4x logical | 7.29 = **1.82 payload** | truncate |
| 6-bit ranks padded: 96 of 128, 192 of 256 | 4.5 = **3.4 payload** | unpack, LUT |
| 6-bit ranks padded: 3072 of 4096 | 5.61 = **4.21 payload** | unpack, LUT |
| 6-bit ranks padded: 64 of 128 | 4.54 = **2.27 payload** | unpack, LUT |

Plane streams were 95-100 MB, so compare them with their 73 MB rate of 4.3. Payload = logical rate x fill.

**Why recoding can't win (measured rates, arithmetic):** the best block tier carries 56 B at 6.33 TB/s, or 2.77 TB/s of information, vs 4.54 for dense random data. Any padded form loses to dense packing at 4.54.

### Layouts (measured, raw scales in their own allocation, 1 unit of 9; two repeats)

| layout | 701 MB us | 73 MB us | 238 MB us |
|---|---|---|---|
| one plain allocation | 153.5-153.7 | 16.86 | 52.38 |
| codes plain + scales plain | 153.6-153.9 | 17.09 | 52.54 |
| codes plain + scales compressible | 153.7-154.0 | 17.25 | 52.38 |
| codes + scales both compressible | 153.8-154.1 | | |
| one compressible allocation, mixed, K from 64 B to 64 KB | | 16.74-17.09 | |
| bound: codes plain + zeros compressible | 137.95 | | |

- Reading two allocations together costs nothing at 701 MB, and +0.2-0.4 us at 73 MB.
- Mixed in one allocation (380 MB): 4.525-4.546, at every block size K.
- The bound (-10%) is out of reach; it needs scales that take no bytes.

### Byte-saving options (time vs raw; inferred from byte counts unless marked)

| option | stream time | decode |
|---|---|---|
| Dense 6-bit scales, plain | -2.8% (0.9722 logical; 148.96 us measured) | unpack 3 bytes to 4, base or LUT |
| 5-bit + plane 5 in a compressible allocation | **147.46 us measured, -1.0% vs 6-bit**; model -0.6%; about 0.3% noise | as above + one-bit plane merge |
| 5-bit + sparse exceptions | -4.2% | unpack, patch rare values; byte<96 occurs at 1.7e-7 (measured), so about 17 per 100M scales (inferred) |
| Entropy limit 3.71 bits | -6.0% | real entropy decoder |

Plane 5 is all ones in 99.98% of 128 B packed blocks (measured). Only byte counts have been modelled for these; no decode was built, and the cost of each decode inside a real kernel is unknown.

## 7. Implied stream time per shape (us; no math, no sync, uniform split)

Measured points are 0-1.2% away in bytes from the shape sizes and are corrected at 4.62 TB/s. Over-empty carries about +-0.3 us.

| shape | MB | raw, 1-thread | **stream, 1-thread, over-empty** | 32 threads + release consumer | consumer + ids read |
|---|---|---|---|---|---|
| d1 | 73 | 21.9 | **17.3** | 17.8 | 18.3 |
| v1 | 238 | 58.0 | **53.5** | 54.0 | 54.5 |
| d16 | 337 | 79.6 | **75.1** | 75.5 | 76.0 |
| d32 | 449 | 103.5 | **98.9** | 99.5 | 100.0 |
| v16 | 637 | 143.6 | **139.0** | 139.7 | 140.2 |
| v32 | 701 | 158.1 | **153.6** | 154.4 | 154.9 |

Against rf_ROOFLINE (19.5, 51.7, 76.1, 100.2, 143.4, 154.8 at slightly different bytes, real routing, raw - 4): roughly 2 us better at d1 and 0-3 elsewhere once bytes are scaled. **Not like for like:** different offset convention, and mine has no expert segmentation. With -2.8% from 6-bit scales the stream drops by that share (inferred, decode not included).

## 8. Not done or not tested

- Tensor-map TMA: not run (rf_ measured 2D boxes at 4.0-4.05).
- Offsets 32 and 48 B.
- Page size control beyond allocation class.
- `no_allocate` (rejected by ptxas).
- Real routing and expert segments: this is a uniform split only. All numbers are memory only.
- Stability on the judge's H200: the fast-SM set and dip behaviour may differ.
