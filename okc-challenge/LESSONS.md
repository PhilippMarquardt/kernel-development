# Lessons from the Open Kernel Challenge

## Measuring

**Match the judge's toolchain first.** A local build with nvcc 12.8 serialized an attention kernel that the judge's CUDA 13 build did not. That made one kernel look 27% slower (491 vs 357 µs) and flipped a design decision. A local result is only comparable once the compiler version matches.

**Reproduce the judge's cache state.** Our harness flushed the L2 cache by writing a buffer, and the judge flushes by reading one. The dirty lines were written back during the timed kernel, which added 1–2.5 µs to small kernels and inflated every estimate of the gap to the floor.

**Replay in the judge's order.** The judge interleaves baseline and submission replays. Doing the same locally moved some timings by about 0.35 µs and reversed one verdict, from -0.49 to +0.26.

**Fixed seeds can bias a result.** Five fixed routing seeds gave per-seed medians from 79 to 113 µs on one shape, and one candidate read as 3 µs faster when it was really 6 µs slower. A fresh pool of 100 routings brought local times within about 2% of the judge on most shapes.

**A leaderboard value isn't a level.** The same MoE file scored from 2.451 to 2.512. I started comparing files on repeated checks and stopped treating a new high as an improvement.

## Deciding

**"At the limit" is a hypothesis.** Agents reported kernels as near their floor, and I accepted it. The gains were there in cheap knobs and in hot-loop instructions nobody had read. A floor claim now needs the disassembly accounted for and the cheap options searched.

**Run cheap experiments before rewrites.** Early reviews ranked big redesigns first, and nearly all of them measured about zero. Searching existing library algorithms and reading the disassembly came first after that.

**Set kill gates at the start.** Three ground-up MoE kernels failed, and the best reached only 0.70x of the best file. One-hour gates ended each of them cleanly.

**Test the model on the target hardware.** All three rewrites relied on a time model of bytes over bandwidth plus a small constant. Fitting time against expert count showed what that hid: one design had a 27–29 µs fixed cost, and another lost on every byte because compute didn't overlap with the memory stream.

**One kernel for all shapes pays everywhere.** The score is a geometric mean, so 1 µs on the smallest shape was worth 8 µs on the largest. Builds specialized for each size saved an estimated 3.5–5 µs on one design.

## Running AI agents

**More agents isn't more progress.** For each report I needed to ask what evidence backed the claim and what cheap experiment could falsify it.

**Evidence has to outlast the agent.** Successors inherited conclusions like "at floor" or "dead" without the numbers behind them and tested nothing again. Notes now record the exact variant, method and numbers.

**Use reviewers who haven't seen your history.** Reviewers given our full history proposed variations on what we'd already tried. Clean-sheet reviewers saw only the problem and the public numbers, and gave different designs. Banning our own measurements went too far, though: agents then paid GPU time to rediscover known costs.

**Check that a test can fail.** One test ran 1/40 of the calls it claimed and exited 0 when it printed FAIL. Independent source review caught it, so I now read call counts and add a positive control.

## Hardware findings

Measured on the H200 unless noted:

- Weight streaming follows `T = 1.7 µs + bytes / 4.62 TB/s`, and no access pattern on real weights beat about 4.7 TB/s. See the [report](reports/weight-streaming-limits.md).
- The best compute core reached 6.91 TB/s-equivalent at N = 8, falling to about 2.0 at N = 128, where the tensor units become the limit. See the [report](reports/compute-core-limits.md).
- `ptxas` can hoist an `asm volatile ld.global.cg` out of a spin loop, which causes an infinite spin. Use relaxed or volatile loads with a time-bounded trap.
- A flag passed across SMs costs about 1.4 µs idle and about 6.3 µs while memory is saturated (figures from another participant's published notes).
- Each extra small graph node costs about 1.7 µs, and a dependent cold DRAM round trip about 0.65 µs. Cold data-TLB misses cost almost nothing, so page-touch tricks never helped.
