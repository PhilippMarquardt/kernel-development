# Open Kernel Challenge: GPU kernel optimization case study

**148th of 2,014 participants (top 7.3%), while seriously pursuing only 2 of the 5 problems.** The other three mostly ran on public engines with light tuning.

| Problem we pursued | Speedup | Rank | Top | Behind the winner |
|---|---|---|---|---|
| BF16 projection | 1.610x | 43 of 1,747 | **2.5%** | **6.8%** (winner 1.728x) |
| MoE | 2.512x | 60 of 1,426 | **4.2%** | **11.7%** (winner 2.844x) |

The Muse Spark Open Kernel Challenge asked 2,014 participants to speed up five inference kernels on an NVIDIA H200, using a judge that measures real speedup against a stock baseline and checks that results stay correct.

The work was done by AI coding agents under my direction on one rented H200, using `kdev`, the toolkit in this repository, which was built for the challenge. My part was setting direction, reviewing evidence and deciding what to pursue.

## Results

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="progress/progress-dark.png">
  <img alt="Line chart of score over time for MoE and BF16, Oct 30 to Nov 5. Both rise steeply in the first day. BF16 keeps rising slowly, and MoE stays flat until a second rise on Nov 5. Final values: MoE 2.512, BF16 1.610." src="progress/progress.png">
</picture>

These are the board values when the kernel track closed on 2026-11-05, and the organizers could still revise them. Each one is the best of several noisy runs: the same MoE file scored between 2.451 and 2.512. The data behind the chart is in [`progress/progress.csv`](progress/progress.csv).

## What's mine and what isn't

The submitted kernels build on public engines from other participants, mainly alygg77, with my tuning on top. Adopting those engines caused the big jump on the first night. **Those submissions aren't in this repository, because they contain other people's code.**

My own work is:

- **`kdev`**, the measurement and multi-agent toolkit, built with AI agents. See the [repository README](../README.md).
- **MoE engines for small batch sizes.** Together with tuning the public engine for large sizes, they raised MoE from 2.366 to a repeat-check level of about 2.49. The sources are in [`engines/`](engines/).
- **BF16 tuning.** Tuning the public engine raised BF16 from 1.539 to a repeat-run level of about 1.60.
- **Hardware measurements** that showed where the time was going, in [`reports/`](reports/).

## How the work ran

1. **Match the judge.** Local timing had to reproduce the judge's toolchain, cache state and replay order before any result could be trusted.
2. **Measure before designing.** Reports like the [weight-streaming limit](reports/weight-streaming-limits.md) separated hard limits from fixable overhead.
3. **Set a kill gate first.** Each design got a pass and kill threshold before it ran, so three ground-up MoE rewrites were dropped quickly. The best of them reached only 0.70x of the best file.
4. **Harden what ships.** Candidates went through poison-scratch, timeout and repeated-replay tests before submission, and independent reviewers read the source.

[`LESSONS.md`](LESSONS.md) covers what this taught me, with the numbers.

## In this folder

| Path | Contents |
|---|---|
| [`LESSONS.md`](LESSONS.md) | What went wrong and what I'd keep |
| [`engines/`](engines/) | MoE engines as shipped: `sc_h.cu` (M = 1), `x3_h.cu` (M up to 8), `x7_h.cu` (M = 8, tuned) |
| [`reports/`](reports/) | H200 measurements: [weight streaming](reports/weight-streaming-limits.md) and [compute core](reports/compute-core-limits.md) |
| [`progress/`](progress/) | Score history, and the script that draws the chart |
