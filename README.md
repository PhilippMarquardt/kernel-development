# kernel-development (`kdev`)

`kdev` is a toolkit that coding agents (Claude Code, Codex) use to write and optimise GPU kernels on a remote
GPU host. It plays the role for kernel work that REA ("Reverse Engineer Anything") plays for reverse
engineering: one stable contract over many backend tools. The backends here are nvcc/ptxas, cuobjdump/nvdisasm,
CUDA graphs, CUPTI, `%globaltimer` instrumentation, compute-sanitizer, and ncu/nsys when they are permitted.

**Status: v0.1, all packages built (P1-P8) and run end-to-end on the OKC H200 pod** (build, check, time,
calib, sass, ab, serve, mcp). Agents: start with [docs/QUICKSTART.md](docs/QUICKSTART.md) (the 10 commands, in
order) and [skills/kdev/SKILL.md](skills/kdev/SKILL.md). Design: [docs/DESIGN.md](docs/DESIGN.md); binding plan:
[docs/PLAN.md](docs/PLAN.md); timing protocol and calibration evidence: [docs/PROTOCOL.md](docs/PROTOCOL.md).

## Install

- On the GPU pod, from a local checkout: `scripts/pod_install.sh [HOST] [PORT]`. The script copies the repo
  (md5-verified) to `/root/kdev` and runs `uv pip install --no-deps -e` into `/root/vsgl`. It aborts if any
  existing package would change. It writes `/root/.kdev/kdev.toml` (lock `/root/gpu.lock`, okc_root
  `/root/okc`) and `/root/.kdev/toolchain.lock` (judge CUDA 13.0 at `/root/cuda130w`). Pod storage is not
  persistent, so re-run it after every pod restart.
- Locally (ledger, sass, dashboard, MCP, tests; no GPU needed): `pip install -e ".[dev,mcp]"`, then `pytest`
  and `ruff check src tests`.

## Design in brief

- **Interface.** `kdev` is a CLI with a JSON result contract, built over a Python library. It runs on the GPU
  pod and is driven over ssh, either directly or through `kdev remote`. A Claude Code skill describes the
  workflow. `kdev mcp` is an MCP stdio server generated from the same command registry.
- **Judge fidelity by default.** Every timing uses:
  - the pinned judge toolchain, recorded from the built `.so`;
  - a read flush of L2;
  - CUDA-graph replay;
  - interleaved A/B with paired ratios and bootstrap CIs;
  - multiple seeds when the input sample changes the work (e.g. MoE routing);
  - overhead subtraction;
  - an exclusive-GPU lock that detects unlocked CUDA processes.

  Opting out of any of these is possible, but it is explicit and recorded in the result.
- **Context-economical.** Each command prints a summary of at most ~40 lines and the paths to its files. The
  full output (samples, SASS, traces) goes to a run directory, and `kdev show` reads it back in pages.
- **Tools.**

  | Tool | What it does |
  |---|---|
  | `doctor` | Checks the GPU, toolchain, lock and permissions |
  | `build` | Compiles only, never under the GPU lock; reports ptxas regs, spills and C751x warnings |
  | `check` | Correctness on multiple seeds, eager and graph, repeated |
  | `time`, `ab` | Judge-style timing of one file, or interleaved comparison of several |
  | `sass` | Hot-loop opcode mix; audits for PDL hoisting, spills and serialization |
  | `timeline` | Per-CTA phase timeline from `%globaltimer` marks |
  | `floor` | Roofline plus a latency skeleton built from measured constants |
  | `sweep` | Grid search over build knobs |
  | `regs` | Register and occupancy report |
  | `sanitize` | Runs compute-sanitizer |
  | `profile` | Uses ncu when permitted, otherwise CUPTI |
  | `dispatch` | Shows which kernels actually run for each shape |
  | `calib` | Compares local timings against the judge |
  | `judge` | Prepares and records judge runs (record-only) |
  | `comply` | Checks the problem's rules |
  | `queue` | GPU lock and queue status |
  | `sync`, `remote` | Copies results between pod and local machine with md5 checks; runs commands on the pod |
  | `note` | Records claims, each tied to the runs that back it |
- **Safety.** `kdev` never submits to a judge. It refuses to compile while holding the GPU lock, guards
  against reward hacking, and declares each tool's effects.
- **Extensible.** Problem adapters supply the reference, checks, inputs and score. The first ones are OKC,
  KernelBench, KernelBot, standalone `.cu` and Triton functions. GPU profiles (`gpus/*.toml`) hold each GPU's
  peak numbers and measured constants.

The requirements come from the Open Kernel Challenge, 2026-10: about 40 agents on one H200. They repeatedly
rebuilt the same harnesses. Several days of results were invalidated by a write flush, the wrong nvcc, timing
on one seed and comparing ratios instead of absolute times. DESIGN.md Section 3 maps each of those mistakes to
a requirement.
