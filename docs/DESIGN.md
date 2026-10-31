# kdev: an agent-facing toolkit for GPU kernel development (design)

Status: design, 2026-10-31. Nothing here is implemented yet.
Sources: Section 2 surveys outside work. Section 3 draws on our own experience: the Muse Spark Open Kernel
Challenge (OKC), 5 problems, 2026-10-22..31, about 40 agents on one H200 pod. The OKC evidence is in
`arcagi3/okc/` (LESSONS.md, TRACKER.md, calibration/all_calib.md, research/tlb/SUMMARY.txt,
muse_attention_decode/.../floor/RESULTS.md, and the per-problem NOTES.md files).

---

## 0. One-paragraph summary

`kdev` is a CLI with a JSON contract and a Python library under it. It is installed on the GPU host. Agents
(Claude Code, Codex) drive it locally, or over ssh through a thin `kdev remote` client. It turns the ~150 ad-hoc
scripts our OKC agents rewrote over and over into about 15 composable tools:

- build (outside the GPU lock, with the judge's pinned toolchain);
- check (multi-seed, eager + graph, repeated);
- time / ab (judge-faithful graph replay, read flush, interleaved, with CIs);
- sass (hot-loop histograms, ptxas warnings, PDL hoist audit);
- timeline (globaltimer instrumentation);
- floor (roofline plus measured microbench constants);
- sweep;
- regs/occupancy;
- sanitize;
- calib (judge-vs-local);
- a GPU queue/lock;
- sync/backup.

Every tool writes its full output to a run directory. It returns a short summary (at most ~40 lines) plus the
file paths. Problems plug in through a small adapter (reference, checks, inputs, scoring). A thin MCP server is
an optional later layer over the same library, not the primary interface.

---

## 1. Goals / non-goals

### Goals
1. **Judge fidelity first.** A local number from `kdev` must predict the judge to about ±3-5% per shape, with
   the error and its source stated. Most of our lost time came from local numbers that lied (Section 3).
2. **Make the right thing the default.** Defaults: read flush, graph replay, the pinned toolchain, interleaved
   A/B, multiple seeds, and building outside the lock. An agent has to *opt out* of fidelity explicitly. Every
   opt-out is recorded in the result, so it can never happen silently.
3. **Context-economical output.** Each call costs the agent at most ~2 KB of context. The details go to files
   the agent can `Read` selectively.
4. **Evidence, not conclusions.** Every result records the toolchain, git/md5 of the inputs, GPU state, method
   and raw samples (REA's "evidence with every conclusion"; our LESSONS 6). Claims such as "at the floor" or
   "dead" must point to a run id.
5. **Multi-agent safety on one GPU.** A queue/lock that also detects unlocked CUDA processes. No compiling while
   holding the lock. A visible list of offenders.
6. **Reusable across problems and GPUs.** OKC's 5 problems are the first adapters. KernelBench-style
   `Model/ModelNew`, a GPU MODE KernelBot problem, or a plain `.cu` + `main()` must each be adaptable in under
   100 lines. H100/H200/B200/RTX are handled as GPU profiles.

### Non-goals
- Not an autonomous optimizer or agent loop. Orchestration stays with the agents and the coordinator. `kdev` is
  the instrument panel. (HotKernel bundles a loop as a *skill*; we ship workflow guidance the same way, as a
  skill, not as code.)
- No kernel generation, LLM calls, or "auto-tune my kernel" magic beyond explicit `sweep` grids.
- **Never submit to a judge.** At most, `kdev` *prepares* a judge payload and records judge results that a
  human-authorised command produced. Any submit or check that consumes a quota is out of scope (Section 7).
- No replacement for Nsight. When counters are permitted, `kdev` wraps `ncu`/`nsys` and parses them into
  summaries. It does not reimplement them.
- No GUI. Timelines are emitted as Perfetto/Chrome-trace JSON plus a text summary.

---

## 2. Survey: what exists and what we take from it

### 2.1 REA, "Reverse Engineer Anything" (github.com/morluto/rea, MIT, 2026)
This is the model the user named. REA is a bridge that exposes Ghidra/IDA/Hopper/JADX/Frida/TShark and built-in
parsers to coding agents through **both MCP and a CLI that share the same workflows**. Its install
(`npx rea-agents setup`) registers the server *and* installs a skill ("investigation instructions") for each
agent. Design points worth copying, from docs/tool-design.md and docs/mcp-contracts.md:
- **Backends behind one contract.** The tool selects a provider (Ghidra vs IDA) per target. Missing backends are
  reported as `capability_unavailable`, not as crashes. -> Our `ncu` may be blocked: fall back to
  globaltimer/torch.profiler and say which backend produced the number.
- **Compact schemas by default** (`REA_MCP_INPUT_SCHEMA_PROFILE=compact`) to cut the permanent context cost of
  tool definitions. Validation stays canonical server-side.
- **Evidence records with ids and a ledger.** Large results are retained under an `evidence_id` and read back
  with paged views (`offset/limit`, `coverage.next_offset`). "Missing coverage is unknown, not absence."
  Truncation is always reported. -> Our run ids + `kdev show <run> --section`.
- **Typed error contract**: `invalid_request`, `resource_constraint`, `provider_timeout`,
  `capability_unavailable`, `cleanup_incomplete`, with a bounded `recovery` hint. -> We copy this enum style.
- **Progress for long jobs**, rate-limited, with cancellation distinct from timeout and partial observations
  kept. -> GPU queue jobs report queue position/elapsed. Partial timings survive a timeout.
- **Truthful effects**: each tool declares its process/filesystem/network/mutation effects. -> We declare
  `gpu: none|queued`, `quota: never`.
- **Small reusable primitives.** A tool is extended only when its intent and result contract match. One tool
  never mixes discovery, execution and mutation modes.

### 2.2 GPU-kernel agent tooling
- **HotKernel** (xinhuagu/hotkernel; MCP, `uvx hotkernel`, Claude Code plugin + "kernel-speedrun" skill) is the
  closest existing analogue. Its tools are `gpu_info`, `roofline`, `torch_trace`, `profile` (nsys/ncu),
  `top_kernels`, `disasm` (SASS/PTX with `max_lines`), `bench`, `verify` (interleaved; a faster/slower verdict
  only beyond 2x the combined SE; `incorrect` if any run fails), `record` (the only writer of an append-only
  leaderboard) and `new_target`. Two lessons: (a) the MCP server is torch-free and calls a user-chosen
  interpreter; (b) the timing protocol was versioned ("protocol 3", rotating buffers >2x L2), and rows from
  the old flawed protocols are marked "not counted". **What it lacks for us**: judge calibration, CUDA-graph
  replay, a read-vs-write flush distinction, multi-agent GPU locking, remote/ssh operation, PDL/SASS audits,
  instrumentation timelines, and the cluster/seed effects.
- **KernelBench** (ScalingIntelligence): `Model` vs `ModelNew`, random-input allclose and CUDA-event timing.
  Its published reward-hack catalogue applies directly: CUDA code that is never called, timing on the wrong
  stream or with missing syncs, lazy/unmaterialised outputs, results that pass the benchmarked inputs and fail
  on other inputs of the same shape. Sakana's AI CUDA Engineer had to revise its results for exactly these
  hacks (a replication found a median of 1.1-1.2x after the fixes). "Measuring the Checker" (arXiv 2609.22220)
  mutation-tests the checkers themselves. -> `check` must use fresh seeds and extra shapes, and must confirm
  that the timed code path is the one it checked (`dispatch` probe).
- **GPU MODE KernelBot / popcorn-cli** and the OKC judge (Paradigm). The judge methodology we must mirror:
  - each shape is captured as a CUDA graph;
  - a **read** of a buffer larger than L2 before every replay;
  - submission and baseline take turns on the same fresh inputs;
  - per-shape median of order-swapped paired ratios;
  - minus ~4 us of replay fixed cost;
  - a geomean over loads of weighted shape sums;
  - a 15-minute total budget including compiling;
  - re-submitting identical code returns `duplicate`.

  (msaroufim/openkernelchallenge-audit is not publicly indexed; the adapter must read the judge spec from each
  problem's TASK.md and the judge JSON, not hard-code it.)
- **Harness-engineering papers** (arXiv 2607.17979, MLSys'26 FlashInfer contest, B200): they split an
  *evaluation harness* (compile, correctness, official-aligned timing, artifact archive) from a
  *profile-backed controller*. Agent-assisted runs beat full-agent runs, and what mattered was references,
  profiler interpretation, **conservative promotion gates** and persistent trajectory memory. CUDA-Agent
  (ByteDance/Tsinghua) and CudaForge follow the same generate -> compile -> verify -> profile -> iterate loop.
  Both assume `ncu` is available, which it is not on rented pods.
- **NVIDIA**: Nsight AI offers a hosted CUDA docs MCP server and a self-hostable Nsight Copilot blueprint. It
  provides docs and ncu interpretation, not a measurement harness. There is no public agentic kernel harness.
- **Vendor/DSL tooling worth wrapping**: Triton `do_bench` (L2 clear, but by a write `zero_()`, the exact
  OKC bug) and TileLang `do_bench` (256 MB flush buffer); Triton **Proton** intra-kernel profiling and KPerfIR
  (region timing, <25 cycles of overhead, post-hoc correction); Quack's two-timer trace (globaltimer anchor +
  clock64 events, emitted as Perfetto); TVM TIRx globaltimer per-warp traces; `cutlass_profiler`; and the
  ThunderKittens/CUTLASS/CuTe-DSL compile flows, which all end in `nvcc`/`ptxas`, so a single SASS stage
  covers them.
- **Binary tools**:
  - `cuobjdump -sass/-ptx`, `nvdisasm -cfg -lineinfo` (control flow, line mapping), `ptxas -v` (regs, spills,
    smem, the C7510-C7515 wgmma-serialization warnings), `cu++filt`;
  - CuAssembler / maxas / turingas (SASS re-assembly; useful only for reading control codes and stall/yield
    bits, not as a dependency);
  - NVBit (binary instrumentation, which does not need profiling-counter permission and gives instruction
    counts/memory traces at a high slowdown);
  - `compute-sanitizer` (memcheck/racecheck/synccheck/initcheck; also works without counter permission).
- **When counters are blocked (`ERR_NVGPUCTRPERM`, `RmProfilingAdminOnly=1`, typical on RunPod)**, these still
  work:
  - CUPTI activity tracing: torch.profiler kernel durations and `nsys` kernel traces (no metrics);
  - CUDA events / graph replay;
  - `%globaltimer`/`%clock64` instrumentation;
  - static SASS analysis;
  - NVBit;
  - sanitizer;
  - nvidia-smi clocks and power.

  `kdev` treats these as the primary backends and ncu as optional.

---

## 3. Requirements mined from our OKC experience

### 3.1 Tools our agents rebuilt repeatedly
Each problem grew its own copies:
- **timing harnesses**: ah.py, hp.py, hm.py, ph.py, cq_bench.py, cq_run.py, run2.py, gv_mb.py, vf_bench.py,
  hx_tc.py, is_tc.py;
- **timelines**: gtprof.py, profclk.py, tl1/tl2/tl18.py, ax_tl.py, vtl.py, ovtl.py, sk_tl.py, lab23.py,
  dmx_dbg.py;
- **SASS/ptxas**: sass1.py, aud/sass.sh + split.py, loops.py, bm_ptxas.py, hx_ptxas.py, buildv.py, b4_ptx.py;
- **compile-only prebuilds**: build6.py, hx_build.py, is_build.py, prebuild_any.py, prebuild_v.py,
  sgljit_prebuild.py;
- **sweeps**: sweep*.sh, sweep_sl.sh;
- **calibration**: calib.py, calib_all.py, calib_seeds.py;
- **floor microbenchmarks**: fb.py/fl_src.py, tlb.cu, ic.py;
- **locking**: gpu_run.sh, q.sh;
- **flush fixes**: read_flush.py (which patched 68 harness files).

Their shared core was always the same: load a submission module, build cases from `reference.generate_input`,
capture a graph, flush, replay interleaved, subtract the fixed cost, take medians/paired ratios, and print a
per-shape table plus a weighted geomean. **That core is `kdev time/ab`.** The variations (modes
check/time/prof/tl, `--seeds`, `--nopdl`, shape filters, `@{override}` knob dicts, `-D` define sets) become
flags.

### 3.2 Mistakes that cost us, and the requirement each one creates
| # | What happened | Cost | Requirement |
|---|---|---|---|
| M1 | Harnesses flushed L2 with `zero_()`/`add_(1)`. Dirty lines were written back during the timed kernel. The judge *reads*. | Small kernels read 1-2.5 us slow, so every "gap to floor" was inflated | Read flush is the only default; a write flush only behind `--flush write`, marked in the output |
| M2 | load_inline used nvcc 12.8 (`/usr/local/cuda`), then 13.4. The judge uses **13.0.88**. 12.8 serialised wgmma (ptxas C7515). | Many "variant slower" conclusions wrong; own attention 27% slower than it really was | Toolchain pinned per problem (`toolchain.lock`); every result prints the nvcc/ptxas version from the built `.so` notes; mismatch with the pin = hard error |
| M3 | The torch extension cache is keyed on source, not on nvcc version, so stale `.so` files survived the toolchain change | Silent stale binaries | Cache dir derived from hash(toolchain + flags + source); `build` reports cache hit/miss and the `.so` notes |
| M4 | Compiling inside the GPU lock (prepare runs, first MoE calibration) | Queue stalls, and contention in other agents' timings | `build` never takes the lock; `time` refuses to compile (fails with `needs_build` unless `--allow-jit`, which is logged) |
| M5 | Bare `flock` while other agents' prepare/warm-up/test ran unlocked | Corrupted timings | Lock = flock + "no other live CUDA context" (gpu_run.sh logic) + an offender log; every timing records `gpu_exclusive: ok/waited/fail` |
| M6 | Promoting on a local ratio of sub/SGLang-baseline, where the local baseline ran 1-3 us slow on short shapes | Wrong direction on swaps | Report absolute per-shape us alongside ratios; A/B decides on absolute candidate-vs-candidate times, not on each one's ratio to the baseline |
| M7 | MoE/decode timed with one routing seed: decode1 1.24 vs 0.96 local/judge | Unusable conclusions | Adapters declare `input_variance: routing` -> default `--seeds 5`; per-seed spread reported |
| M8 | Local +8% FA3 swap, judge -7..-9%; the unmodified baseline was never judged in the same hour | A regressed best | `calib` gate: before promotion, the local per-shape direction must match the judge calibration table; the "same-hour judge baseline" is a required field of `promote` |
| M9 | PDL kernels: nvcc hoisted `LDG.E.CONSTANT` of producer data above `griddepcontrol.wait` (ACQBULK) | Intermittent wrong results (~50% eager) | `sass audit --pdl`: flags any global load before ACQBULK; `check --repeat N` in eager + graph |
| M10 | Judge returned `duplicate` for identical code | Wasted a check | `judge prepare` hashes the payload against a local ledger of judged hashes and warns |
| M11 | Judge noise ±1-2% plus a cluster/node offset of 1-2% | Chasing 1% items | Judge records keep cluster/node; `calib` compares same-node pairs; promotion needs effect > max(3%, CI) |
| M12 | Optimising a code path that never ran (overwritten entry points, embedded alternative sources) | Hours wasted | `dispatch` probe: which kernels actually launch per scored shape (CUPTI names + counts), shown in every `time` summary |
| M13 | "At the floor" accepted without SASS reading or a knob search | Missed the public #1 ideas | `floor` reports the measured skeleton floor; a `limit-claim` record requires a SASS audit run id + a sweep run id |
| M14 | No ncu (ERR_NVGPUCTRPERM) | Every agent improvised globaltimer marks | `timeline` as a first-class tool with a header-only device API and a 32/256 ns resolution report |
| M15 | Prefetch/`rows[32]` register arrays serialised or spilled; ptxas DCE turned microbenchmarks into pure streaming | Misread microbenchmarks | `sass` reports spills/LDL/STL, instruction mix per loop, and **warns when expected opcodes (HGMMA/MMA/LDS) are missing** |
| M16 | `rep` FAILs were fp32 atomic-order nondeterminism (1 ulp at \|x\|>=8), not races | False alarms | `check --repeat` separates bitwise nondeterminism from tolerance failures and reports the max ulp |
| M17 | Pod storage is not persistent; NOTES overwritten from a stale pod copy; `/workspace` (mfs) hung | Lost work | `sync` is bidirectional with md5 and never overwrites a newer local file without keeping a `.conflict` copy; work dirs never on `/workspace` |
| M18 | Cold prepare 228 s against a 15-minute budget | Near timeout | `check --cold` measures prepare/compile time from an empty cache, against the problem's budget |
| M19 | Rules violations that pass tests (bf16 router input where the reference uses fp32) | A non-compliant best | `comply` checklist tool: per-op precision vs reference, no cross-call caching, no work outside the timed call; used as a promotion gate |
| M20 | Thrust headers intermittently missing (alternatives symlink); ptxas from pip vs system mixed | Flaky builds | `doctor` verifies the toolchain tree; `build` retries known-transient errors once |

### 3.3 Measured constants worth shipping as GPU profile data (H200, CUDA 13.0, read flush)
These come from okc/research/tlb/SUMMARY.txt and floor/RESULTS.md:

| quantity | value |
|---|---|
| graph replay fixed cost | ~4.0 us (judge); 5.18-5.22 us for a 1-kernel `add_` graph |
| null kernel | -0.5 us relative to that graph, independent of grid/smem/cluster |
| cold dependent DRAM trip | 0.31-0.35 us (1 thread), ~0.65 us (132 CTAs) |
| L2 hit | 0.15 us |
| kernel boundary within a stream | 2.6-2.9 us, even with PDL |
| extra small graph node | ~1.7 us |
| cooperative launch attribute | +1.3-1.5 us |
| cold icache | <=0.16 us |
| cold data-TLB after flush | ~0 |
| MEMBAR.GPU | 0.16-0.33 us |
| cold atom.acq_rel | 0.57 us |
| barrier.cluster | 0.26-0.5 us |
| streaming bandwidth | 4.11 TB/s for 200 MB |
| globaltimer resolution | 32 or 256 ns, depending on the kernel |

`floor` uses these for latency-bound skeleton estimates. Each GPU profile records how its constants were
measured, and `kdev floor --calibrate` re-measures them on a new GPU.

---

## 4. Interface choice

**Recommendation: a CLI (`kdev`) with a stable JSON contract, over a Python library (`kdev.*`), installed on
the GPU host. A local `kdev remote` client runs commands over ssh. A Claude Code skill (`SKILL.md`) teaches
the workflow. An MCP server is a phase-4 thin adapter over the same library.**

Reasons, for Claude Code agents working over ssh on a remote GPU pod:

1. **Agents already have Bash and ssh.** A CLI needs no per-session registration, so every subagent inherits
   it. MCP servers are configured per client, and the background subagents spawned by the coordinator would
   each need the config.
2. **Tool definitions cost context in every turn.** About 15 MCP tools with full schemas run to thousands of
   tokens per agent for the whole session; REA added a "compact schema" profile for exactly this reason.
   `kdev --help` costs context only when an agent asks for it.
3. **Locality.** GPU, toolchain and files live on the pod. A local MCP server would have to proxy everything
   over ssh anyway, and a server on the pod would need a tunnel per agent. The CLI on the pod plus
   `ssh pod kdev ...` (or `kdev remote`) is the simplest topology. It also survives ssh drops via
   `kdev submit-job`, detached and polled by run id.
4. **Long jobs.** Timing runs take 1-50 minutes. A CLI with run directories, `--detach`, `kdev wait <run>` and
   background Bash fits Claude Code's `run_in_background`/Monitor model. MCP calls block or need progress
   plumbing.
5. **Humans, cron and the watcher use the same commands.** The coordinator's 20-minute watcher and the backup
   cron are just `kdev queue status --json` and `kdev sync`.
6. **Composability.** Agents pipe JSON (`kdev ab ... --json | jq`) and write their own tiny glue, without
   having to fork the harness.

The MCP layer comes later, for clients without a shell and for typed discovery. It wraps the same library
calls, so there is one contract. Following REA, it gets compact schemas, evidence ids, and paging
(`kdev show`).

### 4.1 Output contract (all tools)
- **stdout (default)** is a human/LLM summary of at most ~40 lines: verdict first, then a small table, then
  `run: <id>  dir: <path>`, then `next:` hints (at most 2).
- **`--json`** gives a single JSON object:
  `{tool, version, run_id, status: ok|fail|error, verdict, summary:{...small...}, files:{name:path},
  warnings:[...], fidelity:{toolchain, flush, graph, exclusive, seeds, overhead_us, optouts:[...]},
  error?:{code, message, details, recovery}}`.
- **Error codes** (REA-style): `invalid_request`, `needs_build`, `toolchain_mismatch`, `gpu_busy_timeout`,
  `capability_unavailable` (e.g. ncu), `check_failed`, `build_failed`, `timeout`, `quota_forbidden`.
- **Run dir**: `$KDEV_HOME/runs/<yyyymmdd-hhmmss>-<tool>-<slug>/` containing `meta.json` (inputs, md5s,
  toolchain, GPU, clocks before/after), `result.json`, `raw.jsonl` (all samples), `log.txt`, plus
  tool-specific artifacts (`.sass`, `trace.json`, `ptxas.txt`).
- **`kdev show <run> [--section X] [--offset N --limit M]`** gives paged reads of large outputs, so agents never
  `cat` a 3000-line SASS file.
- **Truncation is always explicit**: `"... 412 more rows in <path>"`. Nothing is ever silently clipped.

---

## 5. Tools: inputs and outputs

Common flags: `--problem P` (or auto-detected from the file's first line or adapter), `--shapes FILTER`,
`--json`, `--detach`, `--run-name`, `--gpu N`.

### 5.1 `kdev doctor`
- **In**: none / `--problem`.
- **Out**: GPU model, SMs, clocks, driver, ncu permission (`RmProfilingAdminOnly`), toolchain pin vs found
  (nvcc/ptxas/nvdisasm/cuobjdump versions and paths), torch/triton versions vs the pin, `TORCH_EXTENSIONS_DIR`
  and cache keying, free disk on non-network paths, lock health and any unlocked CUDA processes.
- PASS/WARN/FAIL lines (this replaces the bootstrap's 36-check verify).

### 5.2 `kdev build FILE [--variant K=V ...] [--defines "-DX=1 ..."] [--verbose]`
- Never takes the GPU lock and never creates a CUDA context. It imports the module with CUDA blocked, or
  extracts the CUDA sources/load_inline calls through the adapter's `build()` hook.
- **Out**:
  - per extension: cache key and hit/miss, compile seconds, and the `.so` toolchain notes ("V13.0.88");
  - ptxas `-v` per kernel: regs, spill bytes, smem, stack, and C751x warnings verbatim with kernel names;
  - paths to `.cubin`/`.ptx`/`.sass`.
- Fails with `toolchain_mismatch` if the notes differ from the pin.

### 5.3 `kdev check FILE [--seeds N] [--repeat R] [--graph/--eager/--both] [--extra-shapes] [--cold]`
- Runs the adapter's tests and checks with the judge's tolerances on fresh seeds, plus extra shapes (held-out
  sizes within the declared shapes), in eager mode and as a captured graph, repeated R times.
- **Out**: PASS/FAIL per case with the max relative-L2 error vs tolerance, separating
  - `nondeterministic` (bitwise differs between repeats, but within tolerance; reports max ulp);
  - `intermittent` (fails k/R, which points to a race);
  - `wrong`.
- `--cold` measures prepare + compile time from an empty cache against the budget.
- Always includes the `dispatch` probe (5.13).
- Takes the GPU queue; prebuild is required.

### 5.4 `kdev time FILE [--base baseline] [--reps N] [--seeds S] [--shapes F]`
Judge-faithful single-file timing. For each scored shape:
- build the inputs;
- capture a graph of `kernel_fn` and of the baseline;
- replay them interleaved;
- read-flush before each replay;
- subtract the overhead (measured empty graph, plus the judge constant);
- record the medians.

**Out**: a per-shape table (abs us sub / base, paired-ratio speedup with a bootstrap 95% CI, judge weight), the
problem score (weighted geomean as the judge defines it), `fidelity{}`, the clocks/power before and after, and
the `dispatch` summary (kernel names x count per shape).

### 5.5 `kdev ab A.py B.py [C.py ...] [--reps N] [--seeds S] [--order abba|rotate]`
The workhorse. N files are interleaved per rep in ABBA order, with a read flush per replay.
- **Out**: per shape, abs us for each file, the B/A ratio with a CI, and a verdict among `faster` / `slower` /
  `same` / `incorrect`. `faster`/`slower` needs |delta| > max(noise floor, 2x the combined SE); `incorrect`
  means any variant failed the correctness check on the timed inputs.
- Also reports the score delta with a CI, and `@override`/`--variant` knob dicts (from cq_bench).
- Decisions are made on **absolute** candidate times, never on the ratio to the baseline (M6).

### 5.6 `kdev sass FILE|CUBIN [--kernel REGEX] [--loops] [--audit pdl,spills,serialize,expect=HGMMA]`
Uses `cuobjdump`/`nvdisasm -cfg -lineinfo`.
- **Out**:
  - per kernel: instruction count, regs, and an opcode-class histogram (MMA/HGMMA, FFMA, MUFU, LDG/LDS/STS,
    BAR/MEMBAR, int/uniform, F2FP, ...);
  - **per hot loop** (from the CFG back edges, ranked by static size x line-info weight): the instruction mix
    per iteration and per element (when `--per N` is given) — the "~15 ops/8 weights" analysis;
  - audits:
    - `pdl`: global loads before ACQBULK/griddepcontrol.wait;
    - `spills`: LDL/STL in loops;
    - `serialize`: ptxas C7510-C7515;
    - `expect`: opcodes that should be present and are absent, i.e. DCE (M15);
    - `diff A B`: a side-by-side histogram delta.
- The full SASS is in a file, and the summary shows only the top loop, at most ~25 lines.

### 5.7 `kdev timeline FILE --marks [auto|header] [--shape S] [--ctas sample]`
- **Device side**: `kdev/marks.cuh` provides `KD_MARK(id)` (globaltimer + clock64, Quack/Proton two-timer
  style) written to a per-CTA/warp-group buffer. It costs nothing when `KDEV_MARKS` is undefined. The adapter
  injects `-DKDEV_MARKS` plus a buffer argument, or the kernel takes it via an env-overridable define
  (dmx_dbg.py style).
- **Out**:
  - a per-phase table (p10/p50/p90 across CTAs, and critical-path CTA), CTA start/end spread, and the
    resolution detected (32/256 ns);
  - `trace.json` (Perfetto);
  - `--ab` interleaved timelines of two define sets (from ovtl.py).
- Also prints the mark overhead, measured as the delta of the marked vs unmarked build.

### 5.8 `kdev floor [--problem P --shape S] | --bytes B --flops F --dtype D`
- Roofline from the GPU profile, using the judge's floor constants (e.g. 4.8 TB/s / 989 TF) and the measured
  ones (4.11 TB/s achievable), **plus the latency skeleton**: dependent trips, kernel boundaries, graph nodes
  and merge primitives from Section 3.3, summed per a declared plan.
- **Out**: floor us (judge-style and achievable), measured vs floor, and which term dominates.
- `--calibrate` runs the microbenchmark suite (null kernel, chase, gather, merge, stream) and writes the
  GPU-profile constants.

### 5.9 `kdev sweep FILE --grid "TILE=64,128 STAGES=2,3,4" [--build-parallel J] [--top K]`
- Builds every point outside the lock in parallel, then checks them (a quick check) and times them with `ab`
  in batches against the base, interleaved.
- **Out**: the top K points with CIs, failed points by reason (build/spill/C751x/check), and the full table in
  CSV. Supports cuBLASLt algorithm enumeration as a grid source (`--grid cublaslt:SHAPE`).
- This is the "exhaustive knob search" that LESSONS rule 1 demands.

### 5.10 `kdev regs FILE|CUBIN`
- Occupancy report per kernel: regs, smem (static/dynamic), launch bounds, block size, CTAs/SM by each limiter,
  and `setmaxnreg` groups, together with the warps-per-SMSP rule (e.g. 5 warps/SMSP caps at 96 regs) and
  spill info.
- Static only: no GPU.

### 5.11 `kdev sanitize FILE [--tool memcheck|racecheck|synccheck|initcheck] [--shape S]`
- Runs compute-sanitizer on one small shape, eager and graph.
- **Out**: the error count by type and the first 5 distinct errors with SASS/source lines. The full log is in a
  file.

### 5.12 `kdev profile FILE [--backend auto|ncu|nsys|cupti] [--metrics set]`
- `auto` uses `ncu` if it is permitted; otherwise it uses CUPTI activity (torch.profiler) kernel times with PDL
  disabled (`--nopdl`) so per-kernel times are separable.
- **Out**: the top kernels by time, plus ncu sections (SOL, occupancy, warp stalls) when available.
- Reports `capability_unavailable: ncu (ERR_NVGPUCTRPERM)` explicitly instead of failing.

### 5.13 `kdev dispatch FILE [--shapes F]`
- Records which kernels and graph nodes launch for each scored shape (CUPTI names, grid/block, count), plus
  the source route if the adapter exposes a route hook.
- Warns when two shapes the agent thinks differ run the same route, or when a file contains unreferenced
  kernels (M12).

### 5.14 `kdev calib FILE:JUDGE.json [FILE2:JUDGE2.json ...] [--seeds S]`
- Judge-vs-local per shape (from calib.py / calib_seeds.py): judge sub/base us, local sub/base us, both
  speedups, local/judge, and flags outside 0.95-1.05. For 2+ files it compares the judge A/B ratio with the
  local A/B ratio and prints `DISAGREE`.
- Appends to the problem's calibration table (`calib/<problem>.jsonl`, keyed by toolchain + GPU + node).

### 5.15 `kdev judge` (record-only)
- `judge prepare FILE` validates the file against the adapter's static rules, records its md5, warns on a
  `duplicate` hash or an unchanged file, and prints the exact command for a human or an authorised
  orchestrator.
- `judge record JOB.json` ingests a judge result into the ledger: per-shape times, cluster/node, toolchain
  probe output.
- `judge ledger` shows the bests and same-hour baselines.
- **There is no `judge submit`/`check` in kdev** (Section 7).

### 5.16 `kdev comply FILE`
- Static plus dynamic checklist against the adapter's rules:
  - per-op precision (dtype probes on intermediate tensors in eager mode);
  - no cross-call state (call twice with different inputs in the same buffers; compare against fresh-process
    outputs);
  - no work outside the timed call (time `prepare` vs `kernel_fn`; flag large prepare-time output-shaped
    buffers);
  - graph-capturable;
  - no host sync.
- **Out**: a PASS/RISK list. RISK items need a human or reviewer.

### 5.17 `kdev queue` / `kdev lock`
- `queue status` shows the holder, waiters, elapsed time, the offender log (unlocked CUDA pids with their
  cmdlines) and GPU utilisation.
- `lock run -- CMD` is the gpu_run.sh semantics: flock + wait for no foreign CUDA contexts (zombies ignored) +
  timeout + `GPU_EXCLUSIVE=OK waited=Ns`.
- Phase 3 adds a FIFO queue daemon with priorities, per-agent attribution, max-hold and preemption warnings.
  It refuses jobs that would compile (no prebuilt cache hit).

### 5.18 `kdev sync` / `kdev remote`
- `sync pull|push|both --since 3h` copies run dirs, ledgers and adapter dirs pod<->local with md5 verification.
  It never overwrites newer files: a conflict gets a `.conflict-<ts>` copy (M17). A cron mode writes a
  heartbeat.
- `remote [--host H --port P] -- kdev ...` runs a command on the pod with `ssh -i key`, streams the summary,
  and optionally pulls the run dir.
- `remote setup` bootstraps a fresh pod: venv pins, toolchain pin, read-flush self-test, doctor. It is
  idempotent with phase markers, from okc_bootstrap.sh.

### 5.19 `kdev note`
- `note add PROBLEM "claim" --evidence run1,run2 [--kind dead|limit|win]` appends to `notes/<problem>.jsonl`
  and renders it into NOTES.md.
- A `limit` claim is refused without a `sass` run and a `sweep` run (LESSONS 1, 6).
- Successors run `note list --kind limit` to see the unverified claims.

---

## 6. Judge-fidelity requirements (mandatory defaults)
1. **Toolchain pinning.** `adapters/<p>/toolchain.lock` holds the nvcc/ptxas version and path, torch, triton
   and library pins, and the flags (e.g. `-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK`). The extension cache is
   keyed on toolchain+flags+source. Each result embeds the `.so` notes. The pin is discovered once via a judge
   probe (the tc_attn.py technique: print the toolchain notes from prepare in a check run) and recorded with
   that job id.
2. **Read flush**: by default a buffer `>= 4x L2` (256 MB on H200), *read* by a reduction into a sink before
   each replay. The size is configurable because it shifts results by 0.2-0.6 us. A self-test asserts no
   dirty-line cost: the delta between read-flush and write-flush on a tiny kernel is reported in doctor.
3. **Graph replay** of `kernel_fn` exactly as the judge captures it (same stream semantics, PDL attributes
   preserved). An eager timing is available only as `--eager` and is marked.
4. **Overhead subtraction**: the judge constant (4.0 us) plus a locally measured empty-graph and 1-kernel-graph
   cost, both reported. Shapes under ~10 us show a sensitivity warning.
5. **Seeds**: the adapter declares the input variance. The default is 1 seed for deterministic shapes and
   **5 for routing/length-dependent ones**. The judge's per-shape seeds are used when known, and the per-seed
   spread is reported.
6. **Interleaving**: ABBA/rotating order per rep; submission and baseline on the same fresh inputs; the paired
   ratio median, like the judge.
7. **Statistics**: medians plus a bootstrap 95% CI of the paired ratios; the score CI propagated through the
   weighted geomean. A verdict needs the CI to exclude 1 and |delta| >= the per-problem noise floor (from
   `calib`).
8. **Exclusivity**: the lock with the foreign-context check; clocks/power logged before and after; a warning
   if SM clocks moved >3% or the GPU was not exclusive.
9. **Calibration gate**: `promote` (phase 3) requires a `calib` record of the candidate (or of its component's
   direction) that agrees with the judge, and a same-hour judge baseline.
10. **Protocol versioning** (from HotKernel): every result carries `protocol: kdev-tp/N`. A protocol change
    marks older ledger rows "not comparable" rather than mixing them.

---

## 7. Safety
- **Never submit.** kdev has no code path that POSTs to a judge's submission or check endpoint, or runs
  `okc submit` / `okc check`. Judge interaction is limited to `judge prepare` (it prints the command) and
  `judge record` (ingest). A config deny-list blocks wrapping these commands through `lock run` too (exit
  code 3 `quota_forbidden`). Quota use stays a human/orchestrator decision, as the OKC rules and the user
  require.
- **Lock discipline**: compiling under the lock is refused. Each lock is held for at most a configurable
  maximum (default 30 min), with a warning to the holder at 80%. Offenders are logged with pid and cmdline.
  Timing results always record exclusivity.
- **Reward-hack guards** (KernelBench/Sakana lessons; OKC rules forbid caching outputs and work outside the
  timed call):
  - `check` uses fresh seeds and extra shapes;
  - `comply` runs the cross-call-state test;
  - `dispatch` proves the code runs;
  - timing verifies that outputs are materialised (checksummed after replay);
  - results "too good to be true" (faster than the `floor`) are flagged `suspect` instead of `faster`.
- **Effects are declared** per tool (`gpu: none|queued`, `writes: run_dir|cache|ledger`, `network: none|ssh`).
- **Untrusted code**: public/competitor kernels are copied into `third_party/` with a licence header
  (Apache-2.0 with credit; unlicensed = ideas only, never copied). Extracted archives are never executed from
  their own directory.
- **Storage**: there is no network drive for work dirs. Backups go through `sync` with md5.

---

## 8. Extensibility: problem adapters and GPU profiles
An adapter is a directory `adapters/<name>/` with `adapter.py` implementing:

```python
class Adapter:
    name: str; gpu: str; budget_s: int                  # e.g. 900 (15 min incl. compile)
    def shapes(self) -> list[Shape]                     # scored shapes: id, case dict, weight, load
    def tests(self) -> list[Case]                       # correctness-only cases (+ extra held-out shapes)
    def make_inputs(self, case, seed) -> Inputs         # e.g. reference.generate_input
    def load(self, path) -> Module                      # submission module (prepare, kernel_fn)
    def reference(self, inputs) -> Out ; def check(self, out, ref) -> CheckResult  # judge tolerances
    def baseline(self) -> Module | None                 # e.g. SGLang baseline.py
    def score(self, per_shape_times) -> float           # judge formula (weighted sums, geomean over loads)
    def input_variance(self) -> {"routing": True, ...}  # drives the default seed count
    def build(self, module) -> list[Ext]                # optional: compile-only hook
    def rules(self) -> ComplyRules                      # precision per op, forbidden patterns
```

- Bundled adapters:
  - `okc` (generic, reading TASK.md/reference.py/checks.py; the five muse_* are configs);
  - `kernelbench` (`Model`/`ModelNew`, `get_inputs`);
  - `cu-main` (a standalone `.cu` with a `main()` that times itself through `kdev/bench.cuh`);
  - `triton-fn` (a function + a reference lambda);
  - `kernelbot` (GPU MODE `reference.py`/`submission.py` layout).
- **GPU profiles** live in `gpus/<name>.toml`: SMs, L2 size (flush size), peak BW/FLOPs per dtype, the measured
  constants (Section 3.3) with the run ids that measured them, and the globaltimer resolution. A new GPU starts
  from a datasheet profile, and `floor --calibrate` fills in the measured fields.
- **Backends** live in `kdev/backends/` with a capability probe: ncu, nsys, cupti, sanitizer, nvbit, nvdisasm,
  cuobjdump. Tools pick the best backend that is available and report which one they used.

---

## 9. Repo layout
```
kernel-development/
  README.md
  docs/DESIGN.md  docs/PROTOCOL.md (timing protocol vN)  docs/ADAPTERS.md
  pyproject.toml                    # package `kdev`, console script `kdev`; core deps: stdlib + numpy (torch optional, host-side)
  kdev/
    cli.py                          # argparse subcommands, --json, summary rendering
    contract.py                     # result envelope, error codes, run dirs, paging (show)
    runs.py  ledger.py  notes.py
    fidelity/  flush.py graph.py overhead.py stats.py (bootstrap CI, paired ratios) toolchain.py
    tools/     doctor build check time ab sass timeline floor sweep regs sanitize profile dispatch calib judge comply
    gpu/       lock.py (flock + foreign-context wait) queue.py (daemon, phase 3) smi.py
    backends/  ncu.py nsys.py cupti.py sanitizer.py nvdisasm.py cuobjdump.py ptxas.py nvbit.py
    sass/      parse.py cfg.py loops.py audits.py
    device/    marks.cuh (KD_MARK) bench.cuh flush.cuh
    remote/    ssh.py sync.py bootstrap/ (pod setup scripts)
    mcp/       server.py (phase 4; thin wrapper over tools/*)
  adapters/  okc/ kernelbench/ cu_main/ triton_fn/ kernelbot/
  gpus/      h200.toml h100.toml b200.toml rtx5090.toml
  skills/    kdev/SKILL.md          # workflow for agents: baseline -> knobs -> SASS -> rewrite; brief checklist
  tests/     unit (sass parser on stored .sass, stats, contract), gpu-marked integration tests
  examples/  okc_attention/ kernelbench_l1/
```

---

## 10. Phased implementation plan (highest value first)

**Phase 1: the trustworthy timer (1-2 days).** It removes M1-M7, the errors that invalidated whole days.
1. `contract.py` + run dirs + `--json` + `show`.
2. `gpu/lock.py` (port of gpu_run.sh) + `queue status`.
3. `fidelity/*`: read flush, graph capture/replay, overhead, ABBA interleave, bootstrap CI, seeds.
4. `toolchain.py` + `build` (compile-only, keyed cache, `.so` notes, ptxas -v parse) + `doctor`.
5. The `okc` adapter (generic over the five problems), with `time`, `ab` and `check` (eager + graph, repeat,
   seeds).
6. Port `calib` and the judge ledger (`judge record/prepare`, duplicate-hash warning).

Acceptance: re-run the 2026-10-31 calibration (calibration/all_calib.md) through `kdev calib` and reproduce
its per-shape local/judge ratios within ±1%.

**Phase 2: seeing inside the kernel (2-3 days).** It removes M9, M12-M16.
7. `sass` (parser, CFG loops, histograms, the audits pdl/spills/serialize/expect, diff) + `regs`.
8. `dispatch` (CUPTI) + `profile` with auto backend fallback.
9. `timeline` (marks.cuh, buffer injection for load_inline sources, Perfetto export, A/B).
10. `sanitize`.
11. The `skills/kdev/SKILL.md` workflow + the LESSONS "brief checklist" adapted to kdev commands.

**Phase 3: search and gates (2 days).**
12. `sweep` (parallel prebuild, batched interleaved timing, cuBLASLt grid source).
13. `floor` (roofline + latency skeleton + `--calibrate` microbench suite from fb.py/tlb.cu).
14. `comply`, `note` (limit-claim gating), and a `promote` gate (check + comply + calib + same-hour baseline).
15. Queue daemon (FIFO, attribution, max-hold, refusal of compile-under-lock).

**Phase 4: reach (as needed).**
16. `remote` + `sync` (md5, conflict copies, cron mode) + pod bootstrap; ports of okc_bootstrap.sh /
    sync_okc.sh.
17. Additional adapters (kernelbench, cu-main, triton-fn, kernelbot) + GPU profiles (H100/B200/RTX).
18. MCP server over the library, with compact schemas, evidence ids, paging and progress notifications.
19. Optional ncu/NVBit backends where permitted; an instruction-count mode via NVBit.

Ordering rationale: Phase 1 alone would have prevented the toolchain/flush/seed/ratio mistakes and the
duplicated harnesses. Phase 2 gives agents the evidence that LESSONS rule 1 requires before any "at the limit"
claim. Search and gates come next, because they consume Phase 1/2 outputs. Remote/MCP is convenience: ssh +
CLI already works.

---

## 11. Open questions
- The judge's exact flush buffer size and the fixed-cost method. We currently assume a read of >L2 and 4.0 us.
  A probe check could print them (the tc_attn technique).
- Whether to depend on torch in the CLI process, or (like HotKernel) keep the CLI torch-free and spawn a
  pinned interpreter for GPU work. Proposed: the CLI is torch-free, and the worker runs in the adapter's pinned
  venv. This also isolates toolchain env vars per job.
- The queue daemon vs flock-only. Start with flock (proven); add the daemon once more than 4 agents share a
  GPU.
