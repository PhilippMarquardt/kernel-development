# kdev quickstart (optimisation agents)

kdev runs on the GPU pod (`ssh -i ~/.ssh/runpod_ed25519 -p PORT root@HOST`; installed by
`scripts/pod_install.sh`). Each command prints at most 40 lines, starting with the verdict and ending with
`run: <id>`. That run id is your evidence. Add `--json` to get one JSON object. Read large outputs with
`kdev show RUN --section S` and never `cat` the files in a run directory. Every error carries a code and a
`recovery` line: do what the recovery line says.

GPU commands (`check`, `time`, `ab`, `calib`, `dispatch`, ...) queue behind the shared `/root/gpu.lock`, the
same lock `/root/gpu_run.sh` uses. They never compile while holding it. Build first; for long runs use
`--detach` and then `kdev job wait JOB`. While waiting they print the queue position and ETA every 30 s
(`kdev queue status` lists the whole lock queue); `--priority N` makes a job go before lower-priority kdev
jobs (gpu_run.sh jobs keep their kernel queue order). The lock is phased (PROTOCOL 10): judge-mode `time`/`ab`
processes prepare under a SHARED lock (prepares of several jobs overlap) and take it EXCLUSIVE only for the
timed section; `kdev build`'s discovery and `kdev check` run SHARED (they never overlap a timed section).
`--no-phased` restores the whole-process exclusive hold.

## The 10 commands, in order

| # | Command | Why |
|---|---|---|
| 1 | `kdev doctor --problem DIR` | Checks the GPU, the judge toolchain pin (nvcc/ptxas 13.0.88), the lock and the ledger. Fix every FAIL first. |
| 2 | `kdev ledger tried FILE` / `kdev idea list --status open` | Shows whether this file, a comment-only copy of it or this knob set was already measured. Claim your idea with `kdev idea claim ID`. |
| 3 | `kdev build FILE [FILE2]` | Compiles with the pinned toolchain into a hash-keyed cache. Reports regs, spills and C751x. Every GPU command needs it (`needs_build` otherwise). Extensions that are built inside `prepare()` or on the first call are found by short GPU discovery jobs: build, discover, rebuild, until the discovery comes back `complete`. Nothing compiles while the GPU lock is held. If a GPU job reports `needs_build`, run `kdev build` again. Pending records belong to the file's content (md5): after an edit, records from the old version are dropped. `--clean-pending` drops all of the file's records before the build. |
| 4 | `kdev check FILE` | Correctness: fresh seeds, extra shapes, eager and graph, repeats. Reports `nondeterministic` (ulp ≤ 1 is fine), `intermittent` or `wrong`. |
| 5 | `kdev time FILE [--judge-mode]` | Judge-protocol timing against `baseline.py`: graph replay, 256 MB read flush, 4.0 us overhead, 5 seeds, exclusive GPU. It also runs a dispatch probe in a separate process and reports which kernels actually ran for each shape. |
| 6 | `kdev calib FILE:JUDGE.json` | Compares local and judge results per shape (sub and baseline us separately) and records the per-shape judge/local ratio in the calibration table (`kdev calib table PROBLEM`). `time`/`ab` then add a `~J` column (estimated judge us; `?` = shape < 6 us, overhead-dominated). Local and judge differ per shape by ~2.5 % median (judge-side: per-shape overhead, node, its faster SGLang baseline), so compare absolute sub us, not local speedups. On a `DISAGREE`, explain it before you trust local A/B on that problem. |
| 7 | `kdev ab BEST.py CAND.py [--set B:K=V]` | On OKC problems: judge mode (PROTOCOL 9): one process per file in ABBA order (A B B A), each with the SGLang baseline prepared first and replayed interleaved, fresh inputs per pair, prepare pads, gate, per-shape `2*one-two`. Prints per-shape RAW deltas (both half-blocks) and the judge-formula us/score. `faster`/`slower` need the CI and both half-blocks to agree. `--fast` = the old single-process mode (quick screening; it misses baseline-interleaving effects of ~0.3 us). This is the only way to compare two files. With 3+ files every X/A is printed per shape (dRAW, speedup, `+ - =`). Each process logs buffer placement (`data_ptr % 2MB`); a `PLACEMENT:` warning means A and B landed differently: re-run with `--pad-mb 16` (same pads) or more `--rounds`. |
| 8 | `kdev sweep FILE --grid "K=a,b K2=c"` | Searches cheap knobs exhaustively before any rewrite. Builds in parallel, runs the prepare-time discovery per point (short GPU jobs; `--no-discover` skips), quick-checks, then batches an ab against the base. |
| 9 | `kdev sass FILE --kernel RE --loops` (+ `kdev sass diff A B`) | Hot-loop opcode mix and audits (pdl, spills, serialize, repeat, invariant). You need it for any "at the limit" claim. The pdl audit names the parameter each pre-wait load reads (`p.seq_lens x8 (cuda.cu:2444 ...)`); a read-only input is allowlisted in the source with `// kdev:pdl-allow seq_lens -- why` (or `FILE.pdl-allow`). |
| 10 | `kdev note "CLAIM" --kind limit --evidence RUN_SASS,RUN_SWEEP` / `kdev idea resolve ID ...` | Writes the result, with its run ids and the exact variant, to the board. |

Supporting commands: `kdev dispatch FILE`, `kdev regs FILE`, `kdev floor --shape S`, `kdev timeline FILE`,
`kdev comply FILE` (must pass before promoting), `kdev sanitize FILE`, `kdev queue status`,
`kdev job list|wait|logs`, `kdev runs list`, `kdev slim STACK.py OUT.py --trace DISPATCH_RUN` (decode + drop dead
routes, size/md5, <1,000,000 B; then build/check/ab OUT), `kdev ablate FILE --variant mma=a_nomma.py --set
epilogue:NO_EP=1 --control a_z0.py --floor d32/o=4.66` (us each component costs per shape: excess budget).

## Example session (attention)

```
P=/root/okc/muse_attention_decode/muse_attention_decode
kdev doctor --problem $P
kdev build $P/best_2.624.py $P/cand.py          # ~50 s per file, CPU only
kdev check $P/cand.py                           # ~2 min under the lock
kdev ab $P/best_2.624.py $P/cand.py             # ABBA, verdict from the CI
kdev sass $P/cand.py --kernel attn_kernel --loops
kdev idea resolve 12 --status refuted --resolution "TILE=128: x0.99 [0.98,1.00], run 2026...-ab-..."
```

## Working with other agents (board + docs, docs/COLLAB.md)

Agents talk to each other directly on the board and write notes with `kdev doc`. Set `KDEV_AGENT=<your id>`
(or pass `--agent`). Remote agents run the same commands through `kdev remote -- kdev board ...`.

```
kdev board inbox; kdev board read --unread --problem $P; kdev board claims --problem $P   # at start + before every GPU job
kdev doc checklist; kdev doc toc notes --problem $P                                        # rules + what is known
kdev board post claim "plan" --problem $P --files cand_x.py --shapes m128                 # before starting; CONFLICT -> ask the owner
kdev board post finding "TILE=256 x0.98 [0.97,0.99]" --problem $P --ref RUN_ID            # every result, with evidence
kdev board post done "summary" --problem $P --ref RUN_ID                                  # closes your claim
kdev doc append notes --problem $P --section "Round N <id>" "method, variant, numbers" --ref RUN_ID
```

Never edit NOTES.md, LESSONS.md or TRACKER.md by hand. `kdev doc append` only adds entries, so nothing is lost.
`kdev doc export notes --problem $P --md PATH` renders the notes as a file.

## Rules

- Measure only through `kdev time` or `kdev ab`. Opt-outs (`--flush write`, `--eager`, `--allow-jit`, a
  profiler session in the same process) are printed in CAPS and make the run `not comparable`.
- Never start CUDA work outside kdev or `/root/gpu_run.sh`. Never run `okc submit` or `okc check`: judge runs
  belong to the coordinator (`kdev judge record JOB.json` files them into the ledger).
- Keep each GPU job short (one file pair, default reps). Other agents are queued behind you.
- A daemon (`kdev serve start`) is optional. Without one, GPU commands run in `--direct` mode under the same
  flock.
- MCP clients: `kdev mcp` exposes the same commands as `kdev_*` tools (see docs/MCP.md), including
  `kdev_board` and `kdev_doc`.
