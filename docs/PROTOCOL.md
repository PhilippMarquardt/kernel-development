# kdev timing protocol `kdev-tp/1`

Owner: P3 (`src/kdev/timing/`, `src/kdev/worker.py`). Every Result carries `"protocol": "kdev-tp/1"`. A change to
anything in sections 2-5 bumps the version; the ledger then marks older rows `comparable=0` (DESIGN 6.10).

## 1. What the OKC judge does (measured from its records, not assumed)

Verified on the five judge JSONs in `tests/fixtures/{judge,timing}/` (`tests/unit/test_p3_stats.py` reproduces
each value to 1e-7):

| quantity | judge record field | formula |
|---|---|---|
| fixed replay cost | `replay_overhead.overhead_ns` | `2*one_ns - two_ns`; one/two = 1- and 2-kernel graph replays, measured per shape (3.4-4.4 us, median ~3.9) |
| per-rep ratio | `ratios[i]` | `(baseline_runs_ns[i] - ovh) / (runs_ns[i] - ovh)` |
| shape speedup | `paired_ratio` = `speedup` | median over k of `sqrt(r[2k] * r[2k+1])` (two reps at a time, order swapped; `setup.first = baseline`) |
| shape times | `baseline_ns`, `median_ns` | `baseline_raw_ns - ovh`; sub = `baseline_ns / paired_ratio` |
| problem speedup | `result.speedup` | geomean over loads of `sum(w*base) / sum(w*sub)` |
| problem CI | `result.speedup_ci` | bootstrap, widened to at least +-1.5 % |
| reps | `runs` | 6 to 100 per shape (time-budgeted; MoE `MAX_REPEATS=100`) |
| gate | `replay_overhead.gate_ns` | ~300 us of GPU work in front of each replay (flush + gate) |

Also from TASK.md: every replay starts from a cold L2 (the judge *reads* a buffer larger than L2), the
submission and the baseline take turns on the same fresh inputs, and **every repetition gets new inputs**
(routing changes per rep, which is why MoE/decode need several seeds locally).

## 2. kdev-tp/1 per shape (engine.py)

1. Seeds: `params.seed_list`, else `seeds_for_shape(judge_seed, n)`, n = `--seeds` or the adapter's
   `input_variance()` (1, or 5 for routing/lengths). Extra seeds rotate `seed + j*0x2545F4914F6CDD1D mod 2^63-1`
   (identical to calib_seeds.py).
2. Per seed: `adapter.make_inputs(shape, seed)`; per file the judge capture recipe (`graph.capture_with_output`):
   eager call + sync, side-stream warm-up, `torch.cuda.CUDAGraph` capture, sync. All files share the inputs.
3. One warm-up replay of every graph, untimed.
4. `reps` repetitions **per seed** (total `reps * seeds`). Rep r uses seed `(r // 2) % seeds`, so each
   order-swapped pair (2j, 2j+1) shares its inputs. In each rep every file is replayed once, order ABBA
   (`--order rotate`: rep r starts at file r % n). The baseline is label `base` and is loaded, prepared and
   ordered LAST (calib.py order; `extra.base_first` flips it). Load order is part of the protocol: it moves
   memory placement, measured at up to 1 us (5 %) on bf16 verify16/pooler with base-first.
5. Each replay: `flush()` (read: `buf.sum()` over a uint8 buffer, default 256 MB = GPU profile `flush_bytes`;
   optional `extra.gate_cycles` of `torch.cuda._sleep`), CUDA event, `g.replay()`, event, synchronize.
6. `us = us_raw - overhead`. Mode `judge` (default) subtracts 4.0 us; `local` subtracts the locally measured
   `2*one - two` (the judge's method); `none` nothing. empty/one/two-kernel graph costs are measured at the
   END of every job (and also at the start in `local` mode) and always reported in `fidelity.overhead_us`;
   measuring them first allocates graph pools ahead of the inputs and shifted a page-gather attention shape.
7. Guard: after timing, each graph is replayed once and `adapter.check(inputs, out)` runs on its output (a
   failure makes the shape `incorrect`). Then the materialisation probe: float outputs (not aliasing inputs)
   are filled with NaN, replayed and checked again; failing only there is a warning (packed outputs such as
   attention's legitimately fail it).

## 3. Statistics (stats.py, pure Python, seeded)

- Per file and shape: median us, bootstrap 95 % CI (2000 resamples), per-seed medians.
- Ratio of label X vs the reference: judge pairing (`paired_ratios` with the per-rep first side), median,
  percentile bootstrap 95 % CI (10 000 resamples, seed = crc32 of shape/labels). Reported as `speedup`
  (ref/X, >1 = X faster) and as the time ratio `median/ci_lo/ci_hi` (X/ref), plus `abs_delta_us` and
  `num_judge_us = ref_median / speedup`.
- Score: `adapter.score(sub_us, base_us)` (load-weighted geomean) with `sub_us = ref_median / pair speedup`,
  CI by resampling the pairs of every shape (2000 draws). `judge_ci` is the same CI widened to +-1.5 %.
- Verdict: `faster`/`slower` only if the CI excludes 1 and |delta| >= max(noise floor (default 3 %,
  `extra.noise_floor`), 2 * SE from the CI); `incorrect` on a failed check; `suspect` when faster than
  `extra.floors_us[shape]`; else `same`. `ab` decides on absolute A-vs-B times (no baseline unless asked).

## 4. Fidelity record and opt-outs

`fidelity`: toolchain (`nvcc` of CUDA_HOME, `so_notes` = versions found in every mapped extension `.so`,
`pin` from the problem's toolchain.lock / `extra.toolchain_pin` / `$KDEV_TOOLCHAIN_PIN`, `match`), flush
`{method, bytes[, gate_cycles]}`, graph, exclusive (`ok|waited|fail|shared|unknown`), waited_s, seeds,
overhead_us, clocks before/after, optouts. Opt-outs (`flush=write|none`, `eager`, `overhead=local|none`,
`allow_jit`) are printed in CAPS and make the row `comparable=0`.

Exclusivity: the lock holder may export `KDEV_EXCLUSIVE`/`KDEV_WAITED_S`; otherwise the worker checks that the
shared lock file (`KDEV_LOCK_PATH`, default `/root/gpu.lock`) is held and that nvidia-smi shows no foreign live
CUDA context at start, between shapes and at the end. A foreign context -> `exclusive=fail` and error
`not_exclusive` (result kept).

No compile under the lock: before any submission is imported the worker installs P5's
`kdev.build.cache.install_load_inline_hook` (mode `use`): `load_inline`/`load` resolve to the prebuilt kdev
cache for the pinned toolchain and the file's knobs/defines; a missing variant -> `needs_build` (exit 4) unless
`--allow-jit`. `extra.legacy_ext` (opt-out `legacy_ext`) uses the process's TORCH_EXTENSIONS_DIR instead,
guarded by `ninja -n` (pending work -> `needs_build`). Other JIT systems (SGLang/Triton caches) are not
intercepted: prebuild them. `fidelity.toolchain.build_cache` and `.exts` record which path was used.

## 5. Worker process contract

`python -m kdev.worker JOB.json [--run-dir DIR]`: chdir to the run dir; writes `meta.json`, `raw.jsonl` (one
Sample per line, appended per shape), `log.txt`, `table.md`/`shapes.json` (time/ab), `calib.md`/`calib.json`
(calib), `result.json`; prints exactly one JSON line (the Result) on stdout; exit status = `Result.exit_code`.
SIGTERM -> `cancelled`, `timeout_s` -> `timeout`, partial samples kept. When the P7 adapter is not available,
the engine uses a private direct OKC adapter (`timing/_direct.py`, calib.py semantics); `extra.adapter =
"direct"` forces it (calib does by default, shapes and seeds from the judge JSON).

## 6. Known differences from the judge (and their size)

- Overhead: constant 4.0 us vs per-shape `2*one-two` (3.4-4.4 us): up to +-0.5 us on the sub time; matters on
  <10 us shapes (warned). Use `--overhead local` to mimic, at the cost of comparability with calibrated rows.
- Inputs: judge refills inputs every repetition; kdev uses `seeds` fixed input sets (5 for routing problems).
- Flush/gate: judge gate ~300 us before each replay; kdev 256 MB read (~65 us). `extra.gate_cycles` adds a gate.
- Judge state copies (`setup.copies`/`prepare_pad`) are not replicated (judge mode, section 9, adds the pads).
- Local baseline (SGLang) runs 1-3 us slower than on the judge on short attention/router shapes
  (all_calib.md): compare absolute submission us, not local speedups, on those shapes.

- Layout sensitivity: attention `decode 1/sliding` (ctx 80000, page 1) baseline reads 19.9-23.0 us depending
  on allocation history in the process (same shape repeated in one job: 20.8, 23.0, 22.8). Judge: 20.9.

## 7. Calibration evidence (pod H200, 2026-10-31, same env as okc/calibration/all: ext13, read flush, 4.0 us)

`kdev calib` (40 reps, 2 runs) vs calib_all.py (40 reps, 2 runs), interleaved under one gpu_run.sh lock,
means of the 2 runs; and vs the published `all_calib.md` local columns:

| problem | metric | vs calib.py (same session) | vs all_calib.md |
|---|---|---|---|
| bf16 (30 shapes) | sub us | median 0.59 %, max 1.78 % (0.16 us), 30/30 within 2 % | median 0.72 %, 29/30 within 2 % |
| bf16 | base us | median 0.50 %, 29/30 within 2 % | median 0.37 %, 28/30 within 2 % |
| bf16 | speedup | median 0.84 %, 29/30 within 2 % | median 0.74 %, 28/30 within 2 % |
| attention (12) | sub us | median 0.18 %, 11/12 within 2 % (d1 sliding 4.72 vs 4.85 us) | median 0.57 %, 11/12 |
| attention | base us | median 0.43 %, 11/12 within 2 % (d1 sliding, layout) | median 0.34 %, 11/12 |

Self-noise for reference: calib.py run-to-run sub us median 0.55 % (max 1.7 %), kdev 0.44 %.

## 8. Judge gap vs local (pod H200, 2026-10-31, same session) and the calibration table

bf16 `best_1.567.py` vs its judge JSON (`62921d1af0fe4904`, sfo3), 40 reps, calib.py and `kdev calib`
interleaved twice (calib.py, kdev, calib.py, kdev), means of the 2 runs:

| comparison (30 shapes) | sub us | base us | speedup |
|---|---|---|---|
| kdev vs calib.py (same session) | median 0.66 %, 28/30 within 2 % | 0.43 %, 29/30 | 0.84 %, 26/30 |
| run-to-run self-noise (calib.py / kdev) | 0.69 % / 0.49 % | 0.47 % / 0.57 % | 0.57 % / 0.99 % |
| kdev vs judge | median 2.62 %, 11/30 within 2 %, max 16.5 % | 2.28 %, 12/30, max 12.9 % | 2.63 %, 13/30 |
| calib.py vs judge | median 2.56 %, 12/30, max 16.2 % | 2.62 %, 14/30, max 12.5 % | 2.29 %, 12/30 |

Verdict: kdev reproduces calib.py (the two >2 % shapes, verify 1/o +0.27 us and verify 1/unpooler +0.1 us,
are consistent across runs: placement-sensitive small shapes, see section 6); both differ from the judge
by the same amount. The gap is judge-side and per shape: local is slower on 3-6 us shapes (unpooler/router
m1 +0.5 us = 16 %; the judge subtracts its per-shape `2*one - two` = 3.8-4.4 us instead of 4.0, which
explains up to 0.4 us of it), and the local SGLang baseline is slower than the judge's on router shapes
(+1.1-1.9 us, 7-13 %) and attention sliding shapes (~5 %). It is therefore recorded, not corrected:
`kdev calib` stores judge/local per case for the submission and the baseline separately in
`$KDEV_HOME/calib/<problem>.json`; `time`/`ab` print `~J` = local us x median ratio (uncertainty
max(1.5 %, half the observed spread), + 0.3 us/us on shapes < 6 us, flagged `?`). Use absolute sub us
(and `~J`) to predict the judge; never local speedups on baseline-gap shapes.

## 9. Judge mode `kdev-tp/1+jm` (`time --judge-mode`, `ab` default on OKC problems; board #49)

The judge's per-run loop, read from its records (`setup`, `replay_overhead`, `runs_ns`/`baseline_runs_ns`) and
rt2_'s `rt2/jf.py`, which reproduced judge pair deltas (`timing/judge_mode.py`):

1. One submission per process. The SGLang baseline is loaded and prepared FIRST (`setup.first = baseline`;
   `--first submission` flips it: the judge picks per run), each `prepare` preceded by a random 8-32 MB pad
   (`setup.prepare_pad`; kept alive; `--no-pad`). Pad sizes come from `extra.pad_seed`.
2. Per shape: inputs from the judge seed (TOML task seed, or the case's seed in `--judge-json J`), plus a pool
   of `--fresh` (8) inputs `seed + 7919*j` (bounded at 2 GB). Graphs are captured submission first, then the
   baseline; one warm-up replay each.
3. `--reps` (60) repetitions; before every even rep the next pool input is copied into the captured input
   tensors (fresh inputs per pair). Even reps replay the baseline first, odd reps the submission first.
4. Every replay: 256 MB read flush + `--gate-us` (235) of `torch.cuda._sleep` (flush + gate ~ the judge's
   `gate_ns` 300 us), events around `g.replay()`.
5. A 1- and a 2-kernel graph are replayed (same flush + gate) every 4th rep; per-shape overhead
   `2*median(one) - median(two)` is subtracted from every replay of that shape (judge `overhead_ns`).
6. Reported per shape: median RAW us of both sides, the overhead, judge pair speedup (median of order-swapped
   pair geomeans), sub us = base / speedup; score = load-weighted geomean (judge formula).

`ab` (judge mode): the worker spawns one judge-mode `time` child process per file in ABBA order (`A B B A`,
`--rounds N` repeats the block; distinct pad seeds) and never touches CUDA itself. Per shape it reports the
RAW delta B-A (mean of each file's processes) with both half-block deltas, the judge-formula us of each file
(mean over its processes) with a bootstrap CI (pairs resampled inside each process), and the score of every
process vs its own baseline; `B/A score` = ratio of the mean scores. A verdict other than `same` also needs
both half-blocks to agree. `--fast` = the old single-process interleaved mode (section 2), still the mode of
`sweep`. Judge-mode rows are tagged `summary.mode = judge`, `protocol kdev-tp/1+jm`.

Why: in one process, interleaving the SGLang baseline replays with the submission moves router m128/256 raw by
~0.35 us; the single-process A/B had flipped a verdict (local -0.49 vs judge +0.26 us).

Validation (pod H200, 2026-10-31; judge pair records p1 = base vs stk3b, p3 = base vs sk2o_l125, each file judged
twice; per-shape RAW delta = mean(B a/b) - mean(A a/b); judge self-noise = |base_b - base_a| per shape):

| pair | method | MAE vs judge (30 shapes) | corr | sign agree (\|dJ\| >= 0.2 us) | MAE on jf's 11 shapes | B/A score (judge) |
|---|---|---|---|---|---|---|
| p1 | `ab` judge mode, 1 round | 0.146 us | 0.67 | 7/9 | 0.244 | x1.0064 (x1.0114) |
| p1 | `ab` judge mode, 2 rounds | 0.142 us | 0.73 | 8/9 | 0.221 | x1.0069 |
| p1 | rt2/jf.py (11 shapes, g0 / gate 235) | - | 0.72 | 6/8 | 0.228 / 0.228 | - |
| p1 | `ab --fast` (old) | 0.189 us | 0.45 | 6/9 | 0.276 | x1.007 |
| p3 | `ab` judge mode, 1 round | 0.147 us | 0.76 | 10/11 | 0.163 | x1.0058 (x1.0059) |
| p3 | `ab --fast` (old) | 0.186 us | 0.69 | 7/11 | 0.175 | x1.005 |

Judge self-noise MAE: p1 0.20 us, p3 0.13 us, so judge mode is at the judge's own noise level. An ABBA run of
30 bf16 shapes takes ~80 s (4 processes x ~20 s).

## 10. Phased GPU lock (lock only the timed part; `service/phase.py`)

The flock on `/root/gpu.lock` stays the one `/root/gpu_run.sh` takes (`flock 9` = exclusive). kdev now also
uses it SHARED:

| job | lock |
|---|---|
| `time --judge-mode`, every child process of a judge-mode `ab` | prepare (CUDA init, load, pads, `prepare()`, flush buffer) SHARED; then `torch.cuda.synchronize()`, park (idle), EXCLUSIVE for the timed section until the process exits (graph capture, warm-up, timed reps, verification, overhead graphs, dispatch probe, CUDA teardown) |
| `discover` (prepare-time extension discovery of `kdev build`), `check` (not `--cold`) | SHARED for the whole job (untimed) |
| everything else, gpu_run.sh | EXCLUSIVE for the whole job (unchanged) |

Guarantees: a timed section never overlaps a prepare/discover/check of any process (SH and EX exclude each
other in the kernel). During a timed section the only other CUDA contexts allowed are *parked/ready* kdev
workers: synchronized, blocked in a flock wait, no queued work (registry `$KDEV_HOME/phase/procs/<pid>.json`,
pid + start-time checked). All foreign-context checks (kdev lock, worker start/mid/end, lane violator poll)
ignore exactly those pids. gpu_run.sh still waits for every context, so a parked worker gives way: if the
lock is held >= 6 s with no kdev holder registered, it exits and is restarted (fresh prepare).

Fairness: a preparer takes one of `$KDEV_PHASE_SLOTS` (1 since 2026-11-01; 6 = concurrent prepares, see Fidelity status) prepare slots (bounds parked GPU memory; an `ab`
parent also needs `KDEV_PHASE_MIN_FREE_MIB` = 30000 free before starting a non-head child), then defers while
an EX waiter that queued before it exists (FIFO with gpu_run.sh / exclusive kdev jobs) or a phased process
is ready to time (writer preference: timed sections run back to back, prepares batch). `ab` judge mode starts
its ABBA children as slots allow; they prepare concurrently and time strictly in A B B A order (go token =
previous process exited). Opt out: `--no-phased`, `KDEV_PHASED=0`. Results carry `summary.phase`
{slot_wait_s, sh_wait_s, prepare_s, parked_s, ex_wait_s, foreign_wait_s, deferred, exclusive}.

Requeue: a head of the ABBA chain that gave way once restarts *whole* (`KDEV_PHASE_WHOLE=1`: EXCLUSIVE from
prepare to exit, FIFO in the kernel queue), so gpu_run.sh jobs queueing during each re-prepare cannot starve
it (pod 2026-11-01: 4 restarts in 4 min). Queue view (`lockq.view`): a holder is an opener whose
`/proc/<pid>/fdinfo/<fd>` lists a granted flock; a parked worker keeps its fd open after LOCK_UN and is NOT a
holder (the 2026-10-31 watchdog kill was this false positive).

Fidelity status (pod 2026-11-01, d1/o, mrg/m1 vs m2, raw per process): old exclusive mode and phased with
`KDEV_PHASE_SLOTS=1` give per-process RAW 8.99-9.09 us, dRAW -0.05..0.0; phased with 6 slots (concurrent
prepares) gives 8.99-9.25 us, dRAW -0.14..+0.07, including a process that timed with no other context alive
(suspect: interleaved physical placement of concurrently prepared processes). Default is therefore 1 slot:
prepares no longer overlap each other, only the EXCLUSIVE hold shrinks (~11 s -> 1-3 s per process).

Placement: judge-mode processes record `summary.placement[label]` (weights = prepare state + module globals,
outputs per shape: top tensors' `data_ptr % 2 MiB`, `% 64 KiB`, class A = 2 MiB aligned / B = 64 KiB / C);
`ab` warns `PLACEMENT: ... class differs` when the compared files' processes differ. `--pad-mb N` uses the
same N MB pad before every prepare (standardized), `--pad-seed S` re-randomizes the 8-32 MB pads.

## 11. Timelines (`kdev timeline`, `kdev.timeline/2`)

Not a timing verdict: a per-CTA account of where one kernel's time goes. Conditions follow the judge where they
matter for phase lengths: every module is prepared once; per shape and seed fresh inputs (`seeds_for_shape`:
judge seed first, then the calib_seeds rotation, so MoE routing changes per seed), warm call, CUDA graph;
`--replays` timed replays after one warm-up round, the instrumented and the unmarked graphs interleaved ABBA,
256 MB read flush before each replay, mark buffer zeroed before the flush (outside the timed region). Phased lock:
load/prepare/capture SHARED, replays EXCLUSIVE. Every statistic is computed per replay (min / CTA at n//2 / max
over CTAs, per-CTA phase sums) and then the median over replays and seeds is reported, as the agents' runners
did (mw_tl_run.py, mtn_tlrun.py). Mark cost = median marked replay - median unmarked replay of the same file
and shape; it is a whole-kernel number, the local inflation next to a mark (gap/tail) can be larger.
