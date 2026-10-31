# Problem adapters, check, sanitize, comply (P7)

Interface: `src/kdev/adapters/base.py` (frozen). Lookup: `kdev.adapters.registry.get_adapter(problem, file,
adapter)`: `adapter:problem`, a problem dir (TASK.md + reference.py), a shipped problem name, a `.cu` with
`main()` (cu_main), a file with `Model`/`ModelNew`/`get_inputs` (kernelbench), the submission's directory walking
upwards, then `config.default_problem` / `$KDEV_PROBLEM`. Problem dirs are searched under `$KDEV_PROBLEMS_ROOT`,
`config.problems_root`, the TOML `search_dirs`, `/root/okc`, `~/okc` (layout `<root>/<p>/<p>/`).

## okc adapter (`adapters/okc.py`)

| problem | kernel_fn | prepare | check | extras |
|---|---|---|---|---|
| muse_attention_decode | `kernel_fn(data)` | none | `check_implementation(data, collect(data, out))` | writes FP8 cache entries |
| muse_bf16_proj | `kernel_fn(data, state)` | `prepare(weights)` | `check_implementation(data, out, w)` | router FP32 tol 1e-4 |
| muse_moe_w4a16 | `kernel_fn(data, state)` | `prepare(weights)` | `check_implementation(data, out, w)` | routing variance |
| muse_decode_layer / muse_prefill_layer | `kernel_fn(data, state)` | `prepare(weights)` | via `collect`, with weights | `case_fields` (local_start) per input |

- Config per problem: `adapters/okc_problems/<p>.toml` (see the README there). Without a TOML the shapes and tests
  are parsed from TASK.md (`parse_task_md`).
- `reference.py` is imported from the problem dir (dir first on sys.path for `checks`/`nvfp4`) under a private
  module name, so a submission's `import reference` is not satisfied by kdev.
- `make_inputs` clears the reference's `functools` caches (decode/prefill cache `generate_input`), so every call
  returns fresh, non-aliased tensors.
- `prepare` passes a shallow copy of the weights dict and records `last_prepare_s`; `comply` verifies the weight
  tensors are unchanged afterwards.
- Seeds: `default_seeds()` = 5 when `input_variance` has routing/lengths, else 1; `judge_seed(shape)` = the judge
  JSON's seed (shapes from `judge_cases`) or the task seed from the TOML. `judge_cases(json)` /
  `judge_tests(json)` rebuild the judge's benchmark/test cases with their seeds (ids matched to the TOML by case).
- `score` = `base.load_weighted_geomean` over the scored shapes (per load: sum(weight * us); geomean over loads of
  base/sub).
- Extra helpers: `judge_repeats(id)`, `floor_us(id)`, `speed_of_light_us(shape)`, `smallest_test()`,
  `find_shape(id)`, `comply_extra()`, `weights()`, `reference()`.

## cu_main, kernelbench

- `cu_main`: one shape `main`; `build_hook` = one nvcc unit (`-arch` from the GPU profile, `-o <stem>`); `call`
  runs the built binary; `check` = exit 0 and no `FAIL` line. Used for build/sass/regs/sanitize; timing is a stretch
  goal.
- `kernelbench`: problem file with `Model`, `get_inputs`, `get_init_inputs`; submission `ModelNew`; models built
  after `torch.manual_seed(42)`, `allclose(atol=rtol=1e-2)`, baseline = the problem's `Model`.

## kdev check (`checks/check.py`, worker kind=check)

Cases = tests + scored (+ extra with `--extra-shapes`; `--tests-only`; `--shapes` filter: `|`-separated
substrings of id/case/load). Fresh unpredictable seeds per run (recorded). Per (case, seed):

- eager: R repeats, each checked; bitwise comparison between repeats;
- graph: warm-up + side-stream warm-up + capture (harness/judge sequence), R replays, each checked and compared
  with the eager output (any bit difference -> at least `nondeterministic`, reported as graph-vs-eager rel L2;
  repeat-to-repeat max ulp <= 1 is the fp32-atomic-order signature);
- graph_fresh: a second seed's inputs are copied into the captured buffers and the graph replayed once: the judge
  replays one graph on fresh inputs, so data-derived caching or value branching fails here.

Classes: `wrong` (all repeats fail), `intermittent` (some fail -> `kdev sass --audit pdl`, racecheck),
`nondeterministic` (all pass, bits differ; max ulp over elements >= 2^-12 of max|x|), `pass`, `error`.
A fatal CUDA error stops the run (the context is gone). `--cold` points TORCH_EXTENSIONS_DIR/TRITON_CACHE_DIR/
SGLANG_JIT_CACHE_DIR/CUDA_CACHE_PATH at empty dirs and reports load + prepare + first call per scored shape vs
`budget_s` (FAIL > budget or prepare > 300 s; RISK > 60% of the budget). The dispatch summary
(`kdev.observe.dispatch`) is attached. Artifacts: `cases.jsonl`.

## kdev sanitize (`checks/sanitize.py`, worker kind=sanitize)

`run_sanitize_inner` re-execs `compute-sanitizer --tool T --print-limit 100 --log-file sanitizer.log python -m
kdev.checks.sanitize job.json --run-dir D` with `KDEV_SANITIZER_INNER=1`; the inner run executes one small shape
(`--shape`, default `smallest_test()`) eagerly and as a graph replay. `parse_log` gives errors (from the SUMMARY
line), counts per type, the first 5 distinct errors (type | location | thread/block). Sanitizer lookup:
`$KDEV_COMPUTE_SANITIZER`, PATH, `$CUDA_HOME/bin`, `/usr/local/cuda*/bin`. cu_main runs the built binary.

## kdev comply (`checks/comply.py`)

PASS/RISK/FAIL items; RISK needs a human. `--static-only` runs in the CLI process.

- Static: entry points (duplicate `kernel_fn`/`prepare` = overwritten entry points), judge hard rules (imports the
  reference, frame reads, network, processes, timing patches, module edits) = FAIL; torch global numerics, env /
  harness I/O, secrets = RISK; forbid_patterns outside prepare; caching heuristics (data_ptr keys, functools caches,
  module containers written per call, `global`); lossy casts inside prepare; kernel inventory (duplicate kernel
  names = embedded alternative sources).
- Dynamic (GPU): prepare time vs 300 s; weights untouched by prepare; state census (float weights kept at their
  dtype or shared storage, FP32 router as FP32 or BF16 hi+lo, no dequantised NVFP4 copies); host syncs (torch sync
  debug mode); aten op/dtype census; graph capture of every scored shape; cross-call state (same buffers refilled
  with another seed vs a fresh call; graph replay on fresh inputs; outputs aliasing state buffers noted); output
  error vs SGLang's baseline on the same inputs (> 3x and > 25% of tol = RISK: activations below BF16?); router
  precision at T <= 8 (decode/prefill: the submission's experts vs FP32 and BF16-simulated routers on discriminating
  tokens; matching BF16 more often = FAIL); every key (bands of large values at the window's oldest edge and middle
  must be attended); active dispatch per scored shape (profiler kernel names vs kernels defined in the file: shapes
  running none of them, kernels never launched).
