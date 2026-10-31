# kdev build plan (Phase 0)

Status: 2026-10-31. This is the binding plan for the parallel builders. DESIGN.md says *why*; this file says
*what, where, and with which contract*. The skeleton in `src/kdev/` already contains every module listed here
with the agreed signatures, dataclasses and docstrings; builders replace `raise NotImplementedError` bodies and
may add **private** helpers (`_name`) and new private modules inside the directories they own. Changing a
public signature, a dataclass field, the ledger schema, the job/result JSON or an error code requires a note in
`docs/CHANGES.md` (create it; append-only) and a heads-up to the coordinator, because other packages depend on it.

Contents: 1 inventory - 2 architecture & module boundaries - 3 data model - 4 CLI tree - 5 MCP - 6 output
shaping - 7 error codes - 8 safety - 9 tests - 10 build packages & file ownership - 11 integration order.

---

## 1. Tool inventory

### 1.1 Own tools (we write them)

| # | Tool (CLI) | Purpose | GPU | Package |
|---|---|---|---|---|
| 1 | `serve` | GPU job service daemon: owns the GPU, parallel compile workers without CUDA context, one serial timing lane with the judge protocol, exclusivity + offender log | owns | P2 |
| 2 | `job submit/status/wait/cancel/list/logs`, `queue status/offenders`, `lock run` | client side of the service; flock fallback (gpu_run.sh semantics) when no daemon | queued | P2 |
| 3 | `time`, `ab` | judge-faithful timing: graph replay, read flush, ABBA/rotate interleave, overhead subtraction, multi-seed, paired ratios, bootstrap CI, score CI | queued | P3 |
| 4 | `calib` | judge-vs-local per shape, DISAGREE detection, calibration table | queued | P3 |
| 5 | `ledger query/tried/best/runs/import/export` | SQLite experiment ledger; every result recorded; "has this been tried" | none | P4 |
| 6 | `idea add/list/show/claim/resolve/evidence` (+ `note` alias) | hypothesis board with status/evidence/owner; limit-claim gating | none | P4 |
| 7 | `judge prepare/record/ledger/pair` | record-only judge integration; duplicate-hash warning; `pair` may call `okc check` only with `--allow-check` | none | P4 |
| 8 | `build` | compile-only, pinned toolchain, hash-keyed cache, `.so` toolchain notes, ptxas -v parse, C751x | none | P5 |
| 9 | `sass`, `sass diff` | hot-loop extraction, per-element histogram by opcode class, build-vs-build diff, repeated-work detector, PDL hoist audit (LDG before ACQBULK), spills/C751x/expect audits | none | P5 |
| 10 | `sweep` | knob grid: parallel builds (via service build lane), quick check, batched interleaved ab vs base | queued | P5 |
| 11 | `timeline` | source annotations -> `%globaltimer` instrumentation -> per-CTA phase breakdown + Perfetto + chart data | queued | P6 |
| 12 | `floor`, `floor calibrate` | roofline + latency skeleton from GPU-profile constants; microbench suite fills profile | queued (calibrate) | P6 |
| 13 | `regs` | static occupancy: regs/smem/launch bounds -> CTAs/SM per limiter, setmaxnreg groups | none | P6 |
| 14 | `profile`, `dispatch` | top kernels (ncu if permitted, else CUPTI/torch.profiler, nsys if present); which kernels launch per scored shape | queued | P6 |
| 15 | `check` | correctness: fresh seeds, extra shapes, eager+graph, repeat; nondeterministic vs intermittent vs wrong; `--cold` | queued | P7 |
| 16 | `sanitize` | compute-sanitizer wrapper, summarised | queued | P7 |
| 17 | `comply` | rules checklist (precision per op, cross-call state, work outside timed call, graph-capturable, host sync) | queued | P7 |
| 18 | adapters | `okc` (generic over muse_* via reference.py/checks.py/TASK.md + `kdev_problem.toml`), `cu_main`, `kernelbench` (stub) | - | P7 |
| 19 | `dashboard build/watch` | static HTML from ledger + queue + optional read-only board scrape | none | P8 |
| 20 | `mcp` | MCP stdio server (official `mcp` SDK, FastMCP), generated from the command registry | - | P8 |
| 21 | `doctor`, `show`, `runs`, `version` | health checks aggregated from every package's `probe()`; paged artifact reads | none | P1 |
| 22 | `sync`, `remote` | md5-verified pod<->local copy with conflict copies; ssh runner; pod bootstrap | none | P2 |
| 23 | `okc status/leaderboard` | read-only pass-through to the `okc` CLI (never submit) | none (network) | P4 |
| 24 | `skills/kdev/SKILL.md` | agent workflow (baseline -> knobs -> SASS -> rewrite), brief checklist mapped to commands | - | P8 |

### 1.2 Existing tools we wrap (never reimplement)

| Existing tool | Wrapped by | How / fallback |
|---|---|---|
| `nvcc`, `ptxas -v`, `cu++filt` | `kdev.build.compile`, `kdev.build.ptxas` | subprocess with the pinned toolchain env; ptxas -v parsed per kernel |
| `torch.utils.cpp_extension` (load/load_inline), NVRTC | `kdev.build.compile` | load_inline sources discovered by importing the module with CUDA blocked (`CUDA_VISIBLE_DEVICES=""`) and an intercepted `load_inline`; cache dir = hash(toolchain+flags+source) |
| `cuobjdump -sass/-ptx/-elf`, `nvdisasm -cfg -lineinfo` | `kdev.sass.disasm` | `.so` -> cubins -> SASS; CFG from nvdisasm when present, else our own back-edge finder |
| CUDA graphs, CUDA events (torch) | `kdev.timing.graph`, `kdev.timing.engine` | judge capture recipe (warm-up, side-stream warm-up, capture) copied from ah.py/calib.py |
| CUPTI via `torch.profiler` | `kdev.observe.cupti` | kernel names/durations/grid when ncu is blocked; `--nopdl` for separable kernel times |
| `ncu` | `kdev.observe.ncu` | only when `RmProfilingAdminOnly=0`; else `capability_unavailable` + fallback |
| `nsys` | `kdev.observe.nsys` | if present: `nsys profile --trace=cuda` + `nsys stats` csv |
| `compute-sanitizer` | `kdev.checks.sanitize` | memcheck/racecheck/synccheck/initcheck on one small shape |
| `nvidia-smi` | `kdev.service.smi` | clocks/power/temp, compute-apps pid list (foreign context detection) |
| `flock` (fcntl) | `kdev.service.lock` | same lock file as gpu_run.sh (`/root/gpu.lock` by default) so legacy scripts and kdev exclude each other |
| NVBit (optional) | `kdev.observe.nvbit` | instruction counts when present; probe only in v1 |
| `okc` CLI | `kdev.judge.okc_cli` | read-only `status`, `leaderboard`, `jobs`; `check` only from `judge pair --allow-check`; `submit` never (hard-coded deny) |
| `ssh`/`scp`/`rsync` | `kdev.remote` | `ssh -i key -p port`; md5 verification after copy |
| `sqlite3` (stdlib) | `kdev.ledger.db`, `kdev.service.queue` | WAL mode, one file per store |
| `mcp` Python SDK (FastMCP) | `kdev.mcp.server` | optional extra `[mcp]` |

Python deps: core = stdlib only (+ `tomli` on <3.11). `numpy` is optional (stats fall back to pure Python).
`torch` is **never imported by the CLI process**; only the GPU worker (`python -m kdev.worker`, run in the
adapter's pinned interpreter) imports torch. This keeps `kdev --help`, ledger, sass, dashboard and MCP fast and
GPU-free, and lets unit tests run on Windows without CUDA.

---

## 2. Architecture and module boundaries

```
 agent ──ssh──> kdev CLI (torch-free) ──┬── static tools: sass, regs, floor, ledger, idea, judge, dashboard, show
                     │                  │
                     │ JobSpec (JSON)   └── build (subprocess nvcc/ptxas; never takes the GPU lock)
                     v
             kdev serve (daemon, owns GPU)        queue.db (SQLite WAL): jobs, events, offenders, heartbeat
               ├─ build lane: N parallel workers, CUDA_VISIBLE_DEVICES="" (compile-only)
               └─ GPU lane: 1 serial worker: lock.acquire(exclusive) -> `python -m kdev.worker job.json`
                                                   (pinned interpreter; imports torch; runs timing/check/...)
                                                   -> result.json in run dir -> ledger.record(result)
```

Layers (imports only go downwards; enforced by `tests/unit/test_layers.py`):

1. `kdev.contract`, `kdev.models`, `kdev.util` - pure data + helpers, no I/O beyond files.
2. `kdev.config`, `kdev.runs`, `kdev.output`, `kdev.registry` - config, run dirs, rendering, command registry.
3. Domain packages: `service`, `timing`, `ledger`, `judge`, `build`, `sass`, `timeline`, `floor`, `observe`,
   `checks`, `adapters`, `remote`, `dashboard`. They may import layer 1-2 and each other's **public**
   functions listed in this plan. `timing`, `checks`, `timeline`, `observe` runtime parts that touch torch are
   only imported inside `kdev.worker` (lazy `import torch` inside functions, never at module top level).
4. `kdev.cli`, `kdev.mcp`, `kdev.worker` - entry points.

### 2.1 Command registry (how CLI and MCP stay one contract)

Each package owns a `commands.py` with `register(reg: Registry) -> None`. It declares each command with
`reg.command(name, help, effects, handler, args=[Arg(...)...], gpu=bool, long_running=bool)`. `kdev.cli`
builds argparse from the registry; `kdev.mcp.server` builds MCP tools from the same registry (Arg -> JSON
schema). **A builder adds/changes flags only in their own `commands.py`**, so nobody collides in `cli.py`.

Handler signature (all commands): `handler(args: Namespace, ctx: Context) -> Result`. Phase-0 handlers are
`registry.todo("<module.function> (Pn)")` placeholders; replace each with a real `cmd_x` in the same file. `Context` carries config,
the output mode, the agent name, and a `submit()` helper that routes GPU work to the service.

### 2.2 GPU routing rule

A command marked `gpu=True` never touches the GPU in the CLI process. It builds a `JobSpec` and calls
`ctx.submit(spec)`:
- service up (fresh heartbeat in `queue.db`) -> enqueue; wait unless `--detach` (prints `job: <id>`);
- service down -> `--direct` behaviour: run `python -m kdev.worker` in the foreground under
  `service.lock.exclusive()` (gpu_run.sh semantics), recording `exclusive: ok|waited|fail`;
- `KDEV_NO_GPU=1` (unit tests) -> `capability_unavailable`.

### 2.3 Worker

`python -m kdev.worker JOB.json` (`src/kdev/worker.py`, P3) loads the job, chdirs to the run dir, resolves the
adapter, and dispatches on `kind`: `time|ab -> timing.engine`, `calib -> timing.calib`, `check|comply ->
checks`, `sanitize -> checks.sanitize` (re-execs under compute-sanitizer), `timeline -> timeline.run`,
`profile|dispatch -> observe`, `floor_calibrate -> floor.microbench`. It writes `result.json`, `raw.jsonl`,
`meta.json`, exits with the error-code exit status. The daemon (or direct mode) ingests `result.json` into the
ledger. The worker refuses to compile: a `needs_build` error unless the job has `allow_jit=true` (logged as an
opt-out).

---

## 3. Data model

All JSON documents carry `"schema": "kdev.<name>/<int>"`. Times are microseconds (`_us`) as floats, wall
times are ISO-8601 UTC strings (`_utc`). Dataclasses live in `src/kdev/models.py` (owned by P1, frozen after
Phase 0) and `src/kdev/contract.py`; every dataclass has `to_dict()`/`from_dict()`.

### 3.1 Job spec (`kdev.job/1`)

```json
{
  "schema": "kdev.job/1",
  "job_id": "j20261031-143012-7f3a",
  "kind": "build|check|time|ab|sweep|calib|sanitize|comply|timeline|profile|dispatch|floor_calibrate|exec",
  "lane": "build|gpu",
  "problem": "muse_moe_w4a16",
  "adapter": "okc",
  "files": [
    {"path": "/root/mad/a.py", "label": "A", "md5": "…", "knobs": {"TILE": 128}, "defines": ["-DX=1"],
     "judge_json": null}
  ],
  "params": {"reps": 40, "seeds": 5, "shapes": "d1", "order": "abba", "flush": "read",
             "flush_bytes": 268435456, "graph": true, "overhead": "judge+local", "base": "baseline",
             "repeat": 1, "allow_jit": false, "extra": {}},
  "agent": "moe-w3", "priority": 0, "timeout_s": 1800, "max_hold_s": 1800,
  "run_name": "tile128", "submitted_utc": "2026-10-31T14:30:12Z",
  "requires_prebuilt": true, "parent_job": null
}
```
`exec` is a raw `lock run -- CMD` job (argv in `params.argv`), checked against the deny-list.

### 3.2 Result envelope (`kdev.result/1`) - stdout of `--json`, and `result.json` in the run dir

```json
{
  "schema": "kdev.result/1", "tool": "ab", "version": "0.1.0", "protocol": "kdev-tp/1",
  "run_id": "20261031-143012-ab-tile128", "job_id": "j…", "status": "ok|fail|error",
  "verdict": "faster|slower|same|incorrect|suspect|pass|fail|…", "exit_code": 0,
  "summary": {"…small, tool-specific, <=2 KB…": null},
  "shapes": [{"shape_id": "d1/sliding", "weight": 46, "load": "d1",
              "per_file": [{"label": "A", "md5": "…", "median_us": 21.3, "ci_lo_us": 21.1, "ci_hi_us": 21.6,
                            "n": 40, "seeds": [11, 12], "per_seed_median_us": [21.2, 21.4]}],
              "ratio": {"num": "B", "den": "A", "median": 0.973, "ci_lo": 0.961, "ci_hi": 0.985},
              "verdict": "faster"}],
  "score": {"per_file": {"A": 2.593, "B": 2.640}, "delta": {"median": 1.018, "ci_lo": 1.004, "ci_hi": 1.031}},
  "fidelity": {"toolchain": {"nvcc": "13.0.88", "pin": "13.0.88", "so_notes": ["V13.0.88"], "match": true},
               "flush": {"method": "read", "bytes": 268435456},
               "graph": true, "exclusive": "ok", "waited_s": 3.1, "seeds": [11, 12, 13, 14, 15],
               "overhead_us": {"judge_const": 4.0, "empty_graph": 0.74, "one_kernel_graph": 5.2, "subtracted": 4.0},
               "clocks": {"before": {"sm_mhz": 1980, "mem_mhz": 3201, "power_w": 120, "temp_c": 41},
                          "after": {"…": 0}},
               "optouts": []},
  "dispatch": {"d1/sliding": [{"kernel": "vjc_k<1>", "count": 1, "grid": [132, 1, 1]}]},
  "files": {"result": "…/result.json", "raw": "…/raw.jsonl", "log": "…/log.txt"},
  "warnings": ["shape v1/sliding < 10 us: overhead-sensitive"],
  "next": ["kdev sass /root/mad/b.py --kernel vjc --audit pdl"],
  "error": null
}
```
`error` = `{"code": "needs_build", "message": "…", "details": {…}, "recovery": "run `kdev build X` first"}`.

### 3.3 Run directory

`$KDEV_HOME/runs/<yyyymmdd-hhmmss>-<tool>-<slug>/`: `meta.json` (job spec, inputs + md5s, toolchain, host,
GPU, clocks), `result.json`, `raw.jsonl` (one sample per line: `{shape_id, label, seed, rep, order_pos,
us_raw, us}`), `log.txt`, tool artifacts (`*.sass`, `ptxas.txt`, `trace.json`, `timeline.csv`, `sweep.csv`,
`sanitizer.log`). Sections for `kdev show` are declared in `result.json["files"]`.

### 3.4 Ledger schema (`$KDEV_HOME/ledger.db`, SQLite WAL) - authoritative copy in `src/kdev/ledger/schema.sql`

```sql
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);  -- schema_version=1
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY, job_id TEXT, tool TEXT NOT NULL, status TEXT NOT NULL, verdict TEXT,
  problem TEXT, adapter TEXT, agent TEXT, host TEXT, gpu_name TEXT, gpu_uuid TEXT,
  created_utc TEXT NOT NULL, finished_utc TEXT, protocol TEXT NOT NULL, comparable INTEGER NOT NULL DEFAULT 1,
  nvcc_version TEXT, toolchain_pin TEXT, toolchain_json TEXT,
  flush_method TEXT, flush_bytes INTEGER, graph INTEGER, exclusive TEXT, waited_s REAL,
  seeds_json TEXT, reps INTEGER, overhead_us REAL, optouts_json TEXT,
  clocks_before_json TEXT, clocks_after_json TEXT, run_dir TEXT, summary_json TEXT, error_code TEXT);
CREATE TABLE IF NOT EXISTS artifacts (               -- each input file of a run
  run_id TEXT NOT NULL REFERENCES runs(run_id), label TEXT NOT NULL, role TEXT NOT NULL,  -- sub|base|variant|cubin
  path TEXT, md5 TEXT NOT NULL, norm_md5 TEXT,        -- norm_md5: comments/whitespace stripped (dedup-proof)
  knobs_json TEXT, defines TEXT, so_md5 TEXT, PRIMARY KEY (run_id, label));
CREATE TABLE IF NOT EXISTS shape_results (
  run_id TEXT NOT NULL, label TEXT NOT NULL, shape_id TEXT NOT NULL, load TEXT, weight REAL,
  seed INTEGER NOT NULL DEFAULT -1,                    -- -1 = pooled over seeds
  median_us REAL, ci_lo_us REAL, ci_hi_us REAL, n INTEGER,
  ratio_den_label TEXT, ratio_median REAL, ratio_ci_lo REAL, ratio_ci_hi REAL,
  check_status TEXT, PRIMARY KEY (run_id, label, shape_id, seed));
CREATE TABLE IF NOT EXISTS scores (
  run_id TEXT NOT NULL, label TEXT NOT NULL, score REAL, ci_lo REAL, ci_hi REAL,
  vs_label TEXT, PRIMARY KEY (run_id, label));
CREATE TABLE IF NOT EXISTS builds (
  run_id TEXT NOT NULL, label TEXT NOT NULL, ext_name TEXT NOT NULL, cache_key TEXT, cache_hit INTEGER,
  seconds REAL, so_path TEXT, so_notes TEXT, PRIMARY KEY (run_id, label, ext_name));
CREATE TABLE IF NOT EXISTS kernels (
  run_id TEXT NOT NULL, label TEXT NOT NULL, kernel TEXT NOT NULL, regs INTEGER, spill_store INTEGER,
  spill_load INTEGER, stack INTEGER, smem_static INTEGER, barriers INTEGER, c751x TEXT,
  sass_instr INTEGER, PRIMARY KEY (run_id, label, kernel));
CREATE TABLE IF NOT EXISTS judge_results (
  judge_id TEXT PRIMARY KEY, problem TEXT, file_md5 TEXT, norm_md5 TEXT, mode TEXT,  -- check|submit
  status TEXT, speedup REAL, speedup_ci_json TEXT, cluster TEXT, node TEXT, toolchain TEXT,
  created_utc TEXT, finished_utc TEXT, duplicate_of TEXT, raw_path TEXT, recorded_utc TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS judge_shapes (
  judge_id TEXT NOT NULL, shape_id TEXT NOT NULL, case_json TEXT, sub_us REAL, base_us REAL, speedup REAL,
  PRIMARY KEY (judge_id, shape_id));
CREATE TABLE IF NOT EXISTS calib (
  run_id TEXT NOT NULL, judge_id TEXT NOT NULL, label TEXT NOT NULL, shape_id TEXT NOT NULL,
  local_speedup REAL, judge_speedup REAL, local_over_judge REAL, flag TEXT,   -- ok|out_of_band|DISAGREE
  PRIMARY KEY (run_id, judge_id, label, shape_id));
CREATE TABLE IF NOT EXISTS ideas (
  idea_id INTEGER PRIMARY KEY AUTOINCREMENT, problem TEXT, title TEXT NOT NULL, hypothesis TEXT,
  kind TEXT NOT NULL DEFAULT 'idea',                   -- idea|dead|limit|win (note-tool kinds)
  status TEXT NOT NULL DEFAULT 'open',                 -- open|claimed|testing|confirmed|refuted|abandoned
  owner TEXT, expected_gain_pct REAL, tags TEXT, variant TEXT,   -- exact variant tested (LESSONS 6)
  created_utc TEXT NOT NULL, updated_utc TEXT NOT NULL, resolution TEXT);
CREATE TABLE IF NOT EXISTS idea_evidence (
  idea_id INTEGER NOT NULL REFERENCES ideas(idea_id), ref_kind TEXT NOT NULL,  -- run|judge|file|url|text
  ref TEXT NOT NULL, note TEXT, supports TEXT,           -- for|against|neutral
  author TEXT, added_utc TEXT NOT NULL);
CREATE VIRTUAL TABLE IF NOT EXISTS ideas_fts USING fts5(title, hypothesis, variant, content='ideas',
  content_rowid='idea_id');                              -- fallback to LIKE if fts5 unavailable
CREATE INDEX IF NOT EXISTS ix_art_md5 ON artifacts(md5);
CREATE INDEX IF NOT EXISTS ix_art_norm ON artifacts(norm_md5);
CREATE INDEX IF NOT EXISTS ix_runs_problem ON runs(problem, created_utc);
CREATE INDEX IF NOT EXISTS ix_judge_md5 ON judge_results(file_md5);
```
"Has this been tried" (`ledger.tried`) matches on: exact md5, norm_md5, (problem, knobs subset), and FTS over
idea text; returns run ids + verdicts + judge ids. Rows with `comparable=0` (old protocol, write flush,
toolchain mismatch, non-exclusive) are shown but marked `not comparable`.

### 3.5 Service queue (`$KDEV_HOME/queue.db`) - authoritative copy in `src/kdev/service/schema.sql`

```sql
CREATE TABLE IF NOT EXISTS jobs (job_id TEXT PRIMARY KEY, lane TEXT NOT NULL, kind TEXT NOT NULL,
  state TEXT NOT NULL,          -- queued|running|done|failed|cancelled|timeout
  priority INTEGER NOT NULL DEFAULT 0, agent TEXT, spec_json TEXT NOT NULL, run_id TEXT, run_dir TEXT,
  submitted_utc TEXT NOT NULL, started_utc TEXT, finished_utc TEXT, pid INTEGER, exit_code INTEGER,
  error_code TEXT, waited_s REAL);
CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, ts_utc TEXT NOT NULL,
  job_id TEXT, kind TEXT NOT NULL, message TEXT);   -- started|finished|warn_hold|offender|refused|…
CREATE TABLE IF NOT EXISTS offenders (id INTEGER PRIMARY KEY AUTOINCREMENT, ts_utc TEXT NOT NULL,
  pid INTEGER NOT NULL, cmdline TEXT, gpu_mem_mib INTEGER, during_job TEXT, action TEXT);  -- waited|aborted
CREATE TABLE IF NOT EXISTS heartbeat (id INTEGER PRIMARY KEY CHECK (id = 1), pid INTEGER, host TEXT,
  ts_utc TEXT NOT NULL, version TEXT, gpu_lane_job TEXT, build_workers INTEGER);
```
Clients write only by `INSERT INTO jobs` (state `queued`) and `UPDATE … SET state='cancelled'` on their own
queued jobs; everything else is the daemon's. Offenders are also appended to the legacy
`/root/gpu_offenders.log` so the existing watcher keeps working.

### 3.6 Problem adapter interface (`src/kdev/adapters/base.py`, frozen in Phase 0)

```python
@dataclass class Shape:  shape_id: str; case: dict; weight: float = 1.0; load: str = ""; scored: bool = True
@dataclass class CheckResult: ok: bool; message: str; max_rel_err: float | None = None; tol: float | None = None
class Adapter(Protocol):
    name: str; problem: str; gpu: str; budget_s: int; protocol: TimingProtocol   # judge constants per problem
    def shapes(self) -> list[Shape]                       # scored shapes with weights and loads
    def tests(self) -> list[Shape]                        # correctness-only cases (+ held-out extra shapes)
    def input_variance(self) -> dict[str, bool]           # {"routing": True} -> default seeds 5
    def judge_seed(self, shape: Shape) -> int | None      # judge's per-shape seed when known
    def load(self, path: str, tag: str) -> Any            # import submission module (prepare, kernel_fn)
    def baseline_path(self) -> str | None
    def prepare(self, module: Any) -> Any                 # untimed state (weights -> prepare(weights)), may be None
    def make_inputs(self, shape: Shape, seed: int) -> Any # reference.generate_input(**case, seed=seed)
    def call(self, module: Any, state: Any, inputs: Any) -> Any   # the timed call: kernel_fn(data[, state])
    def check(self, inputs: Any, output: Any) -> CheckResult       # reference.check_implementation
    def score(self, per_shape_us: dict[str, float], base_us: dict[str, float]) -> float  # judge formula
    def rules(self) -> ComplyRules
    def build_hook(self, path: str) -> list[ExtSpec]      # optional: compile-only extraction
    def judge_cases(self, judge_json: dict) -> list[Shape]# shapes+seeds from an `okc status --json`
```
The `okc` adapter is generic: it imports `reference.py` (+ `checks.py`) from the problem dir and reads
`kdev_problem.toml` (shipped per muse_* under `src/kdev/adapters/okc_problems/`) for the scored shapes,
weights, loads, test cases, variance, the `kernel_fn` arity and the score formula (`load_weighted_geomean`).
Shapes can also come from a judge JSON (`records[event=benchmark].case`, dropping `load`, `repeats`, `weight`,
`router_*`, as calib.py does).

### 3.7 Timing protocol (`kdev-tp/1`, `TimingProtocol` dataclass; documented in docs/PROTOCOL.md by P3)

Per shape: build inputs (fresh per seed) -> for each file: `prepare` (once per process), warm-up call, side-stream
warm-up, capture `torch.cuda.CUDAGraph` -> warm replays -> R reps; each rep replays every file once in ABBA
(even rep forward, odd rep reversed) or rotating order; **before every replay** `flush.read()` (a `.sum()` over
a >= 4x-L2 buffer into a sink, default 256 MB); time with CUDA events around `g.replay()`; subtract overhead
(default: judge constant 4.0 us; local empty-graph and 1-kernel-graph costs are measured and reported). Paired
ratio per rep pair (order-swapped pairs, like the judge), median; bootstrap 95% CI (10k resamples, seeded) of
the paired-ratio median; score via adapter formula with CI propagated by bootstrap over reps. After replay the
output checksum is compared with an eager output (materialisation guard). Verdict `faster|slower` requires CI
excluding 1 and |delta| >= max(noise_floor, 2x combined SE); faster-than-floor -> `suspect`.

### 3.8 GPU profile (`src/kdev/data/gpus/<name>.toml`, overrides in `$KDEV_HOME/gpus/`; loader `kdev.floor.gpus`)

```toml
schema = "kdev.gpu/1"
name = "h200"; marketing = "NVIDIA H200 (SXM, 141 GB)"; sm_count = 132; cc = "9.0"
l2_bytes = 52428800; flush_bytes = 268435456; smem_per_sm = 233472; regs_per_sm = 65536
max_warps_per_sm = 64; max_ctas_per_sm = 32; globaltimer_ns = [32, 256]
[peak]          # datasheet / judge floor constants
dram_tbps = 4.8; bf16_tflops = 989; fp8_tflops = 1979; fp4_tflops = 0
[measured]      # each value: { value = x, unit = "us", run = "<run_id or source>", how = "…" }
graph_fixed_us = { value = 4.0, unit = "us", source = "judge" }
one_kernel_graph_us = { value = 5.2, unit = "us", source = "okc/research/tlb" }
null_kernel_rel_us = { value = -0.5, unit = "us", source = "okc/research/tlb" }
dram_trip_1cta_us = { value = 0.33, unit = "us", source = "okc/research/tlb" }
dram_trip_132cta_us = { value = 0.65, unit = "us", source = "okc/research/tlb" }
l2_hit_us = 0.15; kernel_boundary_us = 2.75; extra_graph_node_us = 1.7; coop_launch_us = 1.4
icache_cold_us = 0.16; membar_gpu_us = 0.25; atom_acqrel_cold_us = 0.57; barrier_cluster_us = 0.38
stream_tbps = 4.11
```
(Values from DESIGN 3.3. Scalars or tables are both accepted by the loader.)

### 3.9 Toolchain lock (`toolchain.lock`, TOML, per problem; P5)

```toml
schema = "kdev.toolchain/1"
nvcc_version = "13.0.88"; cuda_home = "/root/cuda130w"; ptxas = "/root/cuda130w/bin/ptxas"
torch = "2.14.1+cu130"; triton = "3.8.0"; extra = { flashinfer = "0.7.0.post1" }
env = { NVCC_APPEND_FLAGS = "-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK", SGLANG_JIT_CACHE_DIR = "/root/sgljit130" }
python = "/root/venv/bin/python"; source = "judge probe check 44e9e47738ea4634"
```

### 3.10 Config (`kdev.toml`, searched: `$KDEV_CONFIG`, `./kdev.toml`, `$KDEV_HOME/kdev.toml`)

Keys: `home` (default `~/.kdev`, override `$KDEV_HOME`), `lock_path` (`/root/gpu.lock`), `offender_log`,
`max_hold_s` (1800), `build_workers` (8), `problems_root`, `default_problem`, `python` (worker interpreter),
`deny_commands` (["okc submit", "okc check"]), `remote.host/port/key/user/remote_home`, `sync.paths`,
`dashboard.out`, `board.enabled`. Never place `home` on a network drive (`doctor` warns on `/workspace`).

---

## 4. CLI command tree

Global flags (every command): `--json`, `--problem P`, `--agent NAME` (default `$KDEV_AGENT`), `--home DIR`,
`--quiet`. GPU commands also: `--detach`, `--direct` (no daemon, flock path), `--timeout S`, `--priority N`,
`--run-name NAME`, `--shapes FILTER`.

```
kdev version | doctor [--problem P] [--fix-hints]
kdev show RUN_ID [--section NAME] [--offset N] [--limit M]      kdev runs list [--tool T] [--limit N]
kdev build FILE... [--variant K=V]... [--defines "-DX=1 …"] [--verbose] [--jobs J] [--force]
kdev check FILE [--seeds N] [--repeat R] [--mode graph|eager|both] [--extra-shapes] [--cold] [--tests-only]
kdev time FILE [--base PATH|baseline|none] [--reps N] [--seeds S] [--flush read|write|none] [--flush-mb M]
              [--eager] [--overhead judge|local|none] [--allow-jit]
kdev ab A B [C…] [--reps N] [--seeds S] [--order abba|rotate] [--set LABEL:K=V]… (+ time's fidelity flags)
kdev sweep FILE --grid "K=a,b K2=c,d" [--base PATH] [--build-parallel J] [--top K] [--quick-check]
kdev calib FILE:JUDGE.json… [--seeds S] [--reps N]
kdev sass FILE|SO|CUBIN [--kernel RE] [--loops] [--per N] [--audit pdl,spills,serialize,repeat,expect=HGMMA]
kdev sass diff A B [--kernel RE]
kdev regs FILE|SO|CUBIN [--kernel RE] [--block N]
kdev timeline FILE [--shape S] [--marks auto|header] [--ctas all|sample:N] [--ab DEFSET_A DEFSET_B]
kdev floor [--problem P --shape S] [--bytes B --flops F --dtype D] [--plan PLAN.toml] [--gpu NAME]
kdev floor calibrate [--gpu NAME] [--write]
kdev profile FILE [--backend auto|ncu|nsys|cupti] [--metrics SET] [--nopdl]
kdev dispatch FILE [--shapes F]
kdev sanitize FILE [--tool memcheck|racecheck|synccheck|initcheck] [--shape S]
kdev comply FILE [--static-only]
kdev serve start|stop|status|run [--build-workers N] [--foreground]
kdev job submit SPEC.json | status ID | wait ID [--timeout S] | cancel ID | list [--state S] | logs ID [--tail N]
kdev queue status | offenders [--since 3h]
kdev lock run [--timeout S] -- CMD…              (deny-listed: okc submit/check)
kdev ledger query [--problem P] [--file F] [--md5 M] [--tool T] [--since D] [--sql "SELECT…"(read-only)]
kdev ledger tried FILE|TEXT [--knobs K=V…]       kdev ledger best [--problem P]
kdev ledger import RESULT.json|RUN_DIR…          kdev ledger export [--format csv|jsonl]
kdev idea add TITLE [--hypothesis H] [--kind idea|dead|limit|win] [--gain PCT] [--tags T] [--variant V]
kdev idea list [--status S] [--owner O] [--kind K] | show ID | claim ID | release ID | resolve ID --status S [--resolution R]
kdev idea evidence ID --run RUN|--judge JID|--file F|--text T [--supports for|against|neutral]
kdev note …                                      (alias: `idea add --kind …`)
kdev judge prepare FILE | record JOB.json | ledger [--problem P] | pair A [--base B] [--allow-check]
kdev okc status JOB | leaderboard [--problem P]  (read-only)
kdev dashboard build [--out DIR] [--scrape] | watch [--every S]
kdev sync pull|push|both [--since 3h] [--dry-run]   kdev remote [--host H --port P] -- kdev …
kdev remote setup                                kdev mcp [--profile compact|full]
```

---

## 5. MCP server (P8)

Built with the official `mcp` Python SDK (`from mcp.server.fastmcp import FastMCP`), stdio transport, entry
`kdev mcp` or `python -m kdev.mcp`. Tools are **generated from the registry** (same names with `_` instead of
spaces: `kdev_ab`, `kdev_sass_diff`, `kdev_idea_claim`, …), so every CLI command has an MCP twin with the same
params and the same Result envelope. Rules:
- Two profiles (REA style): `full` (one tool per command) and `compact` (default; ~12 tools: `kdev_doctor`,
  `kdev_build`, `kdev_check`, `kdev_time`, `kdev_ab`, `kdev_sass`, `kdev_timeline`, `kdev_ledger`,
  `kdev_idea`, `kdev_job` (status/wait/cancel/list), `kdev_show`, `kdev_run` (generic: `command` + `argv`)).
- GPU tools enqueue and return `{job_id, run_id}` immediately unless `wait_s` (<= 120) is given; long waits
  go through `kdev_job wait`. Progress notifications from queue position/elapsed.
- Large outputs are never inlined: tool results are the <= 40-line summary + `files`; reading goes through
  `kdev_show` with offset/limit. Resources: `kdev://runs/{run_id}/{section}`.
- **Not exposed over MCP**: `judge pair --allow-check`, `lock run`, `serve stop`, `remote`, `sync push`,
  `ledger --sql`. MCP never has a path to `okc check`/`submit`.
- Each tool's description carries its declared effects (`gpu`, `writes`, `network`, `quota: never`).

---

## 6. Output shaping rules (all tools, enforced by `kdev.output`)

1. Default stdout <= 40 lines, <= ~2 KB: line 1 = `VERDICT  tool  key numbers`; then a table of at most 15
   rows; then `run: <id>  dir: <path>`; then at most 2 `next:` hints. Wider tables go to files.
2. `--json` prints exactly one JSON object (the Result envelope) and nothing else on stdout; logs go to stderr
   and `log.txt`.
3. Every truncation is explicit: `… 412 more rows in <path> (kdev show <run> --section rows --offset 15)`.
4. `kdev show RUN --section S --offset N --limit M` (default limit 60 lines) returns a page plus
   `next_offset` (or `end`). Sections: any key of `result.files` plus `summary`, `meta`, `log`, `raw`.
5. Numbers: us with 1 decimal >= 10 us, 2 decimals < 10 us; ratios `x1.234`; CIs `[lo, hi]`.
6. Fidelity line on every timing output: `fidelity: nvcc 13.0.88=pin read-flush 256MB graph excl=ok seeds=5
   overhead=-4.00us optouts=none`. Any opt-out is printed in CAPS (`FLUSH=WRITE`, `EAGER`).
7. No ANSI colours unless stdout is a TTY and `NO_COLOR` is unset.

---

## 7. Error codes (`kdev.contract.ErrorCode`; exit codes are stable)

| code | exit | meaning / typical recovery |
|---|---|---|
| (ok) | 0 | status ok (verdict may still be `slower`) |
| `check_failed` | 1 | correctness failed (`status: fail`) |
| `invalid_request` | 2 | bad args/spec; message names the field |
| `quota_forbidden` | 3 | command would consume a judge quota / deny-listed |
| `needs_build` | 4 | timing job found no prebuilt cache hit; run `kdev build` |
| `toolchain_mismatch` | 5 | `.so` notes != pin |
| `gpu_busy_timeout` | 6 | lock or foreign contexts did not clear in time (offenders logged) |
| `capability_unavailable` | 7 | backend missing (ncu perms, nsys absent, no GPU) - names the fallback |
| `build_failed` | 8 | compiler error; first error lines + log path |
| `timeout` | 9 | job exceeded `timeout_s`; partial samples kept |
| `not_found` | 10 | run/job/idea/file id unknown |
| `conflict` | 11 | sync conflict, idea already claimed, stale version |
| `cancelled` | 12 | job cancelled |
| `not_exclusive` | 13 | timing finished but exclusivity was violated mid-run (result kept, `comparable=0`) |
| `not_implemented` | 14 | stub (Phase 0) |
| `internal` | 15 | bug; traceback in log |

Errors are returned as Results (`status: error`, `error{code,message,details,recovery}`), never as bare
tracebacks. `recovery` is one sentence.

---

## 8. Safety rules (tests must cover each)

1. **Never submit.** No code path runs `okc submit` or POSTs to a judge. `okc check` is reachable only from
   `kdev judge pair --allow-check` (CLI-only, prints the exact commands first, records both job ids, refuses
   duplicate hashes). `kdev.judge.okc_cli` hard-denies `submit`; `lock run`/`exec` jobs deny-list
   `okc submit|check` (exit 3). A unit test greps the source tree for other `okc` invocations.
2. **No compile under the lock**: GPU-lane jobs with `requires_prebuilt` and no cache hit fail `needs_build`;
   `--allow-jit` is an opt-out recorded in fidelity.
3. **Exclusivity**: daemon + flock on the shared lock file + foreign-context wait (zombies ignored) + offender
   log; max hold (default 30 min) with a warning at 80%; every timing records `exclusive`.
4. **Fidelity opt-outs** are explicit, printed in caps, recorded, and set `comparable=0` in the ledger.
5. **Reward-hack guards**: fresh seeds + extra shapes in `check`, output checksum after replay, dispatch probe,
   faster-than-floor -> `suspect`.
6. **Untrusted code**: kdev never executes files from extracted archives' own directory; third-party kernels
   go to `third_party/` with licence header (Apache-2.0 with credit; unlicensed = ideas only).
7. **Storage**: `sync` md5-verifies, never overwrites a newer file without a `.conflict-<ts>` copy; `doctor`
   warns if `KDEV_HOME` is on a network/mfs path.
8. **Read-only by default** for scraping: dashboard/board scrape only uses `okc leaderboard/status`.

---

## 9. Test strategy

- `pytest` runs on Windows/Linux **without a GPU, CUDA, torch or nvcc**. `tests/conftest.py` sets
  `KDEV_HOME` to a tmp dir, `KDEV_NO_GPU=1`, and provides fixtures: `fixtures_dir`, `tmp_home`,
  `fake_run_dir`, `sample_sass`, `sample_ptxas`, `judge_json`, `fake_smi`.
- Real fixtures (copied from OKC): `tests/fixtures/sass/vjc_k_L1_F0_M1.sass` (contains ACQBULK and an
  `LDG.E.CONSTANT` before it at 0x0080 - the PDL hoist positive case), `tests/fixtures/ptxas/vjc.ptxas` and
  `v130_attention.log` (ptxas -v output), `tests/fixtures/judge/judge_moe.json`, `judge_attention.json`
  (`okc status --json`), `cal_moe.json` (calib output). Builders add more fixtures under their own
  subdirectory (`tests/fixtures/<pkg>/`).
- Mocks: subprocess-wrapping code takes a `runner` callable (default `kdev.util.run`) so tests inject canned
  stdout; GPU timing logic is written against a `Timer` protocol (`kdev.timing.engine.Timer`) so
  scheduling/ABBA/overhead/stats are tested with a deterministic fake timer; `kdev.service.smi` takes a
  query function; lock tests use a tmp lock path and a fake `compute_apps()`.
- Markers: `@pytest.mark.gpu` (needs CUDA + torch; skipped unless `KDEV_GPU_TESTS=1`), `@pytest.mark.slow`,
  `@pytest.mark.network`. Integration tests live in `tests/gpu/` and run on the pod.
- Layout: `tests/unit/test_p<N>_<pkg>.py` per package (owned by that package). Phase-0 placeholders assert that
  the module imports and the registry exposes its commands; builders replace them.
- `tests/unit/test_layers.py` (P1): no module outside `kdev.worker`-reachable runtime functions imports torch
  at top level; layering; the `okc` call deny test (P4 owns that one: `test_p4_safety.py`).
- Acceptance on the pod (from DESIGN 10): `kdev calib` reproduces `okc/calibration/all_calib.md` per-shape
  local/judge ratios within +-1%.

---

## 10. Build packages and exact file ownership

Rules: a package owns its files exclusively; other packages only import its public API. Shared files frozen
after Phase 0: `src/kdev/models.py`, `src/kdev/contract.py`, `src/kdev/adapters/base.py`,
`src/kdev/ledger/schema.sql`, `src/kdev/service/schema.sql`, `pyproject.toml` (deps: ask the coordinator).
Any frozen-file change goes through `docs/CHANGES.md` + coordinator. Each package also owns
`tests/unit/test_p<N>_*.py` and `tests/fixtures/<pkg>/`.

### P1 - core, config, CLI dispatch, output shaping, doctor, show
Files: `src/kdev/__init__.py`, `__main__.py`, `cli.py`, `config.py`, `runs.py`, `output.py`, `registry.py`,
`util.py`, `core_commands.py` (doctor/show/runs/version), `doctor.py`; tests `test_p1_core.py`,
`test_layers.py`; docs `docs/CONTRIBUTING.md`.
Deliver: config resolution, run-dir creation, `show` paging, renderer (rules in Section 6), Context.submit
routing (calls `kdev.service.client`), doctor aggregating each package's `probe()` functions
(`service.probe`, `build.probe`, `observe.probe`, `ledger.probe`, `remote.probe`).

### P2 - GPU job service, lock, remote/sync
Files: `src/kdev/service/{__init__,schema.sql,lock.py,smi.py,queue.py,daemon.py,client.py,commands.py}.py`,
`src/kdev/remote/{__init__,ssh.py,sync.py,bootstrap.py,commands.py}`, `scripts/` (pod bootstrap ports of
okc_bootstrap.sh/sync_okc.sh, optional).
Deliver: `kdev serve` (build lane process pool with `CUDA_VISIBLE_DEVICES=""`, serial GPU lane, heartbeat,
max-hold warnings, cancellation, SIGTERM-safe: running job finishes or is marked `timeout`, queue persists),
`lock.exclusive()` context manager = gpu_run.sh semantics, offender log (db + legacy file), client
submit/wait/status, `queue status`, `lock run` with deny-list; `sync` (md5, `.conflict-<ts>`), `remote`.
After a GPU job the daemon calls `kdev.ledger.record.ingest_result(result_path)`.

### P3 - timing engine, statistics, calibration, worker
Files: `src/kdev/timing/{__init__,flush.py,graph.py,overhead.py,stats.py,schedule.py,engine.py,seeds.py,
calib.py,commands.py}`, `src/kdev/worker.py`, `docs/PROTOCOL.md`.
Deliver: protocol kdev-tp/1 exactly as Section 3.7 (port the core of ah.py/calib.py/cq_run.py), `Timer`
protocol with torch and fake implementations, bootstrap CI + score CI, verdict logic, multi-seed, calib
(port calib.py: judge JSON -> shapes, local/judge, DISAGREE), `time`/`ab`/`calib` commands (build JobSpecs),
worker dispatch for all kinds (calls other packages' `run_*` functions listed in their modules).

### P4 - ledger, idea board, judge (record-only), okc read-only CLI
Files: `src/kdev/ledger/{__init__,schema.sql,db.py,record.py,query.py,ideas.py,commands.py}`,
`src/kdev/judge/{__init__,okc_cli.py,prepare.py,record.py,pair.py,commands.py}`; tests incl.
`test_p4_safety.py` (source-tree grep for okc submit/check).
Deliver: schema migrations, `ingest_result()` (Result envelope -> rows), `tried()`, `best()`, ideas CRUD with
claim/resolve semantics (claim fails `conflict` if owned by someone else; `limit` kind requires evidence of a
sass run and a sweep run), judge prepare (static rules via adapter, md5/norm_md5 duplicate warning), record
(okc status JSON -> judge_results/judge_shapes; fixture judge_moe.json), pair (`--allow-check` gate).

### P5 - build wrappers, toolchain, SASS analyser, sweep
Files: `src/kdev/build/{__init__,toolchain.py,compile.py,ptxas.py,cache.py,commands.py}`,
`src/kdev/sass/{__init__,disasm.py,parse.py,cfg.py,classes.py,loops.py,audits.py,diff.py,commands.py}`,
`src/kdev/sweep/{__init__,grid.py,commands.py}`.
Deliver: toolchain lock load/verify, `.so` notes reader (strings scan for "Cuda compilation tools"), compile-only
build of load_inline modules with hash-keyed cache, ptxas -v parser (fixtures), SASS parse/CFG/back-edge loops,
opcode class histogram (per loop, per element with `--per`), audits: pdl (LDG*/LD before ACQBULK reachable in
program order), spills (LDL/STL, esp. in loops), serialize (C7510-C7515), expect (missing opcodes), repeat
(k-gram repeated normalised instruction windows in a hot loop); diff of two builds; sweep grid expansion +
JobSpecs (builds on build lane, then ab batches).

### P6 - timeline, floor/roofline, GPU profiles, occupancy, profile/dispatch backends
Files: `src/kdev/timeline/{__init__,marks.cuh,instrument.py,decode.py,trace.py,run.py,commands.py}`,
`src/kdev/floor/{__init__,gpus.py,roofline.py,skeleton.py,microbench.py,commands.py}`,
`src/kdev/occupancy/{__init__,calc.py,commands.py}`, `src/kdev/observe/{__init__,cupti.py,ncu.py,nsys.py,
nvbit.py,dispatch.py,commands.py}`, `src/kdev/data/gpus/*.toml` (h200.toml seeded in Phase 0).
Deliver: `KD_MARK(id)` device header (globaltimer + clock64, per CTA/warp buffer, zero cost without
`KDEV_MARKS`), source annotation `// @kdev:mark NAME` -> instrumented source + define injection, decoder ->
per-phase p10/p50/p90, critical CTA, resolution detection, Perfetto JSON, chart data (`timeline.csv`);
roofline + skeleton; microbench suite (port fb.py/tlb.cu ideas) writing `[measured]`; occupancy calculator;
CUPTI kernel list via torch.profiler, ncu/nsys wrappers with capability probes, dispatch probe.

### P7 - problem adapters, check, sanitize, comply
Files: `src/kdev/adapters/{__init__,registry.py,okc.py,cu_main.py,kernelbench.py}`,
`src/kdev/adapters/okc_problems/*.toml` (one per muse_* problem), `src/kdev/checks/{__init__,check.py,
sanitize.py,comply.py,commands.py}`, `docs/ADAPTERS.md`. (`adapters/base.py` is frozen; P7 is its steward.)
Deliver: generic okc adapter (reference.py/checks.py import, prepare arity, scored shapes/weights/loads from
TASK.md tables copied into `kdev_problem.toml` + judge JSON), `check` (seeds, extra shapes, eager/graph,
repeat, nondeterministic/intermittent/wrong classification, max ulp, `--cold`), sanitize wrapper, comply
checklist.

### P8 - dashboard, MCP server, skill
Files: `src/kdev/dashboard/{__init__,collect.py,render.py,board.py,commands.py}`,
`src/kdev/dashboard/templates/index.html`, `src/kdev/mcp/{__init__,__main__,server.py,schema.py,commands.py}`,
`skills/kdev/SKILL.md`, `docs/MCP.md`.
Deliver: static HTML (no external network at view time; inline CSS/JS, light/dark) with per-problem best,
recent runs, queue, offenders, ideas, calibration table, board scrape (read-only, optional); MCP server
generated from the registry (Section 5); SKILL.md workflow + brief checklist mapped to kdev commands.

### P9 - collaboration: agent message board + documentation store (added 2026-10-31)
Files: `src/kdev/collab/{__init__,db.py,board.py,docs.py,refs.py,backup.py,commands.py}`, `docs/COLLAB.md`,
`scripts/board_wrapper.sh`, `tests/unit/test_p9_collab.py`.
Deliver: `kdev board` (post/read/tail/inbox/ack/search/thread/claims/release/import; claim-overlap conflicts),
`kdev doc` (append/show/toc/list/search/checklist/import/export; append-only, round-trip markdown), `kdev collab
backup`; `$KDEV_HOME/collab.db` (own numbered migrations, WAL, BEGIN IMMEDIATE + retries); MCP compact tools
`kdev_board`, `kdev_doc`.

---

## 11. Integration order and handoffs

- All packages start in parallel against the frozen models. Stubs return `not_implemented` Results, so the
  CLI is usable end-to-end from day 0.
- Critical path for the "trustworthy timer": P1 (Context.submit, output) -> P2 (lock + direct mode) -> P3
  (engine) -> P7 (okc adapter) -> P4 (ingest). Each of those packages should land a minimal vertical slice
  first (direct mode + one adapter) before polishing.
- Cross-package calls (the only ones allowed):
  - P1 -> `service.client.submit/wait`, every package's `probe()`.
  - P2 -> `worker` (subprocess), `ledger.record.ingest_result`, `build.cache.is_prebuilt`.
  - P3 worker -> `adapters.registry.get_adapter`, `checks.check.run_check`, `checks.comply.run_comply`,
    `checks.sanitize.run_sanitize_inner`, `timeline.run.run_timeline`, `observe.dispatch.run_dispatch`,
    `observe.cupti.run_profile`, `floor.microbench.run_calibrate`, `build.toolchain.verify_so`.
  - P4 -> `adapters.registry.get_adapter` (judge prepare rules).
  - P5 sweep -> `service.client.submit`, `timing.engine` result parsing via models.
  - P6 -> `floor.gpus.load_profile` (also used by P3 for flush size), `sass.parse` (regs via ptxas info).
  - P8 -> `registry`, `ledger.query`, `service.queue.snapshot`, `judge.okc_cli` (read-only).
- Definition of done per package: functions implemented, unit tests pass with no GPU, `kdev <cmd> --help`
  accurate, `--json` envelope valid (`contract.validate_result`), docstrings updated, GPU tests (if any)
  marked and run on the pod once.
