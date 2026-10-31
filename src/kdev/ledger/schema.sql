-- kdev experiment ledger ($KDEV_HOME/ledger.db), SQLite WAL. FROZEN after Phase 0 (docs/PLAN.md 3.4).
-- Owner/steward: P4. Changes: docs/CHANGES.md + migration in kdev.ledger.db.
PRAGMA journal_mode = WAL;
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
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
INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', '1');
