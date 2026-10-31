-- kdev service queue ($KDEV_HOME/queue.db), SQLite WAL. FROZEN after Phase 0 (docs/PLAN.md 3.5).
PRAGMA journal_mode = WAL;
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS jobs (
  job_id TEXT PRIMARY KEY, lane TEXT NOT NULL, kind TEXT NOT NULL,
  state TEXT NOT NULL,                 -- queued|running|done|failed|cancelled|timeout
  priority INTEGER NOT NULL DEFAULT 0, agent TEXT, spec_json TEXT NOT NULL, run_id TEXT, run_dir TEXT,
  submitted_utc TEXT NOT NULL, started_utc TEXT, finished_utc TEXT, pid INTEGER, exit_code INTEGER,
  error_code TEXT, waited_s REAL);
CREATE INDEX IF NOT EXISTS ix_jobs_state ON jobs(lane, state, priority DESC, submitted_utc);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts_utc TEXT NOT NULL, job_id TEXT, kind TEXT NOT NULL, message TEXT);
CREATE TABLE IF NOT EXISTS offenders (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts_utc TEXT NOT NULL, pid INTEGER NOT NULL, cmdline TEXT,
  gpu_mem_mib INTEGER, during_job TEXT, action TEXT);   -- waited|aborted
CREATE TABLE IF NOT EXISTS heartbeat (
  id INTEGER PRIMARY KEY CHECK (id = 1), pid INTEGER, host TEXT, ts_utc TEXT NOT NULL, version TEXT,
  gpu_lane_job TEXT, build_workers INTEGER);
INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', '1');
