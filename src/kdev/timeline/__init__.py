"""Per-CTA phase timelines from %globaltimer marks (M14). Owner: P6.

Flow: source annotations (``// @kdev:mark NAME`` lines or explicit ``KD_MARK(id)``) -> instrument.py writes
an instrumented copy with ``#include "kdev/marks.cuh"`` and -DKDEV_MARKS -> run.py (worker, GPU) allocates
the mark buffer, sets it via the module's ``kd_marks_init`` host hook, replays one shape -> decode.py ->
per-phase table + trace.py (Perfetto JSON + timeline.csv chart data).
"""
