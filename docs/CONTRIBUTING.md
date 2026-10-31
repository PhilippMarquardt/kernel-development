# Contributing to kdev

Read `docs/PLAN.md` first (it is binding), then `docs/DESIGN.md` for the reasons.

## Setup

```
pip install -e ".[dev]"        # add ,mcp for the MCP server
pytest                         # unit tests: no GPU, CUDA, torch or nvcc needed
KDEV_GPU_TESTS=1 pytest tests/gpu   # on the GPU pod only
ruff check src tests
```

## Ownership

- Each build package (P1-P8, PLAN section 10) owns its files exclusively. Import other packages only through
  the public functions listed in PLAN 11; never edit another package's files.
- Frozen files (`models.py`, `contract.py`, `adapters/base.py`, `ledger/schema.sql`, `service/schema.sql`,
  `pyproject.toml` dependencies): change only with an entry in `docs/CHANGES.md` (append-only: date,
  package, change, who must adapt) and a heads-up to the coordinator.
- Commands and flags live in your package's `commands.py` (`register(reg)`); do not edit `cli.py` for them.
- Your tests: `tests/unit/test_p<N>_*.py`, fixtures under `tests/fixtures/<pkg>/`, GPU tests in
  `tests/gpu/test_<pkg>.py`.

## Conventions

- Python >= 3.10, `from __future__ import annotations`, full type hints on public functions, dataclasses for
  structured data, docstrings stating units (`_us`, `_ns`, `_bytes`, `_utc`).
- Core is stdlib-only. `torch` is imported **only inside functions that run in the worker**, never at module
  top level (`tests/unit/test_layers.py` enforces this; it also blocks `mcp` and `numpy`). Optional deps
  (numpy, mcp) must have a fallback or a `capability_unavailable` error.
- Every command returns a `kdev.contract.Result`; failures raise `KdevError(code, message, recovery=...)`
  with a code from PLAN section 7. No bare tracebacks on stdout. `--json` prints exactly one JSON object.
- Output shaping (PLAN 6): <= 40 summary lines, explicit truncation with the path to the full data, large
  artifacts in the run dir, readable through `kdev show`.
- Subprocess calls go through a `runner: Runner = kdev.util.run` parameter (no `shell=True`), so tests inject
  canned output. Never call `okc` except via `kdev.judge.okc_cli` (safety rule 8.1).
- Unit tests must not need a GPU: use fixtures, `FakeTimer`, fake runners and fake `compute_apps`. Mark real
  GPU tests `@pytest.mark.gpu`.
- Timing defaults are the judge protocol (read flush, graph replay, 4.0 us overhead, ABBA, adapter seeds).
  Any deviation is an opt-out recorded in `Fidelity.optouts` and printed in CAPS.
- Files: LF line endings, UTF-8, line length 120. No GPU work, compiles or network in import side effects.

## Definition of done (per package)

Functions implemented; unit tests pass without a GPU; `kdev <cmd> --help` accurate; `--json` passes
`contract.validate_result`; GPU tests (if any) marked and run once on the pod; docs updated.
Do not commit; the coordinator commits.
