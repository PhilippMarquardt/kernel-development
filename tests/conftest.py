"""Shared test fixtures. Unit tests never need a GPU, CUDA, torch or nvcc (docs/PLAN.md section 9)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def pytest_collection_modifyitems(config, items):
    if os.environ.get("KDEV_GPU_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="GPU test: set KDEV_GPU_TESTS=1 on a GPU host")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path, monkeypatch):
    """Every test gets its own KDEV_HOME and KDEV_NO_GPU=1 (unless running GPU tests)."""
    home = tmp_path / "kdev_home"
    monkeypatch.setenv("KDEV_HOME", str(home))
    monkeypatch.setenv("KDEV_AGENT", "pytest")
    monkeypatch.setenv("KDEV_LOCK_PATH", str(tmp_path / "gpu.lock"))
    if os.environ.get("KDEV_GPU_TESTS") != "1":
        monkeypatch.setenv("KDEV_NO_GPU", "1")
    yield


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def tmp_home(tmp_path) -> Path:
    return tmp_path / "kdev_home"


@pytest.fixture
def sample_sass() -> str:
    """cuobjdump -sass of a PDL attention kernel (ACQBULK present; LDG.E.CONSTANT at 0x0080 before it)."""
    return (FIXTURES / "sass" / "vjc_k_L1_F0_M1.sass").read_text(encoding="utf-8", errors="replace")


@pytest.fixture
def sample_ptxas() -> str:
    return (FIXTURES / "ptxas" / "vjc.ptxas").read_text(encoding="utf-8", errors="replace")


@pytest.fixture
def judge_json() -> dict:
    """`okc status --json` of the MoE best (job 2bbe8248)."""
    return json.loads((FIXTURES / "judge" / "judge_moe.json").read_text(encoding="utf-8"))


@pytest.fixture
def fake_runner():
    """Factory for a canned subprocess runner: fake_runner({('nvcc', '--version'): (0, 'out', '')})."""
    from kdev.util import Proc

    def make(table: dict[tuple[str, ...], tuple[int, str, str]]):
        calls: list[list[str]] = []

        def runner(argv, **kw):
            calls.append(list(argv))
            for key, (rc, out, err) in table.items():
                if tuple(argv[: len(key)]) == key or (key and os.path.basename(argv[0]) == key[0]
                                                      and tuple(argv[1:len(key)]) == key[1:]):
                    return Proc(list(argv), rc, out, err)
            return Proc(list(argv), 127, "", f"not found: {argv[0]}")

        runner.calls = calls  # type: ignore[attr-defined]
        return runner

    return make
