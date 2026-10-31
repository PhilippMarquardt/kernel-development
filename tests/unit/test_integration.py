"""Cross-package wiring (integration pass): P3<->P6 taint guard + isolated dispatch, P5 public hook API used by
P6, P2 needs_build gate on the real P5 is_prebuilt, P1 result.json carries Result.lines + summary.txt."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from kdev.contract import ErrorCode, KdevError, Result
from kdev.models import FileRef, JobSpec, TimingParams
from kdev.observe import guard


@pytest.fixture(autouse=True)
def _clean_guard():
    guard._reset_for_tests()
    yield
    guard._reset_for_tests()


def test_timing_refuses_after_profiler():
    from kdev.timing.engine import taint_optouts

    assert taint_optouts("time", TimingParams()) == []
    guard.mark_profiled("test")
    with pytest.raises(KdevError) as e:
        taint_optouts("time", TimingParams())
    assert e.value.code == ErrorCode.INVALID_REQUEST
    assert taint_optouts("time", TimingParams(extra={"allow_profiled": True})) == [guard.OPTOUT]


def test_time_dispatch_runs_in_child(monkeypatch, tmp_path):
    from kdev.observe import dispatch
    from kdev.timing import engine

    seen = {}

    def fake_probe(spec, out_dir, **kw):
        seen["kind"], seen["files"], seen["out"] = spec.kind, [f.path for f in spec.files], out_dir
        return {"s1": [{"kernel": "k", "count": 1, "grid": [1, 1, 1]}]}, ["w"]

    monkeypatch.setattr(dispatch, "probe_isolated", fake_probe)
    spec = JobSpec(kind="time", files=[FileRef(path="/x/a.py", label="A")], params={})
    disp, warns = engine._isolated_dispatch(spec, tmp_path, "time")
    assert disp and warns == ["w"] and seen["kind"] == "dispatch" and seen["files"] == ["/x/a.py"]
    assert spec.kind == "time"                                    # the job spec itself is untouched
    assert engine._isolated_dispatch(spec, tmp_path, "ab") == (None, [])      # ab: off by default
    spec.params = {"extra": {"dispatch": False}}
    assert engine._isolated_dispatch(spec, tmp_path, "time") == (None, [])


def _fake_torch(monkeypatch):
    ce = types.SimpleNamespace(load_inline=lambda *a, **k: "real_inline", load=lambda *a, **k: "real_load")
    utils = types.SimpleNamespace(cpp_extension=ce)
    torch = types.ModuleType("torch")
    torch.utils = utils  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.utils", utils)
    monkeypatch.setitem(sys.modules, "torch.utils.cpp_extension", ce)
    return ce


def test_shim_install_keeps_bound_functions(monkeypatch):
    from kdev.build import _shim, cache

    monkeypatch.setattr(_shim, "_ACTIVE", [])
    ce = _fake_torch(monkeypatch)
    assert cache.active_hook() is None and cache.select_variant(FileRef(path="a")) is None
    h1 = _shim.install(_shim.Hook(mode="use", ext_root="/r1", tc_version="13.0", tc_ptxas="p"))
    bound = ce.load_inline                       # what `from torch.utils.cpp_extension import load_inline` holds
    h2 = _shim.install(_shim.Hook(mode="use", ext_root="/r2", tc_version="13.0", tc_ptxas="p", knobs={"T": 1}))
    assert h2 is h1 and ce.load_inline is bound and h1.ext_root == "/r2" and h1.knobs == {"T": 1}
    h = cache.select_variant(FileRef(path="b", knobs={"T": 2}, defines=["-DX=1"]))
    assert h is h1 and h.knobs == {"T": 2} and h.defines == ["-DX=1"] and cache.active_hook() is h1


def test_gate_uses_real_is_prebuilt(tmp_path):
    from kdev.service import _jobrun

    f = tmp_path / "sub.py"
    f.write_text("x = 1\n", encoding="utf-8")
    err = _jobrun._prebuilt_gate(JobSpec(kind="time", files=[FileRef(path=str(f))], requires_prebuilt=True))
    assert err is not None and err.code == ErrorCode.NEEDS_BUILD


def test_result_json_keeps_lines_and_summary(tmp_path):
    from kdev.runs import write_result_files

    r = Result(tool="time", status="ok", verdict="faster", lines=["tool line 1", "tool line 2"])
    p = write_result_files(tmp_path, r)
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["lines"] == ["tool line 1", "tool line 2"] and "lines" not in json.loads(r.to_json())
    assert Result.from_dict(d).lines == r.lines
    assert "tool line 1" in (tmp_path / "summary.txt").read_text(encoding="utf-8")


def test_base_name_anonymous_namespace():
    from kdev.observe.dispatch import base_name

    rt = "void (anonymous namespace)::attn_kernel<1, false, 13>((anonymous namespace)::Params)"
    st = "_ZN39_GLOBAL__N__f3bab234_7_cuda_cu_0b177e6611attn_kernelILi8ELb1ELi0EEEvNS_6ParamsE"
    assert base_name(rt) == base_name(st) == "attn_kernel"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX flock queueing")
def test_lock_waiter_queues_and_abandons(tmp_path):
    import fcntl
    import os
    import time

    from kdev.service.lock import _FileLock

    p = str(tmp_path / "gpu.lock")
    fd = os.open(p, os.O_WRONLY | os.O_CREAT)
    fcntl.flock(fd, fcntl.LOCK_EX)
    lk = _FileLock(p)
    t0 = time.time()

    def stop():
        if time.time() - t0 > 0.5:
            raise TimeoutError

    with pytest.raises(TimeoutError):
        lk.acquire_waiting(stop, 0.05)
    lk.release()
    fcntl.flock(fd, fcntl.LOCK_UN)
    time.sleep(0.2)                       # the abandoned waiter took and dropped the lock
    lk2 = _FileLock(p)
    lk2.acquire_waiting(lambda: None, 0.05)
    assert lk2.held
    lk2.release()
    os.close(fd)


def test_hook_legacy_fallback_never_compiles(monkeypatch, tmp_path):
    from kdev.build import _shim

    monkeypatch.setenv("KDEV_LEGACY_EXT", "1")
    monkeypatch.setattr(_shim, "_ACTIVE", [])
    ce = _fake_torch(monkeypatch)
    calls = []

    def real_inline(name, *a, **k):          # mimics torch: runs ninja via _run_ninja_build, then imports
        ce._run_ninja_build(str(tmp_path / name), False, "err")
        calls.append(name)
        return f"module:{name}"

    ce.load_inline = real_inline
    ce._run_ninja_build = lambda d, v, e: None
    hook = _shim.install(_shim.Hook(mode="use", ext_root=str(tmp_path / "cache"), tc_version="13.0", tc_ptxas="p"))
    ninja = {"out": "ninja: no work to do."}
    monkeypatch.setattr("subprocess.run", lambda *a, **k: types.SimpleNamespace(stdout=ninja["out"]))
    assert ce.load_inline("ext_a", "cpp", cuda_sources="x") == "module:ext_a"
    assert hook.records[-1]["status"] == "legacy"
    ninja["out"] = "[1/3] nvcc ..."                    # would compile -> needs_build, real build never runs
    with pytest.raises(_shim.NeedsBuild):
        ce.load_inline("ext_b", "cpp", cuda_sources="y")
    assert calls == ["ext_a"] and hook.records[-1]["status"] == "needs_build"


def test_pending_prepare_time_extension_roundtrip(monkeypatch, tmp_path):
    from kdev.build import _shim

    monkeypatch.setattr(_shim, "_ACTIVE", [])
    monkeypatch.delenv("KDEV_LEGACY_EXT", raising=False)
    _fake_torch(monkeypatch)
    root = str(tmp_path / "cache")
    sub = str(tmp_path / "sub.py")
    use = _shim.Hook(mode="use", ext_root=root, tc_version="13.0", tc_ptxas="p")
    use.owner = sub
    with pytest.raises(_shim.NeedsBuild):
        use._call("load_inline", ("prep_ext", "cpp"), {"cuda_sources": "k", "functions": ["f"]})
    pend = list((tmp_path / "cache" / "pending").glob("*.json"))
    assert len(pend) == 1 and json.loads(pend[0].read_text())["kw"]["name"] == "prep_ext"

    built = []
    b = _shim.Hook(mode="build", ext_root=root, tc_version="13.0", tc_ptxas="p")
    monkeypatch.setattr(b, "_build", lambda kind, kw, rec, d, marker, stub: built.append((kw["name"], d)))
    assert b.build_pending(str(tmp_path / "other.py")) == [] and not built      # other owner: untouched
    keys = b.build_pending(sub)
    assert built[0][0] == "prep_ext" and len(keys) == 1 and not pend[0].exists()


def test_build_discover_loop_rebuilds_until_complete(tmp_path):
    from kdev.build.commands import _discover_loop

    f = tmp_path / "sub.py"
    f.write_text("x = 1\n", encoding="utf-8")
    answers = iter([Result(tool="discover", verdict="pending", summary={"pending": 3, "pending_names": ["a"]}),
                    Result(tool="discover", verdict="pending", summary={"pending": 1}),
                    Result(tool="discover", verdict="complete", summary={"pending": 0})])
    subs, rebuilt = [], []
    ctx = types.SimpleNamespace(problem=None, agent="t", submit=lambda spec: (subs.append(spec), next(answers))[1])
    rep = types.SimpleNamespace(ok=True)
    out, rounds, warns = _discover_loop(str(f), rep, lambda p: (rebuilt.append(p), rep)[1], {}, [], ctx)
    assert [r["pending"] for r in rounds] == [3, 1, 0] and len(rebuilt) == 2 and not warns
    assert all(s.kind == "discover" and not s.requires_prebuilt and s.lane == "gpu" for s in subs)

    def boom(spec):
        raise KdevError(ErrorCode.CAPABILITY_UNAVAILABLE, "no gpu")

    ctx.submit = boom
    _, rounds, warns = _discover_loop(str(f), rep, lambda p: rep, {}, [], ctx)
    assert rounds == [] and "discovery skipped" in warns[0]


def test_hook_discover_mode_records_and_stubs(monkeypatch, tmp_path):
    from kdev.build import _shim

    monkeypatch.setattr(_shim, "_ACTIVE", [])
    monkeypatch.delenv("KDEV_LEGACY_EXT", raising=False)
    _fake_torch(monkeypatch)
    h = _shim.Hook(mode="use", ext_root=str(tmp_path / "c"), tc_version="13.0", tc_ptxas="p")
    h.discover = True
    m = h._call("load_inline", ("prep_ext", "cpp"), {"cuda_sources": "k"})
    assert m.anything(1, 2) is None and h.records[-1]["status"] == "pending"
    assert len(list((tmp_path / "c" / "pending").glob("*.json"))) == 1


def test_pending_records_of_edited_file_are_dropped(monkeypatch, tmp_path):
    """Board #78: a pending call recorded by a GPU run for an older content of the file (compile error fixed,
    new ext hash) must not be rebuilt (build_failed for old names) nor count as pending (ab: needs_build)."""
    from kdev.build import _shim, cache
    from kdev.models import Toolchain

    monkeypatch.setattr(_shim, "_ACTIVE", [])
    monkeypatch.delenv("KDEV_LEGACY_EXT", raising=False)
    _fake_torch(monkeypatch)
    tc = Toolchain(nvcc_version="13.0", ptxas="p")
    home = str(tmp_path / "home")
    root = cache.ext_root(home, tc)
    sub = tmp_path / "a_noep.py"
    sub.write_text("SRC = 'broken'\n", encoding="utf-8")
    use = _shim.Hook(mode="use", ext_root=root, tc_version=cache.tc_version(tc), tc_ptxas=cache.tc_ident(tc))
    use.owner = str(sub)
    use.discover = True
    use._call("load_inline", ("proj_sk2o_old", "cpp"), {"cuda_sources": "broken"})
    assert cache.pending_for(str(sub), tc, home) == ["proj_sk2o_old"]

    sub.write_text("SRC = 'fixed'\n", encoding="utf-8")          # source changed: old record superseded
    use.records.clear()
    use._call("load_inline", ("proj_sk2o_new", "cpp"), {"cuda_sources": "fixed"})
    assert cache.pending_for(str(sub), tc, home) == ["proj_sk2o_new"]
    assert len(list((Path(root) / "pending").glob("*.json"))) == 1        # stale record deleted

    built = []
    b = _shim.Hook(mode="build", ext_root=root, tc_version=cache.tc_version(tc), tc_ptxas=cache.tc_ident(tc))
    monkeypatch.setattr(b, "_build", lambda kind, kw, rec, d, marker, stub: built.append(kw["name"]))
    assert len(b.build_pending(str(sub))) == 1 and built == ["proj_sk2o_new"]

    # legacy record (no file_md5) older than the file: stale too; --clean-pending drops current ones as well
    pdir = Path(root) / "pending"
    (pdir / "old.json").write_text(json.dumps({"kind": "load_inline", "kw": {"name": "legacy"}, "key": "old",
                                               "tc_version": cache.tc_version(tc), "tc_ptxas": cache.tc_ident(tc),
                                               "file": str(sub), "ts": 1.0}), encoding="utf-8")
    use._call("load_inline", ("cur", "cpp"), {"cuda_sources": "cur"})
    assert cache.pending_for(str(sub), tc, home) == ["cur"] and not (pdir / "old.json").exists()
    assert len(cache.clean_pending(str(sub), tc, home)) == 1 and cache.pending_for(str(sub), tc, home) == []
