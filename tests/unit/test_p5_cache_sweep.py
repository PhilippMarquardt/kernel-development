"""P5: hash-keyed cache, compile-only builds (fake nvcc / fake torch), load_inline shim, sweep orchestration."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from kdev.build import _shim, cache, compile as comp
from kdev.contract import ErrorCode, KdevError, Result, validate_result
from kdev.models import FileRef, JobSpec, Toolchain
from kdev.sweep import grid
from kdev.util import Proc

FX = Path(__file__).resolve().parents[1] / "fixtures"
TC = Toolchain(nvcc_version="13.0.88", ptxas="/root/cuda130w/bin/ptxas", nvcc="/root/cuda130w/bin/nvcc",
               cuda_home="/root/cuda130w", source="test")
NOTES = b"\0Cuda compilation tools, release 13.0, V13.0.88\0Build cuda_13.0.r13.0/compiler.1_0\0"


# ----------------------------------------------------------------------------------------------- cache keys
def test_cache_key_depends_on_toolchain_flags_knobs_source(tmp_path):
    k = cache.cache_key(b"src", TC, ["-O3"], {"T": 1})
    assert len(k) == 16 and k == cache.cache_key(b"src", TC, ["-O3"], {"T": 1})
    other_tc = Toolchain(nvcc_version="12.8.93", ptxas=TC.ptxas)
    assert len({k, cache.cache_key(b"src", other_tc, ["-O3"], {"T": 1}),           # M3: toolchain in the key
                cache.cache_key(b"src", TC, ["-O2"], {"T": 1}), cache.cache_key(b"src", TC, ["-O3"], {"T": 2}),
                cache.cache_key(b"src2", TC, ["-O3"], {"T": 1}),
                cache.cache_key(b"src", Toolchain(nvcc_version="13.0.88", ptxas="/pip/ptxas"), ["-O3"], {"T": 1})}) == 6
    d = cache.ext_dir(str(tmp_path), TC, k)
    assert Path(d) == tmp_path / "cache" / "ext" / "13.0.88" / k


def _fake_nvcc(log: str = "ptxas info    : Compiling entry function 'k1' for 'sm_90a'\n"
               "ptxas info    : Used 40 registers, used 1 barriers, 1024 bytes smem\n", rc: int = 0):
    calls = []

    def runner(argv, **kw):
        calls.append((list(argv), kw.get("env", {})))
        if "nvcc" in os.path.basename(argv[0]) and rc == 0:
            out = argv[argv.index("-o") + 1]
            Path(out).write_bytes(b"\x7fELF" + NOTES)
        return Proc(list(argv), rc, "", log if rc == 0 else "k.cu(3): error: identifier \"x\" is undefined\n")

    runner.calls = calls
    return runner


def test_build_cu_with_fake_nvcc_cache_hit_and_prebuilt(tmp_path, monkeypatch):
    monkeypatch.setattr(comp, "_nvcc_usable", lambda pin: True)
    src = tmp_path / "k.cu"
    src.write_text("__global__ void k1(float* x) { x[0] = TILE; }\n")
    home = str(tmp_path / "home")
    r = _fake_nvcc()
    rep = comp.build_file(str(src), knobs={"TILE": 128}, home=home, runner=r, toolchain=TC, backend="nvcc")
    assert not rep.errors and rep.toolchain_match and len(rep.exts) == 1
    e = rep.exts[0]
    assert not e.cache_hit and e.kernels[0].regs == 40 and e.kernels[0].smem_static == 1024
    argv, env = r.calls[0]
    assert "-DTILE=128" in argv and "-Xptxas=-v" in argv and env["CUDA_VISIBLE_DEVICES"] == ""
    rep2 = comp.build_file(str(src), knobs={"TILE": 128}, home=home, runner=r, toolchain=TC, backend="nvcc")
    assert rep2.exts[0].cache_hit and rep2.exts[0].kernels[0].regs == 40
    rep3 = comp.build_file(str(src), knobs={"TILE": 64}, home=home, runner=r, toolchain=TC, backend="nvcc")
    assert not rep3.exts[0].cache_hit and rep3.exts[0].cache_key != e.cache_key

    monkeypatch.setenv("KDEV_HOME", home)
    monkeypatch.setattr("kdev.build.toolchain.pin_for", lambda *a, **k: TC)
    spec = JobSpec(kind="time", files=[FileRef(path=str(src), knobs={"TILE": 128})])
    assert cache.is_prebuilt(spec)
    assert not cache.is_prebuilt(JobSpec(kind="time", files=[FileRef(path=str(src), knobs={"TILE": 32})]))
    src.write_text("__global__ void k1(float* x) { x[0] = 2 * TILE; }\n")      # edited after the build
    assert not cache.is_prebuilt(spec)


def test_build_cu_failure_and_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(comp, "_nvcc_usable", lambda pin: True)
    src = tmp_path / "k.cu"
    src.write_text("__global__ void k1() {}\n")
    rep = comp.build_file(str(src), home=str(tmp_path), runner=_fake_nvcc(rc=1), toolchain=TC)
    assert rep.errors and rep.errors[0].startswith("build_failed") and "undefined" in rep.errors[0]
    res = comp.build_result([rep])
    assert res.error.code == "build_failed" and res.exit_code == 8 and validate_result(res.to_dict()) == []
    pin134 = Toolchain(nvcc_version="13.4.92", ptxas=TC.ptxas, nvcc=TC.nvcc, cuda_home=TC.cuda_home)
    rep2 = comp.build_file(str(src), home=str(tmp_path), runner=_fake_nvcc(), toolchain=pin134)
    assert not rep2.toolchain_match and any(e.startswith("toolchain_mismatch") for e in rep2.errors)
    assert comp.build_result([rep2]).error.code == "toolchain_mismatch"


def test_short_name():
    assert comp.short_name("_ZN38_GLOBAL__N__5234c7d1_6_vjc_cu_0b177e6611attn_kernelILi8ELb1ELi0EEEvNS_6ParamsE") == \
        "attn_kernel<8,1,0>"
    assert comp.short_name("pdl_dot") == "pdl_dot"


# ----------------------------------------------------------------------------------------------- shim
def _fake_real_load_inline(log="ptxas info    : Used 12 registers\n"):
    calls = []

    def real(**kw):
        calls.append(kw)
        d = kw["build_directory"]
        Path(d, kw["name"] + ".so").write_bytes(b"\x7fELF" + NOTES)
        os.write(1, log.encode())        # the shim redirects fd 1 into build.log
        return object()

    real.calls = calls
    return real


def test_shim_hook_build_then_hit_then_use(tmp_path):
    h = _shim.Hook(mode="build", ext_root=str(tmp_path), tc_version="13.0.88", tc_ptxas="p", knobs={"T": 4})
    real = _fake_real_load_inline()
    h.real = {"load_inline": real}
    f = h.make("load_inline")
    m = f("ext_a", "cpp", cuda_sources="cuda", extra_cuda_cflags=["-O3"])
    assert isinstance(m, _shim._Stub) and len(real.calls) == 1   # noqa: SLF001
    kw = real.calls[0]
    assert "-DT=4" in kw["extra_cuda_cflags"] and "-Xptxas=-v" in kw["extra_cuda_cflags"]
    rec = h.records[0]
    assert rec["status"] == "ok" and Path(rec["dir"], _shim.MARKER).is_file()
    assert "Used 12 registers" in Path(rec["log_path"]).read_text()
    f("ext_a", "cpp", cuda_sources="cuda", extra_cuda_cflags=["-O3"])
    assert h.records[1]["cache_hit"] and len(real.calls) == 1
    use = _shim.Hook(mode="use", ext_root=str(tmp_path), tc_version="13.0.88", tc_ptxas="p", knobs={"T": 8})
    use.real = {"load_inline": real}
    with pytest.raises(_shim.NeedsBuild):          # other knob value: not prebuilt -> needs_build, no compile
        use.make("load_inline")("ext_a", "cpp", cuda_sources="cuda", extra_cuda_cflags=["-O3"])
    assert len(real.calls) == 1


_FAKE_TORCH = {
    "torch/__init__.py": "from . import cuda, utils\n__version__ = '0.0-fake'\n",
    "torch/cuda/__init__.py": "def get_device_capability(*a, **k):\n    raise RuntimeError('no driver')\n",
    "torch/utils/__init__.py": "from . import cpp_extension\n",
    "torch/utils/cpp_extension.py": (
        "import os\n"
        "NOTES = b'\\0Cuda compilation tools, release 13.0, V13.0.88\\0'\n"
        "def load_inline(name, cpp_sources, cuda_sources=None, functions=None, extra_cflags=None,\n"
        "                extra_cuda_cflags=None, extra_ldflags=None, extra_include_paths=None,\n"
        "                build_directory=None, verbose=False, with_cuda=None, **kw):\n"
        "    if 'BROKEN' in (cuda_sources or ''):\n"
        "        os.write(2, b'cuda.cu(1): error: BROKEN is undefined\\n')\n"
        "        raise RuntimeError('Error building extension ' + name)\n"
        "    open(os.path.join(build_directory, name + '.so'), 'wb').write(b'ELF' + NOTES)\n"
        "    os.write(1, (\"ptxas info    : Compiling entry function '_Z2k1v' for 'sm_90a'\\n\"\n"
        "                 \"ptxas info    : Function properties for _Z2k1v\\n\"\n"
        "                 \"    0 bytes stack frame, 8 bytes spill stores, 8 bytes spill loads\\n\"\n"
        "                 \"ptxas info    : Used 255 registers, used 1 barriers\\n\").encode())\n"
        "    return None\n"
        "def load(*a, **k):\n    raise RuntimeError('unused')\n"),
}

_SUB = """
from torch.utils.cpp_extension import load_inline
import torch
CAP = torch.cuda.get_device_capability()     # patched by the shim: no driver access
CUDA_SRC = "__global__ void k1() {}" + "{extra}"
_MOD = None
def _module():
    global _MOD
    if _MOD is None:
        _MOD = load_inline(name="okc_test_ext", cpp_sources="", cuda_sources=CUDA_SRC, functions=["run"],
                           extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"])
        _MOD.run()          # kernel launch: the stub swallows it
    return _MOD
def kernel_fn(data):
    return _module().run(data)
"""


@pytest.fixture
def fake_torch(tmp_path, monkeypatch):
    root = tmp_path / "faketorch"
    for rel, txt in _FAKE_TORCH.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(txt)
    monkeypatch.setenv("PYTHONPATH", str(root))
    return root


def test_build_py_through_shim(tmp_path, fake_torch):
    sub = tmp_path / "prob" / "sub.py"
    sub.parent.mkdir()
    sub.write_text(_SUB.replace("{extra}", ""))
    home = str(tmp_path / "home")
    rep = comp.build_file(str(sub), knobs={"BLOCK": 64}, home=home, toolchain=TC, python=sys.executable,
                          timeout_s=120)
    assert [e for e in rep.errors if not e.startswith("warning")] == []
    (e,) = rep.exts
    assert e.name == "okc_test_ext" and not e.cache_hit and e.so_path.endswith("okc_test_ext.so")
    assert e.so_notes == ["Cuda compilation tools, release 13.0, V13.0.88"] and rep.toolchain_match
    (k,) = e.kernels
    assert (k.name, k.regs, k.spill_store) == ("_Z2k1v", 255, 8)
    rep2 = comp.build_file(str(sub), knobs={"BLOCK": 64}, home=home, toolchain=TC, python=sys.executable)
    assert rep2.exts[0].cache_hit and rep2.exts[0].kernels[0].regs == 255
    ok, why = cache.file_prebuilt(FileRef(path=str(sub), knobs={"BLOCK": 64}), TC, home)
    assert ok, why
    assert not cache.file_prebuilt(FileRef(path=str(sub), knobs={"BLOCK": 32}), TC, home)[0]
    rows = comp.report_rows([rep])
    assert rows["kernels"][0]["spill_store"] == 8 and rows["builds"][0]["cache_key"] == e.cache_key
    # the GPU worker's hook (hook_env, other process env) computes the same key -> uses the prebuilt entry
    env = cache.hook_env(FileRef(path=str(sub), knobs={"BLOCK": 64}), TC, home)
    kw = {"name": "okc_test_ext", "cpp_sources": "", "cuda_sources": "__global__ void k1() {}", "functions": ["run"],
          "extra_cuda_cflags": ["-O3", "-gencode=arch=compute_90a,code=sm_90a"]}
    key = _shim.call_key("load_inline", kw, env["KDEV_TC_VERSION"], env["KDEV_TC_PTXAS"],
                         json.loads(env["KDEV_KNOBS"]), json.loads(env["KDEV_DEFINES"]))
    assert key == e.cache_key and cache.entry_complete(Path(env["KDEV_EXT_ROOT"]) / key)


def test_build_py_compile_error(tmp_path, fake_torch):
    sub = tmp_path / "sub.py"
    sub.write_text(_SUB.replace("{extra}", "BROKEN"))
    rep = comp.build_file(str(sub), home=str(tmp_path / "home"), toolchain=TC, python=sys.executable)
    errs = [e for e in rep.errors if e.startswith("build_failed")]
    assert errs and "BROKEN is undefined" in errs[0]


def test_build_py_without_torch(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "empty"))
    sub = tmp_path / "sub.py"
    sub.write_text("x = 1\n")
    try:
        import torch  # noqa: F401
        pytest.skip("torch installed")
    except ImportError:
        pass
    with pytest.raises(KdevError) as ei:
        comp.build_file(str(sub), home=str(tmp_path), toolchain=TC, python=sys.executable)
    assert ei.value.code == ErrorCode.CAPABILITY_UNAVAILABLE


# ----------------------------------------------------------------------------------------------- sweep
def test_parse_grid_and_expand():
    g = grid.parse_grid("TILE=64,128 STAGES=2,3;SPLIT=true")
    assert g == {"TILE": [64, 128], "STAGES": [2, 3], "SPLIT": [True]}
    pts = grid.expand(g)
    assert len(pts) == 4 and pts[0].label == "p1" and pts[-1].knobs == {"TILE": 128, "STAGES": 3, "SPLIT": True}
    for bad in ("", "TILE", "TILE=", "A=1 A=2"):
        with pytest.raises(KdevError):
            grid.parse_grid(bad)
    with pytest.raises(KdevError) as ei:
        grid.parse_grid("cublaslt:M=128")
    assert ei.value.code == ErrorCode.NOT_IMPLEMENTED


def _fake_builder(file, knobs, problem=None, home=None):
    from kdev.models import KernelInfo
    rep = comp.BuildReport(file=file, md5="x")
    if knobs.get("TILE") == 256:
        rep.errors.append("build_failed: too much smem")
        return rep
    spill = 16 if knobs.get("STAGES") == 4 else 0
    rep.exts.append(comp.ExtBuild(name="e", cache_key="k", cache_hit=False, seconds=0.1,
                                  kernels=[KernelInfo("k", regs=100 + knobs.get("STAGES", 0), spill_store=spill)]))
    return rep


class _Ctx:
    agent, problem, home = "pytest", None, None

    def __init__(self, fail=False):
        self.specs = []
        self.fail = fail

    def submit(self, spec, **kw):
        if self.fail:
            raise NotImplementedError("Context.submit (P1 + P2)")
        self.specs.append(spec)
        if spec.kind == "check":
            return Result(tool="check", status="ok", verdict="pass", run_id=f"r{len(self.specs)}")
        per = {"base": 2.0}
        for f in spec.files[1:]:
            per[f.label] = 2.0 * (1 + f.knobs["TILE"] / 1000 + f.knobs["STAGES"] / 100)
        return Result(tool="ab", status="ok", verdict="faster", run_id=f"r{len(self.specs)}",
                      score={"per_file": per, "ci_per_file": {k: [v * 0.99, v * 1.01] for k, v in per.items()}})


def test_run_sweep_ranks_and_drops(tmp_path):
    f = tmp_path / "sub.py"
    f.write_text("x = 1\n")
    ctx = _Ctx()
    res = grid.run_sweep(str(f), grid.parse_grid("TILE=64,128,256 STAGES=2,3,4"), base=None, top=3,
                         build_parallel=4, quick_check=True, ctx=ctx, builder=_fake_builder)
    assert validate_result(res.to_dict()) == []
    s = res.summary
    assert s["n_points"] == 9 and s["dropped"] == {"build_failed": 3, "spills": 2, "c751x": 0} and s["timed"] == 4
    assert s["top"][0]["knobs"] == {"TILE": 128, "STAGES": 3} and s["top"][0]["rel"] == pytest.approx(1.158)
    ab = [sp for sp in ctx.specs if sp.kind == "ab"]
    assert len(ab) == 1 and [x.label for x in ab[0].files][0] == "base" and len(ab[0].files) == 5
    assert all(sp.resolved_lane() == "gpu" for sp in ctx.specs)
    csv = Path(res.files["csv"]).read_text()
    assert csv.splitlines()[0].startswith("label,TILE,STAGES,build_status") and len(csv.splitlines()) == 10


def test_run_sweep_without_gpu_route_and_dry_run(tmp_path):
    f = tmp_path / "sub.py"
    f.write_text("x = 1\n")
    res = grid.run_sweep(str(f), {"TILE": [64, 128], "STAGES": [2]}, base=None, top=3, build_parallel=2,
                         quick_check=False, ctx=_Ctx(fail=True), builder=_fake_builder)
    assert res.verdict == "built_only" and res.status == "ok" and any("GPU stage skipped" in w for w in res.warnings)
    dry = grid.run_sweep(str(f), {"TILE": list(range(10))}, base=None, top=3, build_parallel=2, quick_check=False,
                         ctx=_Ctx(), dry_run=True, builder=_fake_builder)
    assert dry.verdict == "dry_run" and dry.summary["ab_jobs"] == 3


def test_cli_sweep_dry_run(tmp_path):
    from kdev.cli import run as cli_run
    f = tmp_path / "sub.py"
    f.write_text("x = 1\n")
    res = cli_run(["sweep", str(f), "--grid", "A=1,2 B=3", "--dry-run"])
    assert res.verdict == "dry_run" and res.summary["n_points"] == 2


def test_cli_build_cu_with_fake_nvcc(tmp_path, monkeypatch):
    from kdev.cli import run as cli_run
    monkeypatch.setattr(comp, "_nvcc_usable", lambda pin: True)
    monkeypatch.setattr("kdev.build.toolchain.pin_for", lambda *a, **k: TC)
    monkeypatch.setattr(comp, "pin_for", lambda *a, **k: TC)
    monkeypatch.setattr(comp, "run", _fake_nvcc())
    monkeypatch.setattr(comp.build_file, "__kwdefaults__", {**comp.build_file.__kwdefaults__, "runner": _fake_nvcc()})
    src = tmp_path / "k.cu"
    src.write_text("__global__ void k1() {}\n")
    res = cli_run(["build", str(src), "--variant", "TILE=64", "--json"])
    assert validate_result(res.to_dict()) == [] and res.status == "ok", res.to_dict()
    b = json.loads(Path(res.files["build"]).read_text())
    assert b["kernels"][0]["regs"] == 40 and b["knobs"] == {"TILE": 64}
    assert res.summary["kernels"][0]["regs"] == 40 and res.summary["builds"][0]["ext_name"] == "k"
    try:
        from kdev.ledger.record import rows_from_result
        rows = rows_from_result(res)
    except NotImplementedError:
        pytest.skip("ledger (P4) not implemented")
    assert rows["kernels"][0]["regs"] == 40 and rows["builds"][0]["cache_key"] == b["builds"][0]["cache_key"]


def test_build_py_default_args_entry_failure_is_not_fatal(tmp_path, fake_torch):
    # OKC moe best: `_ext(defines=None)` builds MAIN_DEFINES, a variant its prepare never uses (and that does not
    # compile). A guessed default-args entry must not fail the file or enter its manifest.
    sub = tmp_path / "sub.py"
    src = _SUB.replace("{extra}", "").replace("def _module():", "def _ext(defines=None):\n"
                                              "    return load_inline(name='okc_guess', cpp_sources='', "
                                              "cuda_sources='BROKEN', functions=['run'])\ndef _module():")
    sub.write_text(src)
    rep = comp.build_file(str(sub), home=str(tmp_path / "home"), toolchain=TC, python=sys.executable)
    assert not [e for e in rep.errors if e.startswith("build_failed")], rep.errors
    assert any("default-args entry build failed" in e for e in rep.errors)
