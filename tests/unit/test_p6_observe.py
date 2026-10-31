"""P6: observe backends (ncu/nsys/nvbit probes + parsers, CUPTI kernel table, dispatch summary). No GPU."""
from __future__ import annotations

import json
from types import SimpleNamespace

from kdev.models import Check
from kdev.observe import _rt, cupti, dispatch, ncu, nsys, nvbit
from kdev.util import Proc


def _runner(rc=0, out="NVIDIA (R) Nsight Compute Command Line Profiler\nVersion 2025.3"):
    return lambda argv, **kw: Proc(list(argv), rc, out, "")


def test_ncu_available_perm_detection(tmp_path):
    params = tmp_path / "params"
    status = tmp_path / "status"
    params.write_text("ResmanDebugLevel: 4294967295\nRmProfilingAdminOnly: 1\n")
    status.write_text("Name:\tpython\nCapEff:\t00000000a80425fb\n")       # docker default: no CAP_SYS_ADMIN
    ok, why = ncu.available(_runner(), params_path=str(params), status_path=str(status), ncu_path="ncu")
    assert not ok and "ERR_NVGPUCTRPERM" in why
    status.write_text("CapEff:\t000001ffffffffff\n")                        # privileged
    assert ncu.available(_runner(), params_path=str(params), status_path=str(status), ncu_path="ncu")[0]
    params.write_text("RmProfilingAdminOnly: 0\n")
    status.write_text("CapEff:\t0000000000000000\n")
    assert ncu.available(_runner(), params_path=str(params), status_path=str(status), ncu_path="ncu") == (True, "ok")
    assert not ncu.available(_runner(rc=1), params_path=str(params), ncu_path="ncu")[0]


def test_ncu_argv_and_perm_error():
    argv = ncu.ncu_argv(["python", "x.py"], metrics="sol,occupancy", kernel_regex="attn", out="r.ncu-rep")
    assert argv[0] == "ncu" and argv[-2:] == ["python", "x.py"]
    assert "SpeedOfLight" in argv and "Occupancy" in argv and "regex:attn" in argv
    assert argv[argv.index("--clock-control") + 1] == "none"
    assert ncu.is_perm_error("==ERROR== ERR_NVGPUCTRPERM - The user does not have permission")
    assert not ncu.is_perm_error("==PROF== Disconnected")


NCU_CSV = '''==PROF== Connected to process 1
"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream","Block Size","Grid Size","Device","CC","Section Name","Metric Name","Metric Unit","Metric Value"
"0","1","python","h","attn_kernel","1","7","(384, 1, 1)","(132, 1, 1)","0","9.0","GPU Speed Of Light Throughput","Duration","us","4.21"
"0","1","python","h","attn_kernel","1","7","(384, 1, 1)","(132, 1, 1)","0","9.0","GPU Speed Of Light Throughput","Memory Throughput","%","61.2"
"1","1","python","h","merge","1","7","(256, 1, 1)","(1, 1, 1)","0","9.0","GPU Speed Of Light Throughput","Duration","us","1.1"
'''


def test_ncu_parse_csv():
    rows = ncu.parse_csv(NCU_CSV)
    assert len(rows) == 3 and rows[0]["Kernel Name"] == "attn_kernel"
    s = ncu.summarize_rows(rows)
    assert [k["kernel"] for k in s] == ["attn_kernel", "merge"]
    assert s[0]["metrics"]["GPU Speed Of Light Throughput/Duration"] == "4.21 us"
    assert ncu.parse_csv("nothing") == []


NSYS_CSV = '''Generating SQLite file nsys.sqlite from nsys.nsys-rep
Processing [nsys.sqlite] with [cuda_gpu_kern_sum.py]...

 ** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):

Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name
30.0,"3,000",3,1000.0,1000.0,900,1100,10.0,small_kernel
70.0,7000,2,3500.0,3500.0,3400,3600,10.0,"big_kernel<1>(Params)"
'''


def test_nsys_parse_and_argv():
    rows = nsys.parse_kern_sum(NSYS_CSV)
    assert [r["name"] for r in rows] == ["big_kernel<1>(Params)", "small_kernel"]
    assert rows[0]["count"] == 2 and rows[0]["avg_us"] == 3.5 and rows[1]["total_us"] == 3.0
    a = nsys.nsys_argv(["python", "t.py"], out="o")
    assert a[:2] == ["nsys", "profile"] and "--trace=cuda" in a and a[-2:] == ["python", "t.py"]


def test_nvbit_available(tmp_path, monkeypatch):
    monkeypatch.delenv("NVBIT_PATH", raising=False)
    assert not nvbit.available()
    (tmp_path / "tools" / "instr_count").mkdir(parents=True)
    (tmp_path / "tools" / "instr_count" / "instr_count.so").write_bytes(b"")
    monkeypatch.setenv("NVBIT_PATH", str(tmp_path))
    assert nvbit.available()


def test_probe_never_raises():
    from kdev.observe import probe
    checks = probe(None)
    assert {c.name for c in checks} >= {"ncu", "nsys", "compute-sanitizer", "nvbit", "cupti"}
    assert all(isinstance(c, Check) and c.status in ("PASS", "WARN", "FAIL", "SKIP") for c in checks)


TRACE = {"traceEvents": [
    {"ph": "X", "cat": "kernel", "name": "void vjc_k<1>(Params)", "dur": 3.0, "args": {"grid": [132, 1, 1],
                                                                                     "block": [384, 1, 1]}},
    {"ph": "X", "cat": "kernel", "name": "void vjc_k<1>(Params)", "dur": 3.4, "args": {"grid": [132, 1, 1],
                                                                                     "block": [384, 1, 1]}},
    {"ph": "X", "cat": "kernel", "name": "merge", "dur": 1.0, "args": {"grid": [1, 1, 1], "block": [256, 1, 1]}},
    {"ph": "X", "cat": "cuda_runtime", "name": "cudaGraphLaunch", "dur": 9.0},
    {"ph": "X", "cat": "gpu_memcpy", "name": "Memcpy DtoD", "dur": 2.0},
]}


def test_kernel_table_chrome_trace_and_fakes():
    t = cupti.kernel_table(TRACE)
    assert [(r.name, r.count) for r in t] == [("void vjc_k<1>(Params)", 2), ("merge", 1)]
    assert abs(t[0].total_us - 6.4) < 1e-9 and t[0].grid == [132, 1, 1] and abs(t[0].mean_us - 3.2) < 1e-9
    fe = [SimpleNamespace(name="k", device_type="DeviceType.CUDA", time_range=SimpleNamespace(elapsed_us=lambda: 2.0)),
          SimpleNamespace(name="cpu_op", device_type="DeviceType.CPU", time_range=SimpleNamespace(elapsed_us=lambda: 9))]
    assert [(r.name, r.total_us) for r in cupti.kernel_table(fe)] == [("k", 2.0)]
    avg = [SimpleNamespace(key="k2", count=4, self_device_time_total=8.0)]
    assert [(r.name, r.count, r.mean_us) for r in cupti.kernel_table(avg)] == [("k2", 4, 2.0)]


def test_base_name():
    assert dispatch.base_name("void ns::foo<1, (bool)0>(ns::Params)") == "foo"
    assert dispatch.base_name("_ZN38_GLOBAL__N__5234c7d1_6_vjc_cu_0b177e6611attn_kernelILi8ELb1ELi0EEEvNS_6ParamsE") \
        == "attn_kernel"
    assert dispatch.base_name("_Z6k_nullv") == "k_null"
    assert dispatch.base_name("_Z3fooILi1EEvPf") == "foo"
    assert dispatch.base_name("merge") == "merge"


def test_summarize_dispatch_warnings():
    per = {"d1": [{"kernel": "void attn_kernel<1>(P)", "count": 1, "grid": [33, 1, 1]}],
           "d16": [{"kernel": "void attn_kernel<1>(P)", "count": 1, "grid": [33, 1, 1]}],
           "v1": [{"kernel": "void attn_kernel<8>(P)", "count": 1, "grid": [132, 1, 1]},
                  {"kernel": "void at::native::vectorized_elementwise_kernel<4>(int)", "count": 1, "grid": [1, 1, 1]}],
           "v16": []}
    compact, w = dispatch.summarize_dispatch(per, ["_Z11attn_kernelILi1EEv1P", "_Z9old_fast3v"])
    assert compact["d1"] == [{"kernel": "void attn_kernel<1>(P)", "count": 1, "grid": [33, 1, 1]}]
    text = "\n".join(w)
    assert "v16: no kernel launched" in text
    assert "same route" in text and "d1, d16" in text
    assert "never launched" in text and "old_fast3" in text and "attn_kernel," not in text
    assert "library/torch kernels" in text
    _, w2 = dispatch.summarize_dispatch({"a": [{"kernel": "k", "grid": [1]}]})
    assert w2 == []


def test_loaded_libraries_from_maps(tmp_path):
    maps = tmp_path / "maps"
    maps.write_text(
        "7f00-7f01 r-xp 00000000 08:01 1 /usr/lib/libc.so.6\n"
        "7f02-7f03 r-xp 00000000 08:01 2 /root/.cache/torch_extensions/x/x.so\n"
        "7f04-7f05 r--p 00000000 08:01 2 /root/.cache/torch_extensions/x/x.so\n"
        "7f06-7f07 rw-p 00000000 00:00 0 [heap]\n")
    assert _rt.loaded_libraries(str(maps)) == ["/usr/lib/libc.so.6", "/root/.cache/torch_extensions/x/x.so"]
    assert _rt.median([3.0, 1.0, 2.0, 10.0]) == 2.5


def test_profile_and_dispatch_cli_route_to_gpu(tmp_path, capsys):
    from kdev.cli import main
    f = tmp_path / "sub.py"
    f.write_text("x = 1\n")
    for cmd in (["profile", str(f), "--backend", "cupti"], ["dispatch", str(f)]):
        main(cmd + ["--json"])
        d = json.loads(capsys.readouterr().out)
        assert d["tool"] == cmd[0] and d["error"]["code"] in ("capability_unavailable", "not_implemented")
    main(["profile", str(tmp_path / "missing.py"), "--json"])
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "not_found"


def test_profiler_taint_guard(tmp_path):
    import pytest

    from kdev.contract import KdevError
    from kdev.models import JobSpec
    from kdev.observe import guard
    guard._reset_for_tests()
    try:
        assert guard.check_clean("time") is None and not guard.profiled()
        guard.mark_profiled("torch.profiler (CUPTI)")
        assert guard.profiled()
        assert guard.check_clean("time", refuse=False) == guard.OPTOUT
        with pytest.raises(KdevError) as e:
            guard.check_clean("time")
        assert "2-5 us" in e.value.message
        # P6's own timing jobs refuse after a profiler session in the same process
        from kdev.timeline.run import run_timeline
        with pytest.raises(KdevError):
            run_timeline(JobSpec(kind="timeline"), tmp_path)
    finally:
        guard._reset_for_tests()


def test_probe_isolated_runs_child_process(tmp_path):
    from kdev.models import FileRef, JobSpec
    from kdev.observe import dispatch
    spec = JobSpec(kind="dispatch", files=[FileRef("/x/sub.py")])
    calls = []

    def fake(argv, **kw):
        calls.append(argv)
        out = argv[-1]
        (tmp_path / "out" / "dispatch_result.json").write_text(json.dumps(
            {"dispatch": {"d1": [{"kernel": "k", "count": 1, "grid": [1]}]}, "warnings": ["w"]}))
        return Proc(list(argv), 0, "", "") if out else None

    comp, warns = dispatch.probe_isolated(spec, str(tmp_path / "out"), python="py", runner=fake)
    assert calls[0][:4] == ["py", "-m", "kdev.observe.dispatch", "--child"]
    assert comp["d1"][0]["kernel"] == "k" and warns == ["w"]
    assert json.loads((tmp_path / "out" / "dispatch_job.json").read_text())["kind"] == "dispatch"
    comp, warns = dispatch.probe_isolated(spec, str(tmp_path / "o2"), runner=lambda a, **k: Proc(a, 3, "", "boom"))
    assert comp == {} and "rc=3" in warns[0] and "boom" in warns[0]
