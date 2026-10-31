"""P7: check classification/reporting, sanitize log parsing + re-exec wrapper, comply static rules and the pure
parts of the dynamic probes, CLI spec building. No GPU, no torch."""
from __future__ import annotations

import json
import textwrap
from argparse import Namespace
from pathlib import Path

import pytest

from kdev.adapters.base import ComplyRules, Shape
from kdev.adapters.okc import OkcAdapter
from kdev.checks import check, comply, sanitize
from kdev.checks.commands import build_spec, check_params, cmd_comply
from kdev.contract import validate_result
from kdev.models import FileRef, JobSpec
from kdev.registry import Context
from kdev.util import Proc

# ------------------------------------------------------------------------------------------------- check


def test_classify():
    assert check.classify([True, True, True], True) == "pass"
    assert check.classify([True, True], False) == "nondeterministic"
    assert check.classify([True, False, True], True) == "intermittent"
    assert check.classify([False, False], True) == "wrong"
    assert check.classify([], True) == "error"
    assert check.worst_class(["pass", "nondeterministic", "intermittent"]) == "intermittent"


def test_ulp_helpers():
    assert check.ulp_distance_f32(1.0, 1.0) == 0
    assert check.ulp_distance_f32(1.0, 1.0000001192092896) == 1      # next float32 after 1.0
    assert check.ulp_distance_f32(8.0, 8.000000953674316) == 1       # 1 ulp at |x| >= 8 (fp32 atomics)
    assert check.ulp_distance_f32(-0.0, 0.0) == 0
    assert check.ulp_distance_f32(-1.0, 1.0) == 2 * 0x3F800000
    assert check.ordered_bits(0x8000_0001, 32) == -1


def test_pick_seeds_fresh_and_explicit():
    a, b = check.pick_seeds(4), check.pick_seeds(4)
    assert len(a) == 4 and a != b and all(0 < s < 2**31 for s in a)
    assert check.pick_seeds(3, [5, 6]) == [5, 6]


def test_select_cases_and_filter():
    a = OkcAdapter("muse_moe_w4a16")
    all_ = check.select_cases(a, extra=True)
    assert len(all_) == len(a.tests()) + len(a.shapes()) + len(a.extra_shapes())
    assert len(check.select_cases(a, tests_only=True)) == 5
    only = check.select_cases(a, filt="d16")
    assert [s.shape_id for s in only] == ["d16/bs16"]
    assert [s.shape_id for s in check.select_cases(a, filt="decode 1")] == ["d1/bs1"]


def _o(sid, klass, **kw):
    return check.CaseOutcome(sid, 1, kw.pop("mode", "eager"), kw.pop("passes", 3), 3, klass, **kw)


def test_summarize_render_and_hints():
    outs = [_o("d1/a", "pass", max_rel_err=1e-3, max_ulp=0, extra={"tol": 0.015}),
            _o("d1/a", "nondeterministic", max_ulp=1, mode="graph", extra={"vs_eager_rel_l2": 2e-4}),
            _o("t0", "intermittent", passes=1, message="relative L2 error 3e-2 > 0.015")]
    s = check.summarize(outs)
    assert s["counts"]["intermittent"] == 1 and s["n_failures"] == 1 and s["max_ulp"] == 1
    assert s["graph_vs_eager_rel_l2"] == 2e-4
    shapes = [Shape("d1/a", {}, 46, "decode 1"), Shape("t0", {}, scored=False)]
    rows = check.per_shape(outs, shapes)
    st = {r["shape_id"]: r["check_status"] for r in rows}
    assert st == {"d1/a": "nondeterministic", "t0": "intermittent"}
    s["problem"] = "p"
    lines = check.render_lines("fail", s, rows)
    assert lines[0].startswith("FAIL  check") and len(lines) <= 40
    assert any("intermittent: t0" in ln for ln in lines)
    assert "--audit pdl" in check.next_hints(s, "x.py")[0]


def test_cold_verdict():
    assert check.cold_verdict(10, 100, 20, 900, 300)["status"] == "PASS"
    assert check.cold_verdict(10, 250, 20, 900, 300)["status"] == "RISK"
    assert check.cold_verdict(10, 310, 20, 900, 300)["status"] == "FAIL"
    assert check.cold_verdict(400, 200, 400, 900, 300)["status"] == "FAIL"


def test_run_check_without_files_is_error_result():
    r = check.run_check(JobSpec(kind="check", problem="muse_moe_w4a16"), None)
    assert r.status == "error" and r.error.code == "invalid_request"
    assert not validate_result(r.to_dict())


def test_artifacts(tmp_path):
    p = check.write_jsonl(tmp_path / "run", "cases.jsonl", [{"a": 1}, {"b": 2}])
    assert Path(p).read_text().count("\n") == 2
    q = check.write_json(tmp_path / "run", "x.json", {"k": 1})
    assert json.loads(Path(q).read_text()) == {"k": 1}
    assert check.write_json(None, "x.json", {}) is None


# ---------------------------------------------------------------------------------------------- sanitize

MEMCHECK_LOG = textwrap.dedent("""\
    ========= COMPUTE-SANITIZER
    ========= Invalid __global__ write of size 4 bytes
    =========     at 0x70 in /root/x.cu:12:kern(float *)
    =========     by thread (31,0,0) in block (0,0,0)
    =========     Address 0x7f0000000080 is out of bounds
    =========
    ========= Invalid __global__ write of size 4 bytes
    =========     at 0x70 in /root/x.cu:12:kern(float *)
    =========     by thread (63,0,0) in block (0,0,0)
    =========
    ========= Invalid __shared__ read of size 16 bytes
    =========     at 0x1a0 in /root/x.cu:40:other()
    =========     by thread (0,0,0) in block (3,0,0)
    =========
    ========= Program hit cudaErrorLaunchFailure (error 719) due to "unspecified launch failure" on CUDA API call to cudaDeviceSynchronize.
    =========
    ========= ERROR SUMMARY: 4 errors
    """)
RACE_LOG = textwrap.dedent("""\
    ========= COMPUTE-SANITIZER
    ========= Error: Race reported between Write access at 0x1a0 in /x.cu:20:kern()
    =========     and Read access at 0x1c0 in /x.cu:22:kern() [128 hazards]
    =========
    ========= Warning: Race reported between Read access at 0x10 in /x.cu:5:k2()
    =========     and Write access at 0x20 in /x.cu:6:k2() [2 hazards]
    =========
    ========= RACECHECK SUMMARY: 2 hazards displayed (1 error, 1 warning)
    """)


def test_parse_memcheck_log():
    s = sanitize.parse_log(MEMCHECK_LOG, "memcheck")
    assert s.errors == 4
    assert s.by_type == {"invalid __global__ write": 2, "invalid __shared__ read": 1, "api cudaErrorLaunchFailure": 1}
    assert len(s.first) == 3                       # the two identical writes collapse into one distinct error
    assert "x.cu:12:kern" in s.first[0] and "by thread (31,0,0)" in s.first[0]


def test_parse_racecheck_log():
    s = sanitize.parse_log(RACE_LOG, "racecheck")
    assert s.errors == 1 and s.warnings == 1
    assert s.by_type == {"race write-read": 1}
    assert sanitize.parse_log("========= COMPUTE-SANITIZER\n========= ERROR SUMMARY: 0 errors\n", "memcheck").errors == 0


def test_sanitizer_argv():
    argv = sanitize.sanitizer_argv("racecheck", ["python", "-m", "x"], "/r/san.log", exe="/cs")
    assert argv[:7] == ["/cs", "--tool", "racecheck", "--print-limit", "100", "--log-file", "/r/san.log"]
    assert argv[-3:] == ["python", "-m", "x"]
    with pytest.raises(Exception):
        sanitize.sanitizer_argv("bogus", [], "x")


def test_sanitize_outer_with_fake_runner(tmp_path):
    sub = tmp_path / "sub.py"
    sub.write_text("def kernel_fn(data, state):\n    return data\n", encoding="utf-8")
    rd = tmp_path / "run"
    calls = []

    def runner(argv, **kw):
        calls.append((argv, kw))
        log = argv[argv.index("--log-file") + 1]
        Path(log).write_text(MEMCHECK_LOG, encoding="utf-8")
        (rd / "sanitize_inner.json").write_text(json.dumps({"shape": "t4", "ok": True, "messages": []}))
        return Proc(argv, 0, "", "")

    spec = JobSpec(kind="sanitize", files=[FileRef(str(sub))], problem="muse_moe_w4a16",
                   params={"extra": {"tool": "memcheck"}})
    r = sanitize.run_sanitize_outer(spec, rd, runner=runner, exe="/usr/bin/compute-sanitizer")
    argv, kw = calls[0]
    assert argv[0] == "/usr/bin/compute-sanitizer" and "kdev.checks.sanitize" in argv
    assert kw["env"][sanitize.INNER_ENV] == "1"
    job = json.loads((rd / "sanitize_job.json").read_text())
    assert job["kind"] == "sanitize" and job["problem"] == "muse_moe_w4a16"
    assert r.status == "fail" and r.verdict == "errors" and r.summary["errors"] == 4
    assert r.summary["shape"] == "t4" and r.lines[0].startswith("ERRORS  sanitize")
    assert not validate_result(r.to_dict())


def test_sanitize_missing_tool(monkeypatch, tmp_path):
    monkeypatch.setattr(sanitize, "find_sanitizer", lambda: None)
    sub = tmp_path / "sub.py"
    sub.write_text("def kernel_fn(d, s):\n    return d\n", encoding="utf-8")
    r = sanitize.run_sanitize_inner(JobSpec(kind="sanitize", files=[FileRef(str(sub))], problem="muse_moe_w4a16"),
                                    tmp_path)
    assert r.status == "error" and r.error.code == "capability_unavailable"


# ------------------------------------------------------------------------------------------------ comply

RULES = ComplyRules(precision={"router": "fp32"}, forbid_patterns=[r"\.item\(\)", r"torch\.cuda\.synchronize"])

CLEAN = textwrap.dedent('''\
    import torch
    from torch.utils.cpp_extension import load_inline
    SRC = """
    __global__ void __launch_bounds__(256) moe_k(float* x) {}
    __global__ void other_k(float* x) {}
    """

    def prepare(weights):
        n = int(weights["w"].sum().item())   # fine in prepare
        torch.cuda.synchronize()
        return {"w": weights["w"], "n": n}

    def kernel_fn(data, state):
        return data["x"] @ state["w"].T
    ''')


def _by(items):
    return {i.rule: i for i in items}


def test_static_clean():
    items = _by(comply.static_checks(CLEAN, RULES))
    assert items["entry_points"].status == "PASS"
    assert items["forbid_patterns"].status == "PASS"           # .item()/synchronize only inside prepare
    for r in ("no_reference_import", "no_frame_reads", "no_network", "no_processes", "no_timing_patches",
              "no_module_edits"):
        assert items[r].status == "PASS", r
    assert items["no_cross_call_caching"].status == "PASS"
    assert items["lossless_weights_static"].status == "PASS"
    assert items["kernel_inventory"].status == "PASS" and "2 kernels" in items["kernel_inventory"].detail
    assert comply.static_kernels(CLEAN) == {"moe_k": 1, "other_k": 1}


DIRTY = textwrap.dedent('''\
    import sys, subprocess, functools
    import torch
    import reference
    _CACHE = {}
    torch.cuda.Event = None
    torch.backends.cuda.matmul.allow_tf32 = True

    def prepare(weights):
        return {"w": weights["w"].to(torch.float8_e4m3fn)}

    def kernel_fn(data, state):          # dead: overwritten below
        return data["x"].cpu()

    def _impl(data, state):
        key = data["x"].data_ptr()
        if key in _CACHE:
            return _CACHE[key]
        n = data["x"].sum().item()
        f = sys._getframe(1)
        _CACHE[key] = data["x"] * n
        return _CACHE[key]

    @functools.lru_cache
    def helper(x):
        return x

    def _debug_dump(data):              # never reached from kernel_fn
        print(data["x"].tolist())

    def kernel_fn(data, state):
        return _impl(data, state)
    ''')


def test_static_dirty():
    items = _by(comply.static_checks(DIRTY, RULES))
    assert items["entry_points"].status == "RISK" and "2x" in items["entry_points"].detail
    for r in ("no_reference_import", "no_frame_reads", "no_processes", "no_timing_patches"):
        assert items[r].status == "FAIL", r
    assert items["torch_global_config"].status == "RISK"
    assert items["forbid_patterns"].status == "RISK" and "kernel_fn" in items["forbid_patterns"].detail
    cc = items["no_cross_call_caching"]
    assert cc.status == "RISK" and "data pointers" in cc.detail and "_CACHE[...]" in cc.detail
    assert items["lossless_weights_static"].status == "RISK"
    assert ".cpu" not in items["forbid_patterns"].detail and "tolist" not in items["forbid_patterns"].detail
    assert comply.verdict_of(list(items.values())) == "fail"


def test_call_graph_and_kernel_names():
    import ast
    src = CG_SRC
    tree = ast.parse(src)
    cg = comply.CallGraph(tree)
    reach = cg.reach("kernel_fn")
    go = cg.methods["go"][0]
    assert id(go) in reach and id(cg.live["helper"]) in reach and id(cg.live["kernel_fn"]) in reach
    assert id(cg.live["unused"]) not in reach
    assert cg.owner(cg.live["unused"].lineno + 1) is cg.live["unused"]
    assert comply.cuda_kernel_names(CU_SRC) == ["k_a", "k_b", "k_c"]


CG_SRC = """class Runner:
    def go(self, x):
        return helper(x)

def helper(x):
    return x

def unused(x):
    return x.item()

def kernel_fn(data, state):
    return Runner().go(data)
"""
CU_SRC = """__global__ void __launch_bounds__(min(256, NT), 1) k_a(float* x) {}
template <int N> __global__ __launch_bounds__(128) void k_b(int n);
extern "C" __global__ void
k_c(const float* __restrict__ x) {}
"""


def test_static_syntax_error():
    items = comply.static_checks("def kernel_fn(:\n", RULES)
    assert items[0].status == "FAIL" and items[0].rule == "parse"


def test_dispatch_items():
    per = {"d1": ["void moe_k<1>(float*)", "elementwise_kernel"], "d16": ["sm90_xmma_gemm", "Memset"]}
    items = _by(comply.dispatch_items(per, {"moe_k": 1, "dead_k": 1}))
    assert items["active_dispatch"].status == "PASS"
    assert items["active_dispatch_own_kernels"].status == "RISK" and "d16" in items["active_dispatch_own_kernels"].detail
    assert items["dead_kernels"].status == "RISK" and "dead_k" in items["dead_kernels"].detail
    assert "active_dispatch_own_kernels" not in _by(comply.dispatch_items({"d1": ["moe_k"]}, {"moe_k": 1}))


def test_routing_verdict():
    assert comply.routing_verdict({"tokens": 64, "discriminating": 6, "match_fp32": 6, "match_lowp": 0}, 8).status == "PASS"
    assert comply.routing_verdict({"tokens": 64, "discriminating": 6, "match_fp32": 1, "match_lowp": 5}, 8).status == "FAIL"
    assert comply.routing_verdict({"tokens": 64, "discriminating": 6, "match_fp32": 4, "match_lowp": 2}, 8).status == "RISK"
    inc = comply.routing_verdict({"tokens": 8, "discriminating": 0, "mismatch_slots": 0}, 8)
    assert inc.status == "PASS" and "inconclusive" in inc.detail


def test_precision_vs_baseline():
    assert comply.precision_vs_baseline([("d1", 2e-3, 1.5e-3, 1.5e-2)]).status == "PASS"
    assert comply.precision_vs_baseline([("d1", 9e-3, 1e-3, 1.5e-2)]).status == "RISK"
    assert comply.precision_vs_baseline([("d1", 2e-3, 1e-4, 1.5e-2)]).status == "PASS"  # < 25% of tol
    assert comply.precision_vs_baseline([("d1", None, None, None)]).status == "RISK"


def test_make_result_valid(tmp_path):
    items = comply.static_checks(CLEAN, RULES) + comply.unverified_precision_items(RULES)
    r = comply.make_result(items, problem="p", file="f.py", mode="static", run_dir=tmp_path)
    assert r.verdict == "risk" and r.status == "ok" and r.summary["counts"]["RISK"] == 1
    assert not validate_result(r.to_dict()) and Path(r.files["comply"]).is_file()
    assert len(r.lines) <= 40


def test_run_comply_static_only_worker(tmp_path):
    sub = tmp_path / "sub.py"
    sub.write_text(CLEAN, encoding="utf-8")
    spec = JobSpec(kind="comply", files=[FileRef(str(sub))], problem="muse_decode_layer",
                   params={"extra": {"static_only": True}})
    r = comply.run_comply(spec, tmp_path / "run")
    rules = {i["rule"] for i in r.summary["items"]}
    assert "precision:router" in rules and r.summary["mode"] == "static"


# ---------------------------------------------------------------------------------------------- commands

def _args(tmp_path, **kw):
    f = tmp_path / "sub.py"
    f.write_text(CLEAN, encoding="utf-8")
    base = dict(file=str(f), problem="muse_moe_w4a16", seeds=3, repeat=2, mode="both", extra_shapes=True,
                tests_only=False, cold=False, shapes=None, detach=False, direct=False, timeout=1200, priority=0,
                run_name="x", static_only=False, no_baseline=False, tool="memcheck", shape=None)
    base.update(kw)
    return Namespace(**base)


def test_build_spec(tmp_path):
    a = _args(tmp_path)
    spec = build_spec("check", a, Context(agent="t"), check_params(a))
    assert spec.kind == "check" and spec.problem == "muse_moe_w4a16" and spec.files[0].md5
    assert spec.params["extra"]["extra_shapes"] is True and spec.params["repeat"] == 2
    assert spec.timeout_s == 1200 and spec.agent == "t" and spec.resolved_lane() == "gpu"
    assert JobSpec.from_dict(spec.to_dict()).files[0].path == spec.files[0].path


def test_cmd_comply_static_only_runs_locally(tmp_path):
    r = cmd_comply(_args(tmp_path, static_only=True), Context())
    assert r.tool == "comply" and r.summary["mode"] == "static" and not validate_result(r.to_dict())


def test_gpu_commands_route_through_submit(tmp_path):
    from kdev.checks.commands import cmd_check, cmd_sanitize
    seen = []

    class Ctx(Context):
        def submit(self, spec, *, detach=False, direct=False, timeout_s=None):
            seen.append(spec)
            from kdev.contract import Result
            return Result(tool=spec.kind)

    cmd_check(_args(tmp_path), Ctx())
    cmd_sanitize(_args(tmp_path, tool="racecheck", shape="t0"), Ctx())
    assert [s.kind for s in seen] == ["check", "sanitize"]
    assert seen[1].params["extra"] == {"tool": "racecheck", "shape": "t0"} and seen[1].timeout_s >= 3600
