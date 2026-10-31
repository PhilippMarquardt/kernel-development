"""P7: adapters (okc, cu_main, kernelbench, registry) against a tiny fake problem dir (torch-free reference.py)."""
from __future__ import annotations

import math
import textwrap

import pytest

from kdev.adapters.base import Adapter, Shape, TimingProtocol, load_weighted_geomean
from kdev.adapters.okc import (OkcAdapter, case_key, discover_extensions, known_problems, parse_case_text,
                               parse_check_message, parse_task_md)
from kdev.adapters.registry import ADAPTERS, detect, get_adapter
from kdev.contract import KdevError
from kdev.registry import build_registry

TASK = textwrap.dedent("""\
    # Fake (`fake_prob`)

    ## Shapes

    | load | shape | weight | floor |
    |---|---|---|---|
    | decode 1 | m 1, layer a | 10 | 1.0 µs |
    | decode 1 | m 1, layer b | 5 | 1.0 µs |
    | verify 16 | m 128, layer a | 10 | 2.0 µs |

    Before timing, a run must pass these 2 tests, some harder than the scored shapes:

    ```
    m 3, layer a
    m 9, layer b, dist outliers
    ```
    """)

REFERENCE = textwrap.dedent('''\
    """fake reference: data is a plain dict, outputs are lists; no torch."""
    from checks import compare
    WEIGHTS_SEED = 7
    REL_L2_TOL = 0.01
    CALLS = []

    def generate_weights(seed, device="cuda"):
        return {"w": [seed, 2.0], "router": [1.0]}

    def generate_input(m, layer, seed, dist="normal", local_start=0, device="cuda", **_):
        CALLS.append((m, layer, seed, local_start))
        return {"x": [float(seed % 97 + i) for i in range(m)], "config": {"m": m, "layer": layer},
                "local_start": local_start}

    def case_fields(case, weights):
        return {"local_start": 32 * (case["seed"] % 2)}

    def ref(data, weights):
        return [v * weights["w"][1] for v in data["x"]]

    def check_implementation(data, output, weights):
        return compare(output, ref(data, weights), REL_L2_TOL)
    ''')

CHECKS = textwrap.dedent('''\
    def compare(out, ref, tol):
        if not isinstance(out, list) or len(out) != len(ref):
            return False, "expected a list"
        num = sum((a - b) ** 2 for a, b in zip(out, ref)) ** 0.5
        den = max(sum(b * b for b in ref) ** 0.5, 1e-12)
        err = num / den
        if err > tol:
            return False, f"relative L2 error {err:.3e} > {tol:g}"
        return True, f"relative L2 error {err:.3e} (worst row {err:.3e})"
    ''')

TOML = textwrap.dedent('''\
    schema = "kdev.okc_problem/1"
    problem = "fake_prob"
    kernel_fn_args = 2
    check_style = "direct"
    weights = true
    case_fields = true
    input_variance = { routing = true }
    [protocol]
    judge_reps = 9
    budget_s = 600
    [[shapes]]
    id = "d1/a"
    load = "decode 1"
    weight = 10
    seed = 11
    repeats = 60
    floor_us = 1.0
    case = { m = 1, layer = "a" }
    [[shapes]]
    id = "d1/b"
    load = "decode 1"
    weight = 5
    seed = 12
    case = { m = 1, layer = "b" }
    [[shapes]]
    id = "v16/a"
    load = "verify 16"
    weight = 10
    case = { m = 128, layer = "a" }
    [[tests]]
    id = "t0"
    case = { m = 3, layer = "a" }
    [[tests]]
    id = "t1"
    case = { m = 9, layer = "b", dist = "outliers" }
    [[extra]]
    id = "x0"
    case = { m = 17, layer = "a" }
    [comply]
    precision = { router = "fp32" }
    forbid_patterns = ['\\.item\\(\\)']
    [comply.routing]
    fp32_max_tokens = 8
    ''')

SUB = textwrap.dedent('''\
    # okc: problem=fake_prob
    def prepare(weights):
        return {"scale": weights["w"][1]}

    def kernel_fn(data, state):
        return [v * state["scale"] for v in data["x"]]
    ''')


@pytest.fixture
def prob(tmp_path):
    d = tmp_path / "fake_prob" / "fake_prob"
    d.mkdir(parents=True)
    (d / "TASK.md").write_text(TASK, encoding="utf-8")
    (d / "reference.py").write_text(REFERENCE, encoding="utf-8")
    (d / "checks.py").write_text(CHECKS, encoding="utf-8")
    (d / "baseline.py").write_text(SUB, encoding="utf-8")
    (d / "submission.py").write_text(SUB, encoding="utf-8")
    cfg = tmp_path / "fake_prob.toml"
    cfg.write_text(TOML, encoding="utf-8")
    return d, cfg


def test_p7_commands_registered():
    assert {"check", "sanitize", "comply"} <= set(build_registry().names())


def test_adapter_interface_frozen_names():
    for meth in ("shapes", "tests", "make_inputs", "call", "check", "score", "judge_cases", "build_hook"):
        assert hasattr(Adapter, meth)
    assert {"okc", "cu_main", "kernelbench"} <= set(ADAPTERS)
    assert TimingProtocol().overhead_us == 4.0
    assert Shape("d1/sliding", {"batch": 1}).weight == 1.0


def test_load_weighted_geomean():
    shapes = [Shape("a", {}, 46, "d1"), Shape("b", {}, 16, "d1"), Shape("c", {}, 1, "d16"),
              Shape("t", {}, 1, "d16", scored=False)]
    base = {"a": 20.0, "b": 30.0, "c": 100.0, "t": 1.0}
    sub = {"a": 10.0, "b": 30.0, "c": 50.0, "t": 100.0}
    d1 = (46 * 20 + 16 * 30) / (46 * 10 + 16 * 30)
    assert load_weighted_geomean(shapes, sub, base) == pytest.approx(math.sqrt(d1 * 2.0))
    assert load_weighted_geomean(shapes, {"a": 10.0}, base) == pytest.approx(2.0)  # partial: one load
    assert math.isnan(load_weighted_geomean(shapes, {}, base))


def test_parse_task_md_and_case_text():
    shapes, tests = parse_task_md(TASK)
    assert [s.shape_id for s in shapes] == ["d1/a", "d1/b", "v16/a"]
    assert shapes[0].case == {"m": 1, "layer": "a"} and shapes[0].weight == 10 and shapes[0].load == "decode 1"
    assert [t.case for t in tests] == [{"m": 3, "layer": "a"}, {"m": 9, "layer": "b", "dist": "outliers"}]
    assert parse_case_text("batch 3, ctx 4000, kv_scale 0.015625") == {"batch": 3, "ctx": 4000, "kv_scale": 0.015625}


def test_parse_check_message():
    n = parse_check_message("relative L2 error 2.532e-03 (worst row 2.602e-03) [bound 0.147: 2.5 x ...]")
    assert n == {"rel_err": 2.532e-3, "bound": 0.147, "worst_row": 2.602e-3}
    assert parse_check_message("malformed output") == {}


def test_okc_adapter_pure(prob):
    d, cfg = prob
    a = OkcAdapter(str(d), str(cfg))
    assert a.problem == "fake_prob" and a.problem_dir == str(d)
    assert [s.shape_id for s in a.shapes()] == ["d1/a", "d1/b", "v16/a"]
    assert all(not s.scored for s in a.tests() + a.extra_shapes())
    assert a.input_variance() == {"routing": True} and a.default_seeds() == 5
    assert a.protocol.judge_reps == 9 and a.protocol.budget_s == 600 and a.protocol.overhead_us == 4.0
    assert a.judge_seed(a.shapes()[0]) == 11 and a.judge_repeats("d1/a") == 60 and a.judge_repeats("v16/a") == 9
    assert a.floor_us("d1/a") == 1.0
    assert a.smallest_test().shape_id == "t0"
    assert a.baseline_path().endswith("baseline.py")
    r = a.rules()
    assert r.precision == {"router": "fp32"} and r.forbid_patterns == [r"\.item\(\)"]
    assert a.comply_extra()["routing"]["fp32_max_tokens"] == 8
    assert a.score({"d1/a": 1, "d1/b": 1, "v16/a": 1}, {"d1/a": 2, "d1/b": 2, "v16/a": 2}) == pytest.approx(2.0)


def test_okc_adapter_runtime_with_fake_reference(prob):
    d, cfg = prob
    a = OkcAdapter(str(d), str(cfg))
    mod = a.load(str(d / "submission.py"), "A")
    state = a.prepare(mod)
    assert state == {"scale": 2.0} and a.last_prepare_s is not None
    data = a.make_inputs(a.tests()[0], seed=5)
    assert data["local_start"] == 32  # case_fields merged (seed 5 is odd)
    out = a.call(mod, state, data)
    res = a.check(data, out)
    assert res.ok and res.max_rel_err == 0.0 and res.tol == 0.01
    bad = a.check(data, [v + 1 for v in out])
    assert not bad.ok and bad.max_rel_err > 0.01
    crash = a.check(data, None)
    assert not crash.ok


def test_reference_not_importable_by_submission(prob):
    import sys
    d, cfg = prob
    a = OkcAdapter(str(d), str(cfg))
    a.reference()
    assert "reference" not in sys.modules or getattr(sys.modules["reference"], "__file__", "") != str(d / "reference.py")


def test_judge_cases_from_fixture(judge_json):
    a = OkcAdapter("muse_moe_w4a16")
    cases = a.judge_cases(judge_json)
    assert [c.shape_id for c in cases] == ["d1/bs1", "d16/bs16", "d32/bs32", "v1/bs8", "v16/bs128", "v32/bs256"]
    assert all(c.seed and c.weight == 60 and "load" not in c.case and "seed" not in c.case for c in cases)
    assert cases[0].load == "decode 1"
    tests = a.judge_tests(judge_json)
    assert len(tests) == 5 and tests[0].shape_id == "t0" and tests[0].seed


def test_shipped_tomls_consistent():
    names = known_problems()
    assert {"muse_attention_decode", "muse_bf16_proj", "muse_decode_layer", "muse_moe_w4a16",
            "muse_prefill_layer"} <= set(names)
    expect = {"muse_attention_decode": (12, 16, 1), "muse_bf16_proj": (30, 9, 2), "muse_decode_layer": (12, 13, 2),
              "muse_moe_w4a16": (6, 5, 2), "muse_prefill_layer": (8, 5, 2)}
    for p, (n_shapes, n_tests, arity) in expect.items():
        a = OkcAdapter(p)
        assert len(a.shapes()) == n_shapes and len(a.tests()) == n_tests and a.kernel_fn_args() == arity
        ids = [s.shape_id for s in a.shapes() + a.tests() + a.extra_shapes()]
        assert len(ids) == len(set(ids)), p
        loads = {s.load for s in a.shapes()}
        assert all(s.weight > 0 for s in a.shapes()) and len(loads) in (3, 6)
        assert len({case_key(s.case) for s in a.shapes()}) == n_shapes
        assert a.rules().forbid_patterns and a.protocol.prepare_budget_s == 300
    assert OkcAdapter("muse_decode_layer").comply_extra()["routing"]["fp32_max_tokens"] == 8


def test_unknown_problem_errors():
    with pytest.raises(KdevError) as e:
        OkcAdapter("no_such_problem")
    assert e.value.code.value == "invalid_request"
    with pytest.raises(KdevError):
        get_adapter(problem="no_such_problem_either")


def test_registry_detect_and_get(prob, tmp_path):
    d, cfg = prob
    assert detect(str(d / "submission.py")) == ("okc", str(d.resolve()))
    a = get_adapter(file=str(d / "submission.py"))
    assert a.name == "okc" and a.problem == "fake_prob" and a.config_path is None  # TASK.md fallback, no TOML
    assert [s.shape_id for s in a.shapes()] == ["d1/a", "d1/b", "v16/a"]
    assert get_adapter(problem="okc:muse_moe_w4a16").problem == "muse_moe_w4a16"
    assert get_adapter(problem="muse_bf16_proj").name == "okc"
    cu = tmp_path / "k.cu"
    cu.write_text("__global__ void k(){}\nint main() { return 0; }\n", encoding="utf-8")
    assert detect(str(cu))[0] == "cu_main"
    assert get_adapter(file=str(cu)).name == "cu_main"
    kb = tmp_path / "kb.py"
    kb.write_text("class Model:\n    pass\n\ndef get_inputs():\n    return []\n", encoding="utf-8")
    assert detect(str(kb))[0] == "kernelbench"
    assert get_adapter(file=str(kb)).name == "kernelbench"
    stray = tmp_path / "stray.py"
    stray.write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(KdevError):
        get_adapter(file=str(stray))


def test_cu_main_adapter(tmp_path, fake_runner):
    from kdev.adapters.cu_main import CuMainAdapter, binary_path
    cu = tmp_path / "k.cu"
    cu.write_text("int main(){return 0;}\n", encoding="utf-8")
    runner = fake_runner({(binary_path(str(cu)),): (0, "PASS all\n", "")})
    a = CuMainAdapter(str(cu), runner=runner)
    spec = a.build_hook(str(cu))[0]
    assert spec.name == "k" and "-arch=sm_90a" in spec.cuda_cflags
    h = a.load(str(cu), "A")
    with pytest.raises(KdevError) as e:
        a.prepare(h)
    assert e.value.code.value == "needs_build"
    out = a.call(h, None, a.make_inputs(a.shapes()[0], 1))
    assert a.check(None, out).ok
    assert a.score({"main": 2.0}, {"main": 4.0}) == 2.0


def test_discover_extensions(tmp_path):
    src = tmp_path / "s.py"
    src.write_text(textwrap.dedent('''\
        from torch.utils.cpp_extension import load_inline
        CUDA = "__global__ void k(float* x) {}"
        CPP = "void f();"
        FLAGS = ["-O3", "-DX=1"]
        def build():
            return load_inline(name="mod_a", cuda_sources=[CUDA], cpp_sources=CPP, extra_cuda_cflags=FLAGS)
        '''), encoding="utf-8")
    specs = discover_extensions(str(src))
    assert len(specs) == 1 and specs[0].name == "mod_a" and specs[0].inline
    assert specs[0].sources == ["__global__ void k(float* x) {}", "void f();"] and specs[0].cuda_cflags == ["-O3", "-DX=1"]
