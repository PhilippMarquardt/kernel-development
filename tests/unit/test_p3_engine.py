"""P3: engine scheduling + summary logic with the deterministic FakeTimer (no GPU)."""
from __future__ import annotations

import pytest

from kdev.adapters.base import Shape
from kdev.contract import validate_result
from kdev.models import JobSpec, TimingParams
from kdev.timing import engine as E


def _p(**kw):
    return TimingParams(**kw)


def test_fake_timer_adds_overhead_and_noise():
    t = E.FakeTimer(means={"A": 10.0}, noise=[0.5, -0.5], overhead=4.0)
    assert t.replay_us("A") == 14.5
    assert t.replay_us("A") == 13.5
    assert t.calls == ["A", "A"]


def test_measure_shape_abba_and_overhead_subtraction():
    t = E.FakeTimer(means={"base": 20.0, "A": 10.0})
    smp = E.measure_shape(t, "s1", 7, _p(reps=4))
    # 1 warm-up rep (2 calls) + 4 reps x 2 files
    assert len(t.calls) == 2 + 8
    assert t.calls[2:] == ["base", "A", "A", "base", "base", "A", "A", "base"]
    assert len(smp) == 8
    assert {s.us for s in smp if s.label == "A"} == {10.0}
    assert {s.us_raw for s in smp if s.label == "base"} == {24.0}
    assert [(s.rep, s.order_pos, s.label) for s in smp[:4]] == [(0, 0, "base"), (0, 1, "A"), (1, 0, "A"),
                                                                 (1, 1, "base")]
    assert all(s.seed == 7 and s.shape_id == "s1" for s in smp)


def test_multi_seed_pairs_share_seed():
    timers = [(1, E.FakeTimer(means={"b": 2.0, "a": 1.0})), (2, E.FakeTimer(means={"b": 4.0, "a": 2.0}))]
    smp = E._measure_interleaved(timers, "s", _p(reps=3))
    reps = sorted({(s.rep, s.seed) for s in smp})
    assert len(reps) == 6                     # reps * seeds
    assert reps[:4] == [(0, 1), (1, 1), (2, 2), (3, 2)]   # (2j, 2j+1) share a seed
    ps = E.pair_speedups(E._group(smp)["s"], "a", "b")
    assert ps == pytest.approx([2.0, 2.0, 2.0])


def _samples(means_by_shape, reps=20, noise=None):
    out = []
    for sid, means in means_by_shape.items():
        t = E.FakeTimer(means=means, noise=noise or [0.0, 0.1, -0.1, 0.05, -0.05, 0.02, -0.03])
        out += E.measure_shape(t, sid, 1, _p(reps=reps))
    return out


def _score(sub, base):
    import math

    vals = [base[s] / sub[s] for s in sub]
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def test_summarize_faster_slower_same_and_score():
    shapes = [Shape("fast", {}, weight=2, load="d1"), Shape("slow", {}, load="d1"), Shape("eq", {}, load="d2")]
    smp = _samples({"fast": {"base": 20.0, "A": 10.0}, "slow": {"base": 10.0, "A": 20.0},
                    "eq": {"base": 30.0, "A": 30.0}})
    rows, score = E.summarize(smp, shapes, _p(reps=20), "base", _score, md5s={"A": "m"})
    v = {r["shape_id"]: r["verdict"] for r in rows}
    assert v == {"fast": "faster", "slow": "slower", "eq": "same"}
    fast = rows[0]
    assert fast["weight"] == 2 and fast["load"] == "d1"
    assert fast["ratio"]["num"] == "A" and fast["ratio"]["den"] == "base"
    assert fast["ratio"]["speedup"] == pytest.approx(2.0, rel=0.03)
    assert fast["ratio"]["median"] == pytest.approx(0.5, rel=0.03)      # time ratio num/den
    assert fast["ratio"]["speedup_ci"][0] <= fast["ratio"]["speedup"] <= fast["ratio"]["speedup_ci"][1]
    a = next(p for p in fast["per_file"] if p["label"] == "A")
    assert a["md5"] == "m" and a["n"] == 20 and a["median_us"] == pytest.approx(10.0, abs=0.1)
    assert score["per_file"]["A"] == pytest.approx(1.0, rel=0.05)       # geomean(2, 0.5, 1)
    assert score["delta"]["num"] == "A"
    assert score["ci"]["A"][0] <= score["per_file"]["A"] <= score["ci"]["A"][1]
    assert score["ci_per_file"]["A"] == score["ci"]["A"]          # sweep (P5) reads ci_per_file
    assert E.overall_verdict(rows, score, 0.03) == "same"


def test_summarize_incorrect_and_suspect_and_warnings():
    shapes = [Shape("s", {}), Shape("t", {})]
    smp = _samples({"s": {"base": 8.0, "A": 4.0}, "t": {"base": 50.0, "A": 25.0}})
    rows, _ = E.summarize(smp, shapes, _p(), "base", None, checks={("s", "A"): False, ("t", "A"): True},
                          floors_us={"t": 30.0})
    by = {r["shape_id"]: r for r in rows}
    assert by["s"]["verdict"] == "incorrect" and by["s"]["check_status"] == "fail:A"
    assert by["t"]["verdict"] == "suspect" and by["t"]["check_status"] == "pass"
    assert any("overhead-sensitive" in w for w in by["s"]["warnings"])
    assert E.overall_verdict(rows, None, 0.03) == "incorrect"


def test_summarize_ab_three_files_absolute():
    shapes = [Shape("s", {})]
    smp = _samples({"s": {"A": 10.0, "B": 9.0, "C": 11.0}}, reps=40)
    rows, score = E.summarize(smp, shapes, _p(reps=40), "A", _score)
    r = rows[0]
    assert [x["num"] for x in r["ratios"]] == ["B", "C"]
    assert r["ratios"][0]["verdict"] == "faster" and r["ratios"][1]["verdict"] == "slower"
    assert r["ratios"][0]["abs_delta_us"] == pytest.approx(-1.0, abs=0.1)
    assert score["per_file"]["B"] == pytest.approx(10 / 9, rel=0.02)
    assert E.overall_verdict(rows, score, 0.03) == "faster"


def test_render_lines_and_fidelity_line():
    from kdev.contract import Fidelity

    shapes = [Shape("s", {})]
    smp = _samples({"s": {"base": 20.0, "A": 10.0}})
    rows, score = E.summarize(smp, shapes, _p(), "base", _score)
    fid = Fidelity(toolchain={"so_notes": ["13.0.88"], "pin": "13.0.88", "match": True},
                   flush={"method": "write", "bytes": 256 << 20}, exclusive="ok", seeds=[1],
                   overhead_us={"subtracted": 4.0}, optouts=["flush=write"])
    lines = E.render_lines("time", "faster", rows, score, fid, ["base", "A"])
    assert lines[0].startswith("FASTER  time")
    assert len(lines) <= 40
    fl = lines[-1]
    assert "13.0.88=13.0.88" in fl and "WRITE-FLUSH 256MB" in fl and "optouts=FLUSH=WRITE" in fl
    assert "overhead=-4.00us" in fl


def test_files_for_orders_base_last_by_default(tmp_path):
    a = tmp_path / "a.py"
    a.write_text("x=1\n")
    b = tmp_path / "baseline.py"
    b.write_text("y=1\n")
    spec = JobSpec(kind="time", files=[], params={"base": "baseline"})
    from kdev.models import FileRef

    spec.files = [FileRef(str(a), "A")]
    files = E._files_for(spec, "time", lambda: str(b))
    assert [f.label for f in files] == ["A", "base"]
    spec.params = {"base": "baseline", "extra": {"base_first": True}}
    assert [f.label for f in E._files_for(spec, "time", lambda: str(b))] == ["base", "A"]
    assert files[0].md5 and files[1].md5
    spec_ab = JobSpec(kind="ab", files=[FileRef(str(a), "A"), FileRef(str(a), "B")], params={})
    assert [f.label for f in E._files_for(spec_ab, "ab", lambda: str(b))] == ["A", "B"]
    spec_nf = JobSpec(kind="time", files=[FileRef(str(a), "A")], params={"base": str(tmp_path / "nope.py")})
    from kdev.contract import KdevError

    with pytest.raises(KdevError):
        E._files_for(spec_nf, "time", lambda: None)


def test_optouts():
    assert E._optouts(_p()) == []
    assert E._optouts(_p(flush="write", graph=False, overhead="local", allow_jit=True)) == \
        ["flush=write", "eager", "overhead=local", "allow_jit"]
    assert E._optouts(_p(extra={"legacy_ext": True})) == ["legacy_ext"]


def test_find_problem_dir_and_direct_adapter_pure(tmp_path):
    pd = tmp_path / "prob"
    pd.mkdir()
    (pd / "reference.py").write_text("")
    (pd / "baseline.py").write_text("")
    from kdev.models import FileRef

    spec = JobSpec(kind="time", files=[FileRef(str(pd / "sub.py"), "A")])
    assert E.find_problem_dir(spec) == str(pd.resolve()) or E.find_problem_dir(spec) == str(pd)
    ad, pdir, src = E.resolve_adapter(spec)
    assert src == "direct" and ad.baseline_path().endswith("baseline.py")
    assert ad.input_variance() == {"routing": False}


def test_select_shapes_from_judge_json(fixtures_dir, tmp_path):
    from kdev.models import FileRef
    from kdev.timing._direct import DirectOkcAdapter

    pd = tmp_path / "muse_moe_w4a16"
    pd.mkdir()
    ad = DirectOkcAdapter(str(pd))
    assert ad.input_variance() == {"routing": True}
    spec = JobSpec(kind="time", files=[FileRef("x.py", "A", judge_json=str(fixtures_dir / "judge/judge_moe.json"))])
    sh = E.select_shapes(ad, spec, TimingParams())
    assert len(sh) == 6 and sh[0].seed is not None and sh[0].weight == 60 and sh[0].load == "decode 1"
    sh2 = E.select_shapes(ad, spec, TimingParams(shapes="verify"))
    assert len(sh2) == 3
    assert ad.shapes() == sh2


def test_result_envelope_valid_from_summary():
    from kdev.contract import Result

    shapes = [Shape("s", {})]
    rows, score = E.summarize(_samples({"s": {"base": 20.0, "A": 10.0}}), shapes, _p(), "base", _score)
    r = Result(tool="time", verdict="faster", shapes=rows, score=score)
    assert validate_result(r.to_dict()) == []
    import json

    json.loads(r.to_json())
