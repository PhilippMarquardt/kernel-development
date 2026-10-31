"""P3: judge JSON parsing, calibration comparison and the DISAGREE rule (fixtures from OKC)."""
from __future__ import annotations

import json

import pytest

from kdev.timing import calib


def _load(fixtures_dir, rel):
    return json.loads((fixtures_dir / rel).read_text(encoding="utf-8"))


def test_parse_judge_moe(judge_json):
    jr = calib.parse_judge_json(judge_json)
    assert jr.judge_id.startswith("2bbe8248")
    assert jr.node and jr.cluster
    assert len(jr.shapes) == 6
    s = jr.shapes[0]
    assert s.shape_id == "decode 1/bs1"
    assert "load" not in s.case and "weight" not in s.case and "seed" in s.case
    assert s.weight == 60 and s.load == "decode 1" and s.seed == s.case["seed"]
    assert s.sub_us == pytest.approx(s.base_us / s.speedup)
    assert s.base_us == pytest.approx(77.104, abs=0.01)
    assert s.overhead_us == pytest.approx(3.808, abs=1e-3)
    assert s.runs == 100
    assert jr.speedup_ci and len(jr.speedup_ci) == 2


def test_parse_judge_ids_unique(fixtures_dir):
    for rel in ("judge/judge_attention.json", "timing/judge_bf16.json", "timing/judge_prefill.json",
                "timing/judge_decode.json"):
        jr = calib.parse_judge_json(_load(fixtures_dir, rel))
        ids = [s.shape_id for s in jr.shapes]
        assert len(ids) == len(set(ids)), rel
    bf = calib.parse_judge_json(_load(fixtures_dir, "timing/judge_bf16.json"))
    assert bf.shapes[0].shape_id == "decode 1/qkv_gate/m1"
    assert len(bf.shapes) == 30


def test_parse_matches_calib_py_output(fixtures_dir):
    """Our judge us/speedups equal the judge columns calib.py wrote (cal_bf16.json, cal_moe.json)."""
    for jrel, crel in (("timing/judge_bf16.json", "timing/cal_bf16.json"),
                       ("judge/judge_moe.json", "judge/cal_moe.json")):
        jr = calib.parse_judge_json(_load(fixtures_dir, jrel))
        cal = _load(fixtures_dir, crel)["rows"]
        assert len(cal) == len(jr.shapes)
        for s, row in zip(jr.shapes, cal):
            assert s.speedup == pytest.approx(row["judge_sp"][0], rel=1e-9)
            assert s.sub_us == pytest.approx(row["judge_us"][0], rel=1e-9)
            assert s.base_us == pytest.approx(row["judge_base_us"], rel=1e-9)


def test_compare_flags(judge_json):
    jr = calib.parse_judge_json(judge_json)
    loc = {s.shape_id: s.speedup for s in jr.shapes}
    loc[jr.shapes[0].shape_id] *= 1.24
    rows = calib.compare(loc, jr)
    assert rows[0]["flag"] == "out_of_band" and rows[0]["local_over_judge"] == pytest.approx(1.24)
    assert all(r["flag"] == "ok" for r in rows[1:])


def test_ab_disagree_rule():
    assert calib.ab_disagree(1.05, 0.95)              # direction flip > 2 %
    assert not calib.ab_disagree(1.01, 0.995)         # flip but within 2 % and gap < 3 %
    assert calib.ab_disagree(1.10, 1.05)              # same direction, gap 4.8 % > 3 %
    assert not calib.ab_disagree(1.04, 1.03)


def test_calib_rows_and_markdown(judge_json):
    jr = calib.parse_judge_json(judge_json)
    rows = []
    for s in jr.shapes:
        rows.append({"shape_id": s.shape_id,
                     "per_file": [{"label": "base", "median_us": s.base_us * 1.01},
                                  {"label": "A", "median_us": s.sub_us}, {"label": "B", "median_us": s.sub_us}],
                     "ratios": [{"num": "A", "den": "base", "speedup": s.speedup * 1.01},
                                {"num": "B", "den": "base", "speedup": s.speedup * 0.9}]})
    judges = {"A": jr, "B": jr}
    cr = calib.calib_rows(rows, judges, ["A", "B"], "base", jr)
    assert len(cr) == 6
    f = cr[0]["files"]
    assert f["A"]["local_over_judge"] == pytest.approx(1.01) and f["A"]["flag"] == "ok"
    assert f["B"]["local_ab"] == pytest.approx(0.9 / 1.01) and f["B"]["judge_ab"] == pytest.approx(1.0)
    assert f["B"]["ab_flag"] == "DISAGREE" and f["B"]["flag"] == "DISAGREE"
    md = calib.render_calib_md(cr, ["A", "B"], "# t")
    assert "| decode 1/bs1 |" in md and "**" in md


def test_judge_only_no_files_errors(tmp_path):
    from kdev.contract import KdevError
    from kdev.models import FileRef, JobSpec

    spec = JobSpec(kind="calib", files=[FileRef(str(tmp_path / "a.py"), "A")])
    with pytest.raises(KdevError):
        calib._parse_files(spec)
