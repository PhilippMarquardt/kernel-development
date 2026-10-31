"""Calibration table: judge/local ratios per shape from calib, applied by time/ab (P3, calibtab)."""
from __future__ import annotations

import pytest

from kdev.adapters.base import Shape
from kdev.timing import calibtab
from kdev.timing.engine import render_lines


def _obs(key, role, loc, jud, run="r1", judge="j1"):
    return {"key": key, "shape_id": "decode 1/router/m1", "role": role, "local_us": loc, "judge_us": jud,
            "run_id": run, "judge_id": judge}


def test_table_roundtrip_estimate_and_small_flag(tmp_path):
    case = {"layer": "router", "m": 1, "seed": 7}
    key = calibtab.case_key(case)
    assert key == calibtab.case_key({"m": 1, "layer": "router", "seed": 99, "load": "decode 1"})
    tab = calibtab.load("p", str(tmp_path))
    assert calibtab.add_observations(tab, [_obs(key, "sub", 4.0, 3.4), _obs(key, "base", 11.25, 10.2)]) == 2
    calibtab.add_observations(tab, [_obs(key, "sub", 4.0, 3.4)])          # same run+judge: replaced
    calibtab.add_observations(tab, [_obs(key, "sub", 4.0, 3.5, run="r2")])
    calibtab.save(tab, "p", str(tmp_path))
    tab = calibtab.load("p", str(tmp_path))
    assert len(tab["shapes"][key]["sub"]) == 2
    e = calibtab.estimate(tab, case, "sub", 3.9)
    assert e["judge_est_us"] == pytest.approx(3.9 * 0.8625) and e["small"] and e["n"] == 2
    assert e["unc"] > 0.07                                                   # small-shape term included
    b = calibtab.estimate(tab, case, "base", 11.0)
    assert b["ratio"] == pytest.approx(10.2 / 11.25) and not b["small"] and b["unc"] == calibtab.JUDGE_NOISE
    assert calibtab.estimate(tab, {"layer": "o", "m": 1}, "sub", 5.0) is None
    lines = calibtab.table_lines(tab)
    assert "small" in lines[1] and "gap>5%" in lines[1]


def test_annotate_and_render_column(tmp_path):
    case = {"layer": "router", "m": 1}
    tab = calibtab.load("p", str(tmp_path))
    calibtab.add_observations(tab, [_obs(calibtab.case_key(case), "sub", 10.0, 9.0)])
    calibtab.save(tab, "p", str(tmp_path))
    rows = [{"shape_id": "d1/router", "per_file": [{"label": "A", "median_us": 10.0}, {"label": "base", "median_us": 20.0}],
             "ratio": None, "verdict": "faster"}]
    info = calibtab.annotate(rows, [Shape("d1/router", dict(case))], "p", "base", str(tmp_path))
    assert info["shapes_calibrated"] == 1 and rows[0]["per_file"][0]["judge_est"]["judge_est_us"] == pytest.approx(9.0)
    assert "judge_est" not in rows[0]["per_file"][1]                          # no base observation
    out = render_lines("time", "faster", rows, None, None, ["A", "base"])
    assert "~J A" in out[1] and "9.00" in out[2]
    assert calibtab.annotate(rows, [], "nope", "base", str(tmp_path)) is None
