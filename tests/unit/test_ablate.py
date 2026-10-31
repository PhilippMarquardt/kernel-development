"""kdev ablate: variant plan (explicit / knob / hook), budget table from a fake ab Result. No GPU."""
from __future__ import annotations

import json
from argparse import Namespace

import pytest

from kdev.ablate import budget as B
from kdev.ablate import commands as C
from kdev.contract import KdevError, Result
from kdev.registry import Context


def _row(sid, us, weight=1.0, ovh=4.0):
    return {"shape_id": sid, "weight": weight,
            "per_file": [{"label": lb, "median_raw_us": v, "median_us": v - ovh, "overhead_us": ovh}
                         for lb, v in us.items()] + [{"label": "base", "median_raw_us": 30.0}],
            "ratios": [{"num": lb, "den": "A", "verdict": "faster" if us["A"] - v > 0.2 else "same"}
                       for lb, v in us.items() if lb != "A"]}


def _ns(**kw):
    d = dict(file=None, component=None, variant=None, set=None, hook=None, control=None, floor=None, rounds=None,
             fast=False, no_build=False, dry_run=False, from_run=None, map=None)
    d.update(kw)
    return Namespace(**d)


def test_budget_saved_excess_remainder_skeleton_noise():
    rows = [_row("d32/qkv", {"A": 14.20, "B": 13.81, "C": 14.14, "D": 13.18, "E": 14.25}, weight=2.0)]
    labels = {"B": "x_loads", "C": "mma", "D": "stream", "E": "control"}
    b = B.budget(rows, "A", labels, floors={"d32/qkv": 7.91})
    r = b["shapes"][0]
    assert r["saved"]["x_loads"] == pytest.approx(0.39)
    assert r["saved"]["mma"] == pytest.approx(0.06) and r["noise"]["mma"] and not r["noise"]["x_loads"]
    assert r["excess_us"] == pytest.approx(14.20 - 4.0 - 7.91)
    assert r["sum_singles"] == pytest.approx(0.45)                       # stream is not a single
    assert r["remainder_us"] == pytest.approx(r["excess_us"] - 0.45)
    assert r["compute_us"] == pytest.approx(1.02) and r["skeleton_us"] == pytest.approx(13.18 - 4.0 - 7.91)
    assert "control" not in b["components"] and b["totals"]["x_loads"] == pytest.approx(0.78)
    lines = B.render(b)
    assert lines[0].split()[:3] == ["shape", "base", "excess"] and "+0.06~" in lines[1]
    assert lines[-2].startswith("ranking") and lines[-2].index("x_loads") < lines[-2].index("mma")
    md = B.markdown(b, "m1.py")
    assert "| d32/qkv | 2 | 14.20 |" in md and "B=x_loads" in md


def test_budget_without_floor_or_raw():
    rows = [{"shape_id": "s", "per_file": [{"label": "A", "median_us": 10.0}, {"label": "B", "median_us": 9.5}]}]
    b = B.budget(rows, "A", {"B": "epilogue"})
    r = b["shapes"][0]
    assert r["saved"]["epilogue"] == pytest.approx(0.5) and r["excess_us"] is None and r["overhead_us"] == 0.0
    assert "excess" not in B.render(b)[0]


def test_plan_explicit_knob_and_hook(tmp_path):
    base = tmp_path / "m1.py"
    base.write_text("x = 1\n")
    v = tmp_path / "a_nox.py"
    v.write_text("x = 2\n")
    (tmp_path / "kdev_ablate.py").write_text(
        "import os\n"
        "def ablate(base, component, out_dir):\n"
        "    if component == 'epilogue':\n"
        "        p = os.path.join(out_dir, 'a_noep.py'); open(p, 'w').write('ep'); return p\n"
        "    if component == 'control':\n"
        "        return {'path': base, 'knobs': {'XS_ABL': 0}}\n"
        "    return None\n")
    pl = C.plan(_ns(file=str(base), component=["nox,epilogue"], variant=[f"x_loads={v}"], set=["mma:XS_ABL=4"]))
    by = {x["component"]: x for x in pl["variants"]}
    assert [x["component"] for x in pl["variants"]] == ["x_loads", "epilogue", "mma", "control"]
    assert by["x_loads"]["source"] == "--variant" and by["mma"]["knobs"] == {"XS_ABL": 4}
    assert by["epilogue"]["path"].endswith("a_noep.py") and by["control"]["knobs"] == {"XS_ABL": 0}
    assert pl["labels"] == {"B": "x_loads", "C": "epilogue", "D": "mma", "E": "control"}
    with pytest.raises(KdevError):
        C.plan(_ns(file=str(base), component=["decode"]))                 # hook returns None -> missing


def test_dry_run_and_from_run(tmp_path):
    base = tmp_path / "m1.py"
    base.write_text("x\n")
    r = C.cmd_ablate(_ns(file=str(base), set=["mma:NO_MMA=1"], dry_run=True), Context())
    assert r.verdict == "plan" and any("mma" in ln and "NO_MMA=1" in ln for ln in r.lines)
    run = tmp_path / "run1"
    run.mkdir()
    res = Result(tool="ab", status="ok", verdict="faster", run_id="run1",
                 shapes=[_row("v1/qkv", {"A": 13.48, "B": 13.12})])
    (run / "result.json").write_text(json.dumps(res.to_dict()))
    out = C.cmd_ablate(_ns(from_run=str(run), map=["B=epilogue,C=mma"], floor=["v1/qkv=7.91"]), Context())
    assert out.verdict == "ok" and (run / "budget.md").is_file() and (run / "budget.json").is_file()
    assert out.summary["budget"]["shapes"][0]["saved"]["epilogue"] == pytest.approx(0.36)


def test_registered():
    from kdev.registry import build_registry

    assert "ablate" in build_registry().names()
