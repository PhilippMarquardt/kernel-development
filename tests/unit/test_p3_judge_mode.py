"""Judge mode (board #49): CLI defaults, ABBA process sequence, per-shape aggregation; board --problem filter
(no GPU)."""
from __future__ import annotations

import random

import pytest

from kdev.collab import board, db
from kdev.contract import KdevError
from kdev.models import Sample
from kdev.timing import judge_mode as JM
from kdev.timing.engine import _fallback_score

from test_p3_timing import _Ctx, _ns  # noqa: E402

from kdev.timing import commands as C


def _okc_dir(tmp_path):
    d = tmp_path / "muse_demo"
    d.mkdir()
    (d / "reference.py").write_text("")
    (d / "baseline.py").write_text("")
    a, b = d / "a.py", d / "b.py"
    a.write_text("")
    b.write_text("")
    return d, a, b


def _jns(**kw):
    base = dict(reps=None, judge_mode=False, fast=False, gate_us=None, fresh=None, first=None, no_pad=False,
                judge_json=None, rounds=None)
    base.update(kw)
    return _ns(**base)


def test_proc_sequence():
    assert JM.proc_sequence(["A", "B"]) == ["A", "B", "B", "A"]
    assert JM.proc_sequence(["A", "B"], 2) == ["A", "B", "B", "A"] * 2
    assert JM.proc_sequence(["A", "B", "C"]) == ["A", "B", "C", "C", "B", "A"]


def test_ab_defaults_to_judge_mode_on_okc(tmp_path):
    d, a, b = _okc_dir(tmp_path)
    ctx = _Ctx()
    C.cmd_ab(_jns(files=[str(a), str(b)]), ctx)
    p = ctx.got[0][0].params
    ex = p["extra"]
    assert ex["judge_mode"] is True and p["reps"] == JM.REPS and ex["fresh"] == JM.FRESH
    assert ex["gate_cycles"] == JM.gate_cycles(JM.GATE_US) and ex["prepare_pad"] is True
    assert ex["first"] == "baseline" and ex["base_first"] is True
    ctx = _Ctx()
    C.cmd_ab(_jns(files=[str(a), str(b)], fast=True), ctx)
    p = ctx.got[0][0].params
    assert not (p.get("extra") or {}).get("judge_mode") and p["reps"] == 40
    ctx = _Ctx()
    C.cmd_ab(_jns(files=[str(a), str(b)], gate_us=0.0, fresh=1, first="submission", no_pad=True, reps=80,
                  rounds=2), ctx)
    p = ctx.got[0][0].params
    assert p["reps"] == 80 and p["extra"]["gate_cycles"] == 0 and p["extra"]["fresh"] == 1
    assert p["extra"]["base_first"] is False and p["extra"]["prepare_pad"] is False and p["extra"]["rounds"] == 2
    with pytest.raises(KdevError):
        C.cmd_ab(_jns(files=[str(a), str(b)], judge_mode=True, fast=True), _Ctx())
    with pytest.raises(KdevError):
        C.cmd_ab(_jns(files=[str(a), str(b)], flush="write"), _Ctx())


def test_ab_non_okc_stays_fast(tmp_path):
    a, b = tmp_path / "a.py", tmp_path / "b.py"
    a.write_text("")
    b.write_text("")
    ctx = _Ctx()
    C.cmd_ab(_jns(files=[str(a), str(b)]), ctx)
    assert not (ctx.got[0][0].params.get("extra") or {}).get("judge_mode")


def test_time_judge_mode_flag(tmp_path, fixtures_dir):
    d, a, _ = _okc_dir(tmp_path)
    ctx = _Ctx()
    C.cmd_time(_jns(file=str(a)), ctx)
    assert not (ctx.got[0][0].params.get("extra") or {}).get("judge_mode")
    ctx = _Ctx()
    j = fixtures_dir / "judge" / "judge_moe.json"
    C.cmd_time(_jns(file=str(a), judge_mode=True, judge_json=str(j)), ctx)
    p = ctx.got[0][0].params
    assert p["extra"]["judge_mode"] and p["base"] == "baseline" and p["extra"]["seeds_from_judge"].endswith(".json")
    with pytest.raises(KdevError):
        C.cmd_time(_jns(file=str(a), judge_mode=True, base="none"), _Ctx())


def _proc(label, sub_us, base_us=20.0, ovh=4.0, reps=60, noise=0.05, seed=0, shapes=("s1", "s2")):
    rng = random.Random(seed)
    smp = []
    for sid in shapes:
        for r in range(reps):
            order = ["base", label] if r % 2 == 0 else [label, "base"]
            for pos, lb in enumerate(order):
                us = (base_us if lb == "base" else sub_us[sid]) + rng.gauss(0, noise)
                smp.append(Sample(shape_id=sid, label=lb, seed=1, rep=r, order_pos=pos, us_raw=us + ovh, us=us))
    return label, JM.proc_shape_data(smp, label)


def test_aggregate_abba_deltas_and_verdict():
    from kdev.adapters.base import Shape

    shapes = [Shape("s1", {}, weight=1.0, load="L"), Shape("s2", {}, weight=1.0, load="L")]
    a = {"s1": 10.0, "s2": 10.0}
    b = {"s1": 9.0, "s2": 10.0}
    procs = [_proc("A", a, seed=1), _proc("B", b, seed=2), _proc("B", b, seed=3), _proc("A", a, seed=4)]
    rows, score, verdict = JM.aggregate(procs, ["A", "B"], shapes, _fallback_score(shapes), 0.01, n_boot=200)
    r1 = next(r for r in rows if r["shape_id"] == "s1")
    r2 = next(r for r in rows if r["shape_id"] == "s2")
    assert r1["ratio"]["delta_raw_us"] == pytest.approx(-1.0, abs=0.05)
    assert len(r1["ratio"]["delta_raw_halves_us"]) == 2 and all(h < -0.9 for h in r1["ratio"]["delta_raw_halves_us"])
    assert r1["verdict"] == "faster" and r2["verdict"] == "same"
    assert r1["ratio"]["speedup"] == pytest.approx(10 / 9, rel=0.01)
    assert score["delta"]["median"] == pytest.approx(20 / 19, rel=0.01) and verdict == "faster"
    assert score["vs_base"]["per_file"]["A"] == pytest.approx(2.0, rel=0.01)
    lines = JM.render_ab(verdict, rows, score, ["A", "B"], JM.proc_sequence(["A", "B"]))
    assert lines[0].startswith("FASTER  ab --judge-mode (A B B A processes)") and len(lines) == 5


def test_aggregate_disagreeing_halves_is_same():
    from kdev.adapters.base import Shape

    shapes = [Shape("s1", {}, load="L")]
    procs = [_proc("A", {"s1": 10.0}, shapes=("s1",), seed=1), _proc("B", {"s1": 9.0}, shapes=("s1",), seed=2),
             _proc("B", {"s1": 11.0}, shapes=("s1",), seed=3), _proc("A", {"s1": 10.0}, shapes=("s1",), seed=4)]
    rows, _, verdict = JM.aggregate(procs, ["A", "B"], shapes, _fallback_score(shapes), 0.01, n_boot=100)
    assert rows[0]["verdict"] == "same" and verdict == "same"


# ------------------------------------------------------------------------------------- board --problem
def test_board_problem_filter_aliases_and_unscoped(tmp_path):
    c = db.connect(tmp_path / "collab.db")
    try:
        board.post(c, agent="coord", kind="ask", body="kdev feature", problem="kdev")
        board.post(c, agent="bfa", kind="finding", body="bf16 scoped", problem="muse_bf16_proj")
        board.post(c, agent="bfa", kind="finding", body="bf16 unscoped legacy")
        board.post(c, agent="coord", kind="decision", body="global rule")
        board.post(c, agent="newbie", kind="finding", body="nobody knows")
        board.post(c, agent="x", kind="finding", body="suffix trap", problem="proj")
        kd = {p.body for p in board.read(c, problem="kdev")[0]}
        assert kd == {"kdev feature", "global rule", "nobody knows"}
        bf = {p.body for p in board.read(c, problem="bf16_proj")[0]}
        assert bf == {"bf16 scoped", "bf16 unscoped legacy", "global rule", "nobody knows"}
        assert {p.body for p in board.read(c, problem="/root/okc/muse_bf16_proj/muse_bf16_proj")[0]} == bf
        assert {p.body for p in board.read(c, problem="proj")[0]} >= {"suffix trap"}
        assert "bf16 scoped" not in {p.body for p in board.read(c, problem="proj")[0]}
        assert {p.body for p in board.search(c, "feature", problem="kdev")} == {"kdev feature"}
    finally:
        c.close()


def test_infer_problem(monkeypatch):
    monkeypatch.delenv("KDEV_PROBLEM", raising=False)
    assert board.infer_problem(["/root/okc/muse_bf16_proj/muse_bf16_proj/pp/x.py"], cwd="/root") == "muse_bf16_proj"
    assert board.infer_problem([], cwd="/root/okc/muse_moe_w4a16/muse_moe_w4a16") == "muse_moe_w4a16"
    assert board.infer_problem([], cwd="/root") is None
    monkeypatch.setenv("KDEV_PROBLEM", "kdev")
    assert board.infer_problem([], cwd="/root/okc/muse_moe_w4a16/x") == "kdev"
