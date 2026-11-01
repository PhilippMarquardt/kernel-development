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


def test_render_ab_three_files_rows_for_each_vs_a():
    """board #90: ab with 3+ files prints every X/A per shape, not only B/A."""
    from kdev.adapters.base import Shape

    shapes = [Shape("s1", {}, weight=1.0, load="L"), Shape("s2", {}, weight=1.0, load="L")]
    a, b, c = {"s1": 10.0, "s2": 10.0}, {"s1": 9.0, "s2": 10.0}, {"s1": 10.0, "s2": 12.0}
    labels = ["A", "B", "C"]
    seq = JM.proc_sequence(labels)
    data = {"A": a, "B": b, "C": c}
    procs = [_proc(lb, data[lb], seed=i) for i, lb in enumerate(seq)]
    rows, score, verdict = JM.aggregate(procs, labels, shapes, _fallback_score(shapes), 0.01, n_boot=100)
    lines = JM.render_ab(verdict, rows, score, labels, seq)
    assert any(ln.strip().startswith("B/A score") for ln in lines)
    assert any(ln.strip().startswith("C/A score") for ln in lines)
    s1 = next(ln for ln in lines if ln.startswith("s1"))
    s2 = next(ln for ln in lines if ln.startswith("s2"))
    assert s1.split()[2:].count("+") == 1 and s1.split()[-1] == "=" and "x1.11" in s1
    assert s2.split()[-1] == "-" and "x0.83" in s2


def test_render_lines_fast_three_files():
    from kdev.timing.engine import render_lines

    rt = lambda num, sp, v: {"num": num, "den": "A", "speedup": sp, "speedup_ci": [sp, sp], "verdict": v}  # noqa: E731
    rows = [{"shape_id": "s1", "per_file": [{"label": x, "median_us": 10.0} for x in "ABC"],
             "ratio": rt("B", 1.1, "faster"), "ratios": [rt("B", 1.1, "faster"), rt("C", 0.9, "slower")],
             "verdict": "faster"}]
    lines = render_lines("ab", "faster", rows, None, None, ["A", "B", "C"])
    assert "B x1.100+ C x0.900-" in lines[-1]


# ------------------------------------------------------------------ routing seeds (decode agent bug, 2026-11-01)
class _FakeTimer:
    made: list[int] = []

    def __init__(self, adapter, modules, states, shape, seed, params, **kw):
        self.seed = seed
        self._outs = {}
        _FakeTimer.made.append(seed)

    def replay_us(self, lb):
        return 10.0 + (self.seed % 7) * 0.01

    def verify(self, lb):
        return {"ok": True}

    def close(self):
        pass


class _FakeAdapter:
    def __init__(self, routing):
        self.routing = routing

    def input_variance(self):
        return {"routing": self.routing}

    def judge_seed(self, shape):
        return 401

    def make_inputs(self, shape, seed):
        return {"seed": seed}


def _judge_session(tmp_path, monkeypatch, *, seeds, routing):
    import sys
    import types
    from types import SimpleNamespace as NS

    from kdev.timing import engine as E
    from kdev.timing import graph as G
    from kdev.timing.engine import Session

    fake_torch = types.ModuleType("torch")
    fake_torch.zeros = lambda *a, **k: NS(add_=lambda x: None)
    fake_torch.cuda = NS(empty_cache=lambda: None)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(G, "capture", lambda fn: object())
    monkeypatch.setattr(G, "time_replay", lambda g, fl: 2.0)
    monkeypatch.setattr(E, "TorchTimer", _FakeTimer)
    monkeypatch.setattr(E, "_tensors", lambda d: [])
    monkeypatch.setattr(JM, "_record_placement", lambda *a, **k: None)
    _FakeTimer.made = []
    params = E.TimingParams(reps=20, seeds=seeds, extra={"judge_mode": True})
    ss = Session(spec=None, params=params, run_dir=str(tmp_path), files=[NS(label="A"), NS(label="base")],
                 adapter=_FakeAdapter(routing), modules={"A": None, "base": None})
    ss.shapes = [NS(shape_id="m1", seed=None)]
    ss.seeds_used, ss.samples, ss.checks, ss.check_msgs = {}, [], {}, []
    ss.log = lambda m: None
    ss.error = None
    return ss


@pytest.mark.parametrize("seeds,routing,n", [(5, False, 5), (None, True, 5), (None, False, 1), (1, True, 1)])
def test_judge_mode_honours_seeds(tmp_path, monkeypatch, seeds, routing, n):
    ss = _judge_session(tmp_path, monkeypatch, seeds=seeds, routing=routing)
    JM.time_shapes_judge(ss)
    assert ss.error is None
    used = ss.seeds_used["m1"]
    assert len(used) == n == len(set(used)) and used[0] == 401
    assert sorted(set(_FakeTimer.made)) == sorted(used)
    by_seed = {}
    for s in ss.samples:
        by_seed.setdefault(s.seed, set()).add((s.label, s.order_pos))
    assert set(by_seed) == set(used)                                   # every seed actually timed
    for v in by_seed.values():                                         # ... in both orders (ABBA within a seed)
        assert v == {("A", 0), ("A", 1), ("base", 0), ("base", 1)}
    assert ss.shape_overhead["m1"]["seeds"] == n
