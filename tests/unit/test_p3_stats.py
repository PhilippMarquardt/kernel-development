"""P3: pure statistics, schedules, seeds and overhead -- including exact reproduction of the judge's
paired-ratio, overhead and score formulas from real `okc status --json` records."""
from __future__ import annotations

import json
import math
import statistics

import pytest

from kdev.models import Interval
from kdev.timing import overhead, schedule, seeds, stats

JUDGES = ["judge/judge_moe.json", "judge/judge_attention.json", "timing/judge_bf16.json",
          "timing/judge_decode.json", "timing/judge_prefill.json"]


def _load(fixtures_dir, rel):
    return json.loads((fixtures_dir / rel).read_text(encoding="utf-8"))


def _bench(d):
    return [x for x in (d.get("result") or d)["records"] if x.get("event") == "benchmark"]


# ------------------------------------------------------------------------------------------- schedule

def test_abba_order():
    assert schedule.order_for_rep(2, 0) == [0, 1]
    assert schedule.order_for_rep(2, 1) == [1, 0]
    assert schedule.order_for_rep(3, 3) == [2, 1, 0]
    assert schedule.order_for_rep(0, 0) == []


def test_rotate_order():
    assert schedule.order_for_rep(3, 0, "rotate") == [0, 1, 2]
    assert schedule.order_for_rep(3, 1, "rotate") == [1, 2, 0]
    assert schedule.order_for_rep(3, 5, "rotate") == [2, 0, 1]
    with pytest.raises(ValueError):
        schedule.order_for_rep(2, 0, "zigzag")


def test_full_schedule_warmup_not_recorded():
    s = schedule.full_schedule(2, 3, "abba", warmup=1)
    assert s[:2] == [(-1, 0, 0), (-1, 1, 1)]
    assert [x for x in s if x[0] >= 0] == [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0), (2, 0, 0), (2, 1, 1)]
    assert schedule.first_index(2, 1) == 1


# ---------------------------------------------------------------------------------------------- seeds

def test_seed_policy():
    assert seeds.default_seed_count({"routing": True}) == 5
    assert seeds.default_seed_count({"routing": False}) == 1
    assert seeds.default_seed_count({}) == 1
    s = seeds.seeds_for_shape(104258936029151888, 3)
    # calib_seeds.py: (seed + j*0x2545F4914F6CDD1D) % (2**63 - 1), j=0 unchanged
    assert s[0] == 104258936029151888
    assert s[1] == (104258936029151888 + 0x2545F4914F6CDD1D) % (2**63 - 1)
    assert s[2] == (104258936029151888 + 2 * 0x2545F4914F6CDD1D) % (2**63 - 1)
    assert seeds.seeds_for_shape(None, 1) == [11]


# ---------------------------------------------------------------------------------------------- stats

def test_median_and_quantile():
    assert stats.median([3, 1, 2]) == 2
    assert stats.median([4, 1, 3, 2]) == 2.5
    assert math.isnan(stats.median([]))
    assert stats.quantile([1, 2, 3, 4, 5], 0.5) == 3
    assert stats.quantile([0, 10], 0.25) == 2.5


def test_paired_ratios_unpaired_and_paired():
    sub = [10.0, 10.0, 12.0, 8.0]
    base = [20.0, 30.0, 24.0, 24.0]
    assert stats.paired_ratios(sub, base) == [2.0, 3.0, 2.0, 3.0]
    pr = stats.paired_ratios(sub, base, order=[0, 1, 0, 1])
    assert pr == pytest.approx([math.sqrt(6), math.sqrt(6)])
    # not order-swapped consecutive reps are skipped (re-sync)
    assert len(stats.paired_ratios(sub, base, order=[0, 0, 1, 0])) == 1
    with pytest.raises(ValueError):
        stats.paired_ratios([1.0], [1.0, 2.0])


@pytest.mark.parametrize("rel", JUDGES)
def test_judge_paired_ratio_formula_exact(fixtures_dir, rel):
    """Judge: r_i = (base_raw_i - ovh)/(sub_raw_i - ovh), ovh = 2*one - two; speedup = median of the
    geomean of consecutive (order-swapped) pairs. Reproduced to 1e-7 on every benchmark record."""
    for rec in _bench(_load(fixtures_dir, rel)):
        ro = rec["replay_overhead"]
        assert ro["overhead_ns"] == pytest.approx(2 * ro["one_ns"] - ro["two_ns"], abs=1e-3)
        ov = ro["overhead_ns"]
        sub = [x - ov for x in rec["runs_ns"]]
        base = [x - ov for x in rec["baseline_runs_ns"]]
        assert stats.paired_ratios(sub, base) == pytest.approx(rec["ratios"], rel=1e-4)
        order = [i % 2 for i in range(len(sub))]
        sp = stats.median(stats.paired_ratios(sub, base, order))
        assert sp == pytest.approx(rec["paired_ratio"], rel=1e-7)
        assert rec["baseline_ns"] == pytest.approx(rec["baseline_raw_ns"] - ov, abs=1e-3)
        assert rec["median_ns"] == pytest.approx(rec["baseline_ns"] / rec["paired_ratio"], rel=1e-7)
        ov_obj = overhead.from_judge_record(rec)
        assert ov_obj.subtracted == pytest.approx(ov / 1000, abs=1e-6)


@pytest.mark.parametrize("rel", JUDGES)
def test_judge_score_formula_exact(fixtures_dir, rel):
    from kdev.timing.calib import judge_score, parse_judge_json

    d = _load(fixtures_dir, rel)
    jr = parse_judge_json(d)
    assert judge_score(jr) == pytest.approx(d["result"]["speedup"], rel=1e-9)
    lo, hi = d["result"]["speedup_ci"]
    s = d["result"]["speedup"]
    assert lo <= s * (1 - stats.JUDGE_MIN_CI_HALF) + 1e-9 and hi >= s * (1 + stats.JUDGE_MIN_CI_HALF) - 1e-9


def test_bootstrap_ci_reproducible_and_contains_median():
    xs = [1.0 + 0.01 * ((i * 7) % 13 - 6) for i in range(40)]
    a = stats.bootstrap_ci(xs, n_boot=2000, seed=3)
    b = stats.bootstrap_ci(xs, n_boot=2000, seed=3)
    assert a == b
    assert a.ci_lo <= a.median <= a.ci_hi
    assert a.median == statistics.median(xs)
    assert a.n == 40
    one = stats.bootstrap_ci([2.0])
    assert one.ci_lo == one.ci_hi == 2.0
    assert math.isnan(stats.bootstrap_ci([]).median)


def test_bootstrap_ci_width_shrinks_with_n():
    import random

    rng = random.Random(1)
    small = [1 + rng.gauss(0, 0.05) for _ in range(10)]
    big = [1 + rng.gauss(0, 0.05) for _ in range(400)]
    w = lambda iv: iv.ci_hi - iv.ci_lo  # noqa: E731
    assert w(stats.bootstrap_ci(big, n_boot=1000)) < w(stats.bootstrap_ci(small, n_boot=1000))


def test_score_ci_propagates_and_widens():
    per = {"a": [2.0, 2.1, 1.9, 2.0], "b": [1.0, 1.05, 0.95, 1.0]}
    geo = lambda d: math.sqrt(d["a"] * d["b"])  # noqa: E731
    iv = stats.score_ci(per, geo, n_boot=500, seed=1)
    assert iv.median == pytest.approx(math.sqrt(2.0))
    assert iv.ci_lo <= iv.median <= iv.ci_hi
    wide = stats.score_ci({"a": [2.0] * 4}, lambda d: d["a"], n_boot=50, min_half_width=0.015)
    assert wide.ci_lo == pytest.approx(2.0 * 0.985) and wide.ci_hi == pytest.approx(2.0 * 1.015)


def test_verdict_rules():
    assert stats.verdict(Interval(1.10, 1.06, 1.14)) == "faster"
    assert stats.verdict(Interval(0.90, 0.86, 0.94)) == "slower"
    assert stats.verdict(Interval(1.02, 1.01, 1.03)) == "same"          # CI excludes 1 but < 3 % floor
    assert stats.verdict(Interval(1.02, 1.01, 1.03), noise_floor=0.01) == "faster"
    assert stats.verdict(Interval(1.10, 0.98, 1.20)) == "same"          # CI includes 1
    assert stats.verdict(Interval(1.05, 1.01, 1.09), combined_se=0.03) == "same"  # needs >= 2 SE = 6 %
    assert stats.verdict(Interval(1.5, 1.4, 1.6), suspect=True) == "suspect"


def test_max_ulp():
    assert stats.max_ulp([1.0, 2.0], [1.0, 2.0]) == 0
    import struct

    nxt = struct.unpack("<f", struct.pack("<i", struct.unpack("<i", struct.pack("<f", 8.0))[0] + 1))[0]
    assert stats.max_ulp([8.0], [nxt]) == 1
    assert stats.max_ulp([-0.0], [0.0]) == 0
    assert stats.max_ulp([float("nan")], [1.0]) == 2**31
    with pytest.raises(ValueError):
        stats.max_ulp([1.0], [])


# ------------------------------------------------------------------------------------------- overhead

def test_overhead_modes():
    o = overhead.Overhead(one_kernel_graph=5.0, two_kernel_graph=6.1, empty_graph=4.4)
    assert o.subtracted == 4.0
    o.mode = "local"
    assert o.subtracted == pytest.approx(3.9)
    o.mode = "none"
    assert o.subtracted == 0.0
    assert overhead.Overhead(mode="local").subtracted == 4.0          # nothing measured -> constant
    assert overhead.Overhead(mode="local", empty_graph=4.2).subtracted == 4.2
    d = overhead.Overhead(one_kernel_graph=5.0, two_kernel_graph=6.0).to_dict()
    assert d["local_estimate"] == 4.0 and d["subtracted"] == 4.0 and d["judge_const"] == 4.0
    with pytest.raises(ValueError):
        _ = overhead.Overhead(mode="bogus").subtracted
