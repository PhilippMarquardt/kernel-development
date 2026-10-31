"""P6: timeline (instrument/decode/trace), floor (profiles, roofline, skeleton, sensitivity). No GPU."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

import kdev
from kdev.contract import KdevError
from kdev.registry import build_registry

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

from kdev.floor import gpus
from kdev.floor.commands import floor_rows
from kdev.floor.roofline import geomean_sensitivity, roofline
from kdev.floor.skeleton import skeleton
from kdev.timeline import decode, instrument, trace
from kdev.timeline.decode import MarkRec


def test_p6_commands_registered():
    assert {"timeline", "floor", "floor calibrate", "regs", "profile", "dispatch"} <= set(build_registry().names())


def test_h200_profile_parses():
    p = Path(kdev.__file__).parent / "data" / "gpus" / "h200.toml"
    d = tomllib.loads(p.read_text(encoding="utf-8"))
    assert d["schema"] == "kdev.gpu/1" and d["sm_count"] == 132
    assert d["measured"]["graph_fixed_us"]["value"] == 4.0


# ---------------------------------------------------------------- GPU profiles

def test_load_profile_and_accessors():
    p = gpus.load_profile("H200")
    assert p.sm_count == 132 and p.flush_bytes == 256 << 20 and p.globaltimer_ns == [32, 256]
    assert p.peak["dram_tbps"] == 4.8 and p.const("stream_tbps") == 4.11
    assert p.measured["dram_trip_1cta_us"].source.startswith("floor/RESULTS")
    assert p.limit("setmaxnreg_min") == 24 and p.const("nope", -1.0) == -1.0


def test_detect_name():
    assert gpus.detect_name("NVIDIA H200") == "h200"
    assert gpus.detect_name("NVIDIA H100 80GB HBM3") == "h100"
    assert gpus.detect_name("NVIDIA GeForce RTX 3090") == "3090"


def test_profile_not_found():
    with pytest.raises(KdevError) as e:
        gpus.load_profile("b999")
    assert e.value.code.value == "not_found"


def test_write_measured_override_roundtrip(tmp_home):
    path = gpus.write_measured("h200", {"stream_tbps": gpus.Measured(4.2, "TB/s", "test", "r1", "x")}, str(tmp_home))
    assert Path(path).is_file()
    p = gpus.load_profile("h200", home=str(tmp_home))
    assert p.const("stream_tbps") == 4.2 and p.measured["stream_tbps"].run == "r1"
    assert p.const("l2_hit_us") == 0.15               # package values survive the merge
    gpus.write_measured("h200", {"l2_hit_us": gpus.Measured(0.2)}, str(tmp_home))
    p = gpus.load_profile("h200", home=str(tmp_home))
    assert p.const("stream_tbps") == 4.2 and p.const("l2_hit_us") == 0.2
    assert "_override_path" in p.raw


# ---------------------------------------------------------------- roofline / skeleton / sensitivity

def test_roofline_memory_compute_mufu():
    p = gpus.load_profile("h200")
    r = roofline(p, 4.8e6, 0.0)
    assert math.isclose(r.mem_us_peak, 1.0) and r.bound == "memory"
    assert math.isclose(r.mem_us_achievable, 4.8e6 / 4.11e12 * 1e6)
    r = roofline(p, 0, 989e6, "bf16")
    assert math.isclose(r.compute_us, 1.0) and r.bound == "compute"
    r = roofline(p, 0, 0, mufu=16 * 132 * 1.98e3)          # one us of SFU work
    assert math.isclose(r.mufu_us, 1.0) and r.bound == "mufu"
    with pytest.raises(KdevError):
        roofline(p, 1, 1, "fp4")                             # no fp4 peak on H200


def test_skeleton_terms_and_missing():
    p = gpus.load_profile("h200")
    s = skeleton(p, {"dram_trips": 3, "merges": 1, "stream_bytes": 4.11e6, "x_us": 0.5, "bogus": 2, "dtype": "fp8"})
    assert math.isclose(s.terms["launch"], 0.7)
    assert math.isclose(s.terms["dram_trips"], 3 * 0.65)
    assert math.isclose(s.terms["stream_bytes"], 1.0)
    assert s.terms["x_us"] == 0.5 and s.dominant == "dram_trips"
    assert any("bogus" in m for m in s.missing_constants)
    assert math.isclose(s.total_us, 0.7 + 1.95 + 1.56 + 1.0 + 0.5)
    assert "launch" not in skeleton(p, {"launch": False, "membar": 1}).terms


def test_geomean_sensitivity_matches_score_derivative():
    rows = [{"shape_id": "a", "load": "x", "weight": 2, "us": 5.0, "floor_us": 4.0},
            {"shape_id": "b", "load": "x", "weight": 1, "us": 10.0, "floor_us": 10.0},
            {"shape_id": "c", "load": "y", "weight": 1, "us": 8.0, "floor_us": 6.0}]
    sens, total = geomean_sensitivity(rows)
    # load x: sub = 2*5 + 10 = 20; d ln S / d us_a = -w/(n_loads*sub) = 2/(2*20) = 5%/us
    assert math.isclose(sens[0].pct_per_us, 5.0)
    assert math.isclose(sens[2].pct_per_us, 100 * 1 / (2 * 8))

    def score(ua, uc):  # geomean over loads of base/sub with base = 1
        return math.sqrt((1 / (2 * ua + 10)) * (1 / uc))
    assert math.isclose(sens[0].gain_pct_at_floor, 100 * (score(4, 8) / score(5, 8) - 1))
    assert math.isclose(total, 100 * (score(4, 6) / score(5, 8) - 1))


def test_floor_rows_binding_and_gap():
    p = gpus.load_profile("h200")
    rows, total = floor_rows(p, [{"shape_id": "d1", "bytes": 4.8e6, "dram_trips": 3, "us": 6.0, "weight": 1}])
    r = rows[0]
    assert r["binding"] == "skeleton" and math.isclose(r["floor_us"], 0.7 + 1.95)
    assert math.isclose(r["gap_us"], 6.0 - r["floor_us"]) and total is not None and total > 0


def test_floor_cli_json(capsys):
    from kdev.cli import main
    code = main(["floor", "--bytes", "4.8e6", "--terms", "dram_trips=2", "--us", "3", "--json"])
    d = json.loads(capsys.readouterr().out)
    assert code == 0 and d["tool"] == "floor" and d["summary"]["rows"][0]["floor_us"] > 1.0


# ---------------------------------------------------------------- instrumentation

CU = """#include <cuda_runtime.h>
__global__ void k(float* x) {
  // @kdev:mark entry
  x[0] = 1.f;  // @kdev:mark stored
  for (int i = 0; i < 4; ++i) // @kdev:mark bad
    x[i] += 1;
  // @kdev:mark sm
  if (threadIdx.x == 64) {
    // @kdev:mark prod thread
  }
  // @kdev:mark entry
  // @kdev:mark done cta
}
"""


def test_find_annotations():
    a = instrument.find_annotations(CU)
    assert [n for _, n in a] == ["entry", "stored", "bad", "sm", "prod", "entry", "done"]
    assert a[0][0] == 3


def test_instrument_source_cuda():
    out, marks = instrument.instrument_source(CU)
    assert "KD_MARK(0); // @kdev:mark entry" in out
    assert "x[0] = 1.f; KD_MARK(1); // @kdev:mark stored" in out
    assert "KD_MARK_T(" in out and "KD_MARK_CTA(" in out and "KD_MARK_SM();" in out
    assert out.count("KD_MARK(0);") == 2                   # same name -> same id
    assert "bad" not in marks.values()                     # trailing after `for (...)` is refused
    assert marks == {0: "entry", 1: "stored", 2: "prod", 3: "done"}
    assert "#define KDEV_MARKS 1" in out and "kd_marks_init" in out
    assert "\\" not in instrument.prelude()                 # safe inside non-raw Python strings


def test_instrument_explicit_and_include():
    src = '#include "kdev/marks.cuh"\n__global__ void k() { KD_MARK(5); // loaded\n  KD_MARK(7);\n  // @kdev:mark x\n}\n'
    out, marks = instrument.instrument_source(src)
    assert '#include "kdev/marks.cuh"' not in out and out.count("kd_marks_init") == 1
    assert marks[5] == "loaded" and marks[7] == "m7" and marks[0] == "x"


PY = '''import torch
from torch.utils.cpp_extension import load_inline
CPP = "void f();"
CUDA = r"""
#include <cuda_runtime.h>
__global__ void k(float* x) {
  // @kdev:mark start
  x[0] = 2.f;
  // @kdev:mark end
}
"""
FMT = """
__global__ void g{name}() {{
  // @kdev:mark fstart
}}
"""
OTHER = "no marks here"
'''


def test_instrument_python_strings():
    out, marks, warns = instrument.instrument_python(PY)
    assert marks == {0: "start", 1: "end", 2: "fstart"}
    assert 'OTHER = "no marks here"' in out and 'CPP = "void f();"' in out
    compile(out, "<x>", "exec")                             # still valid Python
    ns: dict = {}
    exec(compile(out.replace("import torch\nfrom torch.utils.cpp_extension import load_inline\n", ""), "<x>", "exec"),
         ns)
    assert "KD_MARK(0);" in ns["CUDA"] and "kd_marks_init" in ns["CUDA"]
    g = ns["FMT"].format(name="X")                          # braces doubled in the format template
    assert "void gX()" in g and "KD_MARK(2);" in g and "kd_mark_write" in g
    assert not warns


def test_instrument_file(tmp_path):
    src = tmp_path / "src" / "sub.py"
    src.parent.mkdir()
    src.write_text(PY, encoding="utf-8")
    inst = instrument.instrument_file(str(src), str(tmp_path / "out"))
    assert Path(inst.path).name == "sub.py" and Path(inst.marks_json).is_file()
    assert json.loads(Path(inst.marks_json).read_text())["marks"]["1"] == "end"
    with pytest.raises(KdevError):
        instrument.instrument_file(str(src), str(src.parent))


# ---------------------------------------------------------------- decode / trace

def _recs(tick: int = 32, n_cta: int = 4) -> list[MarkRec]:
    """3 marks per CTA; CTA 3 is late (critical); SM records; clock at 2 GHz."""
    out = []
    base = 1_000_000_000
    for c in range(n_cta):
        t0 = base + c * 64 + (2000 if c == 3 else 0)
        for m, dt in ((0, 0), (1, 1000 + 32 * c), (2, 3000 + 64 * c)):
            t = t0 + dt
            t = t - (t % tick)
            out.append(MarkRec(t, (t0 + dt) * 2 & 0xFFFFFFFF, c, 0, m))
        out.append(MarkRec(base + 1, 100 + c % 2, c, 0, decode.SM_MARK))
    return out


def test_buffer_roundtrip_log_and_slot():
    recs = _recs()
    got, dropped = decode.parse_buffer(decode.build_buffer(recs, layout=1))
    assert dropped == 0 and sorted((r.cta, r.mark, r.t_ns) for r in got) == sorted((r.cta, r.mark, r.t_ns) for r in recs)
    got, dropped = decode.parse_buffer(decode.build_buffer(recs, layout=1, capacity=5))
    assert len(got) == 5 and dropped == len(recs) - 5
    got, dropped = decode.parse_buffer(decode.build_buffer(recs, layout=0, max_wg=2, max_marks=4))
    assert dropped == 0 and len(got) == len(recs)
    h = decode.parse_header(decode.build_buffer(recs, layout=0, max_wg=2, max_marks=4))
    assert h["valid"] and h["layout"] == 0 and h["max_wg"] == 2 and h["max_marks"] == 4
    bad = recs + [MarkRec(5, 0, 0, 3, 1)]                  # wg 3 >= max_wg 2 -> dropped
    _, dropped = decode.parse_buffer(decode.build_buffer(bad, layout=0, max_wg=2, max_marks=4))
    assert dropped == 1


def test_slot_layout_recovers_large_cta_ids():
    recs = [MarkRec(10_000 + i, 0, 70000 + i, 0, 0) for i in range(3)]
    got, _ = decode.parse_buffer(decode.build_buffer(recs, layout=0, max_wg=1, max_marks=1))
    assert sorted(r.cta for r in got) == [70000, 70001, 70002]


def test_detect_resolution():
    assert decode.detect_resolution(_recs(32)) == 32
    assert decode.detect_resolution(_recs(256)) == 256
    assert decode.detect_resolution(_recs()[:2]) is None


def test_summarize_fine_tick():
    names = {0: "entry", 1: "loaded", 2: "done"}
    s = decode.summarize(_recs(32), names)
    assert [m.name for m in s.marks] == ["entry", "loaded", "done"]
    assert [p.name for p in s.phases] == ["entry->loaded", "loaded->done"]
    assert s.critical_cta == 3 and s.n_ctas == 4 and s.n_sms == 2
    assert s.duration_source == "globaltimer" and s.resolution_ns == 32
    assert s.cta_start_spread_ns == pytest.approx(2000 + 3 * 64, abs=32)
    assert s.phases[0].p50_ns == pytest.approx(1000 + 32 * 1.5, abs=40)
    assert s.critical_path[0]["phase"] == "start->entry" and len(s.critical_path) == 3
    assert s.sm_mhz == pytest.approx(2000, rel=0.05)


def test_summarize_coarse_tick_uses_clock():
    s = decode.summarize(_recs(256), {})
    assert s.resolution_ns == 256 and s.duration_source == "clock64"
    # clock deltas recover the true 1000 + 32c ns phase lengths despite the 256 ns tick
    assert s.phases[0].p10_ns == pytest.approx(1000 + 32 * 0.3, abs=15)
    assert s.marks[0].name == "m0"


def test_merge_and_sample():
    a = decode.summarize(_recs(32), {})
    b = decode.summarize(_recs(32, n_cta=6), {})
    m = decode.merge_summaries([a, b])
    assert m.replays == 2 and m.critical_cta == 3 and len(m.phases) == 2
    assert decode.merge_summaries([]).replays == 0
    recs = _recs(32, n_cta=10)
    sub = decode.sample_ctas(recs, "sample:3")
    assert {r.cta for r in sub} >= {0, 3, 9} and len({r.cta for r in sub}) <= 4
    with pytest.raises(ValueError):
        decode.sample_ctas(recs, "few")


def test_trace_exports(tmp_path):
    recs = _recs()
    names = {0: "a", 1: "b", 2: "c"}
    tr = trace.to_perfetto(recs, names)
    xs = [e for e in tr["traceEvents"] if e["ph"] == "X"]
    assert len(xs) == 8 and xs[0]["name"] == "a->b" and xs[0]["dur"] > 0
    assert {e["pid"] for e in xs} == {100, 101}
    rows = trace.to_csv_rows(recs, names)
    assert len(rows) == 8 and min(r["start_ns"] for r in rows) == 0
    p = trace.write_csv(str(tmp_path / "t.csv"), rows)
    assert Path(p).read_text().splitlines()[0] == "cta,wg,sm,phase,start_ns,end_ns"
    ch = trace.chart_json(decode.summarize(recs, names), rows, max_gantt_ctas=2)
    assert ch["schema"] == "kdev.timeline_chart/1" and ch["gantt_ctas"] <= 3 and ch["phases"][0]["name"] == "a->b"
    json.dumps(tr)


def test_timeline_cli_instruments_then_submits(tmp_path, capsys):
    """KDEV_NO_GPU=1: the handler instruments + prebuild-skips, then the GPU route reports unavailability."""
    from kdev.cli import main
    src = tmp_path / "sub.cu"
    src.write_text(CU, encoding="utf-8")
    code = main(["timeline", str(src), "--no-build", "--json"])
    d = json.loads(capsys.readouterr().out)
    assert d["tool"] == "timeline" and code != 0
    assert d["error"]["code"] in ("capability_unavailable", "not_implemented")
    copies = list((tmp_path / "kdev_home" / "instrumented").glob("sub-*/sub.cu"))
    assert copies and "KD_MARK(0);" in copies[0].read_text()


def test_timeline_cli_no_marks(tmp_path, capsys):
    from kdev.cli import main
    src = tmp_path / "plain.cu"
    src.write_text("__global__ void k() {}\n", encoding="utf-8")
    main(["timeline", str(src), "--no-build", "--json"])
    d = json.loads(capsys.readouterr().out)
    assert d["error"]["code"] == "invalid_request"


def test_marks_header_present():
    h = (Path(kdev.__file__).parent / "timeline" / "marks.cuh").read_text()
    assert "kd_marks_init" in h and "%%globaltimer" in h and "\\" not in h
    assert "#define KD_MARK(id) do { } while (0)" in h      # zero cost without KDEV_MARKS
