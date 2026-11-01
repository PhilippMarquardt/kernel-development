"""kdev timeline v2: agent mark macros, runtime (load_inline) instrumentation, per-(CTA, wg) log, value/info
records, analysis (marks / phases / slowest CTA / roles), merge, diff, CLI --diff. No CUDA anywhere."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import kdev
from kdev.timeline import analyze as A
from kdev.timeline import decode, instrument
from kdev.timeline.decode import KIND_INFO, KIND_TIME, KIND_VALUE, MarkRec

CUDA_AGENT = r'''
#include <cuda_runtime.h>
struct Params { unsigned long long* dbg; int* flag; };
__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t;
}
#define MWT(c, inf) do { if (p.dbg && threadIdx.x == 0) { unsigned long long* mwb_ = p.dbg + (size_t)blockIdx.x * 512; const unsigned long long n_ = mwb_[0]; if (n_ < 254) { mwb_[2 + 2 * n_] = ((unsigned long long)(c) << 32) | (unsigned)(inf); mwb_[3 + 2 * n_] = gtimer(); mwb_[0] = n_ + 1; } } } while (0)
#define DBG(k) do { if (OKC_DEBUG && threadIdx.x == 0 && p.dbg) p.dbg[blockIdx.x * 8 + (k)] = gtimer(); } while (0)
#define DBGT(k) do { if (OKC_DEBUG && p.dbg) p.dbg[blockIdx.x * 32 + (k)] = gtimer(); } while (0)
#define DBGV(k, v) do { if (OKC_DEBUG && p.dbg) p.dbg[blockIdx.x * 32 + (k)] = (v); } while (0)
#define UNUSED(k) do { if (p.dbg) p.dbg[k] = gtimer(); } while (0)
__global__ void k(Params p) {
  DBG(0);
  MWT(29, 0);
  for (int s = 0; s < 4; ++s) { MWT(1, s); MWT(3, 0); }
  DBGT(10);
  DBGV(26, 123);
  DBG(5);
}
'''


def test_marks_header_wait_helper_and_lesson():
    h = (Path(kdev.__file__).parent / "timeline" / "marks.cuh").read_text()
    assert "\\" not in h
    assert "ld.relaxed.gpu.global" in h and "__trap()" in h and "kd_wait_geq" in h
    assert "HOISTED" in h and "ld.global.cg" in h          # the mtn_ lesson (board #179)
    assert "KD_MARKS_CTALOG" in h and "KD_KIND_VALUE" in h and "KD_ACC_FLUSH" in h
    assert "#define KD_MARK_CTA_I(id, v) do { } while (0)" in h


def test_detect_families():
    fams = {f.name: f for f in instrument.detect_families(CUDA_AGENT)}
    assert fams["MWT"].kind == "info" and fams["MWT"].thread0
    assert fams["DBG"].kind == "timed" and fams["DBG"].thread0 and fams["DBG"].range == 32
    assert fams["DBGT"].kind == "timed" and not fams["DBGT"].thread0
    assert fams["DBGV"].kind == "value" and fams["DBGV"].buffer == "p.dbg"
    out, marks = instrument.instrument_source(CUDA_AGENT)
    assert "UNUSED" not in {m.rstrip("0123456789") for m in marks.values()}   # no call sites -> not rewritten
    assert "#define MWT(c, inf) KD_MARK_CTA_I(" in out and "#define DBGV(k, v) KD_VAL_T(" in out
    assert "#define DBGT(k) KD_MARK_T(" in out
    assert "kd_marks_init" in out and "#define KD_MARKS_CTALOG 1" in out   # MWT = segment log -> ctalog
    by = {v: k for k, v in marks.items()}
    assert by["MWT29"] == by["MWT0"] + 29 and by["DBG5"] == by["DBG0"] + 5
    none, m2 = instrument.instrument_source(CUDA_AGENT, styles="none")
    assert m2 == {} and none == CUDA_AGENT
    only, m3 = instrument.instrument_source(CUDA_AGENT, styles="DBG", layout="slots")
    assert "#define MWT(c, inf) do {" in only and "KD_MARKS_CTALOG" not in only.split("end kdev marks")[0][:400]


def test_lint_cg_poll():
    w: list[str] = []
    src = "__global__ void k(int* f) {\n  int v;\n  do {\n    asm volatile(\"ld.global.cg.s32 %0, [%1];\" : \"=r\"(v) : \"l\"(f));\n  } while (v == 0);\n  // @kdev:mark x\n}\n"
    instrument.instrument_source(src, warnings=w)
    assert any("board #179" in x for x in w)


PY_MOD = '''
CALLS = []
def load_inline(name, cpp_sources, cuda_sources=None, **kw):
    CALLS.append(cuda_sources)
    return name
CUDA_SRC = r"""__global__ void k(Params p) {
  // @kdev:mark entry
  DBG(1);
}
"""
HELPER = '#define DBG(k) do { if (threadIdx.x == 0 && p.dbg) p.dbg[blockIdx.x * 8 + (k)] = gtimer(); } while (0)\\n'
# import-time patch (like mtn_make): the CUDA source the compiler sees is only built here
CUDA_SRC = HELPER + CUDA_SRC.replace("DBG(1);", "DBG(1);\\n  DBG(2);")
OTHER = "__global__ void plain() {}"
def build():
    return load_inline("a", "", cuda_sources=CUDA_SRC), load_inline("b", "", OTHER)
# @kdev:names DBG1=routed DBG2=done
# @kdev:roles 0=consumer 1=producer
'''


def test_runtime_instrumentation_sees_import_time_patches(tmp_path):
    src = tmp_path / "sub.py"
    src.write_text(PY_MOD, encoding="utf-8")
    inst = instrument.instrument_file(str(src), str(tmp_path / "out"))
    text = Path(inst.path).read_text(encoding="utf-8")
    assert text.startswith(PY_MOD.rstrip("\n")) and instrument.FOOTER_TAG in text
    by = {v: k for k, v in inst.marks.items()}
    assert "entry" in by and "DBG2" in by
    assert inst.hints["names"] == {"DBG1": "routed", "DBG2": "done"} and inst.hints["roles"]["1"] == "producer"
    ns: dict = {}
    exec(compile(text, "sub.py", "exec"), ns)                # noqa: S102 - test module
    ns["build"]()
    marked, other = ns["CALLS"]
    assert f"KD_MARK({by['entry']});" in marked and f"#define DBG(k) KD_MARK_CTA({by['DBG0']} + (k))" in marked
    assert marked.count("kd_marks_init(void") == 1
    assert other == "__global__ void plain() {}"             # no marks -> unchanged (same cache entry)
    doc = json.loads(Path(inst.marks_json).read_text())
    assert doc["layout"] == "slots" and doc["families"][0]["name"] == "DBG"
    again = instrument.runtime_transform(marked, instrument.plan_doc(instrument.plan_from_doc(
        {"ids": {str(k): v for k, v in inst.marks.items()}, "families": doc["families"], "n_marks": 40})))
    assert again == marked                                   # idempotent


def test_static_python_single_quoted_strings():
    py = "X = '__global__ void k() {\\n  // @kdev:mark a\\n}\\n'\n"
    out, marks, warns = instrument.instrument_python(py)
    assert marks == {0: "a"} and not warns
    ns: dict = {}
    exec(out, ns)                                            # noqa: S102
    assert "KD_MARK(0);" in ns["X"] and "kd_marks_init" in ns["X"]


def test_hints_and_roles():
    assert instrument.parse_roles("0-2=consumer,3=producer") == {0: "consumer", 1: "consumer", 2: "consumer",
                                                                 3: "producer"}
    h = instrument.hints("// @kdev:names a=b c=d\n# @kdev:t0 MWT29\n")
    assert h == {"names": {"a": "b", "c": "d"}, "t0": "MWT29"}


# ---------------------------------------------------------------- decode

def test_ctalog_roundtrip_with_kinds():
    recs = [MarkRec(1000 + i * 100, i, 3, 0, 1, KIND_TIME) for i in range(5)]
    recs += [MarkRec(5000, 77, 3, 0, 2, KIND_INFO), MarkRec(6000, 4242, 0, 2, 9, KIND_VALUE)]
    buf = decode.build_buffer(recs, layout=2, max_wg=4, cta_recs_log2=3)
    h = decode.parse_header(buf)
    assert h["layout"] == 2 and h["cta_recs"] == 8
    got, dropped = decode.parse_buffer(buf)
    assert dropped == 0
    assert sorted((r.cta, r.wg, r.mark, r.kind, r.clk_lo) for r in got) == \
        sorted((r.cta, r.wg, r.mark, r.kind, r.clk_lo) for r in recs)
    over = [MarkRec(10 + i, 0, 0, 0, 1) for i in range(9)]   # 7 fit (record 0 = counter)
    got, dropped = decode.parse_buffer(decode.build_buffer(over, layout=2, max_wg=1, cta_recs_log2=3))
    assert len(got) == 7 and dropped == 2
    assert decode.used_bytes(buf) == 16 + 16 * 4 * 4 * 8      # CTAs 0..3, 4 wgs, 8 recs each


def test_slots_layout_value_kind():
    recs = [MarkRec(100, 5, 1, 0, 0), MarkRec(900, 2000, 1, 1, 3, KIND_VALUE)]
    got, _ = decode.parse_buffer(decode.build_buffer(recs, layout=0, max_wg=2, max_marks=4))
    assert {(r.mark, r.kind) for r in got} == {(0, KIND_TIME), (3, KIND_VALUE)}
    s = decode.summarize(got, {})
    assert s.n_tracks == 1                                   # value records are not timeline marks


# ---------------------------------------------------------------- analysis

def _replay(late: int = 0, n_cta: int = 4) -> list[MarkRec]:
    """CTA c: start(0) at c*10 ns, two segments seg(1)->end(2) of 1000 ns + 100 ns gap, done(3); CTA n-1 late."""
    out = []
    for c in range(n_cta):
        t = 1_000_000 + c * 10
        out.append(MarkRec(t, 0, c, 0, 0))
        for _ in range(2):
            t += 100
            out.append(MarkRec(t, 0, c, 0, 1))
            t += 1000 + (late if c == n_cta - 1 else 0)
            out.append(MarkRec(t, 0, c, 0, 2))
        out.append(MarkRec(t + 50, 0, c, 0, 3))
        out.append(MarkRec(t, 1980 * 2, c, 1, 9, KIND_VALUE))          # producer: 2 us of waits
        out.append(MarkRec(t, 1980, c, 0, 8, KIND_VALUE))              # consumer: 1 us
    return out


NAMES = {0: "start", 1: "seg", 2: "end", 3: "done", 8: "cons_wait", 9: "prod_free_wait"}


def test_analyze_marks_phases_slowest_roles():
    d = A.analyze(_replay(late=500), NAMES, roles={0: "consumer", 1: "producer"}, t0="start")
    assert d["t0"] == "start" and d["n_ctas"] == 4
    assert d["mark_order"][:2] == ["start", "seg"]
    st = d["marks"]["start"]
    assert st["min"] == 0 and st["max"] == pytest.approx(0.03) and st["n"] == 4
    assert d["marks"]["seg"]["per_cta"] == 2                  # loop iterations kept
    ph = d["phases"]["seg->end"]
    assert ph["occ"] == 2 and ph["cta_mean"] == pytest.approx((2.0 * 3 + 3.0) / 4)
    assert ph["slow"] == pytest.approx(3.0) and ph["cta_max"] == pytest.approx(3.0)
    assert d["phases"]["end->seg"]["cta_mean"] == pytest.approx(0.1)     # the gap between segments
    sl = d["slowest"]
    assert sl["cta"] == 3 and sl["lag_us"] == pytest.approx(1.0, abs=0.05)
    r = d["roles"]
    assert r["producer"]["values"]["prod_free_wait"]["mean"] == pytest.approx(2.0)
    assert r["consumer"]["values"]["cons_wait"]["mean"] == pytest.approx(1.0)
    assert r["producer"]["wait_mean"] == pytest.approx(2.0) and r["consumer"]["wait_max"] == pytest.approx(1.0)
    lines = A.render(A.merge([d]), "d32/bs32", timing={"marked_us": 10.0, "plain_us": 9.5, "overhead_us": 0.5,
                                                       "ns_per_mark": 80.0})
    assert lines[0].startswith("d32/bs32: 1 replays") and "mark cost +0.50 us" in lines[0]
    assert any("slowest CTA 3" in x for x in lines) and any("prod_free_wait" in x for x in lines)
    assert len(lines) <= 20


def test_merge_median_and_t0_fallback():
    ds = [A.analyze(_replay(late=x), NAMES) for x in (0, 500, 1000)]
    m = A.merge(ds)
    assert m["replays"] == 3 and m["t0"] == "first"
    assert m["phases"]["seg->end"]["slow"] == pytest.approx(3.0)
    assert m["slowest"]["cta"] == 3 and m["slowest"]["ctas"][3] >= 2
    assert A.merge([]) == {"empty": True, "replays": 0}
    assert A.analyze([], NAMES) == {"empty": True}


def test_diff():
    a = A.merge([A.analyze(_replay(late=0), NAMES)])
    b = A.merge([A.analyze(_replay(late=1000), NAMES)])
    d = A.diff(a, b)
    assert d["phases"]["seg->end"]["d_slow"] == pytest.approx(2.0)
    assert d["marks"]["done"]["d_max"] == pytest.approx(2.0) and d["end_spread_us"] == pytest.approx(2.0, abs=0.05)
    lines = A.render_diff(d, "v16", timing={"d_plain_us": -1.25, "d_marked_us": -1.0})
    assert lines[0].startswith("v16 B-A:") and "plain replay -1.25 us" in lines[0]


# ---------------------------------------------------------------- CLI

CU = "__global__ void k(float* x) {\n  // @kdev:mark entry\n  x[0] = 1.f;\n  // @kdev:mark done\n}\n"


def test_timeline_cli_diff_instruments_both(tmp_path, capsys):
    from kdev.cli import main
    a, b = tmp_path / "a.cu", tmp_path / "b.cu"
    a.write_text(CU, encoding="utf-8")
    b.write_text(CU.replace("1.f", "2.f"), encoding="utf-8")
    code = main(["timeline", "--diff", str(a), str(b), "--no-build", "--json", "--seeds", "2", "--replays", "3"])
    d = json.loads(capsys.readouterr().out)
    assert d["tool"] == "timeline" and code != 0
    assert d["error"]["code"] in ("capability_unavailable", "not_implemented")
    copies = list((tmp_path / "kdev_home" / "instrumented").glob("*/[ab].cu"))
    assert len(copies) == 2


def test_timeline_cli_diff_needs_two(tmp_path, capsys):
    from kdev.cli import main
    a = tmp_path / "a.cu"
    a.write_text(CU, encoding="utf-8")
    main(["timeline", "--diff", str(a), "--no-build", "--json"])
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "invalid_request"


def test_phase_mode_timeline():
    from types import SimpleNamespace

    from kdev.service import phase
    import sys
    spec = SimpleNamespace(kind="timeline", params={"extra": {}})
    want = None if sys.platform == "win32" else "phased"
    assert phase.mode(spec) == want
