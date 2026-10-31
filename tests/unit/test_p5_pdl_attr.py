"""PDL audit attribution: param/struct-member names for pre-wait loads, per-file allowlist (P5)."""
from __future__ import annotations

from pathlib import Path

from kdev.sass import _pdlsrc, audits
from kdev.sass.parse import parse_sass

FX = Path(__file__).resolve().parents[1] / "fixtures" / "sass" / "pdl_attr"


def _kernel_and_ctx(tmp_path, allow: str = ""):
    src = (FX / "k.cu").read_text() + allow
    (tmp_path / "k.cu").write_text(src)
    text = (FX / "k.nvd").read_text().replace('"k.cu"', f'"{(tmp_path / "k.cu").as_posix()}"')
    (k,) = parse_sass(text)
    return k, _pdlsrc.build_context(None, [k])


def test_short_names():
    f = _pdlsrc.short_kernel_name
    assert f("_ZN39_GLOBAL__N__f3bab234_7_cuda_cu_0b177e6611attn_kernelILi8ELb1ELi0EEEvNS_6ParamsE") == "attn_kernel"
    assert f("_Z7pdl_dotPKfPKjS2_Pfii") == "pdl_dot"
    assert f("pdl_dot") == "pdl_dot"


def test_struct_layout_and_param_map():
    src = (FX / "k.cu").read_text()
    pm = _pdlsrc.param_map({"k.cu": src}, "attn")
    assert pm.name(0x8) == "p.seq_lens" and pm.name(0x10) == "p.page_table" and pm.name(0x1c) == "p.page"
    pm2 = _pdlsrc.param_map({"x.cu": "__global__ void k(const float* __restrict__ x, int n, const int* idx) {}"},
                            "_Z1kPKfiPKi")
    assert [pm2.name(o) for o in (0, 8, 16)] == ["x", "n", "idx"]


def test_pdl_error_names_roots(tmp_path):
    k, ctx = _kernel_and_ctx(tmp_path)
    (f,) = audits.audit_pdl(k, ctx)
    assert f.severity == "error" and "p.seq_lens x1" in f.message and "p.seq_lens + p.page_table x1" in f.message
    d = {x["root"]: x for x in f.details}
    assert d["p.seq_lens"]["line"].endswith("k.cu:9") and "__ldg" in d["p.seq_lens"]["source"]
    assert d["p.seq_lens + p.page_table"]["via_loads"] == ["0x50"]


def test_pdl_allowlist(tmp_path):
    k, ctx = _kernel_and_ctx(tmp_path, "// kdev:pdl-allow seq_lens -- read-only input written at prepare time\n")
    fs = audits.audit_pdl(k, ctx)
    sev = {f.severity: f for f in fs}
    assert "read-only input" in sev["info"].message and sev["info"].addrs == [0x50]
    assert sev["error"].addrs == [0x80]          # page_table is not allowed
    k, ctx = _kernel_and_ctx(tmp_path, "// kdev:pdl-allow seq_lens,p.page_table: both read-only\n")
    assert [f.severity for f in audits.audit_pdl(k, ctx)] == ["info"]


def test_allow_needs_reason_and_sidecar():
    allow, warns = _pdlsrc.parse_allow({"a.py": "# kdev:pdl-allow seq_lens\n",
                                        "a.py.pdl-allow": "kv_indptr: read-only\n# comment\n"})
    assert allow == {"kv_indptr": "read-only"} and "no justification" in warns[0]


def test_pdl_without_context_still_works():
    k = parse_sass((FX / "k.nvd").read_text())[0]
    (f,) = audits.audit_pdl(k)
    assert f.severity == "error" and "param+0x8" in f.message
