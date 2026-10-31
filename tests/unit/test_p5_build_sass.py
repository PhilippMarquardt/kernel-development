"""P5: build/toolchain/ptxas, SASS parser/CFG/loops/audits/diff. No GPU, CUDA or torch needed.

Fixtures: tests/fixtures/sass/vjc_k_L1_F0_M1.sass (PDL hoist positive: LDG.E.CONSTANT at 0x0080 before ACQBULK),
sass/pdl_spill.{cu,sass,nvd,res} (NVRTC 13.4 build of pdl_spill.cu: PDL negative case + local-memory loop),
ptxas/vjc.ptxas, build/nvrtc_spill.ptxas (spills), build/px_c7515.ptxas (C7515 serialisation messages).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from kdev.build import ptxas, toolchain
from kdev.contract import KdevError, validate_result
from kdev.models import KernelInfo, Toolchain
from kdev.registry import build_registry
from kdev.sass import audits, cfg, classes, diff, disasm, loops, parse
from kdev.sass.audits import AUDITS
from kdev.sass.classes import CLASSES
from kdev.sass.parse import parse_sass

FX = Path(__file__).resolve().parents[1] / "fixtures"


def _k(name: str = "pdl_dot"):
    ks = parse_sass((FX / "sass" / "pdl_spill.sass").read_text())
    return next(k for k in ks if k.name == name)


@pytest.fixture
def vjc(sample_sass):
    return parse_sass(sample_sass)[0]


# ----------------------------------------------------------------------------------------------- registry
def test_build_sass_commands_registered():
    assert {"build", "sass", "sass diff", "sweep"} <= set(build_registry().names())


def test_audit_and_class_names():
    assert {"pdl", "spills", "serialize", "expect", "repeat", "invariant"} == set(AUDITS)
    assert "mma" in CLASSES and "ldl_stl" in CLASSES


# ----------------------------------------------------------------------------------------------- parse
def test_parse_headerless_fixture(vjc):
    assert vjc.name == "" and len(vjc.instrs) == 2672
    i = next(x for x in vjc.instrs if x.addr == 0x80)
    assert (i.opcode, i.base, i.operands) == ("LDG.E.CONSTANT", "LDG", ["R100", "desc[UR8][R100.64]"])
    b = next(x for x in vjc.instrs if x.addr == 0x47B0)
    assert b.pred == "@!P0" and b.base == "BRA" and b.target == 0x44C0 and b.operands[0] == "P2"


def test_parse_cuobjdump_headers_and_nvdisasm_agree():
    a = parse_sass((FX / "sass" / "pdl_spill.sass").read_text())
    b = parse_sass((FX / "sass" / "pdl_spill.nvd").read_text())
    assert [k.name for k in a] == [k.name for k in b] == ["spill_k", "pdl_dot"]
    assert a[0].arch == "sm_90a" and b[0].arch == "sm_90a"
    for ka, kb in zip(a, b):
        assert len(ka.instrs) == len(kb.instrs)
        ta = {i.addr: i.target for i in ka.instrs if i.base == "BRA"}
        tb = {i.addr: i.target for i in kb.instrs if i.base == "BRA"}
        assert ta == tb and all(v is not None for v in ta.values())   # `(.L_x_N) labels resolved
    assert any(i.line and i.line.endswith("pdl_spill.cu:5") for i in b[1].instrs)   # //## line info
    assert all(i.line is None for i in a[1].instrs)


def test_normalize_is_register_agnostic():
    x = parse.parse_instr(0, "@P0 IMAD.WIDE R4, R7.reuse, 0x4, c[0x0][0x218]")
    y = parse.parse_instr(16, "@!P3 IMAD.WIDE R12, R9, 0x10, c[0x0][0x220]")
    assert parse.normalize(x) == parse.normalize(y) == "@P IMAD.WIDE R, R, imm, c[]"


# ----------------------------------------------------------------------------------------------- cfg / loops
def test_loops_vjc(vjc):
    ls = cfg.find_loops(vjc)
    assert [hex(lp.head) for lp in ls] == ["0x44c0", "0x4bc0", "0x7ac0", "0x7d20", "0x91a0"]
    nested = next(lp for lp in ls if lp.head == 0x7D20)
    assert nested.depth == 2
    assert all(lp.head != 0xA610 for lp in ls)   # 'BRA self' trap after EXIT is unreachable


def test_loops_pdl_dot_and_hot_loop_ranking():
    k = _k()
    (lp,) = cfg.find_loops(k)
    assert (lp.head, lp.tail) == (0x170, 0x4A0)
    (h,) = loops.hot_loops(k, top=3, per=4)
    assert h.innermost and h.histogram["ldg"] == 5 and h.histogram["fp32"] == 18
    assert h.per_element["ldg"] == pytest.approx(1.25)
    assert loops.hot_loops(_k("spill_k"), top=1)[0].histogram["ldl_stl"] >= 96


def test_reachable_before(vjc):
    c = cfg.build_cfg(vjc)
    before = cfg.reachable_before(vjc, c, 0x6D0)
    assert 0x80 in before and 0x6D0 not in before and 0x9F0 not in before


# ----------------------------------------------------------------------------------------------- classes
@pytest.mark.parametrize("text,cls", [("HGMMA.64x128x16.F32.BF16 R24, gdesc[UR4], RZ", "mma"),
                                      ("UTMALDG.2D [UR8], [UR10]", "tensor_mem"), ("FFMA R1, R2, R3, R4", "fp32"),
                                      ("HFMA2 R1, R2, R3, R4", "fp16x2"), ("MUFU.EX2 R1, R2", "mufu"),
                                      ("F2FP.BF16.F32.PACK_AB R1, R2, R3", "cvt"), ("LDG.E R1, desc[UR4][R2.64]", "ldg"),
                                      ("LDL R1, [R1+0x10]", "ldl_stl"), ("ULDC UR4, c[0x0][0x0]", "ldc"),
                                      ("ACQBULK", "sync"), ("LOP3.LUT R1, R2, R3, RZ, 0xc0, !PT", "int"),
                                      ("UIADD3 UR4, UR4, 0x1, URZ", "uniform"), ("PRMT R1, R2, 0x7610, R3", "move"),
                                      ("BRA 0x100", "control"), ("FOO R1", "other")])
def test_classify(text, cls):
    assert classes.classify(parse.parse_instr(0, text)) == cls


def test_histogram_per():
    ins = [parse.parse_instr(16 * n, t) for n, t in enumerate(["FFMA R1, R2, R3, R4"] * 6 + ["NOP"])]
    assert classes.histogram(ins) == {"fp32": 6.0}
    assert classes.histogram(ins, per=4) == {"fp32": 1.5}


# ----------------------------------------------------------------------------------------------- audits
def test_pdl_positive_on_vjc(vjc):
    (f,) = [x for x in audits.audit_pdl(vjc) if x.severity == "error"]
    assert 0x80 in f.addrs and "LDG.E.CONSTANT" in f.excerpt[0]


def test_pdl_negative_when_load_follows_wait():
    assert audits.audit_pdl(_k("pdl_dot")) == []
    assert audits.audit_pdl(_k("spill_k")) == []    # no ACQBULK: not a PDL consumer


def test_spills_in_loop_and_ptxas_bytes():
    fs = audits.audit_spills(_k("spill_k"), KernelInfo("spill_k", regs=64, spill_store=388, spill_load=404))
    sev = {(f.severity, f.message.split(":")[0]) for f in fs}
    assert ("error", "local memory in loop @0x1a20") in sev or any(f.severity == "error" and "in loop" in f.message
                                                                   for f in fs)
    assert any("ptxas: 388 bytes spill stores" in f.message for f in fs)
    assert audits.audit_spills(_k("pdl_dot")) == []


def test_serialize_from_ptxas_text():
    text = (FX / "build" / "px_c7515.ptxas").read_text()
    name = "_ZN8okc_attn15attn_fwd_kernelILi3ELi0ELb1ELb0ELb0ELb0ELb0ELb0ELb0EEEvNS_8AttnArgsE"
    fs = audits.audit_serialize(name, text)
    assert len(fs) == 1 and fs[0].severity == "error" and "C7515" in fs[0].message
    assert audits.audit_serialize("other_kernel", text) == []
    assert len(audits.audit_serialize("", text)) == 9


def test_expect(vjc):
    assert audits.audit_expect(vjc, ["HMMA", "mma", "MUFU.EX2"]) == []
    (f,) = audits.audit_expect(vjc, ["HGMMA", "HMMA"])
    assert f.severity == "error" and "HGMMA" in f.message and "HMMA," not in f.message


_LOOP = """
        /*0000*/                   S2R R0, SR_TID.X ;
        /*0010*/                   MOV R2, RZ ;
        /*0020*/                   LDG.E.CONSTANT R5, desc[UR4][R8.64] ;
        /*0030*/                   SHF.R.U32.HI R6, RZ, 0x8, R5 ;
        /*0040*/                   LOP3.LUT R6, R6, 0xff, RZ, 0xc0, !PT ;
        /*0050*/                   IMAD.SHL.U32 R6, R6, 0x800000, RZ ;
        /*0060*/                   FFMA R2, R6, R10, R2 ;
        /*0070*/                   SHF.R.U32.HI R7, RZ, 0x8, R5 ;
        /*0080*/                   LOP3.LUT R7, R7, 0xff, RZ, 0xc0, !PT ;
        /*0090*/                   IMAD.SHL.U32 R7, R7, 0x800000, RZ ;
        /*00a0*/                   FFMA R2, R7, R11, R2 ;
        /*00b0*/                   UIADD3 UR6, UR9, 0x40, URZ ;
        /*00c0*/                   IADD3 R12, R20, 0x100, RZ ;
        /*00d0*/                   IMAD R13, R12, R21, RZ ;
        /*00e0*/                   IADD3 R8, P0, R8, 0x4, RZ ;
        /*00f0*/                   IADD3 R0, R0, 0x1, RZ ;
        /*0100*/                   ISETP.GE.AND P1, PT, R0, R13, PT ;
        /*0110*/              @!P1 BRA 0x20 ;
        /*0120*/                   EXIT ;
"""


def test_repeat_detects_redundant_decode_and_unrolled_copies():
    k = parse_sass(_LOOP)[0]
    fs = audits.audit_repeat(k)
    vn = [f for f in fs if "recompute" in f.message]
    assert vn and vn[0].addrs == [0x70, 0x80, 0x90]       # second scale decode == first
    struct = [f for f in fs if "window repeats" in f.message]
    assert struct and "repeats 2x" in struct[0].message


def test_invariant_detects_loop_invariant_math():
    k = parse_sass(_LOOP)[0]
    (f,) = audits.audit_invariant(k)
    assert {0xB0, 0xC0, 0xD0} <= set(f.addrs) and "1 uniform" in f.message
    assert 0x30 not in f.addrs     # depends on the in-loop load


def test_run_audits_and_names(vjc):
    names, exp = audits.parse_audit_names("pdl,spills,expect=HGMMA|LDS")
    assert names == ["pdl", "spills", "expect"] and exp == ["HGMMA", "LDS"]
    fs = audits.run_audits(vjc, ["pdl", "expect=HGMMA"])
    assert [f.severity for f in fs][:2] == ["error", "error"]
    with pytest.raises(KdevError):
        audits.parse_audit_names("pdl,bogus")


# ----------------------------------------------------------------------------------------------- diff
def test_diff_kernels(sample_sass):
    a = parse_sass(sample_sass)
    lines = sample_sass.splitlines()
    cut = "\n".join(ln for ln in lines if "MUFU" not in ln and "/*0080*/" not in ln)
    b = parse_sass(cut)
    (d,) = diff.diff_kernels(a, b)
    assert d.instr_b < d.instr_a and d.class_delta["mufu"] == -55
    (same,) = diff.diff_kernels(a, a)
    assert not same.class_delta and not same.findings_new and not same.findings_gone
    assert any("griddepcontrol.wait" in m for m in d.findings_gone) is False   # LDGSTS still hoisted


def test_stable_name_matching():
    n1 = "_ZN38_GLOBAL__N__5234c7d1_6_vjc_cu_0b177e6611attn_kernelILi8ELb1ELi0EEEvNS_6ParamsE"
    n2 = "_ZN38_GLOBAL__N__99999999_6_vjc_cu_123456ab11attn_kernelILi8ELb1ELi0EEEvNS_6ParamsE"
    assert diff.stable_name(n1) == diff.stable_name(n2)


# ----------------------------------------------------------------------------------------------- ptxas
def test_parse_ptxas_vjc(sample_ptxas):
    ks = ptxas.parse_ptxas(sample_ptxas)
    assert len(ks) > 10
    k = next(k for k in ks if k.name.endswith("attn_kernelILi8ELb1ELi0EEEvNS_6ParamsE"))
    assert (k.regs, k.barriers, k.stack, k.spill_store) == (80, 16, 0, 0)


def test_parse_ptxas_nvrtc_spills():
    ks = {k.name: k for k in ptxas.parse_ptxas((FX / "build" / "nvrtc_spill.ptxas").read_text())}
    s = ks["spill_k"]
    assert (s.regs, s.stack, s.spill_store, s.spill_load) == (64, 784, 388, 404)
    assert ks["pdl_dot"].regs == 29 and ks["pdl_dot"].spill_store == 0


def test_parse_ptxas_c7515_attribution():
    text = (FX / "build" / "px_c7515.ptxas").read_text()
    ks = ptxas.parse_ptxas(text)
    ser = [k for k in ks if any("C7515" in w for w in k.warnings)]
    assert len(ser) == 9 and all("attn_fw" in k.name for k in ser)
    assert len(ptxas.serialization_warnings(text)) == 9
    assert ptxas.serialization_warnings(text)[0][0] == "C7515"
    assert ptxas.parse_ptxas((FX / "ptxas" / "v130_attention.log").read_text()) == []   # no -v in that log


def test_demangle_fallback(fake_runner):
    r = fake_runner({})
    assert ptxas.demangle(["_Z1fv"], runner=r) == {"_Z1fv": "_Z1fv"}
    r2 = fake_runner({("cu++filt",): (0, "f()\n", "")})
    assert ptxas.demangle(["_Z1fv"], runner=r2) == {"_Z1fv": "f()"}


# ----------------------------------------------------------------------------------------------- toolchain
def test_parse_nvcc_version():
    assert toolchain.parse_nvcc_version("Cuda compilation tools, release 13.0, V13.0.88\nBuild x") == "13.0.88"
    assert toolchain.parse_nvcc_version("nothing") == ""


def test_lock_load_and_find(tmp_path, monkeypatch):
    d = tmp_path / "prob"
    d.mkdir()
    (d / "toolchain.lock").write_text(
        'schema = "kdev.toolchain/1"\nnvcc_version = "13.0.88"\ncuda_home = "/root/cuda130w"\n'
        'torch = "2.14.1+cu130"\nextra = { flashinfer = "0.7.0.post1" }\n'
        'env = { NVCC_APPEND_FLAGS = "-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK" }\n'
        'python = "/root/venv/bin/python"\nsource = "judge probe check 44e9e47738ea4634"\n')
    lock = toolchain.find_lock(str(d / "sub.py"))
    assert lock == str(d / "toolchain.lock")
    tc = toolchain.load_lock(lock)
    assert tc.nvcc_version == "13.0.88" and tc.env["NVCC_APPEND_FLAGS"].startswith("-DCCCL")
    assert tc.extra["flashinfer"] == "0.7.0.post1" and tc.ptxas.replace("\\", "/").startswith("/root/cuda130w/bin/")
    assert toolchain.find_lock(str(tmp_path / "nowhere")) is None
    env = toolchain.build_env(tc, "/x/ext")
    assert env["CUDA_VISIBLE_DEVICES"] == "" and env["TORCH_EXTENSIONS_DIR"] == "/x/ext"
    assert env["CUDA_HOME"] == "/root/cuda130w" and env["NVCC_APPEND_FLAGS"].startswith("-DCCCL")


def test_so_notes_and_verify(tmp_path):
    so = tmp_path / "x.so"
    so.write_bytes(b"\x7fELF\0\0junk\0Cuda compilation tools, release 13.0, V13.0.88\0"
                   b"Build cuda_13.0.r13.0/compiler.36424714_0\0ptxas\0")
    notes = toolchain.so_notes(str(so))
    assert "Cuda compilation tools, release 13.0, V13.0.88" in notes
    assert toolchain.verify_so(str(so), Toolchain(nvcc_version="13.0.88")) == (True, notes)
    assert toolchain.verify_so(str(so), Toolchain(nvcc_version="13.4.92"))[0] is False
    empty = tmp_path / "e.so"
    empty.write_bytes(b"\0" * 64)
    assert toolchain.verify_so(str(empty), Toolchain(nvcc_version="13.0.88")) == (False, [])
    assert toolchain.verify_so(str(empty), Toolchain())[0] is True


def test_detect_with_fake_runner(tmp_path, fake_runner, monkeypatch):
    bin_ = tmp_path / "cuda" / "bin"
    bin_.mkdir(parents=True)
    for t in toolchain.TOOLS:
        (bin_ / (t + toolchain._EXE)).write_text("")  # noqa: SLF001
    out = "Cuda compilation tools, release 13.0, V13.0.88"

    def runner(argv, **kw):
        from kdev.util import Proc
        return Proc(list(argv), 0, out if "ptxas" not in argv[0] else out.replace("13.0.88", "13.4.92"), "")

    tc = toolchain.detect(str(tmp_path / "cuda"), runner=runner)
    assert tc.nvcc_version == "13.0.88" and tc.extra["ptxas_version"] == "13.4.92"   # M20 mixed tree visible
    checks = dict((t, s) for t, s, _ in toolchain.check_tools(Toolchain(nvcc_version="13.0.88", nvcc=tc.nvcc,
                                                                        ptxas=tc.ptxas), runner=runner))
    assert checks["nvcc"] == "PASS" and checks["ptxas"] == "FAIL"


def test_build_probe_never_raises():
    from kdev.build import probe
    cs = probe(None)
    assert cs and all(c.status in ("PASS", "WARN", "FAIL", "SKIP") for c in cs)


# ----------------------------------------------------------------------------------------------- disasm
def test_disasm_wrappers(tmp_path, fake_runner):
    r = fake_runner({("cuobjdump", "-xelf"): (0, "Extracting ELF file    1: x.1.sm_90a.cubin\n", ""),
                     ("cuobjdump", "-sass"): (0, (FX / "sass" / "pdl_spill.sass").read_text(), ""),
                     ("cuobjdump", "-res-usage"): (0, (FX / "sass" / "pdl_spill.res").read_text(), "")})
    assert disasm.extract_cubins("a.so", str(tmp_path), runner=r, tool="cuobjdump") == \
        [str(tmp_path / "x.1.sm_90a.cubin")]
    assert disasm.extract_cubins("k.cubin", str(tmp_path), runner=r) == ["k.cubin"]
    only = parse_sass(disasm.sass_text("a.so", kernel_regex="pdl", runner=r, tool="cuobjdump"))
    assert [k.name for k in only] == ["pdl_dot"]
    ru = disasm.res_usage("a.so", runner=r, tool="cuobjdump")
    assert ru["spill_k"]["REG"] == 64 and ru["spill_k"]["STACK"] == 784 and ru["pdl_dot"]["REG"] == 29
    assert disasm.nvdisasm_cfg("k.cubin", runner=r, tool="nvdisasm") is None


# ----------------------------------------------------------------------------------------------- CLI
def test_cli_sass_on_text_fixture(tmp_path):
    from kdev.cli import run as cli_run
    res = cli_run(["sass", str(FX / "sass" / "vjc_k_L1_F0_M1.sass"), "--audit", "pdl,spills,repeat,expect=HMMA",
                   "--loops", "--top", "2", "--per", "8"])
    d = res.to_dict()
    assert validate_result(d) == [] and res.status == "fail" and res.verdict == "fail"
    assert res.summary["errors"] >= 1 and "pdl" in res.summary["audits"]
    assert len(res.lines) <= 32 and Path(res.files["sass_json"]).is_file()
    assert any("per element (/8)" in ln for ln in res.lines)


def test_cli_sass_diff(tmp_path):
    from kdev.cli import run as cli_run
    b = tmp_path / "b.sass"
    b.write_text("\n".join(ln for ln in (FX / "sass" / "vjc_k_L1_F0_M1.sass").read_text().splitlines()
                           if "MUFU" not in ln))
    res = cli_run(["sass", "diff", str(FX / "sass" / "vjc_k_L1_F0_M1.sass"), str(b)])
    assert validate_result(res.to_dict()) == [] and res.verdict == "different"
    assert any("mufu-55" in ln for ln in res.lines)


def test_cli_sass_needs_build(tmp_path):
    from kdev.cli import run as cli_run
    f = tmp_path / "s.py"
    f.write_text("x = 1\n")
    res = cli_run(["sass", str(f)])
    assert res.error is not None and res.error.code == "needs_build" and res.exit_code == 4
