"""P6: occupancy calculator, setmaxnreg pool rule (measured H200 hang limit), `kdev regs`. No GPU."""
from __future__ import annotations

import json

import pytest

from kdev.floor.gpus import load_profile
from kdev.occupancy import calc


@pytest.fixture
def h200():
    return load_profile("h200")


def test_warps_per_smsp_rule(h200):
    assert calc.max_regs_for_warps(h200, 5) == 96          # DESIGN 5.10: 5 warps/SMSP caps at 96
    assert calc.max_regs_for_warps(h200, 4) == 128
    assert calc.max_regs_for_warps(h200, 3) == 168
    assert calc.max_regs_for_warps(h200, 1) == 255


def test_occupancy_limiters(h200):
    r = calc.occupancy(h200, "k", 128, 256, 0)
    assert r.ctas_per_sm["regs"] == 2 and r.ctas_per_sm["warps"] == 8 and r.limiter == "regs"
    assert r.active_ctas == 2 and r.warps_per_smsp == 4
    assert r.reg_cap_for_next_warp == 80                    # 3 CTAs -> 6 warps/SMSP -> 80 regs
    r = calc.occupancy(h200, "k", 32, 128, 100 * 1024)
    assert r.limiter == "smem" and r.active_ctas == 2
    r = calc.occupancy(h200, "k", 16, 32, 0)
    assert r.limiter == "slots" and r.active_ctas == 32


def test_occupancy_launch_failures(h200):
    assert calc.occupancy(h200, "k", 104, 640, 0).errors       # 20 warps x 104 regs do not fit
    assert calc.occupancy(h200, "k", 32, 2048, 0).errors
    assert calc.occupancy(h200, "k", 32, 128, 240 * 1024).errors


def test_setmaxnreg_measured_hang_limit(h200):
    ok = calc.occupancy(h200, "moe", 96, 640, 200 * 1024, [(128, 24), (512, 112)])
    assert not ok.errors and ok.active_ctas == 1
    assert ok.setmaxnreg["pool"] == 640 * 96 and ok.setmaxnreg["max_feasible"][1] == 112
    hang = calc.occupancy(h200, "moe", 96, 640, 200 * 1024, [(128, 24), (512, 120)])
    assert any("HANG" in e for e in hang.errors)
    bad = calc.setmaxnreg_check(h200, 96, 640, [(128, 20), (512, 112)])
    assert any("multiple of 8" in e for e in bad["errors"])
    part = calc.setmaxnreg_check(h200, 96, 640, [(96, 24)])
    assert part["warnings"]


def test_parsers(sample_ptxas, fixtures_dir):
    ks = calc.parse_ptxas_minimal(sample_ptxas)
    assert ks and ks[0]["regs"] == 80 and "attn_kernel" in ks[0]["kernel"]
    res = calc.parse_res_usage((fixtures_dir / "sass" / "pdl_spill.res").read_text())
    assert [(k["kernel"], k["regs"], k["stack"]) for k in res] == [("spill_k", 64, 784), ("pdl_dot", 29, 0)]
    one_line = calc.parse_res_usage("Function _Z1kv:REG:40 STACK:0 SHARED:1024 LOCAL:0")
    assert one_line == [{"kernel": "_Z1kv", "regs": 40, "smem_static": 1024, "stack": 0, "local": 0}]
    assert calc.launch_bounds("__launch_bounds__(384, 1) __global__ void a(); __launch_bounds__( 384 ) void b();") == [384]
    assert calc.parse_groups("128:24, 512:112") == [(128, 24), (512, 112)]


def test_regs_cli_on_ptxas_log(fixtures_dir, capsys):
    from kdev.cli import main
    code = main(["regs", str(fixtures_dir / "ptxas" / "vjc.ptxas"), "--block", "384", "--json"])
    d = json.loads(capsys.readouterr().out)
    assert code == 0 and d["tool"] == "regs"
    ks = d["summary"]["kernels"]
    assert ks and all(k["block"] == 384 for k in ks)
    code = main(["regs", str(fixtures_dir / "ptxas" / "vjc.ptxas"), "--json"])
    d = json.loads(capsys.readouterr().out)
    assert d["error"]["code"] == "invalid_request"            # no block size known


def test_regs_cli_res_usage_with_fake_cuobjdump(tmp_path, monkeypatch, fixtures_dir):
    from kdev.occupancy import commands
    from kdev.util import Proc
    so = tmp_path / "x.so"
    so.write_bytes(b"\0")
    txt = (fixtures_dir / "sass" / "pdl_spill.res").read_text()
    ks, src = commands.kernels_for(str(so), runner=lambda argv, **kw: Proc(list(argv), 0, txt, ""))
    assert src.startswith("cuobjdump") and {k["kernel"] for k in ks} == {"spill_k", "pdl_dot"}
