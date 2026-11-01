"""P1: config, run dirs + paging, output shaping, util helpers, CLI error handling, Context.submit, doctor."""
from __future__ import annotations

import json
import os
import re
import sys
import types
from pathlib import Path

import pytest

from kdev import cli
from kdev.config import load_config
from kdev.contract import EXIT_CODES, ErrorCode, Fidelity, KdevError, Result, validate_result
from kdev.models import Check, FileRef, JobSpec
from kdev.output import MAX_LINES, fidelity_line, fmt_us, render, table
from kdev.registry import Arg, Context, Registry
from kdev.runs import list_runs, new_run, open_run, show
from kdev.util import (atomic_write_text, network_path_reason, norm_md5, norm_source, run, slugify)


def _err(fn, *a, **kw) -> KdevError:
    with pytest.raises(KdevError) as ei:
        fn(*a, **kw)
    return ei.value


# ------------------------------------------------------------------------------------------------ config
def test_config_defaults_and_home_precedence(tmp_home, tmp_path):
    cfg = load_config()
    assert cfg.home == tmp_home and cfg.source == "" and cfg.runs_dir == tmp_home / "runs"
    assert cfg.lock_path == os.environ["KDEV_LOCK_PATH"]
    assert load_config(home=str(tmp_path / "other")).home == tmp_path / "other"


def test_config_file_in_home(tmp_home):
    tmp_home.mkdir()
    (tmp_home / "kdev.toml").write_text(
        'gpu = "h200"\nmax_hold_s = 900\nbogus = 1\n'
        '[toolchain]\nnvcc = "13.0.88"\ncuda_home = "/root/cuda130w"\n'
        '[remote]\nhost = "1.2.3.4"\nport = 22022\n[sync]\npaths = ["/root/mad"]\n', encoding="utf-8")
    cfg = load_config()
    assert cfg.source.endswith("kdev.toml")
    assert (cfg.gpu, cfg.max_hold_s, cfg.nvcc_pin, cfg.cuda_home) == ("h200", 900, "13.0.88", "/root/cuda130w")
    assert (cfg.remote.host, cfg.remote.port, cfg.sync_paths) == ("1.2.3.4", 22022, ["/root/mad"])
    assert any("bogus" in w for w in cfg.warnings)
    assert cfg.home == tmp_home  # file `home` never beats $KDEV_HOME


def test_config_errors(tmp_path, monkeypatch):
    bad = tmp_path / "bad.toml"
    bad.write_text("x = = 1", encoding="utf-8")
    assert _err(load_config, path=str(bad)).code == ErrorCode.INVALID_REQUEST
    typed = tmp_path / "typed.toml"
    typed.write_text('max_hold_s = "soon"', encoding="utf-8")
    e = _err(load_config, path=str(typed))
    assert e.code == ErrorCode.INVALID_REQUEST and e.details["field"] == "max_hold_s"
    monkeypatch.setenv("KDEV_CONFIG", str(tmp_path / "missing.toml"))
    assert _err(load_config).code == ErrorCode.INVALID_REQUEST


def test_config_env_and_problem_dir(tmp_path, monkeypatch):
    (tmp_path / "probs" / "muse_x").mkdir(parents=True)
    cfgf = tmp_path / "k.toml"
    cfgf.write_text(f'problems_root = "{(tmp_path / "probs").as_posix()}"\ndefault_problem = "muse_x"\n',
                    encoding="utf-8")
    monkeypatch.setenv("KDEV_CONFIG", str(cfgf))
    monkeypatch.setenv("KDEV_GPU", "b200")
    cfg = load_config()
    assert cfg.gpu == "b200"
    assert cfg.problem_dir() == tmp_path / "probs" / "muse_x"
    assert cfg.problem_dir("nope") is None


# ------------------------------------------------------------------------------------------------ runs
def test_new_run_ids_unique_and_formatted():
    cfg = load_config()
    a, b = new_run(cfg, "sass diff", "Tile 128!"), new_run(cfg, "sass diff", "Tile 128!")
    assert re.fullmatch(r"\d{8}-\d{6}-sass-diff-tile-128", a.run_id)
    assert a.run_id != b.run_id and a.path.is_dir() and b.path.is_dir()


def test_write_result_open_list():
    cfg = load_config()
    rd = new_run(cfg, "ab", "x")
    rd.append_jsonl("raw.jsonl", {"us": 1.0})
    p = rd.write_result(Result(tool="ab", verdict="faster", summary={"ratio": 0.97}))
    d = json.loads(p.read_text(encoding="utf-8"))
    assert validate_result(d) == [] and d["run_id"] == rd.run_id
    assert d["files"]["dir"] == str(rd.path) and d["files"]["raw"].endswith("raw.jsonl")
    assert open_run(cfg, rd.run_id[:12]).run_id == rd.run_id   # unique prefix
    assert open_run(cfg, str(rd.path)).run_id == rd.run_id     # a path
    new_run(cfg, "time")
    assert _err(open_run, cfg, rd.run_id[:4]).code == ErrorCode.INVALID_REQUEST  # ambiguous
    assert _err(open_run, cfg, "zzz").code == ErrorCode.NOT_FOUND
    rows = list_runs(cfg, tool="ab")
    assert [r["run_id"] for r in rows] == [rd.run_id] and rows[0]["verdict"] == "faster"
    assert len(list_runs(cfg)) == 2 and len(list_runs(cfg, limit=1)) == 1


def test_show_paging_sections_and_safety():
    cfg = load_config()
    rd = new_run(cfg, "sass")
    (rd.path / "k.sass").write_text("".join(f"line {i}\n" for i in range(150)) + "x" * 1000 + "\n",
                                    encoding="utf-8")
    rd.write_result(Result(tool="sass", verdict="ok", files={"sass": str(rd.path / "k.sass")},
                           lines=["hot loop: 42 instr"]))
    pg = show(cfg, rd.run_id, "sass")
    assert (pg.total, len(pg.lines), pg.next_offset) == (151, 60, 60) and pg.lines[0] == "line 0"
    last = show(cfg, rd.run_id, "sass", offset=120, limit=60)
    assert last.next_offset is None and len(last.lines) == 31
    assert "[... +600 chars]" in last.lines[-1]
    assert "hot loop: 42 instr" in show(cfg, rd.run_id).lines   # summary = rendered result
    assert show(cfg, rd.run_id, "k.sass").total == 151          # any file in the run dir
    assert json.loads("\n".join(show(cfg, rd.run_id, "result", limit=1000).lines))["tool"] == "sass"
    outside = rd.path.parent / "secret.txt"
    outside.write_text("no", encoding="utf-8")
    assert _err(show, cfg, rd.run_id, "../secret.txt").code == ErrorCode.NOT_FOUND
    assert _err(show, cfg, rd.run_id, "sass", limit=0).code == ErrorCode.INVALID_REQUEST


def test_cli_show_and_runs_list(capsys):
    cfg = load_config()
    rd = new_run(cfg, "build")
    (rd.path / "log.txt").write_text("".join(f"l{i}\n" for i in range(100)), encoding="utf-8")
    assert cli.main(["show", rd.run_id, "--section", "log", "--offset", "10", "--limit", "5"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[1] == "l10" and "--offset 15" in out[-1]
    assert cli.main(["show", rd.run_id, "--section", "log", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["summary"]["next_offset"] == 60 and len(d["summary"]["lines"]) == 60
    assert cli.main(["runs", "list"]) == 0
    assert rd.run_id in capsys.readouterr().out


# ------------------------------------------------------------------------------------------------ output
def test_fidelity_line_matches_plan_and_caps_optouts():
    f = Fidelity(toolchain={"nvcc": "13.0.88", "pin": "13.0.88", "match": True},
                 flush={"method": "read", "bytes": 268435456}, graph=True, exclusive="ok",
                 seeds=[11, 12, 13, 14, 15], overhead_us={"subtracted": 4.0})
    assert fidelity_line(f) == ("fidelity: nvcc 13.0.88=pin read-flush 256MB graph excl=ok seeds=5 "
                                "overhead=-4.00us optouts=none")
    g = Fidelity(toolchain={"nvcc": "12.8.93", "pin": "13.0.88", "match": False},
                 flush={"method": "write", "bytes": 268435456}, graph=False, exclusive="shared",
                 overhead_us={"subtracted": 0.0}, optouts=["flush=write", "eager"])
    line = fidelity_line(g)
    for s in ("NVCC 12.8.93!=PIN 13.0.88", "FLUSH=WRITE", "EAGER", "EXCL=SHARED", "OVERHEAD=NONE",
              "optouts=FLUSH=WRITE,EAGER"):
        assert s in line


def test_table_truncation_is_explicit():
    lines = table(["a", "b"], [[i, i * 1.5] for i in range(20)], max_rows=15, more_hint="/x/rows.csv")
    assert len(lines) == 17 and lines[-1] == "... 5 more rows in /x/rows.csv"
    assert fmt_us(21.34) == "21.3" and fmt_us(4.256) == "4.26"


def test_render_budget_hints_and_quiet():
    r = Result(tool="sass", verdict="ok", run_id="r1", files={"dir": "/d"}, summary={"instr": 42},
               lines=[f"row {i}" for i in range(100)], warnings=[f"w{i}" for i in range(5)],
               next=["kdev a", "kdev b", "kdev c"])
    out = render(r).splitlines()
    assert len(out) <= MAX_LINES and out[0] == "OK  sass  instr=42"
    assert any("more lines (kdev show r1 --section summary)" in x for x in out)
    assert "... 2 more warnings (--json)" in out and out[-1] == "next: kdev b" and "run: r1  dir: /d" in out
    assert not any(x.startswith("next:") for x in render(r, quiet=True).splitlines())


def test_render_shapes_table():
    r = Result(tool="ab", verdict="faster", shapes=[
        {"shape_id": "d1/sliding", "per_file": [{"label": "A", "median_us": 21.3}, {"label": "B", "median_us": 20.7}],
         "ratio": {"median": 0.973, "ci_lo": 0.961, "ci_hi": 0.985}, "verdict": "faster"}])
    text = render(r)
    assert "d1/sliding" in text and "x0.973" in text and "[0.961, 0.985]" in text


# ------------------------------------------------------------------------------------------------ util
def test_norm_md5_ignores_comments_keeps_preprocessor(tmp_path):
    a, b, c = tmp_path / "a.py", tmp_path / "b.py", tmp_path / "c.py"
    a.write_text('src = """\n#include <cuda.h>\n__global__ void k() {}\n"""\nx = 1\n', encoding="utf-8")
    b.write_text('# v2 resubmit\nsrc = """\n#include <cuda.h>  // hdr\n/* blk */ __global__ void k()  {}\n"""\n'
                 'x  =  1   # same\n', encoding="utf-8")
    c.write_text('src = """\n#include <cuda_fp8.h>\n__global__ void k() {}\n"""\nx = 1\n', encoding="utf-8")
    assert norm_md5(str(a)) == norm_md5(str(b)) != norm_md5(str(c))
    assert norm_source("#pragma unroll\nfoo  # c") == "#pragmaunrollfoo"


def test_util_run_missing_exe_and_atomic_write(tmp_path):
    p = run(["definitely-not-a-binary-kdev"])
    assert p.returncode == 127
    p = run([sys.executable, "-c", "print('hi')"])
    assert p.returncode == 0 and p.stdout.strip() == "hi"
    f = atomic_write_text(tmp_path / "d" / "x.txt", "a\nb\n")
    assert f.read_bytes() == b"a\nb\n" and not [x for x in f.parent.iterdir() if x.suffix == ".tmp"]
    assert slugify("Sass Diff / X!") == "sass-diff-x"


def test_network_path_reason(tmp_path):
    mounts = tmp_path / "mounts"
    mounts.write_text("overlay / overlay rw 0 0\nmfs#x /mnt/net fuse.mfs rw 0 0\nsrv:/e /nfs nfs4 rw 0 0\n",
                      encoding="utf-8")
    assert "network volume" in network_path_reason("/workspace/kdev", str(mounts))
    assert "fuse.mfs" in network_path_reason("/mnt/net/a/b", str(mounts))
    assert "nfs4" in network_path_reason("/nfs/x", str(mounts))
    assert network_path_reason("/root/.kdev", str(mounts)) is None


# ------------------------------------------------------------------------------------------------ CLI
def _mini_registry(handler) -> Registry:
    reg = Registry()
    reg.command("hello", "test", handler, args=[Arg(("--n",), "n", type="int", default=1)])
    reg.command("grp a", "sub a", handler)
    reg.command("grp b", "sub b", handler)
    return reg


def test_cli_json_is_one_object_even_if_handler_prints(monkeypatch, capsys):
    def handler(args, ctx):
        print("noise from a handler")
        return Result(tool="hello", summary={"n": args.n, "agent": ctx.agent})

    monkeypatch.setattr(cli, "build_registry", lambda: _mini_registry(handler))
    assert cli.main(["hello", "--n", "3", "--json"]) == 0
    cap = capsys.readouterr()
    d = json.loads(cap.out)
    assert d["summary"] == {"n": 3, "agent": "pytest"} and "noise" in cap.err


def test_cli_errors_are_results(monkeypatch, capsys):
    def boom(args, ctx):
        raise RuntimeError("kaputt")

    monkeypatch.setattr(cli, "build_registry", lambda: _mini_registry(boom))
    assert cli.main(["hello", "--json"]) == EXIT_CODES[ErrorCode.INTERNAL]
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "internal"
    assert cli.main(["hello", "--n", "x", "--json"]) == 2          # argparse error -> invalid_request
    d = json.loads(capsys.readouterr().out)
    assert d["error"]["code"] == "invalid_request" and validate_result(d) == []
    assert cli.main(["helo", "--json"]) == 2
    assert "hello" in json.loads(capsys.readouterr().out)["error"]["details"]["suggestions"]
    assert cli.main(["grp"]) == 0 and "grp a" in capsys.readouterr().out          # group help
    assert cli.main(["grp", "zz", "--json"]) == 2
    assert json.loads(capsys.readouterr().out)["error"]["details"]["subcommands"] == ["grp a", "grp b"]


def test_cli_bad_config_is_invalid_request(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("KDEV_CONFIG", str(tmp_path / "nope.toml"))
    assert cli.main(["runs", "list", "--json"]) == 2
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "invalid_request"


# ------------------------------------------------------------------------------------------------ submit
def test_submit_no_gpu_is_capability_unavailable():
    ctx = Context(agent="me")
    e = _err(ctx.submit, JobSpec(kind="time", files=[FileRef("/a.py")]))
    assert e.code == ErrorCode.CAPABILITY_UNAVAILABLE and e.details["job_id"].startswith("j")


def test_submit_routes_to_service_client(monkeypatch):
    from kdev.service import client

    monkeypatch.delenv("KDEV_NO_GPU")
    seen = {}

    def fake(cfg, spec, *, detach=False, direct=False, timeout_s=None):
        seen.update(spec=spec, detach=detach, direct=direct, home=cfg.home)
        return Result(tool=spec.kind, verdict="pending", job_id=spec.job_id)

    monkeypatch.setattr(client, "submit_and_maybe_wait", fake)
    ctx = Context(agent="moe-w3", problem="muse_moe")
    args = types.SimpleNamespace(detach=True, direct=False, timeout=60, priority=5, run_name="tile128")
    r = ctx.submit_from_args(JobSpec(kind="ab"), args)
    s = seen["spec"]
    assert r.verdict == "pending" and seen["detach"] is True
    assert (s.agent, s.problem, s.priority, s.run_name, s.timeout_s) == ("moe-w3", "muse_moe", 5, "tile128", 60)
    assert s.job_id and s.submitted_utc.endswith("Z")


# ------------------------------------------------------------------------------------------------ doctor
NVCC_OUT = "nvcc: NVIDIA (R) Cuda compiler driver\nCuda compilation tools, release 13.0, V13.0.88\n"
PTXAS_OUT = "ptxas: NVIDIA (R) Ptx optimizing assembler\nCuda compilation tools, release 13.0, V13.0.88\n"


def _doctor(monkeypatch, runner, modules=("kdev.service", "kdev.build", "kdev.remote"), **kw):
    from kdev.doctor import run_doctor
    return run_doctor(Context(), runner=runner, modules=modules, **kw)


def _by_name(r: Result) -> dict[str, dict]:
    return {c["name"]: c for c in r.summary["checks"]}


def _stub_probes(monkeypatch):
    """Make the three fallback areas use the P1 fallback regardless of whether P2/P5 have landed."""
    for name in ("kdev.service", "kdev.build", "kdev.remote"):
        mod = types.ModuleType(name)

        def probe(ctx, problem=None, _n=name):
            raise NotImplementedError(_n)

        mod.probe = probe
        monkeypatch.setitem(sys.modules, name, mod)


def test_doctor_toolchain_matches_pin(monkeypatch, fake_runner, tmp_home):
    _stub_probes(monkeypatch)
    tmp_home.mkdir()
    (tmp_home / "toolchain.lock").write_text('schema = "kdev.toolchain/1"\nnvcc_version = "13.0.88"\n',
                                             encoding="utf-8")
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", "/root/.cache/torch_ext_cu130")
    runner = fake_runner({("nvcc", "--version"): (0, NVCC_OUT, ""), ("ptxas", "--version"): (0, PTXAS_OUT, "")})
    r = _doctor(monkeypatch, runner)
    c = _by_name(r)
    assert c["toolchain.pin"]["status"] == "PASS" and c["toolchain.nvcc"]["status"] == "PASS"
    assert "13.0.88=pin" in c["toolchain.nvcc"]["detail"] and c["ext.cache"]["status"] == "PASS"
    assert c["gpu.visible"]["status"] == "SKIP"                  # KDEV_NO_GPU=1 in unit tests
    assert c["lock"]["status"] == "PASS" and c["disk"]["status"] in ("PASS", "WARN")
    assert validate_result(r.to_dict()) == [] and r.status == "ok"
    assert not any("okc" in " ".join(a) for a in runner.calls)   # okc presence is never executed


def test_doctor_toolchain_mismatch_fails(monkeypatch, fake_runner):
    _stub_probes(monkeypatch)
    monkeypatch.delenv("KDEV_NO_GPU")
    monkeypatch.delenv("TORCH_EXTENSIONS_DIR", raising=False)
    cfgf = Path(os.environ["KDEV_HOME"]).parent / "k.toml"
    cfgf.write_text('[toolchain]\nnvcc = "13.0.88"\n', encoding="utf-8")
    monkeypatch.setenv("KDEV_CONFIG", str(cfgf))
    runner = fake_runner({
        ("nvcc", "--version"): (0, NVCC_OUT.replace("13.0, V13.0.88", "12.8, V12.8.93"), ""),
        ("ptxas", "--version"): (0, PTXAS_OUT, ""),
        ("nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total"):
            (0, "NVIDIA H200, GPU-abc, 580.65, 143771 MiB\n", ""),
        ("nvidia-smi", "--query-compute-apps=pid,process_name,used_memory"): (0, "1234, python, 2000 MiB\n", ""),
    })
    r = _doctor(monkeypatch, runner, fix_hints=True)
    c = _by_name(r)
    assert c["toolchain.nvcc"]["status"] == "FAIL" and "12.8.93 != pin 13.0.88" in c["toolchain.nvcc"]["detail"]
    assert c["toolchain.mixed"]["status"] == "WARN" and c["ext.cache"]["status"] == "WARN"
    assert c["gpu.visible"]["status"] == "PASS" and "H200" in c["gpu.visible"]["detail"]
    assert c["gpu.foreign"]["status"] == "WARN"
    assert r.status == "fail" and r.verdict == "fail" and r.exit_code == 1
    assert r.lines[0].startswith("FAIL") and any("hint:" in x for x in r.lines)
    assert len(render(r).splitlines()) <= MAX_LINES


def test_doctor_probe_results_and_crashes(monkeypatch, fake_runner):
    good = types.ModuleType("kdev_fake_good")
    good.probe = lambda ctx, problem=None: [Check("x.ok", "PASS", "fine"), {"name": "x.d", "status": "WARN"}]
    bad = types.ModuleType("kdev_fake_bad")

    def boom(ctx, problem=None):
        raise ValueError("oops")

    bad.probe = boom
    stub = types.ModuleType("kdev_fake_stub")

    def nyi(ctx, problem=None):
        raise NotImplementedError

    stub.probe = nyi
    for m in (good, bad, stub):
        monkeypatch.setitem(sys.modules, m.__name__, m)
    r = _doctor(monkeypatch, fake_runner({}), modules=("kdev_fake_good", "kdev_fake_bad", "kdev_fake_stub"))
    c = _by_name(r)
    assert c["x.ok"]["status"] == "PASS" and c["x.d"]["status"] == "WARN"
    assert c["kdev_fake_bad.probe"]["status"] == "FAIL" and "oops" in c["kdev_fake_bad.probe"]["detail"]
    assert c["kdev_fake_stub.probe"]["status"] == "SKIP"
    assert {"config", "home", "gpu.profile", "okc"} <= set(c)


def test_doctor_gpu_profile_and_okc_root(monkeypatch, fake_runner, tmp_path):
    cfgf = tmp_path / "k.toml"
    cfgf.write_text(f'gpu = "h200"\nokc_root = "{(tmp_path / "okc").as_posix()}"\n', encoding="utf-8")
    monkeypatch.setenv("KDEV_CONFIG", str(cfgf))
    c = _by_name(_doctor(monkeypatch, fake_runner({}), modules=()))
    assert c["gpu.profile"]["status"] == "PASS" and "132 SMs" in c["gpu.profile"]["detail"]
    assert c["okc"]["status"] == "WARN"                         # okc_root missing
    cfgf.write_text('gpu = "nosuchgpu"\n', encoding="utf-8")
    assert _by_name(_doctor(monkeypatch, fake_runner({}), modules=()))["gpu.profile"]["status"] == "FAIL"


def test_doctor_cli_end_to_end(capsys):
    code = cli.main(["doctor", "--json"])
    d = json.loads(capsys.readouterr().out)
    assert validate_result(d) == [] and d["tool"] == "doctor" and code == d["exit_code"]


def test_every_command_help_renders():
    """board #155: a bare '%' in a help string crashed `kdev floor --help` (argparse %-formats help)."""
    from kdev.cli import build_parser
    from kdev.registry import build_registry

    for cmd in build_registry().all():
        build_parser(cmd).format_help()
