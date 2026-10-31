"""kdev slim (CPU only): blob decoding, dead-route dropping per a dispatch trace, refusals, CLI."""
from __future__ import annotations

import base64
import hashlib
import json
import zlib
from argparse import Namespace

import pytest

from kdev.contract import KdevError
from kdev.slim import core


def blob(s: str) -> str:
    return base64.b64encode(zlib.compress(s.encode())).decode()


EMB = '''
DEAD3_SRC = "__global__ void emb_dead_k(int* q) {}"
EMB_LIVE_SRC = "__global__ void emb_live_k(int* q) {}"
def emb_dead_build():
    return load_inline(name="e", cuda_sources=DEAD3_SRC)
def emb_live_build():
    return load_inline(name="f", cuda_sources=EMB_LIVE_SRC)
def emb_prepare():
    try:
        emb_dead_build()
    except Exception:
        pass
    return emb_live_build()
'''

PAD = "// " + "x" * 3000 + "\n"


def stack(dead_src: str = "__global__ void dead_k(float* y) {}") -> str:
    return f'''import base64 as _b, zlib as _z
import os


def _zsrc(p):
    return _z.decompress(_b.b64decode(p)).decode()


LIVE_SRC = _zsrc("{blob("__global__ void __launch_bounds__(128) live_k(int* x) {}" + PAD)}")
DEAD_SRC = {dead_src!r}
DEAD2_SRC = "__global__ void dead2_k(float* y) {{}}"
_EMB = "{blob(EMB)}"


def build_live():
    return load_inline(name="l", cuda_sources=LIVE_SRC)


def build_dead():
    return load_inline(name="d", cuda_sources=DEAD_SRC)


def build_dead2():
    return load_inline(name="d2", cuda_sources=DEAD2_SRC)


def prepare():
    try:
        build_dead()
    except Exception:
        pass
    build_dead2()
    return build_live()


NS = {{"load_inline": load_inline}}
exec(compile(_z.decompress(_b.b64decode(_EMB)).decode(), "emb", "exec"), NS)
'''


def trace(*kernels: str) -> dict:
    ks = [{"kernel": f"void {k}<1, 2>(int*)", "count": 1} for k in kernels]
    return {"eager": {"s1": ks}, "graph": {"s1": ks}, "static_kernels": []}


def run_module(src: str) -> dict:
    calls: list[str] = []

    def load_inline(name, cuda_sources):
        calls.append(name)
        return cuda_sources

    g = {"load_inline": load_inline, "__name__": "m"}
    exec(compile(src, "m", "exec"), g)
    g["_calls"] = calls
    return g


TR = trace("live_k", "emb_live_k")


def test_decode_blobs_plain_and_equivalent() -> None:
    src = stack()
    out, n = core.decode_blobs(src)
    assert n == 2
    assert "b64decode" not in out and "decompress(" not in out and "_zsrc" not in out
    assert "import base64" not in out and "import os" in out
    a, b = run_module(src), run_module(out)
    assert a["LIVE_SRC"] == b["LIVE_SRC"] and a["NS"]["EMB_LIVE_SRC"] == b["NS"]["EMB_LIVE_SRC"]
    assert 'exec(compile(_EMB, "emb", "exec"), NS)' in out


def test_drop_dead_routes_safe_only() -> None:
    out, rep = core.slim(stack(), TR, allow_grow=True)
    names = {(d["where"], d["name"]) for d in rep.dropped}
    assert ("main", "build_dead") in names and ("main:_EMB", "emb_dead_build") in names
    assert "DEAD_SRC =" not in out and "dead_k" not in out and "emb_dead_k" not in out
    kept = {d["name"]: d["reason"] for d in rep.kept_dead}
    assert "outside try" in kept["build_dead2"]                     # unconditional call: not safe
    g = run_module(out)
    assert g["prepare"]() == g["LIVE_SRC"]                         # degraded try path, live route intact
    assert g["NS"]["emb_prepare"]() == g["NS"]["EMB_LIVE_SRC"]
    with pytest.raises(RuntimeError, match="kdev slim: route build_dead removed"):
        g["build_dead"]()
    # force + all-dead
    out2, rep2 = core.slim(stack(), TR, allow_grow=True, force=("build_dead2",))
    assert "build_dead2" in {d["name"] for d in rep2.dropped}
    _, rep3 = core.slim(stack(), TR, allow_grow=True, all_dead=True)
    assert "build_dead2" in {d["name"] for d in rep3.dropped} and rep3.warnings


def test_traced_kernel_keeps_route_and_name_dispatch_keeps() -> None:
    _, rep = core.slim(stack(), trace("live_k", "emb_live_k", "dead_k"), allow_grow=True)
    assert "build_dead" not in {d["name"] for d in rep.dropped}
    src = stack() + '\nPREBUILD = ("build_dead",)\n'
    _, rep = core.slim(src, TR, allow_grow=True)
    assert "dynamic" in {d["name"]: d["reason"] for d in rep.kept_dead}["build_dead"]
    _, rep = core.slim(stack(), TR, allow_grow=True, keep=("build_dead",))
    assert "build_dead" not in {d["name"] for d in rep.dropped}


def test_refusals() -> None:
    with pytest.raises(core.SlimRefused, match="does not parse"):
        core.slim("def x(:\n", TR)
    with pytest.raises(core.SlimRefused, match="dynamic blob"):
        core.slim("import base64, zlib\nv = 'x'\ns = zlib.decompress(base64.b64decode(v + 'y')).decode()\n", TR)
    with pytest.raises(core.SlimRefused, match="larger than the input"):
        core.slim(stack(), TR, drop=False)                            # PAD compresses: decoding grows the file
    with pytest.raises(core.SlimRefused, match="submission limit"):
        core.slim(stack(), TR, allow_grow=True, limit=1000)

    def bad_hook(src: str, ctx: dict) -> str:
        return src.replace("live_k", "other_k")
    with pytest.raises(core.SlimRefused, match="launched in the trace would be removed: emb_live_k, live_k"):
        core.slim(stack(), TR, allow_grow=True, hook=bad_hook)


def test_kernels_in_and_lit() -> None:
    txt = "template<int N> __global__ void __launch_bounds__(256, 1) foo(int* x) {}\n__global__ void bar(){}"
    assert core.kernels_in(txt) == {"foo", "bar"}
    for s in ("a'''b", 'x"""y', "ends\\", "tab\t'\"", "plain"):
        assert eval(core.lit(s)) == s


def _args(**kw) -> Namespace:
    d = {"stack": "", "out": "", "trace": None, "drop": None, "keep": None, "no_drop": False, "hook": None,
         "max_bytes": None, "allow_grow": True, "any_trace": False, "force": False}
    d.update(kw)
    return Namespace(**d)


class _Ctx:
    problem = None
    cfg = None


def test_cli(tmp_path) -> None:
    from kdev.slim.commands import cmd_slim

    s = tmp_path / "stk.py"
    s.write_text(stack(), encoding="utf-8")
    o = tmp_path / "out.py"
    with pytest.raises(KdevError) as e:
        cmd_slim(_args(stack=str(s), out=str(o)), _Ctx())
    assert "kdev dispatch" in e.value.recovery
    run = tmp_path / "run"
    run.mkdir()
    (run / "dispatch.json").write_text(json.dumps(TR))
    (run / "job.json").write_text(json.dumps({"files": [{"md5": "0" * 32}]}))
    with pytest.raises(KdevError, match="another file version"):
        cmd_slim(_args(stack=str(s), out=str(o), trace=str(run)), _Ctx())
    (run / "job.json").write_text(json.dumps({"files": [{"md5": hashlib.md5(s.read_bytes()).hexdigest()}]}))
    r = cmd_slim(_args(stack=str(s), out=str(o), trace=str(run)), _Ctx())
    assert r.verdict == "ok" and o.exists() and r.summary["md5_out"] == hashlib.md5(o.read_bytes()).hexdigest()
    assert any("dropped main:build_dead" in ln for ln in r.lines)
    with pytest.raises(KdevError, match="exists"):
        cmd_slim(_args(stack=str(s), out=str(o), trace=str(run)), _Ctx())
    r = cmd_slim(_args(stack=str(s), out=str(tmp_path / "o2.py"), trace=str(run), allow_grow=False, no_drop=True),
                 _Ctx())
    assert r.verdict == "refused" and not (tmp_path / "o2.py").exists()
