"""`kdev build`: compile-only builds, never under the GPU lock, never with a CUDA context. Owner: P5.

Strategy for torch load_inline/load modules: import the module in a subprocess (the pinned interpreter) with
CUDA_VISIBLE_DEVICES="" and an intercepted ``torch.utils.cpp_extension.load_inline`` (``kdev/build/_shim.py``)
that builds every extension into the hash-keyed cache dir with ``-Xptxas -v -lineinfo`` (log captured), then
skips kernel launches. Builds are triggered by import side effects and by calling the module's zero-argument
build entry points (``_module``, ``_build*``, ``_ext*``, ...). Standalone .cu files: nvcc -cubin (and an
executable when the file has ``main()``) with the pinned toolchain, or NVRTC when nvcc cannot run (Windows
without cl.exe). Known-transient errors (missing thrust headers) are retried once (M20).

After a build every extension is checked: ``.so`` notes vs the pin (M2), ptxas -v per kernel (regs, spills,
C751x), cubins extracted and SASS dumped next to it (``sass.txt``) for ``kdev sass``.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from kdev.build import _shim, cache
from kdev.build.ptxas import parse_ptxas, serialization_by_kernel
from kdev.build.toolchain import build_env, pin_for, verify_so
from kdev.contract import ErrorCode, KdevError, Result
from kdev.models import JobSpec, KernelInfo, Toolchain
from kdev.util import Runner, md5_file, run

INFO = "kdev_info.json"
_ERR_LINE = re.compile(r"(error|fatal|undefined reference|No such file)", re.I)


@dataclass
class ExtBuild:
    name: str
    cache_key: str
    cache_hit: bool
    seconds: float
    so_path: str = ""
    so_notes: list[str] = field(default_factory=list)
    kernels: list[KernelInfo] = field(default_factory=list)
    log_path: str = ""
    cubins: list[str] = field(default_factory=list)


@dataclass
class BuildReport:
    file: str
    md5: str
    exts: list[ExtBuild] = field(default_factory=list)
    toolchain_match: bool = True
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def ok(self) -> bool:
        return not any(e.startswith(("build_failed", "toolchain_mismatch")) for e in self.errors)


def _arch(pin: Toolchain) -> str:
    return os.environ.get("KDEV_ARCH") or pin.extra.get("arch") or "sm_90a"


def _knob_flags(knobs: dict[str, Any] | None, defines: list[str] | None) -> list[str]:
    return _shim._knob_defines(dict(knobs or {}), list(defines or []))  # noqa: SLF001


def _first_errors(text: str, n: int = 8) -> list[str]:
    lines = [ln.rstrip() for ln in text.splitlines() if _ERR_LINE.search(ln) and "warning" not in ln.lower()]
    return lines[:n] or [ln.rstrip() for ln in text.splitlines()[-n:]]


def _read(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _postprocess(ext: ExtBuild, d: str, pin: Toolchain, runner: Runner, report: BuildReport, *,
                 refresh: bool) -> None:
    """ptxas parse, .so notes vs pin, cubin extraction + sass.txt; cached in kdev_info.json."""
    info_p = Path(d) / INFO
    if not refresh and info_p.is_file():
        try:
            info = json.loads(info_p.read_text(encoding="utf-8"))
            ext.kernels = [KernelInfo.from_dict(k) for k in info.get("kernels", [])]
            ext.so_notes = info.get("so_notes", [])
            ext.cubins = [c for c in info.get("cubins", []) if os.path.isfile(c)]
            match = info.get("toolchain_match", True)
            if not match:
                report.toolchain_match = False
                report.errors.append(f"toolchain_mismatch: {ext.name} notes {ext.so_notes} != pin "
                                     f"V{pin.nvcc_version}")
            return
        except (OSError, ValueError):
            pass
    log = _read(ext.log_path)
    ext.kernels = parse_ptxas(log)
    art = ext.so_path or (ext.cubins[0] if ext.cubins else "")
    match = True
    if art and pin.nvcc_version:
        match, ext.so_notes = verify_so(art, pin)
        if not match and ext.so_notes:
            report.toolchain_match = False
            report.errors.append(f"toolchain_mismatch: {ext.name} notes {ext.so_notes} != pin V{pin.nvcc_version}")
        elif not match:
            match = True   # no notes (e.g. NVRTC cubin / pure C++): unverifiable, not a mismatch
            ext.so_notes = ["(no toolkit notes)"]
    if ext.so_path and not ext.cubins:
        from kdev.sass.disasm import extract_cubins  # noqa: PLC0415

        try:
            ext.cubins = extract_cubins(ext.so_path, str(Path(d) / "cubins"), runner, tool=pin.cuobjdump or None)
        except KdevError:
            ext.cubins = []
    sass_src = ext.so_path or (ext.cubins[0] if ext.cubins else "")
    if sass_src:
        from kdev.sass.disasm import sass_text  # noqa: PLC0415

        try:
            (Path(d) / "sass.txt").write_text(sass_text(sass_src, runner=runner, tool=pin.cuobjdump or None),
                                              encoding="utf-8")
        except KdevError:
            pass
    sass_p = Path(d) / "sass.txt"
    if sass_p.is_file():
        from kdev.sass.parse import parse_sass  # noqa: PLC0415

        counts = {k.name: sum(1 for i in k.instrs if i.base != "NOP") for k in parse_sass(sass_p.read_text())}
        for k in ext.kernels:
            k.sass_instr = counts.get(k.name.split("@")[0])
    info = {"kernels": [k.to_dict() for k in ext.kernels], "so_notes": ext.so_notes, "cubins": ext.cubins,
            "toolchain_match": match, "pin": pin.nvcc_version, "ptxas_log": ext.log_path}
    info_p.write_text(json.dumps(info, indent=1), encoding="utf-8")


# ----------------------------------------------------------------------------------------------- .cu builds
def _nvcc_usable(pin: Toolchain) -> bool:
    if not pin.nvcc or not os.path.exists(pin.nvcc):
        return False
    return os.name != "nt" or bool(shutil.which("cl"))


def _build_cu(path: str, report: BuildReport, pin: Toolchain, home: str | None, knobs: dict, defines: list[str],
              force: bool, runner: Runner, backend: str) -> None:
    src = Path(path).read_bytes()
    arch = _arch(pin)
    flags = ["-O3", f"-arch={arch}", "-lineinfo", "-Xptxas=-v", *_knob_flags(knobs, defines)]
    key = cache.cache_key(src, pin, flags, knobs)
    d = cache.ext_dir(home or str(cache._home()), pin, key)  # noqa: SLF001
    name = Path(path).stem
    ext = ExtBuild(name=name, cache_key=key, cache_hit=False, seconds=0.0, log_path=str(Path(d) / "build.log"))
    report.exts.append(ext)
    if cache.entry_complete(d) and not force:
        ext.cache_hit = True
        info = json.loads((Path(d) / _shim.MARKER).read_text(encoding="utf-8"))
        ext.cubins = [info["cubin"]]
        ext.so_path = info.get("exe", "")
        _postprocess(ext, d, pin, runner, report, refresh=False)
        return
    Path(d).mkdir(parents=True, exist_ok=True)
    cubin = str(Path(d) / f"{name}.cubin")
    use_nvcc = backend == "nvcc" or (backend == "auto" and _nvcc_usable(pin))
    t0 = time.monotonic()
    log = ""
    exe = ""
    for attempt in (1, 2):
        if use_nvcc:
            env = build_env(pin, d)
            p = runner([pin.nvcc, "-cubin", *flags, path, "-o", cubin], timeout=3600, env=env)
            log = p.stdout + p.stderr
            ok = p.returncode == 0 and os.path.isfile(cubin)
            if ok and re.search(rb"\bint\s+main\s*\(", src):
                exe = str(Path(d) / (name + (".exe" if os.name == "nt" else "")))
                p2 = runner([pin.nvcc, "-O3", f"-arch={arch}", "-lineinfo", *_knob_flags(knobs, defines), path,
                             "-o", exe], timeout=3600, env=env)
                log += p2.stdout + p2.stderr
                ok = p2.returncode == 0
        else:
            from kdev.build import _nvrtc  # noqa: PLC0415

            if not pin.cuda_home:
                raise KdevError(ErrorCode.CAPABILITY_UNAVAILABLE, "no CUDA toolkit found for NVRTC",
                                recovery="set KDEV_CUDA_HOME or write a toolchain.lock")
            try:
                rc, log, cub = _nvrtc.compile_cubin(src.decode("utf-8", "replace"), d, cuda_home=pin.cuda_home,
                                                    arch=arch, flags=_knob_flags(knobs, defines),
                                                    name=f"{name}.cu")
            except (OSError, _nvrtc.NvrtcUnavailable) as e:
                raise KdevError(ErrorCode.CAPABILITY_UNAVAILABLE, f"NVRTC unavailable: {e}",
                                recovery="install the CUDA toolkit (nvcc or libnvrtc)") from e
            ok = rc == 0
            cubin = cub or cubin
        if ok or attempt == 2 or not _shim._TRANSIENT.search(log):  # noqa: SLF001
            break
    ext.seconds = round(time.monotonic() - t0, 3)
    Path(ext.log_path).write_text(log, encoding="utf-8")
    if not ok:
        report.errors.append(f"build_failed: {name}: " + " | ".join(_first_errors(log, 5)))
        return
    ext.cubins = [cubin]
    ext.so_path = exe
    marker = {"name": name, "kind": "cu", "key": key, "cubin": cubin, "exe": exe, "so_path": "",
              "seconds": ext.seconds, "tc_version": pin.nvcc_version, "flags": flags,
              "backend": "nvcc" if use_nvcc else "nvrtc"}
    (Path(d) / _shim.MARKER).write_text(json.dumps(marker, indent=1), encoding="utf-8")
    _postprocess(ext, d, pin, runner, report, refresh=True)


# ----------------------------------------------------------------------------------------------- .py builds
def _build_py(path: str, report: BuildReport, pin: Toolchain, home: str | None, knobs: dict, defines: list[str],
              force: bool, runner: Runner, python: str | None, timeout_s: float) -> list[dict]:
    root = cache.ext_root(home, pin)
    os.makedirs(root, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="kdev_build_", dir=os.path.join(str(cache._home(home)), "cache"))  # noqa: SLF001
    spec = {"file": os.path.abspath(path), "mode": "build", "ext_root": root, "tc_version": cache.tc_version(pin),
            "tc_ptxas": cache.tc_ident(pin), "knobs": knobs, "defines": defines, "force": force,
            "out": os.path.join(tmp, "out.json"), "cap": [int(x) for x in re.findall(r"\d", _arch(pin))[:2]] or [9, 0]}
    sp = os.path.join(tmp, "spec.json")
    py = python or pin.python or os.environ.get("KDEV_PYTHON") or sys.executable
    env = build_env(pin, root)
    shim = str(Path(_shim.__file__).resolve())
    n_pend = len(cache.pending_for(path, pin, home))
    shards = min(8, max(1, (os.cpu_count() or 4) // 4), n_pend) if n_pend > 1 else 0
    if shards:
        spec["skip_pending"] = True            # built below by `shards` parallel shim processes
    Path(sp).write_text(json.dumps(spec), encoding="utf-8")
    p = runner([py, shim, sp], timeout=timeout_s, env=env)
    pend_outs: list[dict[str, Any]] = []
    if shards:
        pend_outs = _parallel_pending(py, shim, spec, tmp, shards, env, runner, timeout_s)
    out: dict[str, Any] = {}
    try:
        out = json.loads(Path(spec["out"]).read_text(encoding="utf-8"))
        for po in pend_outs:
            out.setdefault("exts", []).extend(po.get("exts", []))
            out.setdefault("errors", []).extend(po.get("errors", []))
            out["pending_built"] = list(out.get("pending_built") or []) + list(po.get("pending_built") or [])
    except (OSError, ValueError):
        report.errors.append(f"build_failed: shim crashed (rc {p.returncode}): "
                             + " | ".join(_first_errors(p.stdout + p.stderr, 6)))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if p.returncode == 3 or (out.get("import_error") or "").startswith("torch unavailable"):
        raise KdevError(ErrorCode.CAPABILITY_UNAVAILABLE, f"{py}: {out.get('import_error') or 'torch unavailable'}",
                        recovery="set `python` in toolchain.lock (or KDEV_PYTHON) to the interpreter with torch")
    seen: set[str] = set()
    recs: list[dict] = []
    for r in out.get("exts", []):
        if r["key"] in seen:
            continue
        seen.add(r["key"])
        if not (r.get("status") != "ok" and r.get("origin") == "entry"):
            recs.append(r)                 # a failed default-args guess is not part of the file's manifest
        ext = ExtBuild(name=r["name"], cache_key=r["key"], cache_hit=bool(r.get("cache_hit")),
                       seconds=float(r.get("seconds", 0.0)), so_path=r.get("so_path", ""),
                       log_path=r.get("log_path", ""))
        report.exts.append(ext)
        if r.get("status") != "ok":
            first = " | ".join(_first_errors(_read(ext.log_path), 5))
            if r.get("origin") == "entry":
                # built by calling a build entry point with its default arguments (a guess): the file's real
                # prepare may never make this call, so it does not fail the build; discovery finds the real ones
                report.errors.append(f"warning: default-args entry build failed (not fatal): {r['name']}: "
                                     f"{first[:200]}")
            else:
                report.errors.append(f"build_failed: {r['name']}: {first}")
            continue
        _postprocess(ext, r["dir"], pin, runner, report, refresh=not ext.cache_hit)
    if out.get("import_error") and not recs:
        report.errors.append("build_failed: import failed: " + out["import_error"].splitlines()[0][:300])
    for e in out.get("errors", []):
        if e.startswith("extension ") and not any(e[10:40] in x for x in report.errors):
            report.errors.append(f"build_failed: {e[:300]}")
    pp = out.get("prepare_pass")
    if pp:
        ok_steps, errs = pp.get("steps") or [], pp.get("errors") or []
        msg = (f"warning: cpu prepare pass ({pp.get('seconds', 0)}s): {len(ok_steps)} step(s) ok, {len(errs)} "
               "failed (extensions built before a failure are cached; the rest are recorded by the first GPU run "
               "and built by the next `kdev build`)")
        if errs:
            msg += f"; first: {errs[0][:160]}"
        report.errors.append(msg)
    if out.get("pending_built"):
        report.errors.append(f"warning: built {len(out['pending_built'])} pending call(s) recorded by a GPU run")
    report.extra_entries = out.get("entries_called", [])  # type: ignore[attr-defined]
    return recs


def _parallel_pending(py: str, shim: str, spec: dict[str, Any], tmp: str, shards: int, env: dict[str, str],
                      runner: Runner, timeout_s: float) -> list[dict[str, Any]]:
    """Compile pending (prepare-time) calls in ``shards`` concurrent shim processes (one compile each at a time;
    a bf16 best records ~30). Returns each shard's out.json."""
    from concurrent.futures import ThreadPoolExecutor

    def one(i: int) -> dict[str, Any]:
        s = dict(spec, pending_only=True, shard=i, shards=shards, out=os.path.join(tmp, f"pend{i}.json"))
        sp = os.path.join(tmp, f"pend{i}_spec.json")
        Path(sp).write_text(json.dumps(s), encoding="utf-8")
        runner([py, shim, sp], timeout=timeout_s, env=env)
        try:
            return json.loads(Path(s["out"]).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"errors": [f"pending shard {i} crashed"]}

    with ThreadPoolExecutor(max_workers=shards) as ex:
        return list(ex.map(one, range(shards)))


def build_file(path: str, *, knobs: dict[str, Any] | None = None, defines: list[str] | None = None,
               problem: str | None = None, force: bool = False, verbose: bool = False, home: str | None = None,
               runner: Runner = run, backend: str = "auto", python: str | None = None,
               toolchain: Toolchain | None = None, timeout_s: float = 3600) -> BuildReport:
    """Build one submission (.py with load_inline/load, or .cu) into the cache; never touches the GPU.
    ``backend`` (.cu only): auto | nvcc | nvrtc. Errors are collected in ``report.errors`` with the prefixes
    'build_failed:' / 'toolchain_mismatch:'; missing tools raise KdevError(capability_unavailable)."""
    if not os.path.isfile(path):
        raise KdevError(ErrorCode.NOT_FOUND, f"no such file: {path}")
    knobs = dict(knobs or {})
    defines = [d for d in (defines or []) if d]
    pin = toolchain or pin_for(problem, path, runner=runner)
    md5 = md5_file(path)
    report = BuildReport(file=os.path.abspath(path), md5=md5)
    recs: list[dict] = []
    if path.endswith(".cu"):
        _build_cu(path, report, pin, home, knobs, defines, force, runner, backend)
        recs = [{"name": e.name, "key": e.cache_key,
                 "dir": cache.ext_dir(home or str(cache._home()), pin, e.cache_key)}  # noqa: SLF001
                for e in report.exts]
    elif path.endswith(".py"):
        recs = _build_py(path, report, pin, home, knobs, defines, force, runner, python, timeout_s)
    else:
        raise KdevError(ErrorCode.INVALID_REQUEST, f"cannot build {path}: expected .py or .cu",
                        recovery="pass a submission .py (load_inline) or a .cu file")
    if pin.source == "detect":
        report.errors.append("warning: no toolchain.lock: built with the detected toolchain "
                             f"V{pin.nvcc_version or '?'} (unpinned)")
    status = "ok" if not any(e.startswith(("build_failed", "toolchain_mismatch")) for e in report.errors) else \
        "failed"
    cache.write_manifest(home, cache.file_key(md5, knobs, defines, pin), {
        "file": report.file, "md5": md5, "knobs": knobs, "defines": defines, "status": status,
        "toolchain": {"nvcc_version": pin.nvcc_version, "ptxas": pin.ptxas, "source": pin.source},
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "exts": [{"name": r["name"], "key": r["key"], "dir": r["dir"]} for r in recs],
        "errors": report.errors})
    return report


# --------------------------------------------------------------------------------------------- reporting
def report_rows(reports: list[BuildReport], labels: list[str] | None = None) -> dict[str, list[dict]]:
    """Ledger-shaped rows (PLAN 3.4 builds/kernels columns) for result.json / build.json."""
    builds, kernels = [], []
    for n, rep in enumerate(reports):
        label = (labels[n] if labels else None) or ("A" if len(reports) == 1 else chr(ord("A") + n))
        for e in rep.exts:
            builds.append({"label": label, "file": rep.file, "md5": rep.md5, "ext_name": e.name,
                           "cache_key": e.cache_key, "cache_hit": int(e.cache_hit), "seconds": e.seconds,
                           "so_path": e.so_path, "so_notes": "; ".join(e.so_notes), "cubins": e.cubins,
                           "log": e.log_path})
            ser = serialization_by_kernel(_read(e.log_path)) if e.log_path else {}
            for k in e.kernels:
                c751 = sorted(set(re.findall(r"C751\d", " ".join(k.warnings))) | set(ser.get(k.name, [])))
                kernels.append({"label": label, "ext_name": e.name, "kernel": k.name, "regs": k.regs,
                                "spill_store": k.spill_store, "spill_load": k.spill_load, "stack": k.stack,
                                "smem_static": k.smem_static, "barriers": k.barriers, "c751x": ",".join(c751),
                                "sass_instr": k.sass_instr})
    return {"builds": builds, "kernels": kernels}


def build_result(reports: list[BuildReport], labels: list[str] | None = None, verbose: bool = False) -> Result:
    """Result envelope (tool 'build') with a <=40-line summary; full rows go to build.json by callers."""
    rows = report_rows(reports, labels)
    errs = [e for r in reports for e in r.errors]
    fail = [e for e in errs if e.startswith(("build_failed", "toolchain_mismatch"))]
    mismatch = any(e.startswith("toolchain_mismatch") for e in errs)
    n_ext = len(rows["builds"])
    hits = sum(b["cache_hit"] for b in rows["builds"])
    res = Result(tool="build", status="fail" if fail else "ok",
                 verdict=("toolchain_mismatch" if mismatch else "build_failed") if fail else "built")
    if fail:
        code = ErrorCode.TOOLCHAIN_MISMATCH if mismatch else ErrorCode.BUILD_FAILED
        res.status = "error"
        res.error = KdevError(code, fail[0].split(": ", 1)[-1][:400], details={"errors": fail[:10]},
                              recovery=("rebuild with the pinned toolchain (toolchain.lock) after clearing the "
                                        "cache entry" if mismatch else "fix the first compiler error; full log "
                                        "under the ext dir (build.log)")).info()
    spills = [k for k in rows["kernels"] if k["spill_store"] or k["spill_load"]]
    c751 = [k for k in rows["kernels"] if k["c751x"]]
    res.summary = {"files": len(reports), "exts": n_ext, "cache_hits": hits,
                   "seconds": round(sum(b["seconds"] for b in rows["builds"]), 1),
                   "n_kernels": len(rows["kernels"]), "spilling_kernels": len(spills), "c751x_kernels": len(c751),
                   "toolchain_match": all(r.toolchain_match for r in reports),
                   "so_notes": sorted({n for b in rows["builds"] for n in b["so_notes"].split("; ") if n})[:4],
                   # ledger rows (P4 ingest_result reads summary.builds / summary.kernels)
                   "builds": [{k: v for k, v in b.items() if k not in ("cubins", "log", "file")}
                              for b in rows["builds"][:16]],
                   "kernels": rows["kernels"][:64]}
    lines = [f"{len(reports)} file(s)  {n_ext} ext(s) ({hits} cache hit)  {res.summary['seconds']}s  "
             f"notes: {', '.join(res.summary['so_notes']) or '-'}"]
    for b in rows["builds"][:6]:
        lines.append(f"  {b['label']} {b['ext_name'][:34]:34s} key {b['cache_key']}  "
                     f"{'hit ' if b['cache_hit'] else 'miss'} {b['seconds']:6.1f}s")
    if len(rows["builds"]) > 6:
        lines.append(f"  ... {len(rows['builds']) - 6} more exts in build.json")
    ks = sorted(rows["kernels"], key=lambda k: (-(k["spill_store"] + k["spill_load"]), -bool(k["c751x"]),
                                                -(k["regs"] or 0)))
    if ks:
        lines.append(f"  kernels ({len(ks)}): name                                regs spill(st/ld) stack C751x")
        for k in ks[: (15 if verbose else 8)]:
            lines.append(f"    {short_name(k['kernel'])[:38]:38s} {k['regs'] or '-':>4} "
                         f"{k['spill_store']}/{k['spill_load']:<6} {k['stack']:>5} {k['c751x'] or '-'}")
        if len(ks) > (15 if verbose else 8):
            lines.append(f"    ... {len(ks) - (15 if verbose else 8)} more kernels in build.json")
    for e in [x for x in errs if not x.startswith("warning: ")][:4]:
        lines.append(f"  ! {e[:150]}")
    if spills:
        res.warnings.append(f"{len(spills)} kernel(s) spill: {', '.join(short_name(k['kernel']) for k in spills[:3])}")
    if c751:
        res.warnings.append(f"wgmma serialised (C751x) in {len(c751)} kernel(s): "
                            f"{', '.join(short_name(k['kernel']) for k in c751[:3])}")
    res.warnings += [e[9:] for e in errs if e.startswith("warning: ")]
    res.lines = lines
    if reports and not fail:
        res.next.append(f"kdev sass {reports[0].file} --loops")
    return res


def short_name(mangled: str) -> str:
    """Readable short form of an Itanium-mangled kernel name without a demangler:
    '_ZN38_GLOBAL__N__..11attn_kernelILi8ELb1ELi0EEEvNS_6ParamsE' -> 'attn_kernel<8,1,0>'. Non-mangled names
    are returned unchanged."""
    if not mangled.startswith("_Z"):
        return mangled
    rest = mangled[2:].lstrip("NKL")
    pos, names = 0, []
    while pos < len(rest):
        mm = re.match(r"\d+", rest[pos:])
        if not mm:
            break
        n = int(mm.group(0))
        pos += len(mm.group(0))
        ident = rest[pos:pos + n]
        pos += n
        if not ident.startswith(("_GLOBAL__N", "_INTERNAL_")):
            names.append(ident)
    base = names[-1] if names else mangled
    targs: list[str] = []
    if rest[pos:pos + 1] == "I":
        targs = re.findall(r"L[a-z](-?\d+)E", rest[pos:].split("EE", 1)[0] + "E")
    return base + (f"<{','.join(targs)}>" if targs else "")


def build_job(spec: JobSpec, run_dir: Any) -> Result:
    """Build-lane entry (daemon worker): builds every file of the spec; status fail on build_failed /
    toolchain_mismatch. Writes build.json into ``run_dir`` (RunDir or path) and returns the Result."""
    reports: list[BuildReport] = []
    labels: list[str] = []
    for f in spec.files:
        if f.path in ("baseline", "none"):
            continue
        reports.append(build_file(f.path, knobs=f.knobs, defines=f.defines, problem=spec.problem,
                                  force=bool(spec.params.get("force")), timeout_s=spec.timeout_s))
        labels.append(f.label)
    res = build_result(reports, labels)
    res.job_id = spec.job_id or None
    rd = Path(getattr(run_dir, "path", run_dir)) if run_dir else None
    if rd is not None:
        rd.mkdir(parents=True, exist_ok=True)
        p = rd / "build.json"
        p.write_text(json.dumps({"schema": "kdev.build/1", **report_rows(reports, labels),
                                 "reports": [r.to_dict() for r in reports]}, indent=1, default=str),
                     encoding="utf-8")
        res.files["build"] = str(p)
    return res
