"""compute-sanitizer wrapper. Owner: P7.

The worker re-execs itself as ``compute-sanitizer --tool T --print-limit 100 --log-file san.log python -m
kdev.worker job.json --inner`` on one small shape (eager + graph); ``run_sanitize_inner`` is the code that
runs under the sanitizer; ``parse_log`` summarises errors by type with the first 5 distinct errors.

Implementation: the worker calls :func:`run_sanitize_inner` for kind='sanitize'. Outside the sanitizer
(``KDEV_SANITIZER_INNER`` unset) it re-execs ``compute-sanitizer ... python -m kdev.checks.sanitize job.json
--run-dir D`` with ``KDEV_SANITIZER_INNER=1`` (so it does not depend on worker flags), parses the log and
returns the summary Result. Inside, it runs the shape (``params.extra.shape``, default the adapter's smallest
test) eagerly and as a graph replay and writes ``sanitize_inner.json``. For cu_main the built binary itself
runs under the sanitizer.
"""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kdev.contract import ErrorCode, KdevError, Result
from kdev.models import JobSpec
from kdev.util import Runner, run

TOOLS = ("memcheck", "racecheck", "synccheck", "initcheck")
INNER_ENV = "KDEV_SANITIZER_INNER"
PREFIX = "========="
_SKIP_HEADS = ("COMPUTE-SANITIZER", "ERROR SUMMARY", "RACECHECK SUMMARY", "LEAK SUMMARY", "Target application",
               "Saved host backtrace", "This tool is deprecated")


@dataclass
class SanitizerSummary:
    tool: str
    errors: int
    by_type: dict[str, int] = field(default_factory=dict)
    first: list[str] = field(default_factory=list)
    warnings: int = 0


def find_sanitizer() -> str | None:
    """$KDEV_COMPUTE_SANITIZER, PATH, $CUDA_HOME/bin, /usr/local/cuda/bin, newest /usr/local/cuda-*/bin."""
    env = os.environ.get("KDEV_COMPUTE_SANITIZER")
    if env and os.path.isfile(env):
        return env
    hit = shutil.which("compute-sanitizer")
    if hit:
        return hit
    cands = [os.path.join(os.environ.get("CUDA_HOME", ""), "bin", "compute-sanitizer"),
             "/usr/local/cuda/bin/compute-sanitizer"]
    cands += sorted(glob.glob("/usr/local/cuda-*/bin/compute-sanitizer"), reverse=True)
    return next((c for c in cands if c and os.path.isfile(c)), None)


def sanitizer_argv(tool: str, inner_argv: list[str], log_file: str, exe: str | None = None) -> list[str]:
    if tool not in TOOLS:
        raise KdevError(ErrorCode.INVALID_REQUEST, f"unknown sanitizer tool {tool!r}", details={"tools": TOOLS})
    return [exe or "compute-sanitizer", "--tool", tool, "--print-limit", "100", "--log-file", log_file] + list(inner_argv)


def _norm_type(head: str) -> str:
    h = head.strip()
    rules = [
        (r"Invalid (__\w+__) (read|write|atomic)", lambda m: f"invalid {m.group(1)} {m.group(2)}"),
        (r"Misaligned", lambda m: "misaligned address"),
        (r"Uninitialized (__\w+__) memory read", lambda m: f"uninitialized {m.group(1)} read"),
        (r"Race reported between (\w+) access(?:.*?and (\w+) access)?",
         lambda m: f"race {m.group(1).lower()}" + (f"-{m.group(2).lower()}" if m.group(2) else "")),
        (r"Barrier error|Divergent thread", lambda m: "barrier/divergence"),
        (r"Program hit (\w+)", lambda m: f"api {m.group(1)}"),
        (r"Leaked \d+ bytes", lambda m: "leak"),
        (r"(cudaError\w+|CUDA_ERROR_\w+)", lambda m: f"api {m.group(1)}"),
    ]
    for rx, fn in rules:
        m = re.search(rx, h)
        if m:
            return fn(m)
    return re.sub(r"0x[0-9a-fA-F]+|\d+", "N", h)[:60]


def _location(lines: list[str]) -> str:
    for ln in lines:
        m = re.match(r"\s*at (?:0x[0-9a-fA-F]+ in )?(.*)", ln)
        if m:
            return re.sub(r"0x[0-9a-fA-F]+", "", m.group(1)).strip()[:120]
    return ""


def parse_log(text: str, tool: str) -> SanitizerSummary:
    """Summarise a compute-sanitizer log: error count (from the SUMMARY line when present), counts per error type,
    and the first 5 distinct errors (type @ location, plus the thread/block line)."""
    blocks: list[tuple[str, list[str]]] = []
    for raw in text.splitlines():
        if not raw.startswith(PREFIX):
            continue
        body = raw[len(PREFIX):]
        body = body[1:] if body.startswith(" ") else body
        if not body.strip():
            continue
        if not body.startswith(" ") and not body.startswith("\t"):
            head = body.strip()
            if any(head.removeprefix("Error: ").startswith(s) for s in _SKIP_HEADS):
                blocks.append(("__skip__", [head]))
                continue
            blocks.append((head, []))
        elif blocks:
            blocks[-1][1].append(body)
    errs = [(h, ls) for h, ls in blocks if h != "__skip__" and not h.startswith("Warning:")]
    warns = [h for h, _ in blocks if h.startswith("Warning:")]
    by_type: dict[str, int] = {}
    first: list[str] = []
    seen: set[tuple[str, str]] = set()
    for head, ls in errs:
        t = _norm_type(head.removeprefix("Error: ") + " " + " ".join(x.strip() for x in ls[:1]))
        by_type[t] = by_type.get(t, 0) + 1
        loc = _location(ls if ls else [head])
        key = (t, loc)
        if key in seen or len(first) >= 5:
            continue
        seen.add(key)
        thr = next((x.strip() for x in ls if x.strip().startswith("by thread")), "")
        first.append(" | ".join(x for x in (t, loc, thr) if x))
    count = len(errs)
    m = re.search(r"ERROR SUMMARY: (\d+) error", text)
    if m:
        count = int(m.group(1))
    m = re.search(r"RACECHECK SUMMARY: \d+ hazards? displayed \((\d+) errors?, (\d+) warnings?\)", text)
    if m:
        count = int(m.group(1))
        return SanitizerSummary(tool, count, by_type, first, int(m.group(2)))
    return SanitizerSummary(tool, count, by_type, first, len(warns))


def inner_argv(job_path: str, run_dir: str, python: str | None = None) -> list[str]:
    return [python or sys.executable, "-m", "kdev.checks.sanitize", job_path, "--run-dir", run_dir]


def _run_dir_path(run_dir: Any) -> Path:
    if run_dir is None:
        import tempfile
        return Path(tempfile.mkdtemp(prefix="kdev_san_"))
    p = Path(getattr(run_dir, "path", run_dir))
    p.mkdir(parents=True, exist_ok=True)
    return p


def _adapter(spec: JobSpec) -> Any:
    from kdev.adapters.registry import get_adapter
    return get_adapter(problem=spec.problem, file=spec.files[0].path if spec.files else None, adapter=spec.adapter)


def run_sanitize_inner(spec: JobSpec, run_dir: Any) -> Result:
    """Worker entry for kind='sanitize' (see module docstring). Never raises."""
    try:
        if os.environ.get(INNER_ENV) == "1":
            return _inner(spec, run_dir)
        return run_sanitize_outer(spec, run_dir)
    except KdevError as e:
        return Result.from_error("sanitize", e)


def run_sanitize_outer(spec: JobSpec, run_dir: Any, runner: Runner = run, exe: str | None = None) -> Result:
    if not spec.files:
        raise KdevError(ErrorCode.INVALID_REQUEST, "sanitize needs a submission file")
    ex = dict(spec.params.get("extra") or {})
    tool = str(ex.get("tool") or "memcheck")
    exe = exe or find_sanitizer()
    if not exe:
        raise KdevError(ErrorCode.CAPABILITY_UNAVAILABLE, "compute-sanitizer not found",
                        recovery="install the CUDA toolkit's compute-sanitizer or set KDEV_COMPUTE_SANITIZER")
    rd = _run_dir_path(run_dir)
    log = rd / "sanitizer.log"
    adapter = _adapter(spec)
    if adapter.name == "cu_main":
        binary = adapter.load(spec.files[0].path, "A")["binary"]
        if not os.path.isfile(binary):
            raise KdevError(ErrorCode.NEEDS_BUILD, f"no built binary {binary}",
                            recovery=f"run `kdev build {spec.files[0].path}` first")
        inner = [binary]
    else:
        job_path = rd / "sanitize_job.json"
        job_path.write_text(json.dumps(spec.to_dict(), default=str), encoding="utf-8")
        inner = inner_argv(str(job_path), str(rd), ex.get("python"))
    argv = sanitizer_argv(tool, inner, str(log), exe)
    env = dict(os.environ, **{INNER_ENV: "1"})
    pkg_root = str(Path(__file__).resolve().parents[2])   # .../src (kdev may not be pip-installed)
    env["PYTHONPATH"] = os.pathsep.join(x for x in (pkg_root, env.get("PYTHONPATH", "")) if x)
    proc = runner(argv, timeout=float(spec.timeout_s or 1800), env=env)
    (rd / "sanitize_stdout.txt").write_text((proc.stdout or "") + "\n--- stderr ---\n" + (proc.stderr or ""),
                                            encoding="utf-8")
    text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else (proc.stdout or "")
    summ = parse_log(text, tool)
    inner_res: dict[str, Any] = {}
    ip = rd / "sanitize_inner.json"
    if ip.is_file():
        try:
            inner_res = json.loads(ip.read_text(encoding="utf-8"))
        except ValueError:
            inner_res = {}
    warnings = []
    if adapter.name != "cu_main" and not inner_res:
        warnings.append(f"inner run produced no result (exit {proc.returncode}); see sanitize_stdout.txt")
    if inner_res and not inner_res.get("ok", True):
        warnings.append("inner correctness: " + "; ".join(inner_res.get("messages", []))[:200])
    if "Target application terminated before first instrumented API" in text:
        warnings.append("the inner process exited before any CUDA call (see sanitize_stdout.txt)")
    failed = summ.errors > 0
    verdict = "errors" if failed else ("clean" if (inner_res or adapter.name == "cu_main") else "incomplete")
    summary = {"tool": tool, "errors": summ.errors, "warnings": summ.warnings, "by_type": summ.by_type,
               "first": summ.first, "shape": inner_res.get("shape"), "inner": inner_res, "exit": proc.returncode,
               "sanitizer": exe}
    lines = [f"{verdict.upper()}  sanitize  {tool}  errors={summ.errors}"
             + (f" warnings={summ.warnings}" if summ.warnings else "")
             + (f"  shape={inner_res.get('shape')}" if inner_res.get("shape") else "")]
    for t, n in sorted(summ.by_type.items(), key=lambda kv: -kv[1])[:8]:
        lines.append(f"  {n:6d}  {t}")
    lines += [f"  #{i + 1} {x[:150]}" for i, x in enumerate(summ.first)]
    nxt = [f"kdev sanitize {spec.files[0].path} --tool racecheck"] if tool == "memcheck" else []
    res = Result(tool="sanitize", status="fail" if failed or verdict == "incomplete" else "ok", verdict=verdict,
                 job_id=spec.job_id or None, summary=summary, warnings=warnings, next=nxt[:2],
                 files={"log": str(log), "stdout": str(rd / "sanitize_stdout.txt")})
    res.lines = lines
    return res


def _inner(spec: JobSpec, run_dir: Any) -> Result:
    """Runs under compute-sanitizer: one small shape, eager then graph replay."""
    import torch

    from kdev.checks.check import capture_graph, clone_output

    rd = _run_dir_path(run_dir)
    ex = dict(spec.params.get("extra") or {})
    adapter = _adapter(spec)
    fref = spec.files[0]
    module = adapter.load(fref.path, fref.label or "A")
    state = adapter.prepare(module)
    sid = ex.get("shape")
    shape = adapter.find_shape(sid) if sid and hasattr(adapter, "find_shape") else None
    if sid and shape is None:
        raise KdevError(ErrorCode.INVALID_REQUEST, f"unknown shape {sid!r}")
    if shape is None:
        shape = adapter.smallest_test() if hasattr(adapter, "smallest_test") else adapter.tests()[0]
    seed = int(adapter.config.get("default_seed", 1234)) if hasattr(adapter, "config") else 1234
    data = adapter.make_inputs(shape, seed)
    msgs, oks = [], []
    out = clone_output(adapter.call(module, state, data))
    torch.cuda.synchronize()
    r = adapter.check(data, out)
    oks.append(r.ok)
    msgs.append(f"eager: {r.message[:160]}")
    if spec.params.get("mode", "both") != "eager":
        g, gout = capture_graph(adapter, module, state, data)
        g.replay()
        torch.cuda.synchronize()
        r = adapter.check(data, clone_output(gout))
        oks.append(r.ok)
        msgs.append(f"graph: {r.message[:160]}")
    doc = {"shape": shape.shape_id, "seed": seed, "ok": all(oks), "messages": msgs}
    (rd / "sanitize_inner.json").write_text(json.dumps(doc), encoding="utf-8")
    return Result(tool="sanitize", status="ok" if all(oks) else "fail", verdict="inner", summary=doc)


def main(argv: list[str] | None = None) -> int:
    """``python -m kdev.checks.sanitize JOB.json --run-dir DIR`` (the process compute-sanitizer runs)."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("usage: python -m kdev.checks.sanitize JOB.json --run-dir DIR", file=sys.stderr)
        return 2
    run_dir = args[args.index("--run-dir") + 1] if "--run-dir" in args else None
    spec = JobSpec.from_dict(json.loads(Path(args[0]).read_text(encoding="utf-8")))
    os.environ[INNER_ENV] = "1"
    res = run_sanitize_inner(spec, run_dir)
    print(res.to_json())
    return res.exit_code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
