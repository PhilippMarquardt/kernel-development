"""`kdev doctor`: health checks aggregated from each package's ``probe()``. Owner: P1.

Probe providers (each returns ``list[kdev.models.Check]`` and must never raise or need a GPU context):
``kdev.service.probe`` (GPU visible, lock health, foreign CUDA processes, ncu permission),
``kdev.build.probe`` (toolchain pin vs found, TORCH_EXTENSIONS_DIR keying, thrust headers),
``kdev.observe.probe`` (ncu/nsys/cupti/nvbit/sanitizer availability), ``kdev.ledger.probe`` (db
writable, schema version), ``kdev.remote.probe`` (KDEV_HOME not on a network/mfs path, free disk).

P1 always adds the core checks (config, KDEV_HOME, worker python, problem, GPU profile, okc presence). While
a package's probe is still a Phase-0 stub (raises NotImplementedError), P1 runs a minimal fallback for that
area (service -> GPU visible + foreign processes + lock; build -> nvcc/ptxas vs pin + extension-cache keying;
remote -> network path + free disk), so `kdev doctor` is useful from day 0. The fallbacks disappear as soon
as the owning package implements its probe. Doctor never executes `okc` (presence check only, PLAN 8.1).
"""
from __future__ import annotations

import importlib
import os
import re
from pathlib import Path
from typing import Any, Callable

from kdev.config import Config, no_gpu
from kdev.contract import Result
from kdev.models import Check
from kdev.registry import Context
from kdev.util import Runner, disk_free_bytes, existing_parent, human_bytes, load_toml, network_path_reason, run
from kdev.util import which as _which

PROBE_MODULES = ("kdev.service", "kdev.build", "kdev.observe", "kdev.ledger", "kdev.remote")
STATUS_ORDER = {"FAIL": 0, "WARN": 1, "PASS": 2, "SKIP": 3}
DISK_WARN_BYTES = 10 << 30
DISK_FAIL_BYTES = 2 << 30


# ----------------------------------------------------------------------------------------------- helpers
def _writable_dir(p: Path) -> tuple[bool, str]:
    """(ok, detail) for a directory that may not exist yet."""
    if p.exists():
        if not p.is_dir():
            return False, f"{p} exists and is not a directory"
        return os.access(p, os.W_OK), f"{p}"
    parent = existing_parent(p)
    ok = parent.is_dir() and os.access(parent, os.W_OK)
    return ok, f"{p} (will be created under {parent})"


def _first_line(text: str) -> str:
    for ln in text.splitlines():
        if ln.strip():
            return ln.strip()[:160]
    return ""


_VER = re.compile(r"V(\d+\.\d+\.\d+)")
_REL = re.compile(r"release (\d+\.\d+)")


def parse_cuda_tool_version(text: str) -> str:
    """'Cuda compilation tools, release 13.0, V13.0.88' -> '13.0.88' ('13.0' if only the release)."""
    m = _VER.search(text)
    if m:
        return m.group(1)
    m = _REL.search(text)
    return m.group(1) if m else ""


def _version_match(found: str, pin: str) -> bool:
    return bool(found) and (found == pin or found.startswith(pin + ".") or pin.startswith(found + "."))


def resolve_pin(cfg: Config, problem: str | None) -> dict[str, str]:
    """Toolchain pin for doctor: ``{nvcc_version, cuda_home, source}``. Order: cfg.toolchain_lock,
    <problem_dir>/toolchain.lock, $KDEV_HOME/toolchain.lock, then kdev.toml [toolchain] (nvcc_pin)."""
    cands: list[Path] = []
    if cfg.toolchain_lock:
        cands.append(Path(cfg.toolchain_lock).expanduser())
    pdir = cfg.problem_dir(problem)
    if pdir is not None:
        cands.append(pdir / "toolchain.lock")
    cands.append(cfg.home / "toolchain.lock")
    for c in cands:
        if c.is_file():
            try:
                d = load_toml(c)
            except (ValueError, OSError) as e:
                return {"nvcc_version": "", "cuda_home": "", "source": str(c), "error": str(e)}
            return {"nvcc_version": str(d.get("nvcc_version", "")), "cuda_home": str(d.get("cuda_home", "")),
                    "source": str(c)}
    if cfg.nvcc_pin or cfg.cuda_home:
        return {"nvcc_version": cfg.nvcc_pin, "cuda_home": cfg.cuda_home, "source": cfg.source or "config"}
    return {"nvcc_version": "", "cuda_home": "", "source": ""}


def gpu_profile_path(cfg: Config, name: str) -> Path | None:
    """$KDEV_HOME/gpus/<name>.toml (measured override) or the packaged src/kdev/data/gpus/<name>.toml."""
    for p in (cfg.gpus_dir / f"{name}.toml", Path(__file__).parent / "data" / "gpus" / f"{name}.toml"):
        if p.is_file():
            return p
    return None


def packaged_gpu_profiles() -> list[str]:
    d = Path(__file__).parent / "data" / "gpus"
    return sorted(p.stem for p in d.glob("*.toml")) if d.is_dir() else []


# ----------------------------------------------------------------------------------------------- core (P1)
def core_checks(cfg: Config, problem: str | None, runner: Runner = run) -> list[Check]:
    out: list[Check] = []
    out.append(Check("config", "PASS", f"loaded {cfg.source}" if cfg.source else "defaults (no kdev.toml found)",
                     hint="" if cfg.source else "create ./kdev.toml or $KDEV_HOME/kdev.toml (PLAN 3.10)"))
    for w in cfg.warnings:
        out.append(Check("config.keys", "WARN", w, hint="remove or rename the key (see kdev/config.py)"))
    ok, detail = _writable_dir(cfg.home)
    out.append(Check("home", "PASS" if ok else "FAIL", f"KDEV_HOME={detail}",
                     hint="" if ok else "set KDEV_HOME (or --home) to a writable local directory"))
    if cfg.python:
        exists = Path(cfg.python).expanduser().is_file()
        out.append(Check("python", "PASS" if exists else "FAIL", f"worker interpreter {cfg.python}",
                         hint="" if exists else "fix `python` in kdev.toml / $KDEV_PYTHON"))
    if problem:
        pdir = cfg.problem_dir(problem)
        out.append(Check("problem", "PASS" if pdir else "WARN",
                         f"{problem} -> {pdir}" if pdir else f"{problem}: no directory (problems_root="
                         f"{cfg.problems_root or 'unset'})",
                         hint="" if pdir else "set problems_root in kdev.toml or pass a problem directory"))
    out.append(_gpu_profile_check(cfg))
    out.append(_okc_check(cfg))
    return out


def _gpu_profile_check(cfg: Config) -> Check:
    if not cfg.gpu:
        avail = ", ".join(packaged_gpu_profiles()) or "none"
        return Check("gpu.profile", "SKIP", f"not set (auto-detect from nvidia-smi); packaged: {avail}",
                     hint="set gpu = \"h200\" in kdev.toml to pin the profile")
    p = gpu_profile_path(cfg, cfg.gpu)
    if p is None:
        return Check("gpu.profile", "FAIL", f"profile {cfg.gpu!r} not found (packaged: "
                     f"{', '.join(packaged_gpu_profiles()) or 'none'})",
                     hint=f"add {cfg.gpus_dir / (cfg.gpu + '.toml')} or fix `gpu` in kdev.toml")
    try:
        d = load_toml(p)
    except (ValueError, OSError) as e:
        return Check("gpu.profile", "FAIL", f"{p}: {e}", hint="fix the TOML syntax")
    missing = [k for k in ("sm_count", "l2_bytes", "flush_bytes") if k not in d]
    if d.get("schema") != "kdev.gpu/1" or missing:
        return Check("gpu.profile", "WARN", f"{p}: schema={d.get('schema')!r} missing={missing}",
                     hint="see PLAN 3.8 for the kdev.gpu/1 keys")
    return Check("gpu.profile", "PASS", f"{cfg.gpu}: {d.get('sm_count')} SMs, L2 {human_bytes(d.get('l2_bytes'))}, "
                 f"flush {human_bytes(d.get('flush_bytes'))} ({p.name})")


def _okc_check(cfg: Config) -> Check:
    """Presence only: the okc CLI is never executed here (safety rule 8.1)."""
    exe = _which("okc")
    root = Path(cfg.okc_root).expanduser() if cfg.okc_root else None
    parts = []
    if exe:
        parts.append(f"okc CLI at {exe} (read-only use; submit is denied)")
    if root is not None:
        parts.append(f"okc_root {root} {'present' if root.is_dir() else 'MISSING'}")
    if root is not None and not root.is_dir():
        return Check("okc", "WARN", "; ".join(parts), hint="fix okc_root in kdev.toml")
    if not parts:
        return Check("okc", "SKIP", "okc CLI not on PATH and okc_root unset (judge/okc commands unavailable)")
    return Check("okc", "PASS", "; ".join(parts))


# ----------------------------------------------------------------------------------------------- fallbacks
def fallback_service(cfg: Config, problem: str | None, runner: Runner = run) -> list[Check]:
    out: list[Check] = []
    if no_gpu():
        out.append(Check("gpu.visible", "SKIP", "KDEV_NO_GPU=1"))
    else:
        p = runner(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total",
                    "--format=csv,noheader"], timeout=20)
        rows = [r.strip() for r in p.stdout.splitlines() if r.strip()]
        if p.returncode == 0 and rows:
            names = sorted({r.split(",")[0].strip() for r in rows})
            drv = rows[0].split(",")[2].strip() if rows[0].count(",") >= 2 else "?"
            out.append(Check("gpu.visible", "PASS", f"{len(rows)}x {', '.join(names)} (driver {drv})"))
            q = runner(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                        "--format=csv,noheader"], timeout=20)
            # Windows/WDDM lists every graphics process with memory "[N/A]"; only real CUDA contexts count
            apps = [r.strip() for r in q.stdout.splitlines()
                    if r.strip() and "No running" not in r and not r.rstrip().endswith("[N/A]")]
            if q.returncode == 0:
                out.append(Check("gpu.foreign", "WARN" if apps else "PASS",
                                 f"{len(apps)} CUDA process(es): {'; '.join(apps[:3])}" if apps
                                 else "no CUDA processes",
                                 hint="timings are not exclusive while these run (kdev queue offenders)"
                                 if apps else ""))
        elif p.returncode == 127:
            out.append(Check("gpu.visible", "WARN", "nvidia-smi not found (no GPU on this host)",
                             hint="run GPU commands on the pod (`kdev remote -- kdev ...`)"))
        else:
            out.append(Check("gpu.visible", "FAIL", f"nvidia-smi rc={p.returncode}: "
                             f"{_first_line(p.stderr or p.stdout)}", hint="check the driver / container GPU"))
    lp = Path(cfg.lock_path).expanduser()
    if lp.exists():
        ok = lp.is_file() and os.access(lp, os.W_OK)
        out.append(Check("lock", "PASS" if ok else "FAIL", f"{lp} {'writable' if ok else 'NOT writable'}",
                         hint="" if ok else f"chmod/chown {lp} (shared with gpu_run.sh)"))
    else:
        ok = lp.parent.is_dir() and os.access(lp.parent, os.W_OK)
        out.append(Check("lock", "PASS" if ok else "WARN",
                         f"{lp} (absent, will be created)" if ok else f"{lp}: parent dir missing or read-only",
                         hint="" if ok else "set lock_path in kdev.toml / $KDEV_LOCK_PATH"))
    return out


def fallback_build(cfg: Config, problem: str | None, runner: Runner = run) -> list[Check]:
    out: list[Check] = []
    pin = resolve_pin(cfg, problem)
    pin_v = pin.get("nvcc_version", "")
    if pin.get("error"):
        out.append(Check("toolchain.pin", "FAIL", f"{pin['source']}: {pin['error']}", hint="fix toolchain.lock"))
    elif pin_v:
        out.append(Check("toolchain.pin", "PASS", f"nvcc {pin_v} from {pin['source']}"))
    else:
        out.append(Check("toolchain.pin", "WARN", "no toolchain pin (toolchain.lock or [toolchain] nvcc)",
                         hint="add <problem>/toolchain.lock (PLAN 3.9) or [toolchain] nvcc in kdev.toml"))
    cuda_home = pin.get("cuda_home") or cfg.cuda_home or os.environ.get("CUDA_HOME", "")
    bindirs = [str(Path(cuda_home) / "bin")] if cuda_home else []
    found: dict[str, str] = {}
    for tool in ("nvcc", "ptxas"):
        exe = _which(tool, bindirs) or tool
        p = runner([exe, "--version"], timeout=30)
        v = parse_cuda_tool_version(p.stdout + p.stderr) if p.returncode == 0 else ""
        if not v:
            status = "FAIL" if pin_v and tool == "nvcc" and not no_gpu() else "WARN"
            out.append(Check(f"toolchain.{tool}", status, f"{tool} not found or unparsable"
                             f"{' under ' + cuda_home if cuda_home else ''}",
                             hint="set cuda_home in toolchain.lock / kdev.toml or put the pinned CUDA on PATH"))
            continue
        found[tool] = v
        if pin_v and not _version_match(v, pin_v):
            out.append(Check(f"toolchain.{tool}", "FAIL", f"{tool} {v} != pin {pin_v} ({exe})",
                             hint="use the pinned toolchain (wrong nvcc serialised wgmma before, DESIGN M2)"))
        else:
            out.append(Check(f"toolchain.{tool}", "PASS", f"{tool} {v}{'=pin' if pin_v else ''} ({exe})"))
    if "nvcc" in found and "ptxas" in found and not _version_match(found["ptxas"], found["nvcc"]):
        out.append(Check("toolchain.mixed", "WARN", f"ptxas {found['ptxas']} != nvcc {found['nvcc']}",
                         hint="pip ptxas shadows the toolkit one (DESIGN M20); fix PATH order"))
    ext = os.environ.get("TORCH_EXTENSIONS_DIR", "")
    key = pin_v or found.get("nvcc", "")
    if not ext:
        out.append(Check("ext.cache", "WARN", "TORCH_EXTENSIONS_DIR unset: torch's default cache is keyed on "
                         "source only (stale .so after a toolchain change, DESIGN M3)",
                         hint=f"kdev build keys its own cache under {cfg.cache_dir}; for raw load_inline set "
                         "TORCH_EXTENSIONS_DIR per toolchain"))
    else:
        tags = {key, key.replace(".", ""), "".join(key.split(".")[:2]), "cu" + "".join(key.split(".")[:2])} - {""}
        keyed = any(t in ext for t in tags)
        out.append(Check("ext.cache", "PASS" if keyed else "WARN",
                         f"TORCH_EXTENSIONS_DIR={ext} {'names' if keyed else 'does not name'} the toolchain"
                         f"{' ' + key if key else ''}",
                         hint="" if keyed else "use a per-toolchain dir, e.g. ~/.cache/torch_ext_cu130"))
    ok, detail = _writable_dir(cfg.cache_dir)
    out.append(Check("kdev.cache", "PASS" if ok else "FAIL", f"build cache {detail} (keyed on toolchain+flags+source)",
                     hint="" if ok else "make KDEV_HOME writable"))
    return out


def fallback_remote(cfg: Config, problem: str | None, runner: Runner = run) -> list[Check]:
    out: list[Check] = []
    why = network_path_reason(cfg.home)
    out.append(Check("home.local", "WARN" if why else "PASS",
                     f"KDEV_HOME {cfg.home}: {why}" if why else f"KDEV_HOME {cfg.home} is local",
                     hint="move KDEV_HOME off network storage (/workspace hung in OKC) and use `kdev sync`"
                     if why else ""))
    free = disk_free_bytes(cfg.home)
    if free is None:
        out.append(Check("disk", "WARN", f"cannot stat {existing_parent(cfg.home)}"))
    else:
        st = "FAIL" if free < DISK_FAIL_BYTES else "WARN" if free < DISK_WARN_BYTES else "PASS"
        out.append(Check("disk", st, f"{human_bytes(free)} free at {existing_parent(cfg.home)}",
                         hint="" if st == "PASS" else "free disk (old runs: $KDEV_HOME/runs, ext caches)"))
    if cfg.remote.host and cfg.remote.key:
        k = Path(cfg.remote.key).expanduser()
        out.append(Check("remote.key", "PASS" if k.is_file() else "FAIL", f"ssh key {k}",
                         hint="" if k.is_file() else "fix remote.key in kdev.toml"))
    return out


Fallback = Callable[[Config, "str | None", Runner], list[Check]]
FALLBACKS: dict[str, Fallback] = {
    "kdev.service": fallback_service,
    "kdev.build": fallback_build,
    "kdev.remote": fallback_remote,
}


# ----------------------------------------------------------------------------------------------- public
def _clip(line: str, width: int = 200) -> str:
    return line if len(line) <= width else line[: width - 12] + " [...--json]"


def _as_check(x: Any, mod: str) -> Check:
    if isinstance(x, Check):
        return x
    if isinstance(x, dict) and "name" in x and "status" in x:
        return Check.from_dict(x)
    return Check(f"{mod}.probe", "FAIL", f"probe returned {type(x).__name__}, not Check")


def collect_checks(ctx: Context, problem: str | None = None, *, runner: Runner = run,
                   modules: tuple[str, ...] = PROBE_MODULES) -> list[Check]:
    """Import each PROBE_MODULES entry and call ``probe(ctx, problem)``; a probe that raises becomes a FAIL
    line naming the module. P1 core checks come first; stub probes use the P1 fallback (module docstring)."""
    cfg: Config = ctx.cfg
    problem = problem or ctx.problem_name
    checks = core_checks(cfg, problem, runner)
    for mod in modules:
        short = mod.rsplit(".", 1)[-1]
        try:
            fn = importlib.import_module(mod).probe
            got = fn(ctx, problem)
            checks.extend(_as_check(c, short) for c in (got or []))
        except NotImplementedError:
            fb = FALLBACKS.get(mod)
            if fb is None:
                checks.append(Check(f"{short}.probe", "SKIP", f"{mod}.probe not implemented yet"))
                continue
            try:
                checks.extend(fb(cfg, problem, runner))
            except Exception as e:  # noqa: BLE001
                checks.append(Check(f"{short}.fallback", "FAIL", f"{type(e).__name__}: {e}"))
        except Exception as e:  # noqa: BLE001 - a broken probe must not break doctor
            checks.append(Check(f"{short}.probe", "FAIL", f"{mod}.probe raised {type(e).__name__}: {e}",
                                hint=f"bug in {mod}; report it"))
    for c in checks:
        if c.status not in STATUS_ORDER:
            c.detail = f"(bad status {c.status!r}) {c.detail}"
            c.status = "WARN"
    return checks


def run_doctor(ctx: Context, problem: str | None = None, fix_hints: bool = False, *,
               runner: Runner = run, modules: tuple[str, ...] = PROBE_MODULES) -> Result:
    """PASS/WARN/FAIL summary; status fail if any FAIL."""
    checks = collect_checks(ctx, problem, runner=runner, modules=modules)
    counts = {s: sum(1 for c in checks if c.status == s) for s in STATUS_ORDER}
    verdict = "fail" if counts["FAIL"] else "warn" if counts["WARN"] else "pass"
    ordered = sorted(checks, key=lambda c: STATUS_ORDER.get(c.status, 9))
    lines: list[str] = []
    for c in ordered:
        lines.append(_clip(f"{c.status:4s}  {c.name:18s} {c.detail}"))
        if fix_hints and c.hint and c.status in ("FAIL", "WARN"):
            lines.append(_clip(f"      hint: {c.hint}"))
    nxt = []
    if not fix_hints and any(c.hint for c in checks if c.status in ("FAIL", "WARN")):
        nxt.append("kdev doctor --fix-hints")
    return Result(tool="doctor", status="fail" if counts["FAIL"] else "ok", verdict=verdict,
                  summary={"fail": counts["FAIL"], "warn": counts["WARN"], "pass": counts["PASS"],
                           "skip": counts["SKIP"], "checks": [c.to_dict() for c in checks]},
                  lines=lines, next=nxt)
