"""Toolchain pin (toolchain.lock, PLAN 3.9), detection and `.so` notes. Owner: P5.

LESSONS 9 / M2: every timed kernel must be built with the judge's toolchain (CUDA 13.0, nvcc V13.0.88). The pin
lives in ``<problem_dir>/toolchain.lock`` (or ``$KDEV_HOME/toolchain.lock``); ``verify_so`` checks the toolkit
notes embedded in a built ``.so``/``.cubin`` against it without loading the binary.
"""
from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path

from kdev.models import Toolchain
from kdev.util import Runner, run

if sys.version_info >= (3, 11):
    import tomllib as _toml
else:  # pragma: no cover
    import tomli as _toml  # type: ignore[no-redef]

TOOLS = ("nvcc", "ptxas", "cuobjdump", "nvdisasm")
DISASM_TOOLS = ("cuobjdump", "nvdisasm")   # read binaries only: any version works
_EXE = ".exe" if os.name == "nt" else ""
_VER_RE = re.compile(r"release\s+([\d.]+),\s*V([\d.]+)")
_NOTE_RE = re.compile(rb"(?:Cuda compilation tools[^\x00\n]{0,60}|Build cuda_[^\x00\n]{0,40})")
_TOOLKIT_RE = re.compile(r"V(\d+\.\d+\.\d+)")
_BUILD_RE = re.compile(r"Build cuda_(\d+\.\d+)")


def load_lock(path: str) -> Toolchain:
    """Parse a toolchain.lock TOML into Toolchain. Missing tool paths default to ``<cuda_home>/bin/<tool>``."""
    with open(path, "rb") as f:
        d = _toml.load(f)
    tc = Toolchain.from_dict({k: v for k, v in d.items() if k not in ("schema", "env", "extra")})
    tc.env = {str(k): str(v) for k, v in (d.get("env") or {}).items()}
    tc.extra = {str(k): str(v) for k, v in (d.get("extra") or {}).items()}
    if tc.cuda_home:
        for tool in TOOLS:
            if not getattr(tc, tool):
                cand = Path(tc.cuda_home) / "bin" / (tool + _EXE)
                if _EXE and not cand.exists():
                    cand = Path(tc.cuda_home) / "bin" / tool
                if not cand.exists() and tool in DISASM_TOOLS:
                    # a trimmed toolkit (e.g. /root/cuda13w: nvcc+ptxas only) has no disassembler; any
                    # cuobjdump/nvdisasm reads the cubins (it never makes SASS), so fall back (board #155)
                    cand = Path(find_tool(tool, tc.cuda_home, python=tc.python) or cand)
                setattr(tc, tool, str(cand))
    if not tc.source:
        tc.source = f"lock:{path}"
    tc.extra.setdefault("lock_path", str(path))
    return tc


def find_lock(problem_dir: str | None) -> str | None:
    """<problem_dir>/toolchain.lock, then $KDEV_HOME/toolchain.lock."""
    cands: list[Path] = []
    if problem_dir:
        p = Path(problem_dir)
        cands.append((p if p.is_dir() else p.parent) / "toolchain.lock")
    home = os.environ.get("KDEV_HOME") or "~/.kdev"
    cands.append(Path(home).expanduser() / "toolchain.lock")
    for c in cands:
        if c.is_file():
            return str(c)
    return None


def _disasm_fallbacks(tool: str, python: str | None = None) -> list[Path]:
    """Other places a cuobjdump/nvdisasm lives: /usr/local/cuda*/bin (newest first) and triton's bundled
    ``triton/backends/nvidia/bin`` next to this interpreter or the pinned ``python``."""
    out = sorted(Path("/usr/local").glob(f"cuda*/bin/{tool}{_EXE}"), reverse=True)
    prefixes = {sys.prefix}
    if python:
        prefixes.add(str(Path(python).parent.parent))      # venv/bin/python -> venv (not resolved: venv)
    for pre in sorted(prefixes):
        out += sorted(Path(pre).glob(f"lib/python3*/site-packages/triton/backends/nvidia/bin/{tool}{_EXE}"))
        out += sorted(Path(pre).glob(f"Lib/site-packages/triton/backends/nvidia/bin/{tool}{_EXE}"))
    return out


def find_tool(tool: str, cuda_home: str | None = None, python: str | None = None) -> str:
    """Path of a CUDA tool under ``cuda_home/bin`` (``$KDEV_CUDA_HOME``, ``$CUDA_HOME``), else on PATH; ''.
    Disassemblers (``DISASM_TOOLS``) also fall back to other CUDA trees / triton's bundled copy."""
    for home in (cuda_home, os.environ.get("KDEV_CUDA_HOME"), os.environ.get("CUDA_HOME")):
        if not home:
            continue
        for name in (tool + _EXE, tool):
            p = Path(home) / "bin" / name
            if p.is_file():
                return str(p)
    found = shutil.which(tool) or ""
    if not found and tool in DISASM_TOOLS:
        found = next((str(p) for p in _disasm_fallbacks(tool, python) if p.is_file()), "")
    return found


def detect(cuda_home: str | None = None, runner: Runner = run) -> Toolchain:
    """Versions/paths of nvcc, ptxas, cuobjdump, nvdisasm under cuda_home (or $KDEV_CUDA_HOME/$CUDA_HOME/PATH).

    ``nvcc_version`` is nvcc's version (ptxas's when nvcc is absent); per-tool versions go to
    ``extra['<tool>_version']`` so a pip ptxas mixed with a system nvcc (M20) is visible."""
    cuda_home = cuda_home or os.environ.get("KDEV_CUDA_HOME") or os.environ.get("CUDA_HOME") or None
    tc = Toolchain(cuda_home=cuda_home or "", source="detect")
    for tool in TOOLS:
        path = find_tool(tool, cuda_home)
        setattr(tc, tool, path)
        if not path:
            continue
        try:
            p = runner([path, "--version"], timeout=30)
        except Exception:  # noqa: BLE001 - detection never raises
            continue
        if p.returncode == 0:
            v = parse_nvcc_version(p.stdout + p.stderr)
            if v:
                tc.extra[f"{tool}_version"] = v
    tc.nvcc_version = tc.extra.get("nvcc_version") or tc.extra.get("ptxas_version", "")
    if not tc.cuda_home and (tc.nvcc or tc.ptxas):
        tc.cuda_home = str(Path(tc.nvcc or tc.ptxas).resolve().parent.parent)
    return tc


def parse_nvcc_version(text: str) -> str:
    """'Cuda compilation tools, release 13.0, V13.0.88' -> '13.0.88' ('' when absent)."""
    m = _VER_RE.search(text or "")
    return m.group(2) if m else ""


def so_notes(path: str) -> list[str]:
    """Scan a built .so/.cubin for embedded toolkit strings ('Cuda compilation tools, release X, VX.Y.Z',
    'Build cuda_...') without loading it (the tc_attn.py probe technique). Sorted, unique."""
    try:
        b = Path(path).read_bytes()
    except OSError:
        return []
    return sorted({x.decode(errors="replace").strip() for x in _NOTE_RE.findall(b)})[:16]


def notes_versions(notes: list[str]) -> list[str]:
    """Toolkit versions named by .so notes: 'V13.0.88' -> '13.0.88'; 'Build cuda_13.0.r13.0' -> '13.0'."""
    out: set[str] = set()
    for n in notes:
        if "compilation tools" in n:
            out.update(_TOOLKIT_RE.findall(n))
        else:
            out.update(_BUILD_RE.findall(n))
    return sorted(out)


def versions_match(versions: list[str], pin_version: str) -> bool:
    """Every named version agrees with the pin ('X.Y.Z' equal; an 'X.Y' build tag must prefix the pin)."""
    if not versions:
        return False
    return all((v == pin_version) if v.count(".") == 2 else
               (pin_version == v or pin_version.startswith(v + ".")) for v in versions)


def verify_so(path: str, pin: Toolchain) -> tuple[bool, list[str]]:
    """(match, notes). No pin version -> (True, notes). No toolkit notes -> (False, []): unverifiable; build
    reports it as a warning for objects without device code, toolchain_mismatch otherwise."""
    notes = so_notes(path)
    if not pin.nvcc_version:
        return True, notes
    return versions_match(notes_versions(notes), pin.nvcc_version), notes


def build_env(pin: Toolchain, ext_dir: str) -> dict[str, str]:
    """Environment for compile subprocesses: CUDA_HOME, PATH, NVCC_APPEND_FLAGS, TORCH_EXTENSIONS_DIR=ext_dir,
    CUDA_VISIBLE_DEVICES='' (no CUDA context), plus pin.env (pin.env wins, except CUDA_VISIBLE_DEVICES)."""
    env = dict(os.environ)
    if pin.cuda_home:
        env["CUDA_HOME"] = pin.cuda_home
        env["CUDA_PATH"] = pin.cuda_home
        env["PATH"] = str(Path(pin.cuda_home) / "bin") + os.pathsep + env.get("PATH", "")
    elif pin.nvcc:
        env["PATH"] = str(Path(pin.nvcc).parent) + os.pathsep + env.get("PATH", "")
    env["TORCH_EXTENSIONS_DIR"] = ext_dir
    env.update({str(k): str(v) for k, v in pin.env.items()})
    env.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
    env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def pin_for(problem: str | None = None, file: str | None = None, runner: Runner = run) -> Toolchain:
    """The pinned toolchain for a problem dir / submission file (toolchain.lock), else $KDEV_HOME's lock,
    else the detected one (``source='detect'``: unpinned, callers warn)."""
    for d in (problem, file):
        if d and os.path.exists(d):
            lock = find_lock(d)
            if lock:
                return load_lock(lock)
    lock = find_lock(None)
    if lock:
        return load_lock(lock)
    return detect(runner=runner)


def check_tools(pin: Toolchain, runner: Runner = run) -> list[tuple[str, str, str]]:
    """[(tool, PASS|WARN|FAIL, detail)] per CUDA tool: present and reporting the pinned version."""
    out: list[tuple[str, str, str]] = []
    for tool in TOOLS:
        path = getattr(pin, tool) or find_tool(tool, pin.cuda_home or None)
        if not path or not (os.path.isfile(path) or shutil.which(path)):
            out.append((tool, "FAIL" if tool in ("nvcc", "ptxas") else "WARN", f"{tool} not found"))
            continue
        try:
            p = runner([path, "--version"], timeout=30)
            v = parse_nvcc_version(p.stdout + p.stderr)
        except Exception as e:  # noqa: BLE001
            out.append((tool, "FAIL", f"{path}: {e}"))
            continue
        if pin.nvcc_version and v != pin.nvcc_version:
            # nvcc/ptxas make the binary (M3: a different ptxas = different SASS); cuobjdump/nvdisasm only
            # read it, and a newer disassembler reads older cubins, so their mismatch is a warning.
            sev = "FAIL" if tool in ("nvcc", "ptxas") else "WARN"
            out.append((tool, sev, f"{path} is V{v or '?'} != pin V{pin.nvcc_version}"
                        + ("" if sev == "FAIL" else " (disassembler only: does not affect builds)")))
        else:
            out.append((tool, "PASS", f"{path} V{v or '?'}"))
    return out
