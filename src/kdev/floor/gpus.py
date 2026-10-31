"""GPU profile loader (`kdev.gpu/1`, PLAN 3.8). Owner: P6.

Search: $KDEV_HOME/gpus/<name>.toml (measured overrides) then package data src/kdev/data/gpus/<name>.toml.
The package file is the base; a home file is merged over it (top-level keys replace; ``[peak]``,
``[limits]`` and ``[measured]`` entries replace per key). ``detect_name()`` maps nvidia-smi names
('NVIDIA H200') to profile names ('h200').

Units: every ``[measured]`` entry is ``{value, unit, source, run, how}`` (a bare number is accepted and means
``unit = "us"``). Times are "judge units" (raw graph-replay us minus the judge's 4.0 us fixed cost) unless the
key says otherwise (``*_rel_us`` = relative to the 1-kernel graph).
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kdev.contract import ErrorCode, KdevError

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

PROFILE_SCHEMA = "kdev.gpu/1"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "gpus"


@dataclass
class Measured:
    value: float
    unit: str = "us"
    source: str = ""
    run: str = ""
    how: str = ""


@dataclass
class GpuProfile:
    name: str
    sm_count: int
    l2_bytes: int
    flush_bytes: int
    cc: str = ""
    smem_per_sm: int = 0
    regs_per_sm: int = 65536
    max_warps_per_sm: int = 64
    max_ctas_per_sm: int = 32
    globaltimer_ns: list[int] = field(default_factory=list)
    peak: dict[str, float] = field(default_factory=dict)
    measured: dict[str, Measured] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    def const(self, key: str, default: float | None = None) -> float | None:
        """Value of a measured constant, or ``default`` when the profile lacks it."""
        m = self.measured.get(key)
        return m.value if m is not None else default

    def limit(self, key: str, default: Any = None) -> Any:
        """A ``[limits]`` entry (occupancy rules: smem per CTA, setmaxnreg bounds, ...)."""
        return (self.raw.get("limits") or {}).get(key, default)


def _home_gpus(home: str | None) -> Path | None:
    h = home or os.environ.get("KDEV_HOME")
    return Path(h).expanduser() / "gpus" if h else None


def _measured_from(d: dict[str, Any]) -> dict[str, Measured]:
    out: dict[str, Measured] = {}
    for k, v in (d or {}).items():
        if isinstance(v, dict):
            if "value" not in v:
                continue
            out[k] = Measured(value=float(v["value"]), unit=str(v.get("unit", "us")), source=str(v.get("source", "")),
                              run=str(v.get("run", "")), how=str(v.get("how", "")))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = Measured(value=float(v))
    return out


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            m = dict(out[k])
            m.update(v)
            out[k] = m
        else:
            out[k] = v
    return out


def profile_from_dict(d: dict[str, Any]) -> GpuProfile:
    """GpuProfile from a parsed TOML dict (pure)."""
    if d.get("schema", PROFILE_SCHEMA) != PROFILE_SCHEMA:
        raise KdevError(ErrorCode.INVALID_REQUEST, f"GPU profile schema {d.get('schema')!r} != {PROFILE_SCHEMA}")
    try:
        l2 = int(d.get("l2_bytes", 0))
        return GpuProfile(
            name=str(d["name"]), sm_count=int(d["sm_count"]), l2_bytes=l2,
            flush_bytes=int(d.get("flush_bytes", max(4 * l2, 256 << 20))),
            cc=str(d.get("cc", "")), smem_per_sm=int(d.get("smem_per_sm", 0)),
            regs_per_sm=int(d.get("regs_per_sm", 65536)), max_warps_per_sm=int(d.get("max_warps_per_sm", 64)),
            max_ctas_per_sm=int(d.get("max_ctas_per_sm", 32)),
            globaltimer_ns=[int(x) for x in d.get("globaltimer_ns", [])],
            peak={k: float(v) for k, v in (d.get("peak") or {}).items()
                  if isinstance(v, (int, float)) and not isinstance(v, bool)},
            measured=_measured_from(d.get("measured") or {}), raw=d)
    except KeyError as e:
        raise KdevError(ErrorCode.INVALID_REQUEST, f"GPU profile missing key {e.args[0]!r}") from e


def _read_toml(p: Path) -> dict[str, Any]:
    try:
        return tomllib.loads(p.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as e:
        raise KdevError(ErrorCode.INVALID_REQUEST, f"bad GPU profile {p}: {e}",
                        recovery="fix or delete the file") from e


def available_profiles(home: str | None = None) -> list[str]:
    names = {p.stem for p in DATA_DIR.glob("*.toml")}
    hd = _home_gpus(home)
    if hd is not None and hd.is_dir():
        names |= {p.stem for p in hd.glob("*.toml")}
    return sorted(names)


def load_profile(name: str, home: str | None = None) -> GpuProfile:
    """Load profile ``name`` (case-insensitive; an nvidia-smi name such as 'NVIDIA H200' is mapped)."""
    key = name.strip().lower()
    if not re.fullmatch(r"[a-z0-9_.-]+", key):
        key = detect_name(name)
    pkg = DATA_DIR / f"{key}.toml"
    hd = _home_gpus(home)
    hp = hd / f"{key}.toml" if hd is not None else None
    base = _read_toml(pkg) if pkg.is_file() else {}
    over = _read_toml(hp) if hp is not None and hp.is_file() else {}
    if not base and not over:
        raise KdevError(ErrorCode.NOT_FOUND, f"no GPU profile {key!r} (have: {', '.join(available_profiles(home))})",
                        recovery="copy src/kdev/data/gpus/h200.toml to $KDEV_HOME/gpus/<name>.toml and edit it")
    d = _merge(base, over)
    d.setdefault("name", key)
    prof = profile_from_dict(d)
    if over:
        prof.raw["_override_path"] = str(hp)
    return prof


def detect_name(smi_name: str) -> str:
    """'NVIDIA H200' -> 'h200'. Profiles' ``smi_names`` first, then a heuristic (first model token)."""
    low = smi_name.strip().lower()
    for p in sorted(DATA_DIR.glob("*.toml")):
        try:
            d = tomllib.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - one broken file must not break detection
            continue
        for n in d.get("smi_names", []):
            n = str(n).lower()
            if low == n or low.startswith(n + " ") or low.startswith(n + "-"):
                return p.stem
    while True:
        new = re.sub(r"^(nvidia|tesla|geforce|rtx|quadro)\s+", "", low)
        if new == low:
            break
        low = new
    tok = re.split(r"[\s\-_]+", low)
    return re.sub(r"[^a-z0-9]", "", tok[0]) if tok and tok[0] else "unknown"


def _toml_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ") + '"'


def _toml_scalar(v: Any) -> str | None:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return _toml_str(v)
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, list) and all(isinstance(x, (int, float, str)) and not isinstance(x, bool) for x in v):
        return "[" + ", ".join(_toml_str(x) if isinstance(x, str) else repr(x) for x in v) + "]"
    return None


def _toml_measured(k: str, m: Measured) -> str:
    parts = [f"value = {m.value!r}", f"unit = {_toml_str(m.unit)}"]
    for f in ("source", "run", "how"):
        v = getattr(m, f)
        if v:
            parts.append(f"{f} = {_toml_str(v)}")
    return f"{k} = {{ {', '.join(parts)} }}"


def write_measured(name: str, values: dict[str, Measured], home: str) -> str:
    """Write/merge $KDEV_HOME/gpus/<name>.toml [measured] (floor calibrate --write); returns the path.

    Only the override file is written (the package profile is never modified). Existing override entries are
    kept unless replaced by ``values``."""
    hd = Path(home).expanduser() / "gpus"
    hd.mkdir(parents=True, exist_ok=True)
    path = hd / f"{name}.toml"
    existing = _read_toml(path) if path.is_file() else {}
    measured = _measured_from(existing.get("measured") or {})
    measured.update(values)
    lines = [f"# kdev GPU profile override for {name} (written by `kdev floor calibrate --write`)",
             f"schema = {_toml_str(PROFILE_SCHEMA)}", f"name = {_toml_str(name)}"]
    for k, v in existing.items():
        if k in ("schema", "name") or isinstance(v, dict):
            continue
        s = _toml_scalar(v)
        if s is not None:
            lines.append(f"{k} = {s}")
    for k, v in existing.items():
        if k == "measured" or not isinstance(v, dict):
            continue
        lines += ["", f"[{k}]"]
        for kk, vv in v.items():
            s = _toml_scalar(vv)
            if s is not None:
                lines.append(f"{kk} = {s}")
    lines += ["", "[measured]"] + [_toml_measured(k, measured[k]) for k in sorted(measured)]
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, path)
    return str(path)
