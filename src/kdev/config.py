"""Configuration (kdev.toml + environment). Owner: P1. See docs/PLAN.md section 3.10.

Search order: $KDEV_CONFIG, ./kdev.toml, $KDEV_HOME/kdev.toml (first existing file wins; none = defaults).
Precedence: explicit ``home=`` argument (``--home``) > environment > kdev.toml > defaults.
Env overrides: KDEV_HOME, KDEV_AGENT, KDEV_NO_GPU, KDEV_GPU_TESTS, KDEV_LOCK_PATH, KDEV_PYTHON, KDEV_GPU,
KDEV_CUDA_HOME, KDEV_TOOLCHAIN_LOCK.

Example ``kdev.toml``::

    home = "/root/.kdev"            # never on /workspace (network volume)
    lock_path = "/root/gpu.lock"
    problems_root = "/root/okc/problems"
    okc_root = "/root/okc"
    default_problem = "muse_moe_w4a16"
    gpu = "h200"                    # GPU profile name (src/kdev/data/gpus/<gpu>.toml); "" = detect
    python = "/root/venv/bin/python"

    [toolchain]                     # fallback pin when the problem has no toolchain.lock
    nvcc = "13.0.88"
    cuda_home = "/root/cuda130w"
    lock = "/root/toolchain.lock"

    [remote]
    host = "1.2.3.4"
    port = 22022
    key = "~/.ssh/runpod_ed25519"

    [sync]
    paths = ["/root/mad", "/root/.kdev/runs"]
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from kdev.contract import ErrorCode, KdevError


@dataclass
class RemoteConfig:
    host: str = ""
    port: int = 22
    user: str = "root"
    key: str = ""
    remote_home: str = "/root/.kdev"


@dataclass
class Config:
    home: Path = field(default_factory=lambda: Path(os.environ.get("KDEV_HOME", "~/.kdev")).expanduser())
    lock_path: str = "/root/gpu.lock"            # shared with legacy gpu_run.sh
    offender_log: str = "/root/gpu_offenders.log"
    max_hold_s: int = 1800
    build_workers: int = 8
    foreign_wait_s: int = 600                     # gpu_run.sh aborts after 600 s of foreign contexts
    problems_root: str = ""
    default_problem: str = ""
    python: str = ""                              # worker interpreter ("" = sys.executable)
    deny_commands: list[str] = field(default_factory=lambda: ["okc submit", "okc check"])
    remote: RemoteConfig = field(default_factory=RemoteConfig)
    sync_paths: list[str] = field(default_factory=list)
    dashboard_out: str = ""
    board_enabled: bool = False
    # P1 additions (not frozen): GPU profile, fallback toolchain pin, okc checkout.
    gpu: str = ""                                 # GPU profile name ("" = detect from nvidia-smi)
    nvcc_pin: str = ""                            # fallback pin when no toolchain.lock is found ("13.0.88")
    cuda_home: str = ""                           # fallback CUDA home for doctor/build when no lock
    toolchain_lock: str = ""                      # explicit toolchain.lock path (else per-problem search)
    okc_root: str = ""                            # OKC checkout (read-only use; never `okc submit`)
    source: str = ""                              # kdev.toml that was loaded ("" = defaults only)
    warnings: list[str] = field(default_factory=list)  # unknown keys etc. (doctor prints them)

    @property
    def runs_dir(self) -> Path:
        return self.home / "runs"

    @property
    def ledger_path(self) -> Path:
        return self.home / "ledger.db"

    @property
    def queue_path(self) -> Path:
        return self.home / "queue.db"

    @property
    def cache_dir(self) -> Path:
        return self.home / "cache"

    @property
    def gpus_dir(self) -> Path:
        """Measured GPU-profile overrides ($KDEV_HOME/gpus/<name>.toml)."""
        return self.home / "gpus"

    def problem_dir(self, problem: str | None = None) -> Path | None:
        """Directory of a problem: an existing path as given, else ``problems_root/<name>``; None if unknown.
        ``problem=None`` uses ``default_problem``."""
        name = problem or self.default_problem
        if not name:
            return None
        p = Path(name).expanduser()
        if p.is_dir():
            return p
        if self.problems_root:
            q = Path(self.problems_root).expanduser() / name
            if q.is_dir():
                return q
        return None

    def to_dict(self) -> dict[str, Any]:
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        d["home"] = str(self.home)
        d["remote"] = vars(self.remote).copy()
        return d


# kdev.toml key -> Config attribute (nested tables are flattened as "table.key").
_KEY_MAP = {
    "home": "home", "lock_path": "lock_path", "offender_log": "offender_log", "max_hold_s": "max_hold_s",
    "build_workers": "build_workers", "foreign_wait_s": "foreign_wait_s", "problems_root": "problems_root",
    "default_problem": "default_problem", "python": "python", "deny_commands": "deny_commands",
    "sync_paths": "sync_paths", "sync.paths": "sync_paths", "dashboard_out": "dashboard_out",
    "dashboard.out": "dashboard_out", "board_enabled": "board_enabled", "board.enabled": "board_enabled",
    "gpu": "gpu", "gpu.name": "gpu", "okc_root": "okc_root", "okc.root": "okc_root",
    "nvcc_pin": "nvcc_pin", "toolchain.nvcc": "nvcc_pin", "toolchain.nvcc_version": "nvcc_pin",
    "cuda_home": "cuda_home", "toolchain.cuda_home": "cuda_home",
    "toolchain_lock": "toolchain_lock", "toolchain.lock": "toolchain_lock",
}
_ENV_MAP = {
    "KDEV_LOCK_PATH": "lock_path", "KDEV_PYTHON": "python", "KDEV_GPU": "gpu", "KDEV_CUDA_HOME": "cuda_home",
    "KDEV_TOOLCHAIN_LOCK": "toolchain_lock",
}
_REMOTE_KEYS = {f.name for f in fields(RemoteConfig)}


def find_config_file(home: str | os.PathLike[str] | None = None) -> Path | None:
    """First existing of $KDEV_CONFIG, ./kdev.toml, <home>/kdev.toml. A set but missing $KDEV_CONFIG is an
    invalid_request (a typo there must not silently fall back to defaults)."""
    env = os.environ.get("KDEV_CONFIG")
    if env:
        p = Path(env).expanduser()
        if not p.is_file():
            raise KdevError(ErrorCode.INVALID_REQUEST, f"KDEV_CONFIG={env} does not exist",
                            details={"field": "KDEV_CONFIG"}, recovery="fix or unset KDEV_CONFIG")
        return p
    cands = [Path.cwd() / "kdev.toml"]
    if home is not None:
        cands.append(Path(home).expanduser() / "kdev.toml")
    for c in cands:
        if c.is_file():
            return c
    return None


def _coerce(cfg: Config, attr: str, value: Any, where: str) -> Any:
    cur = getattr(cfg, attr)
    try:
        if attr == "home":
            return Path(str(value)).expanduser()
        if isinstance(cur, bool):
            if isinstance(value, str):
                if value.lower() not in ("1", "0", "true", "false", "yes", "no"):
                    raise ValueError(value)
                return value.lower() in ("1", "true", "yes")
            return bool(value)
        if isinstance(cur, int):
            if isinstance(value, bool):
                raise ValueError(value)
            return int(value)
        if isinstance(cur, list):
            if isinstance(value, str):
                return [value]
            if not isinstance(value, list):
                raise ValueError(value)
            return [str(v) for v in value]
        if isinstance(value, (dict, list)):
            raise ValueError(value)
        return str(value)
    except (TypeError, ValueError):
        raise KdevError(ErrorCode.INVALID_REQUEST, f"{where}: bad value {value!r} for {attr!r}",
                        details={"field": attr, "source": where},
                        recovery=f"set {attr} to a {type(cur).__name__} in kdev.toml") from None


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def apply_dict(cfg: Config, data: dict[str, Any], where: str = "kdev.toml") -> Config:
    """Apply a parsed kdev.toml dict onto ``cfg`` (unknown keys -> ``cfg.warnings``)."""
    data = dict(data)
    remote = data.pop("remote", None)
    if remote is not None:
        if not isinstance(remote, dict):
            raise KdevError(ErrorCode.INVALID_REQUEST, f"{where}: [remote] must be a table",
                            details={"field": "remote"}, recovery="use a [remote] table in kdev.toml")
        for k, v in remote.items():
            if k not in _REMOTE_KEYS:
                cfg.warnings.append(f"{where}: unknown key remote.{k}")
                continue
            cur = getattr(cfg.remote, k)
            try:
                setattr(cfg.remote, k, int(v) if isinstance(cur, int) else str(v))
            except (TypeError, ValueError):
                raise KdevError(ErrorCode.INVALID_REQUEST, f"{where}: bad value {v!r} for remote.{k}",
                                details={"field": f"remote.{k}"}, recovery="fix kdev.toml") from None
    for key, value in _flatten(data).items():
        attr = _KEY_MAP.get(key)
        if attr is None or key == "schema":
            if key != "schema":
                cfg.warnings.append(f"{where}: unknown key {key}")
            continue
        setattr(cfg, attr, _coerce(cfg, attr, value, where))
    return cfg


def load_config(home: str | None = None, path: str | None = None) -> Config:
    """Load kdev.toml (tomllib) + env overrides. Does not create directories.

    ``home`` (``--home``) wins over $KDEV_HOME and the file's ``home``; ``path`` is an explicit config file
    (wins over the search order). Bad TOML or a bad value raises ``KdevError(invalid_request)``.
    """
    from kdev.util import load_toml

    env_home = os.environ.get("KDEV_HOME")
    search_home = home or env_home or "~/.kdev"
    cfg = Config(home=Path(search_home).expanduser())
    cfg_path = Path(path).expanduser() if path else find_config_file(search_home)
    if cfg_path is not None:
        if not cfg_path.is_file():
            raise KdevError(ErrorCode.INVALID_REQUEST, f"config file {cfg_path} does not exist",
                            details={"field": "config"}, recovery="pass an existing kdev.toml")
        try:
            data = load_toml(cfg_path)
        except (ValueError, OSError) as e:
            raise KdevError(ErrorCode.INVALID_REQUEST, f"cannot parse {cfg_path}: {e}",
                            details={"field": "kdev.toml", "path": str(cfg_path)},
                            recovery="fix the TOML syntax in kdev.toml") from None
        apply_dict(cfg, data, str(cfg_path))
        cfg.source = str(cfg_path)
    for var, attr in _ENV_MAP.items():
        v = os.environ.get(var)
        if v:
            setattr(cfg, attr, _coerce(cfg, attr, v, f"${var}"))
    if env_home:
        cfg.home = Path(env_home).expanduser()
    if home:
        cfg.home = Path(home).expanduser()
    return cfg


def no_gpu() -> bool:
    """True when KDEV_NO_GPU=1 (unit tests): GPU commands return capability_unavailable."""
    return os.environ.get("KDEV_NO_GPU", "") == "1"
