"""`kdev remote setup`: idempotent pod bootstrap with phase markers. Owner: P2.

Port of arcagi3/scripts/okc_bootstrap.sh + okc_pod_setup.sh: venv pins, judge toolchain (CUDA 13.0.88 pip
wheels + wrapper dir), TORCH_EXTENSIONS_DIR per toolchain, read-flush self-test, `kdev doctor`. Each phase
writes ``$KDEV_HOME/bootstrap/<phase>.done`` and is skipped when present.

Implementation: the ``venv`` + ``toolchain`` phases wrap the existing, tested okc_bootstrap.sh (run locally
with bash; it uploads okc_pod_setup.sh, runs it detached on the pod and polls). Located via
$KDEV_BOOTSTRAP_SCRIPT, ./scripts/okc_bootstrap.sh or ~/source/repos/arcagi3/scripts/okc_bootstrap.sh.
``kdev_install`` packs this kdev source tree, uploads it md5-verified and pip-installs it into the pod python
($KDEV_REMOTE_PYTHON, default /root/vsgl/bin/python). ``flush_selftest`` runs a 256 MB read-flush probe under
``kdev lock run`` (never a bare GPU process). ``doctor`` runs ``kdev doctor --json`` on the pod.
Markers live on the pod under ``remote.remote_home/bootstrap/``.
"""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

from kdev.config import Config
from kdev.contract import ErrorCode, KdevError
from kdev.remote import ssh
from kdev.util import Runner, run

PHASES = ("venv", "toolchain", "kdev_install", "flush_selftest", "doctor")
_runner: Runner = run
REMOTE_SRC = "/root/kdev_src"
FLUSH_PROBE = ("import torch,time;b=torch.empty(256<<20,dtype=torch.uint8,device='cuda');s=torch.zeros(1,device='cuda');"
               "torch.cuda.synchronize();t=time.perf_counter();[s.add_(b.sum()) for _ in range(10)];"
               "torch.cuda.synchronize();print('READ_FLUSH_OK',round((time.perf_counter()-t)*1e5,1),'us/read')")


def find_bootstrap_script() -> str | None:
    cands = [os.environ.get("KDEV_BOOTSTRAP_SCRIPT", ""), "scripts/okc_bootstrap.sh",
             os.path.expanduser("~/source/repos/arcagi3/scripts/okc_bootstrap.sh")]
    for c in cands:
        if c and os.path.isfile(c):
            return os.path.abspath(c)
    return None


def remote_python() -> str:
    return os.environ.get("KDEV_REMOTE_PYTHON", "/root/vsgl/bin/python")


def _marker(cfg: Config, phase: str) -> str:
    return cfg.remote.remote_home.rstrip("/") + f"/bootstrap/{phase}.done"


def _done(cfg: Config, phase: str, runner: Runner) -> bool:
    return runner(ssh.ssh_argv(cfg.remote, ["test", "-f", _marker(cfg, phase)]), timeout=60).returncode == 0


def _mark(cfg: Config, phase: str, runner: Runner) -> None:
    m = _marker(cfg, phase)
    runner(ssh.ssh_shell_argv(cfg.remote, f"mkdir -p {shlex.quote(os.path.dirname(m))} && date -u > {shlex.quote(m)}"),
           timeout=60)


def _okc_bootstrap(cfg: Config, force: bool, runner: Runner) -> None:
    script = find_bootstrap_script()
    if not script:
        raise KdevError(ErrorCode.CAPABILITY_UNAVAILABLE, "okc_bootstrap.sh not found",
                        recovery="set KDEV_BOOTSTRAP_SCRIPT to arcagi3/scripts/okc_bootstrap.sh")
    bash = shutil.which("bash") or "bash"
    env = dict(os.environ)
    if force:
        env["FORCE"] = "1"
    p = runner([bash, script, cfg.remote.host, str(cfg.remote.port)], timeout=6000, env=env,
               cwd=str(Path(script).parents[1]))
    if p.returncode != 0:
        raise KdevError(ErrorCode.INTERNAL, f"okc_bootstrap.sh exited {p.returncode}: {p.stdout[-400:]}{p.stderr[-400:]}")


def _kdev_install(cfg: Config, runner: Runner) -> None:
    repo = Path(__file__).resolve().parents[3]            # .../kernel-development (src/kdev/remote -> repo)
    with tempfile.TemporaryDirectory(prefix="kdev_inst_") as td:
        tgz = os.path.join(td, "kdev_src.tgz")
        with tarfile.open(tgz, "w:gz") as tf:
            for name in ("pyproject.toml", "README.md", "src"):
                if (repo / name).exists():
                    tf.add(repo / name, arcname=name,
                           filter=lambda ti: None if "__pycache__" in ti.name or ti.name.endswith(
                               (".pyc", ".egg-info")) or ".egg-info" in ti.name else ti)
        ssh.push_verified(cfg.remote, tgz, "/tmp/kdev_src.tgz", runner)
    py = remote_python()
    cmd = (f"rm -rf {REMOTE_SRC} && mkdir -p {REMOTE_SRC} && tar --no-same-owner -xzf /tmp/kdev_src.tgz -C {REMOTE_SRC} && "
           f"{shlex.quote(py)} -m pip install -q --no-deps -e {REMOTE_SRC} && {shlex.quote(py)} -m kdev version")
    p = runner(ssh.ssh_shell_argv(cfg.remote, cmd), timeout=900)
    if p.returncode != 0:
        raise KdevError(ErrorCode.INTERNAL, f"kdev install failed: {p.stderr[-400:]}")


def _remote_kdev(cfg: Config, args: list[str]) -> list[str]:
    return [remote_python(), "-m", "kdev", *args]


def _flush_selftest(cfg: Config, runner: Runner) -> None:
    p = runner(ssh.ssh_argv(cfg.remote, _remote_kdev(cfg, ["lock", "run", "--timeout", "120", "--",
                                                           remote_python(), "-c", FLUSH_PROBE])), timeout=900)
    if p.returncode != 0 or "READ_FLUSH_OK" not in p.stdout + p.stderr:
        raise KdevError(ErrorCode.INTERNAL, f"read-flush self-test failed: {(p.stdout + p.stderr)[-400:]}")


def _doctor(cfg: Config, runner: Runner) -> None:
    p = runner(ssh.ssh_argv(cfg.remote, _remote_kdev(cfg, ["doctor", "--json"])), timeout=300)
    if p.returncode not in (0, 1):            # 1 = some FAIL lines: reported, not a bootstrap failure
        raise KdevError(ErrorCode.INTERNAL, f"kdev doctor exited {p.returncode}: {p.stderr[-300:]}")


def setup(cfg: Config, phases: tuple[str, ...] = PHASES, force: bool = False) -> dict[str, str]:
    """Run phases on the remote host; returns {phase: done|skipped|failed}."""
    ssh._check(cfg.remote)
    runner = _runner
    unknown = [p for p in phases if p not in PHASES]
    if unknown:
        raise KdevError(ErrorCode.INVALID_REQUEST, f"unknown phases {unknown}; known {PHASES}")
    out: dict[str, str] = {}
    failed = False
    for phase in PHASES:
        if phase not in phases:
            continue
        if failed:
            out[phase] = "skipped"
            continue
        if not force and _done(cfg, phase, runner):
            out[phase] = "skipped"
            continue
        try:
            if phase == "venv":
                _okc_bootstrap(cfg, force, runner)          # does venv AND toolchain in one idempotent run
            elif phase == "toolchain":
                if out.get("venv") != "done":
                    _okc_bootstrap(cfg, force, runner)
            elif phase == "kdev_install":
                _kdev_install(cfg, runner)
            elif phase == "flush_selftest":
                _flush_selftest(cfg, runner)
            elif phase == "doctor":
                _doctor(cfg, runner)
            _mark(cfg, phase, runner)
            out[phase] = "done"
        except (KdevError, subprocess.TimeoutExpired, OSError) as e:
            out[phase] = "failed"
            out[f"{phase}_error"] = getattr(e, "message", str(e))[:300]
            failed = True
    return out
