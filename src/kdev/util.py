"""Small shared helpers. Owner: P1.

Subprocess-wrapping code everywhere takes ``runner: Runner = run`` so tests can inject canned output.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import secrets
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence


@dataclass
class Proc:
    """Completed subprocess (what a Runner returns)."""

    argv: list[str]
    returncode: int
    stdout: str
    stderr: str
    seconds: float = 0.0


Runner = Callable[..., Proc]


def run(argv: Sequence[str], *, timeout: float | None = None, env: dict[str, str] | None = None,
        cwd: str | None = None) -> Proc:
    """Run a command (never through a shell) and capture text output.

    A missing executable returns ``returncode=127`` and a timeout ``returncode=124`` (like coreutils
    ``timeout``) instead of raising, so callers (and doctor probes) report instead of crashing.
    """
    t0 = time.monotonic()
    try:
        cp = subprocess.run(list(argv), capture_output=True, text=True, timeout=timeout, env=env, cwd=cwd,
                            check=False, encoding="utf-8", errors="replace")
    except FileNotFoundError as e:
        return Proc(list(argv), 127, "", f"not found: {e}", time.monotonic() - t0)
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        err = e.stderr.decode("utf-8", "replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        return Proc(list(argv), 124, out, err + f"\ntimeout after {timeout} s", time.monotonic() - t0)
    return Proc(list(argv), cp.returncode, cp.stdout or "", cp.stderr or "", time.monotonic() - t0)


def md5_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()  # noqa: S324 - identity hash, not security
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


_C_BLOCK = re.compile(r"/\*.*?\*/", re.S)
_C_LINE = re.compile(r"//[^\n]*")
# '#' starts a comment unless it is a C preprocessor directive at the start of a line (CUDA inside
# load_inline strings keeps '#include <...>' etc., so genuinely different sources do not collapse).
_PP = r"(?:include|define|undef|if|ifdef|ifndef|elif|else|endif|pragma|error|warning|line)\b"
_HASH = re.compile(r"^([ \t]*#[ \t]*" + _PP + r"[^\n]*)|#[^\n]*", re.M)
_WS = re.compile(r"\s+")


def norm_source(text: str) -> str:
    """Source with comments and all whitespace removed (heuristic, not a parser): C ``/* */`` and ``//``
    comments and Python ``#`` comments (C preprocessor lines are kept). Comment-proof identity."""
    text = _C_BLOCK.sub("", text)
    text = _C_LINE.sub("", text)
    text = _HASH.sub(lambda m: m.group(1) or "", text)
    return _WS.sub("", text)


def norm_md5(path: str) -> str:
    """md5 of the source with comments and whitespace removed (Python '#', C '//' and '/* */'), so a
    comment-only change (the judge 'duplicate' workaround) maps to the same identity."""
    with open(path, "rb") as f:
        text = f.read().decode("utf-8", errors="replace")
    return hashlib.md5(norm_source(text).encode("utf-8")).hexdigest()  # noqa: S324


def utc_now() -> str:
    """ISO-8601 UTC timestamp with 'Z', second resolution."""
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_id(prefix: str) -> str:
    """Sortable id: <prefix><yyyymmdd-hhmmss>-<4 hex>, e.g. j20261031-143012-7f3a."""
    return f"{prefix}{_dt.datetime.now(_dt.timezone.utc).strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}"


def slugify(text: str, max_len: int = 40) -> str:
    """Lowercase ``[a-z0-9._-]`` slug for run ids and file names ('sass diff' -> 'sass-diff')."""
    s = re.sub(r"[^a-z0-9._-]+", "-", str(text).lower()).strip("-.")
    return s[:max_len].rstrip("-.")


def atomic_write_text(path: str | os.PathLike[str], text: str) -> Path:
    """Write ``text`` to ``path`` atomically (tmp file in the same dir + os.replace), UTF-8, LF endings."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{p.name}.", suffix=".tmp", dir=str(p.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.replace(tmp, p)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return p


def atomic_write_json(path: str | os.PathLike[str], obj: Any, indent: int | None = 2) -> Path:
    """Atomic JSON write; non-JSON values are converted with str()."""
    return atomic_write_text(path, json.dumps(obj, indent=indent, default=str) + "\n")


def read_json(path: str | os.PathLike[str]) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_toml(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Parse a TOML file with tomllib (py>=3.11) or tomli. Raises ValueError on a syntax error."""
    try:
        import tomllib as _toml
    except ModuleNotFoundError:  # pragma: no cover - python 3.10
        import tomli as _toml  # type: ignore[no-redef]
    with open(path, "rb") as f:
        try:
            return _toml.load(f)
        except _toml.TOMLDecodeError as e:
            raise ValueError(f"{path}: {e}") from e


def which(name: str, extra_dirs: Sequence[str] = ()) -> str | None:
    """Executable path: first in ``extra_dirs`` (e.g. ``<cuda_home>/bin``), then PATH."""
    for d in extra_dirs:
        if not d:
            continue
        for cand in (Path(d) / name, Path(d) / f"{name}.exe"):
            if cand.is_file() and os.access(cand, os.X_OK):
                return str(cand)
    return shutil.which(name)


def existing_parent(path: str | os.PathLike[str]) -> Path:
    """The path itself or its nearest existing ancestor (disk/mount checks of not-yet-created dirs)."""
    p = Path(path).expanduser().absolute()
    while not p.exists() and p.parent != p:
        p = p.parent
    return p


def disk_free_bytes(path: str | os.PathLike[str]) -> int | None:
    try:
        return shutil.disk_usage(existing_parent(path)).free
    except OSError:
        return None


NETWORK_FS = ("nfs", "nfs4", "cifs", "smbfs", "smb3", "mfs", "lustre", "gpfs", "glusterfs", "ceph", "9p",
              "juicefs")
NETWORK_PREFIXES = ("/workspace",)   # RunPod network volume (mfs; hung in OKC, DESIGN M17)


def mount_fstype(path: str | os.PathLike[str], mounts_file: str = "/proc/mounts") -> str | None:
    """Filesystem type of the mount containing ``path`` (Linux /proc/mounts); None if unknown."""
    try:
        with open(mounts_file, encoding="utf-8", errors="replace") as f:
            rows = [ln.split() for ln in f if ln.strip()]
    except OSError:
        return None
    p = str(path) if str(path).startswith("/") else existing_parent(path).as_posix()
    best, best_len = None, -1
    for r in rows:
        if len(r) < 3:
            continue
        mnt = r[1].replace("\\040", " ")
        inside = p == mnt or mnt == "/" or p.startswith(mnt.rstrip("/") + "/")
        if inside and len(mnt) > best_len:
            best, best_len = r[2], len(mnt)
    return best


def network_path_reason(path: str | os.PathLike[str], mounts_file: str = "/proc/mounts") -> str | None:
    """Why ``path`` looks like network/mfs storage (pod network volumes hang and are not a safe home), or
    None if it looks local."""
    raw = str(path)
    posix = raw.replace("\\", "/")
    for pre in NETWORK_PREFIXES:
        if posix == pre or posix.startswith(pre + "/"):
            return f"under {pre} (network volume)"
    if raw.startswith("\\\\") or raw.startswith("//"):
        return "UNC network path"
    fs = mount_fstype(path, mounts_file)
    if fs and (fs in NETWORK_FS or fs.startswith("fuse.")):
        return f"filesystem type {fs}"
    return None


def human_bytes(n: float | None) -> str:
    if n is None:
        return "?"
    x = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if abs(x) < 1024:
            return f"{x:.0f}{unit}" if unit in ("B", "KB") else f"{x:.1f}{unit}"
        x /= 1024
    return f"{x:.1f}TB"
