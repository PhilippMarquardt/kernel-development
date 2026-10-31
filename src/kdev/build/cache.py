"""Hash-keyed extension cache (M3: torch keys on source only). Owner: P5.

key = sha256(toolchain.nvcc_version + ptxas path + sorted flags/defines/knobs + source bytes)[:16];
dir = $KDEV_HOME/cache/ext/<nvcc_version>/<key>/ (the build_directory of that extension; complete when it
holds ``kdev_ext.json`` and the library).

Per submission file a manifest ``$KDEV_HOME/cache/files/<file_key>.json`` lists the extensions its build
produced (file_key = sha256(file md5 + knobs + defines + toolchain)); :func:`is_prebuilt` checks it.

GPU worker integration (P3): before importing a submission call :func:`install_load_inline_hook` (mode
``use``): intercepted ``load_inline``/``load`` calls then import the prebuilt library from the cache and raise
``needs_build`` (via :class:`kdev.build._shim.NeedsBuild`) when it is missing, unless ``allow_jit``.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from kdev.build import _shim
from kdev.models import FileRef, JobSpec, Toolchain
from kdev.util import md5_file

MANIFEST_SCHEMA = "kdev.build_manifest/1"


def _home(home: str | None = None) -> Path:
    return Path(home or os.environ.get("KDEV_HOME") or "~/.kdev").expanduser()


def tc_version(toolchain: Toolchain) -> str:
    return toolchain.nvcc_version or "unknown"


def tc_ident(toolchain: Toolchain) -> str:
    """Toolchain identity beyond the version: ptxas path + the pinned env (NVCC_APPEND_FLAGS, arch list)."""
    env = dict(toolchain.env)
    env.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
    return toolchain.ptxas + "|" + json.dumps(sorted(env.items()))


def cache_key(source_bytes: bytes, toolchain: Toolchain, flags: list[str], knobs: dict) -> str:
    """16 hex chars; changes with the toolchain version, the ptxas path / pinned env, any flag/define/knob or
    the source."""
    return _shim.ext_key(source_bytes, tc_version(toolchain), tc_ident(toolchain), list(flags), dict(knobs or {}))


def ext_root(home: str | None, toolchain: Toolchain) -> str:
    return str(_home(home) / "cache" / "ext" / tc_version(toolchain))


def ext_dir(home: str, toolchain: Toolchain, key: str) -> str:
    return str(Path(ext_root(home, toolchain)) / key)


def entry_complete(d: str | Path) -> bool:
    """A cache dir is complete when its marker exists and the artifact it names exists."""
    d = Path(d)
    m = d / _shim.MARKER
    if not m.is_file():
        return False
    try:
        info = json.loads(m.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    art = info.get("so_path") or info.get("cubin") or ""
    return bool(art) and Path(art).is_file()


def file_key(md5: str, knobs: dict | None, defines: list[str] | None, toolchain: Toolchain) -> str:
    h = hashlib.sha256()
    h.update(f"{md5}|{tc_version(toolchain)}|{toolchain.ptxas}|".encode())
    h.update(json.dumps(dict(knobs or {}), sort_keys=True, default=str).encode())
    h.update(("|" + "\x1f".join(defines or [])).encode())
    return h.hexdigest()[:16]


def manifest_path(home: str | None, fkey: str) -> Path:
    return _home(home) / "cache" / "files" / f"{fkey}.json"


def write_manifest(home: str | None, fkey: str, doc: dict[str, Any]) -> Path:
    p = manifest_path(home, fkey)
    p.parent.mkdir(parents=True, exist_ok=True)
    doc = {"schema": MANIFEST_SCHEMA, **doc}
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, p)
    # also index by path so `kdev sass FILE.py` finds the latest build of a file regardless of knobs
    idx = _home(home) / "cache" / "by_path"
    idx.mkdir(parents=True, exist_ok=True)
    pk = hashlib.sha256(os.path.abspath(str(doc.get("file", ""))).encode()).hexdigest()[:16]
    (idx / f"{pk}.json").write_text(json.dumps({"file": doc.get("file"), "manifest": str(p)}), encoding="utf-8")
    return p


def read_manifest(home: str | None, fkey: str) -> dict[str, Any] | None:
    p = manifest_path(home, fkey)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def latest_manifest_for(path: str, home: str | None = None) -> dict[str, Any] | None:
    """Manifest of the most recent build of ``path`` (any knobs), or None."""
    pk = hashlib.sha256(os.path.abspath(path).encode()).hexdigest()[:16]
    p = _home(home) / "cache" / "by_path" / f"{pk}.json"
    try:
        ref = json.loads(p.read_text(encoding="utf-8"))
        return json.loads(Path(ref["manifest"]).read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError):
        return None


def file_prebuilt(f: FileRef, toolchain: Toolchain, home: str | None = None) -> tuple[bool, str]:
    """(prebuilt, reason) for one job file."""
    if not f.path or not os.path.isfile(f.path):
        return False, f"missing file {f.path}"
    md5 = md5_file(f.path)
    if f.md5 and f.md5 != md5:
        return False, f"{f.path} changed since submit (md5 {md5} != {f.md5})"
    man = read_manifest(home, file_key(md5, f.knobs, f.defines, toolchain))
    if not man:
        return False, f"no build of {os.path.basename(f.path)} for V{tc_version(toolchain)} (knobs {f.knobs})"
    if man.get("status") != "ok":
        return False, f"last build of {os.path.basename(f.path)} {man.get('status')}"
    for e in man.get("exts", []):
        if not entry_complete(e.get("dir", "")):
            return False, f"cache entry {e.get('key')} of {e.get('name')} incomplete"
    pend = pending_for(f.path, toolchain, home)
    if pend:
        return False, f"{len(pend)} prepare-time extension(s) recorded by a GPU run, not built yet: {pend[0]}"
    return True, "ok"


def pending_for(path: str, toolchain: Toolchain, home: str | None = None) -> list[str]:
    """Names of extension calls a GPU worker recorded as missing for ``path`` (``<ext_root>/pending``; the
    next ``kdev build`` of the file compiles them). Records made for an older content of ``path`` (owner md5
    differs: the source was edited since that GPU run) are superseded and deleted here (board #78)."""
    recs, _ = _shim.scan_pending(ext_root(home, toolchain), path, tc_version(toolchain), tc_ident(toolchain))
    return [str((doc.get("kw") or {}).get("name") or doc.get("key")) for _, doc in recs]


def clean_pending(path: str, toolchain: Toolchain, home: str | None = None) -> list[str]:
    """Delete every pending record owned by ``path`` (any content; ``kdev build --clean-pending``) and the
    stale ones; returns the removed record paths."""
    _, removed = _shim.scan_pending(ext_root(home, toolchain), path, tc_version(toolchain), tc_ident(toolchain),
                                    clean=True)
    return removed


def is_prebuilt(spec: JobSpec) -> bool:
    """True if every file in the GPU job has a complete cache entry for the pinned toolchain (the daemon's
    needs_build gate, PLAN 8.2). Files whose path is 'baseline'/'none' or not a .py/.cu are ignored."""
    from kdev.build.toolchain import pin_for  # noqa: PLC0415

    for f in spec.files:
        if not f.path or f.path in ("baseline", "none") or not f.path.endswith((".py", ".cu")):
            continue
        tc = pin_for(spec.problem, f.path)
        ok, _ = file_prebuilt(f, tc)
        if not ok:
            return False
    return True


def hook_env(f: FileRef, toolchain: Toolchain, home: str | None = None, *, allow_jit: bool = False) -> dict[str, str]:
    """Environment variables carrying the hook config into a worker process (read by
    :func:`install_load_inline_hook` when called without arguments)."""
    return {"KDEV_EXT_ROOT": ext_root(home, toolchain), "KDEV_TC_VERSION": tc_version(toolchain),
            "KDEV_TC_PTXAS": tc_ident(toolchain), "KDEV_KNOBS": json.dumps(dict(f.knobs or {}), default=str),
            "KDEV_DEFINES": json.dumps(list(f.defines or [])), "KDEV_ALLOW_JIT": "1" if allow_jit else "0"}


def install_load_inline_hook(f: FileRef | None = None, toolchain: Toolchain | None = None, home: str | None = None,
                             *, allow_jit: bool = False) -> _shim.Hook:
    """Worker-side (imports torch): route load_inline/load to the prebuilt cache (mode ``use``). With no
    arguments the config comes from :func:`hook_env` variables. ``hook.records`` lists what was loaded."""
    if f is not None and toolchain is not None:
        env = hook_env(f, toolchain, home, allow_jit=allow_jit)
    else:
        env = {k: os.environ.get(k, "") for k in ("KDEV_EXT_ROOT", "KDEV_TC_VERSION", "KDEV_TC_PTXAS", "KDEV_KNOBS",
                                                 "KDEV_DEFINES", "KDEV_ALLOW_JIT")}
    hook = _shim.Hook(mode="use", ext_root=env["KDEV_EXT_ROOT"], tc_version=env["KDEV_TC_VERSION"],
                      tc_ptxas=env["KDEV_TC_PTXAS"], knobs=json.loads(env["KDEV_KNOBS"] or "{}"),
                      defines=json.loads(env["KDEV_DEFINES"] or "[]"),
                      allow_jit=allow_jit or env.get("KDEV_ALLOW_JIT") == "1")
    hook.owner = os.path.abspath(f.path) if f is not None and f.path else ""
    return _shim.install(hook)


def active_hook() -> _shim.Hook | None:
    """The load_inline hook installed in this (worker) process, or None. Public P5 API (P6 use_prebuilt)."""
    return _shim.active()


def select_variant(f: FileRef) -> _shim.Hook | None:
    """Switch the installed hook to FileRef ``f``'s knobs/defines, so the next load_inline/load in this process
    resolves to that file's prebuilt variant (several files of one job: timeline --ab, marked + plain).
    Returns the hook, or None when no hook is installed (call :func:`install_load_inline_hook` then)."""
    hook = _shim.active()
    if hook is not None:
        hook.knobs = dict(f.knobs or {})
        hook.defines = list(f.defines or [])
        hook.owner = os.path.abspath(f.path) if f.path else ""
    return hook
