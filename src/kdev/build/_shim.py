"""Compile-only shim around ``torch.utils.cpp_extension.load_inline``/``load`` (P5, private; stdlib only).

Runs in the problem's pinned interpreter (which may not have kdev installed), therefore self-contained:

    python _shim.py SPEC.json        # SPEC: {file, mode, ext_root, tc_id, knobs, defines, out, force, cap, entries}

mode ``build`` (build lane, CUDA_VISIBLE_DEVICES=''): imports the submission with the hook installed, calls its
build entry points (``_module``, ``_build*``, ``_ext*``, ...), and builds every intercepted extension into
``<ext_root>/<key>/`` (key = :func:`ext_key` over toolchain + flags + knobs + sources, never source alone: M3)
with ``-Xptxas -v -lineinfo``; stdout/stderr of the compile go to ``build.log`` there. Kernel launches never
happen (the hook returns a stub module in build mode).

mode ``use`` (GPU worker, via :func:`install`): an intercepted call whose key has a complete entry imports the
cached library directly (no ninja, no nvcc); a missing entry raises :class:`NeedsBuild` unless allow_jit.
"""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import os
import re
import sys
import time
import traceback

MARKER = "kdev_ext.json"
DEFAULT_ENTRIES = ("_module", "_get_module", "_mod", "_load", "_ext", "_get_ext", "_build", "_compile", "build",
                   "get_module", "load_module")
_ENTRY_RE = re.compile(r"^_?(build|module|ext|load|compile|get_module|get_ext|jit)\w*$")
_TRANSIENT = re.compile(r"thrust/[\w/.]+: No such file|fatal error: thrust/|cub/[\w/.]+: No such file|"
                        r"Text file busy|resource temporarily unavailable", re.I)


class NeedsBuild(RuntimeError):
    """Raised in ``use`` mode when an extension has no complete cache entry (kdev error needs_build)."""


class ExtBuildFailed(RuntimeError):
    """Raised by the hook in ``build`` mode when the compiler fails (kdev error build_failed)."""


def ext_key(source_bytes: bytes, tc_version: str, tc_ptxas: str, flags: list[str], knobs: dict) -> str:
    """sha256(toolchain version + ptxas path + sorted flags + sorted knobs + source)[:16] (PLAN cache.py)."""
    h = hashlib.sha256()
    h.update(f"tc={tc_version}|ptxas={tc_ptxas}\n".encode())
    h.update(("flags=" + "\x1f".join(sorted(str(f) for f in flags)) + "\n").encode())
    h.update(("knobs=" + json.dumps({str(k): knobs[k] for k in sorted(knobs, key=str)}, sort_keys=True,
                                     default=str) + "\n").encode())
    h.update(source_bytes)
    return h.hexdigest()[:16]


def file_md5(path: str) -> str:
    """md5 of a file's bytes, '' if unreadable (pending records key their owner file's content on it)."""
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()  # noqa: S324 - identity hash
    except OSError:
        return ""


def pending_state(doc: dict, path: str, tc_version: str, tc_ptxas: str) -> str:
    """How a pending record relates to submission ``path`` under this toolchain: 'match' (to build),
    'other' (another toolchain or owner file), or 'stale' (owned by ``path`` but recorded for different file
    content: the source changed since the GPU run, so the call is superseded and must be dropped, board #78).
    Records without ``file_md5`` (older kdev) are stale when the file was modified after they were written."""
    if doc.get("tc_version") != tc_version or doc.get("tc_ptxas") != tc_ptxas:
        return "other"
    owner = doc.get("file") or ""
    if not owner:
        return "match"
    if os.path.abspath(owner) != os.path.abspath(path):
        return "other"
    want = doc.get("file_md5")
    if want:
        cur = file_md5(path)
        return "stale" if cur and cur != want else "match"
    try:
        return "stale" if os.path.getmtime(path) > float(doc.get("ts") or 0) else "match"
    except OSError:
        return "match"


def scan_pending(ext_root: str, path: str, tc_version: str, tc_ptxas: str, *, prune: bool = True,
                 clean: bool = False) -> tuple[list[tuple[str, dict]], list[str]]:
    """([(record path, doc)] matching ``path``, [removed record paths]). ``prune`` deletes stale records of
    ``path``; ``clean`` deletes every record owned by ``path`` (``kdev build --clean-pending``)."""
    d = os.path.join(ext_root, "pending")
    keep: list[tuple[str, dict]] = []
    removed: list[str] = []
    if not os.path.isdir(d):
        return keep, removed
    for fn in sorted(x for x in os.listdir(d) if x.endswith(".json")):
        p = os.path.join(d, fn)
        try:
            with open(p, encoding="utf-8") as f:
                doc = json.load(f)
        except (OSError, ValueError):
            continue
        st = pending_state(doc, path, tc_version, tc_ptxas)
        owned = bool(doc.get("file")) and st != "other"
        if (clean and owned) or (prune and st == "stale"):
            try:
                os.remove(p)
                removed.append(p)
            except OSError:
                pass
            continue
        if st == "match":
            keep.append((p, doc))
    return keep, removed


def _jsonable(v) -> bool:
    if v is None or isinstance(v, (str, int, float, bool)):
        return True
    if isinstance(v, (list, tuple)):
        return all(_jsonable(x) for x in v)
    if isinstance(v, dict):
        return all(isinstance(k, str) and _jsonable(x) for k, x in v.items())
    return False


def _as_list(x) -> list:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _knob_defines(knobs: dict, defines: list[str]) -> list[str]:
    out = []
    for k in sorted(knobs, key=str):
        v = knobs[k]
        if isinstance(v, bool):
            v = int(v)
        out.append(f"-D{k}={v}")
    out += [d for d in defines if d]
    return out


def call_key(kind: str, kw: dict, tc_version: str, tc_ptxas: str, knobs: dict, defines: list[str]) -> str:
    """Key of one intercepted load/load_inline call (before kdev adds -v/-lineinfo; knobs/defines included).
    ``tc_ptxas`` is the toolchain identity (ptxas path + pinned env, see kdev.build.cache.tc_ident); the process
    environment is deliberately not part of the key so build lane and GPU worker agree."""
    parts: list[bytes] = [f"kind={kind}|name={kw.get('name')}\n".encode()]
    if kind == "load_inline":
        for k in ("cpp_sources", "cuda_sources", "functions"):
            for s in _as_list(kw.get(k)):
                parts.append(f"<{k}>".encode() + str(s).encode())
    else:
        for p in _as_list(kw.get("sources")):
            try:
                with open(p, "rb") as f:
                    parts.append(f"<src {os.path.basename(str(p))}>".encode() + f.read())
            except OSError:
                parts.append(f"<src {p}>".encode())
    flags = []
    for k in ("extra_cflags", "extra_cuda_cflags", "extra_ldflags", "extra_include_paths"):
        flags += [f"{k}:{f}" for f in _as_list(kw.get(k))]
    flags.append(f"with_cuda:{kw.get('with_cuda')}")
    flags += [f"define:{d}" for d in defines if d]
    return ext_key(b"\0".join(parts), tc_version, tc_ptxas, flags, knobs)


class _Stub:
    """Returned instead of the extension module in build mode: any attribute is a no-op callable."""

    def __init__(self, name: str) -> None:
        self.__name__ = name

    def __getattr__(self, item):
        if item.startswith("__"):
            raise AttributeError(item)
        return lambda *a, **k: None


_NAMES = {"load_inline": ("name", "cpp_sources", "cuda_sources", "functions", "extra_cflags",
                          "extra_cuda_cflags", "extra_ldflags", "extra_include_paths", "build_directory",
                          "verbose", "with_cuda", "is_python_module", "with_pytorch_error_handling",
                          "keep_intermediates", "use_pch"),
          "load": ("name", "sources", "extra_cflags", "extra_cuda_cflags", "extra_ldflags", "extra_include_paths",
                   "build_directory", "verbose", "with_cuda", "is_python_module", "is_standalone",
                   "keep_intermediates")}


def _bind(kind: str, args: tuple, kw: dict) -> dict:
    out = dict(zip(_NAMES[kind], args))
    out.update(kw)
    return out


class Hook:
    """State of an installed hook (records every intercepted extension)."""

    def __init__(self, *, mode: str, ext_root: str, tc_version: str, tc_ptxas: str, knobs: dict | None = None,
                 defines: list[str] | None = None, force: bool = False, allow_jit: bool = False,
                 extra_cuda_flags: list[str] | None = None) -> None:
        self.mode = mode
        self.ext_root = ext_root
        self.tc_version = tc_version
        self.tc_ptxas = tc_ptxas
        self.knobs = dict(knobs or {})
        self.defines = list(defines or [])
        self.force = force
        self.allow_jit = allow_jit
        self.extra_cuda_flags = list(extra_cuda_flags if extra_cuda_flags is not None else
                                     ["-Xptxas=-v", "-lineinfo"])
        self.records: list[dict] = []
        # use mode: fall back to an up-to-date torch build in TORCH_EXTENSIONS_DIR (never compiles). Opt-in
        # (KDEV_LEGACY_EXT=1): torch rewrites the sources of a shared build dir when they differ, which makes
        # another agent's next load recompile; the content-keyed kdev cache + pending records are the default.
        self.legacy_fallback = os.environ.get("KDEV_LEGACY_EXT") == "1"
        self.owner = ""          # submission file whose calls this hook currently serves (pending records)
        self.discover = False    # use mode, discovery job: missing entries -> pending record + stub module
        self.real: dict = {}

    # -- the replacement for load_inline / load ------------------------------------------------------------
    def make(self, kind: str):
        def hooked(*args, **kwargs):
            return self._call(kind, args, kwargs)
        hooked.__name__ = kind
        hooked.__kdev_hook__ = True  # type: ignore[attr-defined]
        return hooked

    def _call(self, kind: str, args: tuple, kwargs: dict):
        kw = _bind(kind, args, kwargs)
        name = str(kw.get("name"))
        key = call_key(kind, kw, self.tc_version, self.tc_ptxas, self.knobs, self.defines)
        d = os.path.join(self.ext_root, key)
        marker = os.path.join(d, MARKER)
        rec = {"name": name, "kind": kind, "key": key, "dir": d, "cache_hit": False, "seconds": 0.0,
               "status": "ok", "log_path": os.path.join(d, "build.log"), "so_path": ""}
        complete = os.path.isfile(marker) and _so_in(d, name)
        if self.mode == "use":
            if not complete:
                if not self.allow_jit and self.legacy_fallback and not (self.knobs or self.defines):
                    mod = self._legacy(kind, args, kwargs, rec)
                    if mod is not None:
                        return mod
                if self.discover:          # discovery job: record, hand back a stub, keep going (no compile)
                    rec["status"] = "pending"
                    self.records.append(rec)
                    self._record_pending(kind, kw, key)
                    return _Stub(name)
                if not self.allow_jit:
                    rec["status"] = "needs_build"
                    self.records.append(rec)
                    self._record_pending(kind, kw, key)
                    raise NeedsBuild(f"extension {name!r} (key {key}) has no prebuilt entry in {self.ext_root}; "
                                     f"run `kdev build` first")
                return self._build(kind, kw, rec, d, marker, stub=False)
            rec["cache_hit"] = True
            rec["so_path"] = _so_in(d, name)
            self.records.append(rec)
            from torch.utils import cpp_extension as ce  # noqa: PLC0415 - worker only
            return ce._import_module_from_library(name, d, kw.get("is_python_module", True))  # noqa: SLF001
        if complete and not self.force:
            rec["cache_hit"] = True
            rec["so_path"] = _so_in(d, name)
            self.records.append(rec)
            return _Stub(name)
        return self._build(kind, kw, rec, d, marker, stub=True)

    def _record_pending(self, kind: str, kw: dict, key: str) -> None:
        """Remember a call the GPU worker could not serve (typically an extension built inside prepare(), with
        GPU tensors, which the CPU-only build lane never reaches) so the next ``kdev build`` of the owning file
        compiles exactly this call: ``<ext_root>/pending/<key>.json``."""
        try:
            d = os.path.join(self.ext_root, "pending")
            os.makedirs(d, exist_ok=True)
            safe = {k: v for k, v in kw.items() if _jsonable(v)}
            doc = {"kind": kind, "kw": safe, "key": key, "knobs": self.knobs, "defines": self.defines,
                   "tc_version": self.tc_version, "tc_ptxas": self.tc_ptxas, "file": self.owner,
                   "file_md5": file_md5(self.owner) if self.owner else "", "ts": time.time()}
            with open(os.path.join(d, key + ".json"), "w", encoding="utf-8") as f:
                json.dump(doc, f)
        except Exception:  # noqa: BLE001 - best effort; the NeedsBuild error still names the extension
            pass

    def build_pending(self, file: str, shard: int = 0, shards: int = 1) -> list[str]:
        """Build mode: compile the pending calls recorded for ``file`` (or with no owner) for this toolchain
        (every ``shards``-th one starting at ``shard``: parallel shim processes split the list).
        Returns the keys built; a built call's pending file is removed, records of an older content of
        ``file`` are dropped unbuilt (stale)."""
        done: list[str] = []
        recs, _ = scan_pending(self.ext_root, file, self.tc_version, self.tc_ptxas, prune=shard == 0)
        knobs, defines = self.knobs, self.defines
        for j, (p, doc) in enumerate(recs):
            if j % max(1, shards) != shard:
                continue
            self.knobs, self.defines = dict(doc.get("knobs") or {}), list(doc.get("defines") or [])
            try:
                self._call(doc["kind"], (), dict(doc["kw"]))
                key = call_key(doc["kind"], doc["kw"], self.tc_version, self.tc_ptxas, self.knobs, self.defines)
                if key == doc.get("key"):
                    done.append(key)
                    os.remove(p)
            except Exception as e:  # noqa: BLE001
                self.records.append({"name": doc.get("kw", {}).get("name"), "kind": doc.get("kind"),
                                     "key": doc.get("key"), "status": "failed", "error": str(e)[:300],
                                     "cache_hit": False, "seconds": 0.0, "dir": "", "log_path": "", "so_path": ""})
            finally:
                self.knobs, self.defines = knobs, defines
        return done

    def _legacy(self, kind: str, args: tuple, kwargs: dict, rec: dict):
        """Missing from the kdev cache: load torch's own up-to-date build (TORCH_EXTENSIONS_DIR, e.g. one the
        pod's prebuild scripts made) through the real load_inline, but only if ninja has no work to do, so
        nothing compiles in the GPU lane. Extensions built at prepare time (with GPU tensors) never reach the
        CPU-only build lane; this is how they load. Returns None (caller raises NeedsBuild) when a compile
        would be needed. The .so's toolchain notes are still checked against the pin by the timing engine."""
        try:
            from torch.utils import cpp_extension as ce  # noqa: PLC0415 - worker only
        except Exception:  # noqa: BLE001
            return None
        orig = getattr(ce, "_run_ninja_build", None)
        if orig is None:
            return None
        import subprocess  # noqa: PLC0415
        seen: list[str] = []

        def guarded(build_directory, verbose, error_prefix):
            seen.append(build_directory)
            try:
                out = subprocess.run(["ninja", "-C", build_directory, "-n"], capture_output=True, text=True,
                                     timeout=60).stdout
            except Exception as e:  # noqa: BLE001
                out = str(e)
            if "no work to do" not in out:
                raise NeedsBuild(f"extension {rec['name']!r}: not in the kdev cache and the legacy build in "
                                 f"{build_directory} is not up to date; run `kdev build`")
            return orig(build_directory, verbose, error_prefix)

        ce._run_ninja_build = guarded
        try:
            mod = self.real[kind](*args, **kwargs)
        except NeedsBuild:
            return None
        finally:
            ce._run_ninja_build = orig
        rec["status"] = "legacy"
        rec["dir"] = seen[-1] if seen else ""
        rec["so_path"] = _so_in(rec["dir"], rec["name"]) if rec["dir"] else ""
        self.records.append(rec)
        return mod

    def _build(self, kind: str, kw: dict, rec: dict, d: str, marker: str, *, stub: bool):
        os.makedirs(d, exist_ok=True)
        lock = os.path.join(d, "lock")
        if os.path.exists(lock) and time.time() - os.path.getmtime(lock) > 600:
            os.remove(lock)   # stale torch FileBaton from a killed build
        kw = dict(kw)
        kw["build_directory"] = d
        kw["verbose"] = True
        dflags = _knob_defines(self.knobs, self.defines)
        kw["extra_cuda_cflags"] = _as_list(kw.get("extra_cuda_cflags")) + dflags + self.extra_cuda_flags
        kw["extra_cflags"] = _as_list(kw.get("extra_cflags")) + dflags
        real = self.real[kind]
        log_path = rec["log_path"]
        t0 = time.monotonic()
        err = ""
        result = None
        for attempt in (1, 2):
            err = ""
            with open(log_path, "ab") as logf:
                logf.write(f"### kdev build attempt {attempt}: {kind} {rec['name']} key={rec['key']}\n".encode())
                logf.flush()
                sys.stdout.flush()
                sys.stderr.flush()
                saved = os.dup(1), os.dup(2)
                os.dup2(logf.fileno(), 1)
                os.dup2(logf.fileno(), 2)
                try:
                    result = real(**kw)
                except Exception as e:  # noqa: BLE001
                    err = f"{type(e).__name__}: {e}"
                finally:
                    sys.stdout.flush()
                    sys.stderr.flush()
                    os.dup2(saved[0], 1)
                    os.dup2(saved[1], 2)
                    os.close(saved[0])
                    os.close(saved[1])
            if not err:
                break
            if attempt == 1 and _TRANSIENT.search(err + _tail(log_path, 200)):
                rec["retried"] = True
                continue
            break
        rec["seconds"] = round(time.monotonic() - t0, 3)
        if err:
            rec["status"] = "failed"
            rec["error"] = err[-4000:]
            self.records.append(rec)
            raise ExtBuildFailed(f"extension {rec['name']} failed to build: {err[:400]}")
        rec["so_path"] = _so_in(d, rec["name"])
        info = {k: rec[k] for k in ("name", "kind", "key", "seconds", "so_path")}
        info.update({"tc_version": self.tc_version, "tc_ptxas": self.tc_ptxas, "knobs": self.knobs,
                     "defines": self.defines, "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     "cuda_flags": kw["extra_cuda_cflags"]})
        tmp = marker + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=1)
        os.replace(tmp, marker)
        self.records.append(rec)
        return _Stub(rec["name"]) if stub else result


def _tail(path: str, n: int) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except OSError:
        return ""


def _so_in(d: str, name: str) -> str:
    for ext in (".so", ".pyd", ".dll"):
        p = os.path.join(d, name + ext)
        if os.path.isfile(p):
            return p
    return ""


_ACTIVE: list = []          # [Hook] once installed in this process


_CONFIG_FIELDS = ("mode", "ext_root", "tc_version", "tc_ptxas", "knobs", "defines", "force", "allow_jit",
                  "extra_cuda_flags", "owner")


def active():
    """The hook installed in this process, or None."""
    return _ACTIVE[0] if _ACTIVE else None


def install(hook: Hook):
    """Patch torch.utils.cpp_extension.load_inline/load with ``hook`` (imports torch). Returns the active hook.

    If a hook is already installed, its configuration is replaced by ``hook``'s and the *installed* hook is
    returned: submissions bind ``load_inline`` at import time (``from torch.utils.cpp_extension import ...``),
    so the patched functions must stay the same objects for the whole process."""
    from torch.utils import cpp_extension as ce  # noqa: PLC0415 - runs in the pinned interpreter only

    cur = active()
    if cur is not None and all(getattr(getattr(ce, k), "__kdev_hook__", False) for k in ("load_inline", "load")):
        for k in _CONFIG_FIELDS:
            setattr(cur, k, getattr(hook, k))
        return cur
    for kind in ("load_inline", "load"):
        f = getattr(ce, kind)
        if getattr(f, "__kdev_hook__", False):
            continue
        hook.real[kind] = f
        setattr(ce, kind, hook.make(kind))
    _ACTIVE[:] = [hook]
    return hook


def _call_entries(mod, entries: list[str], errors: list[str], hook: "Hook | None" = None) -> list[str]:
    called = []
    names = list(entries)
    for n, obj in vars(mod).items():
        if callable(obj) and _ENTRY_RE.match(n) and n not in names and getattr(obj, "__module__", None) == mod.__name__:
            names.append(n)
    for n in names:
        fn = getattr(mod, n, None)
        if not callable(fn) or inspect.isclass(fn):
            continue
        try:
            sig = inspect.signature(fn)
            if any(p.default is p.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
                   for p in sig.parameters.values()):
                continue
        except (TypeError, ValueError):
            continue
        called.append(n)
        guess = bool(sig.parameters)        # only defaulted params: we guess the arguments (may be unused)
        n0 = len(hook.records) if hook is not None else 0
        try:
            fn()
        except ExtBuildFailed as e:
            errors.append(("guess: " if guess else "") + str(e))
        except Exception as e:  # noqa: BLE001 - kernel launches etc. fail without CUDA; expected
            errors.append(f"entry {n}: {type(e).__name__}: {str(e)[:200]}")
        if guess and hook is not None:
            for r in hook.records[n0:]:
                r["origin"] = "entry"     # default-args call: may be a variant the file never uses
    return called


def main(argv: list[str]) -> int:
    with open(argv[1], encoding="utf-8") as f:
        spec = json.load(f)
    out = {"exts": [], "errors": [], "import_error": None, "entries_called": []}
    if spec.get("mode", "build") == "build":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""      # never a CUDA context in the build lane
    hook = Hook(mode=spec.get("mode", "build"), ext_root=spec["ext_root"], tc_version=spec.get("tc_version", ""),
                tc_ptxas=spec.get("tc_ptxas", ""), knobs=spec.get("knobs") or {}, defines=spec.get("defines") or [],
                force=bool(spec.get("force")))
    try:
        import torch  # noqa: PLC0415
        cap = tuple(spec.get("cap") or (9, 0))
        torch.cuda.get_device_capability = lambda *a, **k: cap  # compile only: never touch the driver
        install(hook)
    except Exception as e:  # noqa: BLE001
        out["import_error"] = f"torch unavailable: {type(e).__name__}: {e}"
        _write(spec["out"], out, hook)
        return 3
    path = os.path.abspath(spec["file"])
    if spec.get("pending_only"):            # one shard of a parallel pending build (see compile._build_py)
        out["pending_built"] = hook.build_pending(path, int(spec.get("shard", 0)), int(spec.get("shards", 1)))
        _write(spec["out"], out, hook)
        return 0
    sys.path.insert(0, os.path.dirname(path))
    os.chdir(os.path.dirname(path))
    os.environ["KDEV_KNOBS"] = json.dumps(spec.get("knobs") or {})
    try:
        s = importlib.util.spec_from_file_location("kdev_build_target", path)
        mod = importlib.util.module_from_spec(s)
        sys.modules["kdev_build_target"] = mod
        s.loader.exec_module(mod)  # type: ignore[union-attr]
    except ExtBuildFailed as e:
        out["errors"].append(str(e))
        mod = None
    except Exception as e:  # noqa: BLE001
        out["import_error"] = f"{type(e).__name__}: {e}\n" + traceback.format_exc()[-2000:]
        mod = None
    if mod is not None:
        out["entries_called"] = _call_entries(mod, list(spec.get("entries") or DEFAULT_ENTRIES), out["errors"], hook)
        out["entry_errors"] = [e for e in out["errors"] if e.startswith("guess: ")]
        out["errors"] = [e for e in out["errors"] if not e.startswith("guess: ")]
    if hook.mode == "build" and spec.get("prepare_pass", os.environ.get("KDEV_CPU_PREPARE") == "1")             and not out["import_error"]:
        # prepare-time / first-call extensions: run load -> prepare -> kernel_fn per shape on the CPU with a
        # faked torch.cuda (no CUDA context), so the hook sees and builds those load_inline calls too
        try:
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            from kdev.build import _cpuprep  # noqa: PLC0415
            out["prepare_pass"] = _cpuprep.run(path, cap=tuple(spec.get("cap") or (9, 0)),
                                               budget_s=float(spec.get("prepare_budget_s", 900)))
        except Exception as e:  # noqa: BLE001
            out["prepare_pass"] = {"errors": [f"unavailable: {type(e).__name__}: {e}"], "steps": []}
    if hook.mode == "build" and not spec.get("skip_pending"):
        out["pending_built"] = hook.build_pending(path)
    _write(spec["out"], out, hook)
    return 0


def _write(path: str, out: dict, hook: Hook) -> None:
    out["exts"] = hook.records
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
