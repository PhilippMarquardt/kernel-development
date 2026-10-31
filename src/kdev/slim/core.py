"""`kdev slim` core: plain-source, dead-route-free copy of a submission stack file. CPU only.

Generalised from okc muse_bf16_proj pp/slim.sh (mkpp.py + mkslim.py, Round 10 pp_):

1. **decode** every zlib+base64 source blob into a plain string literal: calls of a helper whose body is
   ``return zlib.decompress(base64.b64decode(arg)).decode()`` and inline ``*.decompress(*.b64decode(X)).decode()``
   expressions (X a string constant, or a module-level name bound to one: the name is rebound to the decoded
   text). Every literal is verified (``literal_eval(lit) == decoded``). Decoded strings that are Python
   source (embedded modules run via exec/compile) are decoded recursively. Helpers and base64/zlib imports
   that become unused are removed.
2. **drop dead routes** per a dispatch trace (``kdev dispatch FILE`` -> dispatch.json): a *builder* is a
   top-level function that calls ``load_inline``/``load``; its kernels are the ``__global__`` names in the
   strings it references. A builder none of whose kernels launched on any traced shape is dead. It is
   dropped (body -> ``raise RuntimeError``, the CUDA/C++ strings only it referenced are deleted) only if it is
   safe: its name never appears as a string (getattr/globals dispatch), it is not decorated, and every
   reference to it sits inside a ``try`` body (a failing build degrades instead of crashing). Embedded
   module sources (string constants passed to exec/compile) are slimmed the same way.
3. **refuse** unsafe results: unparsable input/output, a kernel that launched in the trace no longer defined,
   a dynamic blob (decode of a non-constant), output larger than the input (``allow_grow``), an OKC file
   >= 1,000,000 bytes (warn >= 950,000).

Per-problem rewrites (e.g. mkslim's port of one embedded route into the main module) plug in as a hook
module with ``def slim(src: str, ctx: dict) -> str`` applied after decoding.
"""
from __future__ import annotations

import ast
import base64
import hashlib
import importlib.util
import json
import os
import re
import zlib
from dataclasses import dataclass, field
from typing import Any, Callable

OKC_LIMIT = 1_000_000
OKC_MARGIN = 950_000
LEFTOVERS = ("b64decode", "decompress(", "exec(", "\r")
_LOAD_FNS = ("load_inline", "load")


class SlimRefused(Exception):
    """An unsafe case: the message says why; nothing was written."""


@dataclass
class SlimReport:
    decoded: int = 0
    dropped: list[dict[str, Any]] = field(default_factory=list)      # {name, kernels, bytes, where}
    kept_dead: list[dict[str, Any]] = field(default_factory=list)    # dead per trace but unsafe to drop
    live_builders: int = 0
    warnings: list[str] = field(default_factory=list)
    leftovers: dict[str, int] = field(default_factory=dict)
    hook: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


# --------------------------------------------------------------------------------------------- text edits
def _line_offsets(b: bytes) -> list[int]:
    offs = [0]
    for i, ch in enumerate(b):
        if ch == 10:
            offs.append(i + 1)
    return offs


def _span(offs: list[int], node: ast.AST) -> tuple[int, int]:
    """Byte span of a node (ast col offsets are utf-8 byte offsets)."""
    return offs[node.lineno - 1] + node.col_offset, offs[node.end_lineno - 1] + node.end_col_offset  # type: ignore[attr-defined]


def _apply(src: str, edits: list[tuple[int, int, str]]) -> str:
    b = src.encode("utf-8")
    for a, e, txt in sorted(edits, key=lambda t: t[0], reverse=True):
        b = b[:a] + txt.encode("utf-8") + b[e:]
    return b.decode("utf-8")


def lit(s: str) -> str:
    """Shortest-risk literal for ``s`` whose value is verified to be exactly ``s``."""
    for pre in ("r", ""):
        for q in ('"""', "'''"):
            if q in s or s.endswith(q[0]) or (pre == "r" and s.endswith("\\")):
                continue
            cand = pre + q + s + q
            try:
                if ast.literal_eval(cand) == s:
                    return cand
            except (SyntaxError, ValueError):
                pass
    return repr(s)


# --------------------------------------------------------------------------------------------- 1. decode
def _decode_payload(p: str) -> str | None:
    try:
        return zlib.decompress(base64.b64decode(p, validate=True)).decode("utf-8")
    except Exception:  # noqa: BLE001
        return None


def _is_decode_chain(node: ast.AST) -> ast.AST | None:
    """``<x>.decompress(<y>.b64decode(ARG)).decode(...)`` -> ARG node."""
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "decode"):
        return None
    inner = node.func.value
    if not (isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute) and inner.func.attr == "decompress"
            and len(inner.args) == 1):
        return None
    b = inner.args[0]
    if not (isinstance(b, ast.Call) and isinstance(b.func, ast.Attribute) and b.func.attr == "b64decode"
            and len(b.args) == 1):
        return None
    return b.args[0]


def _helpers(tree: ast.Module) -> set[str]:
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and len(n.args.args) == 1:
            body = [s for s in n.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
            if len(body) == 1 and isinstance(body[0], ast.Return) and body[0].value is not None:
                arg = _is_decode_chain(body[0].value)
                if isinstance(arg, ast.Name) and arg.id == n.args.args[0].arg:
                    out.add(n.name)
    return out


def _module_strs(tree: ast.Module) -> dict[str, ast.Assign]:
    out = {}
    for n in tree.body:
        if (isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)):
            out[n.targets[0].id] = n
    return out


def _names_used(tree: ast.AST) -> dict[str, int]:
    cnt: dict[str, int] = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            cnt[n.id] = cnt.get(n.id, 0) + 1
        elif isinstance(n, ast.Attribute):
            pass
    return cnt


def decode_blobs(src: str, depth: int = 0) -> tuple[str, int]:
    """Replace every zlib+base64 blob by its plain text (see module doc). Returns (source, n_decoded)."""
    total = 0
    for _ in range(20):                                     # passes until stable
        tree = ast.parse(src)
        offs = _line_offsets(src.encode("utf-8"))
        helpers = _helpers(tree)
        mstr = _module_strs(tree)
        edits: list[tuple[int, int, str]] = []
        rebound: set[str] = set()
        taken: list[tuple[int, int]] = []
        for n in ast.walk(tree):
            arg = None
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in helpers and len(n.args) == 1:
                arg = n.args[0]
            elif _is_decode_chain(n) is not None:
                arg = _is_decode_chain(n)
            if arg is None:
                continue
            if any(fn.name in helpers and fn.lineno <= n.lineno <= (fn.end_lineno or fn.lineno)
                   for fn in ast.walk(tree) if isinstance(fn, ast.FunctionDef)):
                continue                                    # the helper's own body
            a, e = _span(offs, n)
            if any(a < te and ta < e for ta, te in taken):
                continue
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                txt = _decode_payload(arg.value)
                if txt is None:
                    raise SlimRefused(f"line {n.lineno}: blob does not decode as zlib+base64 utf-8 text")
                edits.append((a, e, lit(_maybe_recurse(txt, depth))))
            elif isinstance(arg, ast.Name) and arg.id in mstr:
                txt = _decode_payload(mstr[arg.id].value.value)  # type: ignore[union-attr]
                if txt is None:
                    raise SlimRefused(f"line {n.lineno}: {arg.id} does not decode as zlib+base64 utf-8 text")
                if arg.id not in rebound:
                    va, ve = _span(offs, mstr[arg.id].value)
                    edits.append((va, ve, lit(_maybe_recurse(txt, depth))))
                    rebound.add(arg.id)
                edits.append((a, e, arg.id))
            else:
                raise SlimRefused(f"line {n.lineno}: dynamic blob (decode of a non-constant expression): refusing")
            taken.append((a, e))
            total += 1
        if not edits:
            break
        src = _apply(src, edits)
        ast.parse(src)
    return _drop_unused_helpers(src), total


def _maybe_recurse(txt: str, depth: int) -> str:
    if depth < 3 and ("b64decode" in txt or "decompress(" in txt):
        try:
            ast.parse(txt)
        except SyntaxError:
            return txt
        return decode_blobs(txt, depth + 1)[0]
    return txt


def _drop_unused_helpers(src: str) -> str:
    for _ in range(5):
        tree = ast.parse(src)
        offs = _line_offsets(src.encode("utf-8"))
        used = _names_used(tree)
        edits = []
        helpers = _helpers(tree)
        for n in tree.body:
            if isinstance(n, ast.FunctionDef) and n.name in helpers and not used.get(n.name):
                edits.append(_stmt_span(offs, n, src))
        for n in tree.body:
            if isinstance(n, (ast.Import,)) and all(a.name in ("base64", "zlib") for a in n.names):
                names = [a.asname or a.name for a in n.names]
                if not any(used.get(x) or _attr_used(tree, x) for x in names):
                    edits.append(_stmt_span(offs, n, src))
        if not edits:
            return src
        src = _apply(src, [(a, e, "") for a, e in edits])
    return src


def _attr_used(tree: ast.AST, name: str) -> bool:
    return any(isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == name
               for n in ast.walk(tree))


def _stmt_span(offs: list[int], node: ast.stmt, src: str) -> tuple[int, int]:
    """Whole lines of a top-level statement (decorators included), plus its trailing newline."""
    first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    a = offs[first - 1]
    e = offs[node.end_lineno] if node.end_lineno < len(offs) else len(src.encode("utf-8"))  # type: ignore[operator]
    return a, e


# --------------------------------------------------------------------------------------------- 2. routes
_LB = re.compile(r"__launch_bounds__\s*\([^)]*\)")
_GLOBAL = re.compile(r"__global__\s+[^;{(]*?\b([A-Za-z_]\w*)\s*\(")


def kernels_in(text: str) -> set[str]:
    return set(_GLOBAL.findall(_LB.sub(" ", text)))


def launched_kernels(trace: dict[str, Any]) -> set[str]:
    from kdev.observe.dispatch import base_name

    out = set()
    for mode in ("eager", "graph"):
        for ks in (trace.get(mode) or {}).values():
            for k in ks or []:
                out.add(base_name(str(k.get("kernel", ""))).split("::")[-1])
    return out


def _calls_load(fn: ast.FunctionDef) -> bool:
    for n in ast.walk(fn):
        if isinstance(n, ast.Call):
            f = n.func
            if (isinstance(f, ast.Name) and f.id in _LOAD_FNS) or (isinstance(f, ast.Attribute) and f.attr == "load_inline"):
                return True
    return False


def _strings_of(fn: ast.AST, mstr: dict[str, ast.Assign]) -> tuple[str, set[str]]:
    texts, refs = [], set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            texts.append(n.value)
        elif isinstance(n, ast.Name) and n.id in mstr:
            refs.add(n.id)
            texts.append(mstr[n.id].value.value)  # type: ignore[union-attr]
    return "\n".join(texts), refs


def _parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    par = {}
    for p in ast.walk(tree):
        for c in ast.iter_child_nodes(p):
            par[c] = p
    return par


def _in_try_body(node: ast.AST, par: dict[ast.AST, ast.AST]) -> bool:
    child, p = node, par.get(node)
    while p is not None:
        if isinstance(p, ast.Try) and child in p.body:
            return True
        child, p = p, par.get(p)
    return False


def _str_constants(tree: ast.AST) -> set[str]:
    return {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and len(n.value) < 200}


def _embedded_modules(tree: ast.Module) -> list[str]:
    """Names of top-level string constants that are Python source run through exec()/compile()."""
    mstr = _module_strs(tree)
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("exec", "compile") and n.args:
            a = n.args[0]
            if isinstance(a, ast.Name) and a.id in mstr and a.id not in out:
                try:
                    ast.parse(mstr[a.id].value.value)  # type: ignore[union-attr]
                except SyntaxError:
                    continue
                out.append(a.id)
    return out


def drop_dead(src: str, launched: set[str], rep: SlimReport, *, force: tuple[str, ...] = (), where: str = "main",
              depth: int = 0, keep: tuple[str, ...] = (), all_dead: bool = False) -> str:
    """Drop the safe dead builders of ``src`` (and of its embedded modules). Mutates ``rep``."""
    tree = ast.parse(src)
    offs = _line_offsets(src.encode("utf-8"))
    mstr = _module_strs(tree)
    par = _parents(tree)
    strs = _str_constants(tree)
    edits: list[tuple[int, int, str]] = []
    freed: set[str] = set()
    for fn in tree.body:
        if not (isinstance(fn, ast.FunctionDef) and _calls_load(fn)):
            continue
        text, refs = _strings_of(fn, mstr)
        ks = kernels_in(text)
        if not ks:
            continue                                         # kernels unknown: never dead
        if ks & launched:
            rep.live_builders += 1
            continue
        why = ""
        if fn.name in keep:
            why = "kept (--keep)"
        elif fn.name in strs:
            why = "name used as a string (dynamic dispatch)"
        elif fn.decorator_list:
            why = "decorated"
        else:
            refs_out = [n for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == fn.name
                        and isinstance(n.ctx, ast.Load) and not _in_try_body(n, par)]
            if refs_out and fn.name not in force and not all_dead:
                why = f"called outside try/except (line {refs_out[0].lineno}); --drop {fn.name} or --all-dead"
        if not why and (not fn.body or fn.body[0].lineno == fn.lineno):
            why = "one-line def"
        info = {"name": fn.name, "kernels": sorted(ks)[:6], "where": where}
        if why:
            rep.kept_dead.append(dict(info, reason=why))
            continue
        a, e = _span(offs, fn)
        indent = " " * (fn.body[0].col_offset if fn.body else 4)
        head = src.encode("utf-8")[a:offs[fn.body[0].lineno - 1]].decode("utf-8")
        body = (f"{indent}raise RuntimeError(\"kdev slim: route {fn.name} removed (none of its kernels "
                f"launched in the dispatch trace)\")")
        edits.append((a, e, head + body))
        info["bytes"] = e - a
        rep.dropped.append(info)
        freed |= refs
    if edits:
        src = _apply(src, edits)
        src = _drop_unused_strings(src, freed)
    for f in force:
        if f not in {d["name"] for d in rep.dropped} and where == "main" and depth == 0:
            if not any(isinstance(n, ast.FunctionDef) and n.name == f for n in ast.walk(ast.parse(src))):
                rep.warnings.append(f"--drop {f}: no such top-level builder")
    if depth < 2:
        tree = ast.parse(src)
        mstr = _module_strs(tree)
        offs = _line_offsets(src.encode("utf-8"))
        emb_edits = []
        for nm in _embedded_modules(tree):
            inner = mstr[nm].value.value  # type: ignore[union-attr]
            new = drop_dead(inner, launched, rep, force=force, where=f"{where}:{nm}", depth=depth + 1, keep=keep,
                            all_dead=all_dead)
            if new != inner:
                va, ve = _span(offs, mstr[nm].value)
                emb_edits.append((va, ve, lit(new)))
        if emb_edits:
            src = _apply(src, emb_edits)
    return src


def _drop_unused_strings(src: str, names: set[str]) -> str:
    tree = ast.parse(src)
    used = _names_used(tree)
    offs = _line_offsets(src.encode("utf-8"))
    edits = []
    for nm, node in _module_strs(tree).items():
        if nm in names and not used.get(nm) and nm not in _str_constants(tree):
            edits.append(_stmt_span(offs, node, src))
    return _apply(src, [(a, e, "") for a, e in edits]) if edits else src


# --------------------------------------------------------------------------------------------- trace + all
def load_trace(ref: str, cfg: Any = None) -> tuple[dict[str, Any], str, str]:
    """dispatch.json path | run dir | run id -> (trace, path, traced file md5 or '')."""
    p = os.path.expanduser(ref)
    cands = []
    if os.path.isfile(p):
        cands = [p]
    else:
        d = p
        if not os.path.isdir(d):
            if cfg is None:
                from kdev.config import load_config
                cfg = load_config()
            from kdev.runs import open_run
            d = str(open_run(cfg, ref).path)
        cands = [os.path.join(d, "dispatch.json"), os.path.join(d, "dispatch", "dispatch.json")]
    path = next((c for c in cands if os.path.isfile(c)), None)
    if path is None:
        raise SlimRefused(f"no dispatch.json for {ref!r}")
    with open(path, encoding="utf-8") as f:
        trace = json.load(f)
    if not isinstance(trace, dict) or not (trace.get("eager") or trace.get("graph")):
        raise SlimRefused(f"{path}: not a kdev dispatch trace (no eager/graph kernels)")
    md5 = ""
    for jn in ("job.json", os.path.join("..", "job.json")):
        jp = os.path.join(os.path.dirname(path), jn)
        if os.path.isfile(jp):
            try:
                with open(jp, encoding="utf-8") as f:
                    files = json.load(f).get("files") or []
                md5 = (files[0].get("md5") or "") if files else ""
            except (OSError, ValueError, AttributeError):
                pass
            break
    return trace, path, md5


def md5s(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


def load_hook(path: str) -> Callable[[str, dict[str, Any]], str]:
    spec = importlib.util.spec_from_file_location("kdev_slim_hook", path)
    if spec is None or spec.loader is None:
        raise SlimRefused(f"cannot load hook {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, "slim", None)
    if not callable(fn):
        raise SlimRefused(f"hook {path} has no slim(src, ctx) function")
    return fn


def slim(src: str, trace: dict[str, Any], *, drop: bool = True, force: tuple[str, ...] = (),
         keep: tuple[str, ...] = (), hook: Callable[[str, dict[str, Any]], str] | None = None,
         limit: int | None = None, allow_grow: bool = False, all_dead: bool = False) -> tuple[str, SlimReport]:
    """The whole recipe on a source text; raises SlimRefused on any unsafe case."""
    rep = SlimReport()
    try:
        ast.parse(src)
    except SyntaxError as e:
        raise SlimRefused(f"input does not parse: {e}") from e
    n_in = len(src.encode("utf-8"))
    out = src.replace("\r\n", "\n")
    out, rep.decoded = decode_blobs(out)
    defined_in = kernels_in(_all_strings(out))
    launched = launched_kernels(trace)
    if hook is not None:
        out = hook(out, {"trace": trace, "launched": sorted(launched)})
        rep.hook = getattr(hook, "__module__", "hook")
        try:
            ast.parse(out)
        except SyntaxError as e:
            raise SlimRefused(f"hook output does not parse: {e}") from e
    if drop:
        out = drop_dead(out, launched, rep, force=force, keep=keep, all_dead=all_dead)
        if all_dead and rep.dropped:
            rep.warnings.append("--all-dead: a dropped builder that prepare() calls unconditionally now raises; "
                                "verify with kdev check OUT before anything else")
    try:
        compile(out, "<slim>", "exec")
    except SyntaxError as e:
        raise SlimRefused(f"output does not compile: {e}") from e
    defined_out = kernels_in(_all_strings(out))
    lost = sorted((launched & defined_in) - defined_out)
    if lost:
        raise SlimRefused("kernels that launched in the trace would be removed: " + ", ".join(lost[:8]))
    if not (launched & defined_in):
        rep.warnings.append("none of the traced kernels is defined in this file: is the trace for this file?")
    n_out = len(out.encode("utf-8"))
    if n_out > n_in and not allow_grow:
        raise SlimRefused(f"output ({n_out:,} B) is larger than the input ({n_in:,} B): decoding alone does not "
                          "slim this file (no dead routes were safe to drop); --allow-grow to write it anyway")
    if limit is not None:
        if n_out >= limit:
            raise SlimRefused(f"output {n_out:,} B >= the {limit:,}-byte submission limit")
        if n_out >= OKC_MARGIN and limit == OKC_LIMIT:
            rep.warnings.append(f"output {n_out:,} B is within 5 % of the {limit:,}-byte limit")
    rep.leftovers = {w if w != "\r" else "CR": _count_code(out, w) for w in LEFTOVERS}
    rep.leftovers = {k: v for k, v in rep.leftovers.items() if v}
    return out, rep


def _all_strings(src: str) -> str:
    return "\n".join(n.value for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Constant) and isinstance(n.value, str))


def _count_code(src: str, word: str) -> int:
    return sum(1 for ln in src.split("\n") if word in ln and not ln.lstrip().startswith("#"))
