"""`kdev comply`: rules checklist (M19, LESSONS 5). Owner: P7. Output: PASS/RISK items; RISK needs a human.

Static (no GPU): forbid_patterns (host syncs, .item(), printing tensors, caching dicts keyed on data
pointers), weight precision stored in prepare (lossless rule). Dynamic (worker kind='comply'): per-op
precision probes on intermediate dtypes in eager mode vs adapter rules; cross-call state (call twice with
different inputs in the same buffers, compare against a fresh process); work outside the timed call
(prepare time vs kernel_fn, prepare-time output-shaped buffers); graph capturability.

Static items (``static_checks``): parse; entry points (one kernel_fn / prepare: M12 overwritten entry points);
hard judge rules from TASK.md "Checked before running" (imports the reference, reads harness frames, patches
timing, edits loaded modules, starts processes, uses the network) -> FAIL; the adapter's forbid_patterns
outside prepare -> RISK; caching heuristics (data_ptr keys, functools caches, module-level containers written in
kernel_fn, ``global``) -> RISK; lossy casts inside prepare -> RISK; kernel inventory (duplicate kernel names =
embedded alternative sources) -> RISK.

Dynamic items (``dynamic_checks``, GPU, in the worker): prepare time vs prepare_budget_s (300 s); prepare leaves
the weights untouched; state census (lossless storage heuristics: float weights kept at their dtype, router FP32 as
FP32 or a BF16 hi+lo pair, no dequantised NVFP4 copy); host syncs in kernel_fn (torch sync debug mode);
aten op/dtype census (TorchDispatchMode; custom kernels are invisible to it); graph capture of every scored shape;
cross-call state (eager: the same buffers refilled with another seed; graph: replay on fresh inputs); output
precision vs SGLang's baseline on the same inputs (BF16-activations rule); router precision at T <= 8 (the
submission's experts vs FP32 and BF16-simulated routers on discriminating tokens); every-key probe (a band of
large values at the window's oldest edge and the middle must change the output as the reference says); active
dispatch per scored shape (torch.profiler kernel names vs the kernels the file defines).
"""
from __future__ import annotations

import ast
import json
import re
import time
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from kdev.adapters.base import ComplyRules
from kdev.contract import ErrorCode, KdevError, Result
from kdev.models import JobSpec

STATUSES = ("PASS", "RISK", "FAIL")


@dataclass
class ComplyItem:
    rule: str
    status: str          # PASS | RISK | FAIL
    detail: str = ""


# ------------------------------------------------------------------------------------------------ static

HARD_RULES: list[tuple[str, str, str]] = [
    ("no_reference_import", r"^\s*(?:from\s+reference\s+import|import\s+reference\b)|import_module\(\s*['\"]reference",
     "imports the reference (judge rejects)"),
    ("no_frame_reads", r"sys\._getframe|inspect\.(?:currentframe|stack|getouterframes|getframeinfo)\b|\.f_back\b"
     r"|\.f_globals\b|\.f_locals\b|gc\.get_referrers", "reads harness frames (judge rejects)"),
    ("no_network", r"^\s*(?:import|from)\s+(?:socket|requests|urllib|http\.client|httpx|aiohttp|ftplib)\b"
     r"|socket\.socket\(|urlopen\(", "uses the network (judge rejects)"),
    ("no_processes", r"\bsubprocess\.|\bPopen\(|os\.system\(|os\.popen\(|os\.fork\(|\bmultiprocessing\b"
     r"|os\.exec\w*\(|os\.spawn\w*\(|^\s*import\s+subprocess", "starts processes (judge rejects)"),
    ("no_timing_patches", r"torch\.cuda\.(?:Event|CUDAGraph|synchronize|graph|graphs)\s*=(?!=)"
     r"|\btime\.(?:perf_counter|time|monotonic|perf_counter_ns)\s*=(?!=)|\.elapsed_time\s*=(?!=)"
     r"|\.replay\s*=(?!=)|\.record\s*=(?!=)", "patches timing (judge rejects)"),
    ("no_module_edits", r"^\s*(?:torch|triton|sys|os|time|builtins|np|numpy)\.(?!backends\b)[\w.]+\s*=(?!=)"
     r"|setattr\(\s*(?:torch|triton|sys|builtins)\b|sys\.modules\[[^\]]+\]\s*=(?!=)",
     "edits loaded modules (judge rejects)"),
]
IMPORT_RULES = {"no_reference_import": ("reference",),
                "no_processes": ("subprocess", "multiprocessing", "pty", "pexpect"),
                "no_network": ("socket", "requests", "urllib", "urllib3", "httpx", "aiohttp", "ftplib", "http")}
SOFT_RULES: list[tuple[str, str, str]] = [
    ("torch_global_config", r"torch\.backends\.[\w.]+\s*=(?!=)|set_float32_matmul_precision",
     "changes torch global numerics (e.g. TF32): check the precision rules"),
    ("env_or_harness_io", r"os\.environ\[[^\]]+\]\s*=(?!=)|os\.environ\.(?:update|setdefault)\(|putenv\("
     r"|open\(\s*['\"]/(?:proc|dev|etc|root|tmp/okc)", "edits the environment / touches harness files"),
    ("secrets", r"(?i)os\.environ(?:\.get)?\s*[\[(]\s*['\"][^'\"]*(?:token|secret|key|okc)", "reads secrets"),
]
CACHE_RULES: list[tuple[str, str, str]] = [
    ("cache_on_data_ptr", r"\[[^\]\n]*\.data_ptr\(\)|\.get\([^)\n]*\.data_ptr\(\)|\bkey\w*\s*=.*\.data_ptr\(\)",
     "dict keyed on data pointers (fresh inputs reuse addresses)"),
    ("functools_cache", r"@(?:functools\.)?(?:lru_)?cache\b", "functools cache on a function"),
    ("cache_on_id", r"\bid\(\s*(?:data|inputs|x|d|inp)\b", "keyed on id(data)"),
]
LOSSY_CAST = (r"\.to\(\s*(?:dtype\s*=\s*)?(?:torch\.)?(?:float8\w*|float16|half|int8|uint8|int4)\b|\.half\(\)"
              r"|quantiz|\.to\(\s*(?:dtype\s*=\s*)?(?:torch\.)?bfloat16\b|\.bfloat16\(\)")
_TRITON_RE = re.compile(r"@triton\.(?:jit|autotune)[^\n]*\n(?:@[^\n]*\n)*def\s+(\w+)")


def _func_ranges(tree: ast.Module) -> list[tuple[str, int, int]]:
    out = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append((node.name, node.lineno, getattr(node, "end_lineno", node.lineno)))
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out.append((f"{node.name}.{sub.name}", sub.lineno, getattr(sub, "end_lineno", sub.lineno)))
    return out


def _enclosing(ranges: list[tuple[str, int, int]], line: int) -> str:
    best = ""
    for name, a, b in ranges:
        if a <= line <= b:
            best = name
    return best


def _is_prepare_fn(name: str) -> bool:
    base = name.split(".")[-1]
    return base == "prepare" or base.startswith(("prepare_", "_prepare", "_prep_", "_repack", "_pack"))


def _hits(source: str, pattern: str, flags: int = re.M) -> list[int]:
    """1-based line numbers of regex hits, ignoring whole-line comments."""
    rx = re.compile(pattern, flags)
    lines = source.splitlines()
    out = []
    for m in rx.finditer(source):
        ln = source.count("\n", 0, m.start()) + 1
        text = lines[ln - 1].lstrip() if ln - 1 < len(lines) else ""
        if text.startswith(("#", "//")):
            continue
        out.append(ln)
    return sorted(set(out))


def _fmt_lines(lines: list[int], ranges: list[tuple[str, int, int]] | None = None, limit: int = 5) -> str:
    parts = []
    for ln in lines[:limit]:
        fn = _enclosing(ranges, ln) if ranges else ""
        parts.append(f"L{ln}" + (f"({fn})" if fn else ""))
    more = f" +{len(lines) - limit} more" if len(lines) > limit else ""
    return ", ".join(parts) + more


_ATTR_MACROS = ("__launch_bounds__", "__maxnreg__", "__cluster_dims__")


def _strip_macro_calls(seg: str) -> str:
    """Remove ``__launch_bounds__(...)``-style attributes (balanced parentheses)."""
    for mac in _ATTR_MACROS:
        while True:
            i = seg.find(mac)
            if i < 0:
                break
            j = seg.find("(", i)
            if j < 0:
                seg = seg[:i] + seg[i + len(mac):]
                continue
            depth, k = 0, j
            while k < len(seg):
                depth += {"(": 1, ")": -1}.get(seg[k], 0)
                if depth == 0:
                    break
                k += 1
            seg = seg[:i] + " " + seg[k + 1:]
    return seg


def cuda_kernel_names(source: str) -> list[str]:
    """Names of ``__global__`` functions (one entry per definition/declaration site)."""
    out = []
    for m in re.finditer(r"__global__", source):
        seg = _strip_macro_calls(source[m.end(): m.end() + 800])
        j = seg.find("(")
        if j < 0:
            continue
        ids = re.findall(r"[A-Za-z_]\w*", seg[:j])
        if ids and not ids[-1].startswith("__"):
            out.append(ids[-1])
    return out


def static_kernels(source: str) -> dict[str, int]:
    """{kernel name: definitions} for CUDA ``__global__`` and Triton ``@triton.jit`` kernels in the source."""
    out: dict[str, int] = {}
    for name in cuda_kernel_names(source) + [m.group(1) for m in _TRITON_RE.finditer(source)]:
        out[name] = out.get(name, 0) + 1
    return out


class CallGraph:
    """Live top-level functions (the last definition of a name wins, like at import) and what is reachable from the
    timed entry point (kernel_fn) vs from prepare. Methods are reached through attribute names or their class."""

    def __init__(self, tree: ast.Module) -> None:
        self.live: dict[str, ast.AST] = {}
        self.classes: dict[str, ast.ClassDef] = {}
        self.methods: dict[str, list[ast.AST]] = {}
        self.all_defs: list[ast.AST] = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.live[node.name] = node
                self.all_defs.append(node)
            elif isinstance(node, ast.ClassDef):
                self.classes[node.name] = node
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        self.methods.setdefault(sub.name, []).append(sub)
                        self.all_defs.append(sub)
            elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Name):
                for t in node.targets:
                    if isinstance(t, ast.Name) and node.value.id in self.live:
                        self.live[t.id] = self.live[node.value.id]

    def _callees(self, node: ast.AST) -> list[ast.AST]:
        out: list[ast.AST] = []
        for n in ast.walk(node):
            name = n.id if isinstance(n, ast.Name) else (n.attr if isinstance(n, ast.Attribute) else None)
            if name is None:
                continue
            if name in self.live:
                out.append(self.live[name])
            if name in self.classes:
                out += [s for s in self.classes[name].body if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))]
            if isinstance(n, ast.Attribute):
                out += self.methods.get(name, [])
        return out

    def reach(self, root: str) -> set[int]:
        start = self.live.get(root)
        if start is None:
            return set()
        seen, todo = {id(start)}, [start]
        while todo:
            for c in self._callees(todo.pop()):
                if id(c) not in seen:
                    seen.add(id(c))
                    todo.append(c)
        return seen

    def owner(self, line: int) -> ast.AST | None:
        best = None
        for d in self.all_defs:
            if d.lineno <= line <= getattr(d, "end_lineno", d.lineno):
                if best is None or d.lineno >= best.lineno:
                    best = d
        return best


def _module_containers(tree: ast.Module) -> set[str]:
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            mutable = isinstance(value, (ast.Dict, ast.List, ast.Set)) or (
                isinstance(value, ast.Call) and getattr(value.func, "id", getattr(value.func, "attr", "")) in
                ("dict", "list", "set", "defaultdict", "OrderedDict", "WeakValueDictionary", "WeakKeyDictionary"))
            if mutable:
                names |= {t.id for t in targets if isinstance(t, ast.Name)}
    return names


def _kernel_fn_writes(tree: ast.Module, containers: set[str], timed: set[int] | None = None) -> list[tuple[int, str]]:
    """(line, what) for module-level container writes / ``global`` inside functions reachable from kernel_fn (all
    non-prepare functions when the call graph is unknown)."""
    out: list[tuple[int, str]] = []
    funcs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    for fn in funcs:
        if timed is not None and id(fn) not in timed:
            continue
        if timed is None and _is_prepare_fn(fn.name):
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.Global):
                out.append((node.lineno, f"global {','.join(node.names)} in {fn.name}"))
            elif isinstance(node, (ast.Assign, ast.AugAssign)):
                tgts = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in tgts:
                    if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and t.value.id in containers:
                        out.append((node.lineno, f"{t.value.id}[...] = in {fn.name}"))
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                v = node.func.value
                if isinstance(v, ast.Name) and v.id in containers and node.func.attr in (
                        "setdefault", "update", "append", "add", "__setitem__", "extend"):
                    out.append((node.lineno, f"{v.id}.{node.func.attr}() in {fn.name}"))
    return sorted(set(out))


def static_checks(source: str, rules: ComplyRules) -> list[ComplyItem]:
    """Static rule items for a submission source (pure; see module docstring)."""
    items: list[ComplyItem] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [ComplyItem("parse", "FAIL", f"syntax error line {e.lineno}: {e.msg}")]
    ranges = _func_ranges(tree)
    cg = CallGraph(tree)
    k_reach, p_reach = cg.reach("kernel_fn"), cg.reach("prepare")

    def timed(ln: int) -> bool:
        if not k_reach:
            return not _is_prepare_fn(_enclosing(ranges, ln))
        o = cg.owner(ln)
        return o is not None and id(o) in k_reach

    def prep_only(ln: int) -> bool:
        o = cg.owner(ln)
        if not p_reach:
            return _is_prepare_fn(_enclosing(ranges, ln))
        return o is not None and id(o) in p_reach and not timed(ln)

    top = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    kf = [n.lineno for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "kernel_fn"]
    kf += [n.lineno for n in tree.body if isinstance(n, ast.Assign)
           and any(isinstance(t, ast.Name) and t.id == "kernel_fn" for t in n.targets)]
    pf = [n.lineno for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "prepare"]
    if not kf:
        items.append(ComplyItem("entry_points", "FAIL", "no top-level kernel_fn"))
    elif len(kf) > 1 or len(pf) > 1:
        items.append(ComplyItem("entry_points", "RISK",
                                f"kernel_fn defined {len(kf)}x (lines {sorted(kf)}), prepare {len(pf)}x: the LAST "
                                "definition wins; confirm the active path (M12, `kdev dispatch`)"))
    else:
        items.append(ComplyItem("entry_points", "PASS", f"kernel_fn L{kf[0]}" + (f", prepare L{pf[0]}" if pf else "")))

    imports: dict[str, list[int]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for al in node.names:
                imports.setdefault(al.name.split(".")[0], []).append(node.lineno)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            imports.setdefault(node.module.split(".")[0], []).append(node.lineno)
    for name, rx, why in HARD_RULES:
        h = _hits(source, rx)
        h = sorted(set(h) | {ln for mod in IMPORT_RULES.get(name, ()) for ln in imports.get(mod, [])})
        items.append(ComplyItem(name, "FAIL" if h else "PASS", f"{why}: {_fmt_lines(h, ranges)}" if h else ""))
    for name, rx, why in SOFT_RULES:
        h = _hits(source, rx)
        if h:
            items.append(ComplyItem(name, "RISK", f"{why}: {_fmt_lines(h, ranges)}"))

    bad: list[tuple[str, list[int]]] = []
    for pat in rules.forbid_patterns:
        try:
            h = [ln for ln in _hits(source, pat) if timed(ln)]
        except re.error as e:
            items.append(ComplyItem("forbid_patterns", "RISK", f"bad pattern {pat!r}: {e}"))
            continue
        if h:
            bad.append((pat, h))
    if bad:
        items.append(ComplyItem("forbid_patterns", "RISK", "; ".join(f"{p} at {_fmt_lines(h, ranges, 3)}"
                                                                     for p, h in bad[:4])
                                + " (reachable from kernel_fn: host syncs break graph capture)"))
    else:
        items.append(ComplyItem("forbid_patterns", "PASS",
                                f"{len(rules.forbid_patterns)} patterns, none in code reachable from kernel_fn"))

    if rules.no_cross_call_state:
        cache_hits = []
        for name, rx, why in CACHE_RULES:
            h = [ln for ln in _hits(source, rx) if timed(ln)]
            if h:
                cache_hits.append(f"{why} at {_fmt_lines(h, ranges, 3)}")
        writes = _kernel_fn_writes(tree, _module_containers(tree), k_reach or None)
        if writes:
            cache_hits.append("module state written per call: " + ", ".join(f"L{ln} {w}" for ln, w in writes[:4]))
        items.append(ComplyItem("no_cross_call_caching", "RISK" if cache_hits else "PASS",
                                "; ".join(cache_hits) if cache_hits else "no data-keyed caches found"))

    if rules.lossless_weights:
        h = [ln for ln in _hits(source, LOSSY_CAST) if prep_only(ln)]
        items.append(ComplyItem("lossless_weights_static", "RISK" if h else "PASS",
                                f"precision-lowering casts in prepare at {_fmt_lines(h, ranges)}: weights must keep "
                                "their precision (router FP32 may become a BF16 hi+lo pair)" if h
                                else "no precision-lowering casts in prepare"))

    ks = static_kernels(source)
    dup = {k: n for k, n in ks.items() if n > 1}
    n_inline = len(_hits(source, r"\bload_inline\("))
    if dup:
        items.append(ComplyItem("kernel_inventory", "RISK",
                                f"{len(ks)} kernels, defined more than once: {', '.join(list(dup)[:6])} (embedded "
                                "alternative sources: confirm which one runs)"))
    else:
        items.append(ComplyItem("kernel_inventory", "PASS",
                                f"{len(ks)} kernels, {n_inline} load_inline call(s), {len(top)} top-level functions"))
    return items


# ----------------------------------------------------------------------------------------------- results

def counts(items: list[ComplyItem]) -> dict[str, int]:
    c = {s: 0 for s in STATUSES}
    for it in items:
        c[it.status] = c.get(it.status, 0) + 1
    return c


def verdict_of(items: list[ComplyItem]) -> str:
    c = counts(items)
    return "fail" if c["FAIL"] else ("risk" if c["RISK"] else "pass")


def render_lines(items: list[ComplyItem], problem: str, file: str, mode: str) -> list[str]:
    c = counts(items)
    v = verdict_of(items)
    lines = [f"{v.upper()}  comply  {problem}  {mode}  PASS={c['PASS']} RISK={c['RISK']} FAIL={c['FAIL']}  {file}"]
    order = sorted(items, key=lambda it: -STATUSES.index(it.status) if it.status in STATUSES else 0)
    for it in order[:18]:
        lines.append(f"{it.status:4s}  {it.rule:26s} {it.detail[:150]}")
    if len(order) > 18:
        lines.append(f"... {len(order) - 18} more items in comply.json")
    return lines


def make_result(items: list[ComplyItem], *, problem: str, file: str, mode: str, run_dir: Any = None,
                job_id: str | None = None, extra: dict[str, Any] | None = None) -> Result:
    from kdev.checks.check import write_json

    v = verdict_of(items)
    summary = {"problem": problem, "file": file, "mode": mode, "counts": counts(items),
               "items": [asdict(i) for i in items]}
    if extra:
        summary.update(extra)
    res = Result(tool="comply", status="fail" if v == "fail" else "ok", verdict=v, job_id=job_id, summary=summary,
                 next=([f"kdev comply {file}"] if mode == "static" else [])
                 + (["RISK items need a human/reviewer decision before promotion"] if v == "risk" else []))
    p = write_json(run_dir, "comply.json", summary)
    if p:
        res.files["comply"] = p
    res.lines = render_lines(items, problem, file, mode)
    return res


def run_static(adapter: Any, path: str) -> list[ComplyItem]:
    try:
        source = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        raise KdevError(ErrorCode.NOT_FOUND, f"cannot read {path}: {e}") from e
    items = static_checks(source, adapter.rules())
    return items


def unverified_precision_items(rules: ComplyRules) -> list[ComplyItem]:
    return [ComplyItem(f"precision:{op}", "RISK", f"{req} -- not verified statically (run `kdev comply` on the GPU)")
            for op, req in rules.precision.items()]


# ---------------------------------------------------------------------------------------------- dynamic

def _walk_tensors(obj: Any, path: str = "state", depth: int = 0, seen: set[int] | None = None) -> list[tuple[str, Any]]:
    import torch
    seen = seen if seen is not None else set()
    if depth > 5 or id(obj) in seen:
        return []
    seen.add(id(obj))
    if isinstance(obj, torch.Tensor):
        return [(path, obj)]
    out: list[tuple[str, Any]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out += _walk_tensors(v, f"{path}.{k}", depth + 1, seen)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out += _walk_tensors(v, f"{path}[{i}]", depth + 1, seen)
    elif hasattr(obj, "__dict__") and not isinstance(obj, (type, types.ModuleType)) and not callable(obj):
        for k, v in vars(obj).items():
            out += _walk_tensors(v, f"{path}.{k}", depth + 1, seen)
    elif isinstance(obj, torch.nn.Module):
        for k, v in obj.state_dict().items():
            out.append((f"{path}.{k}", v))
    return out


def _checksums(weights: Any) -> dict[str, float]:
    import torch
    out = {}
    for name, t in _walk_tensors(weights, "w"):
        try:
            v = t.view(torch.uint8) if t.element_size() == 1 else t
            out[name] = float(v.double().sum()) + float((v.double() * 0.37).abs().sum())
        except Exception:  # noqa: BLE001
            pass
    return out


def lossless_census(weights: Any, state: Any) -> list[ComplyItem]:
    """Heuristics (RISK only): float weights kept at their own dtype (or the same storage); FP32 router as FP32 or a
    BF16 hi+lo pair; no dequantised (float) copy of NVFP4/packed weights."""
    import torch
    w = dict(_walk_tensors(weights, "w"))
    st = _walk_tensors(state, "state")
    if not w:
        return []
    ptrs = {t.data_ptr() for _, t in st if t.numel()}
    risks, notes = [], []
    for name, t in w.items():
        if not t.is_floating_point() or t.element_size() == 1 or t.numel() < 4096:
            continue
        if t.data_ptr() in ptrs:
            continue
        n = t.numel()
        near = [(p, s) for p, s in st if 0.95 * n <= s.numel() <= 1.05 * n]
        same = [p for p, s in near if s.dtype == t.dtype]
        if same:
            continue
        if t.dtype == torch.float32:
            bf = [p for p, s in near if s.dtype == torch.bfloat16]
            if len(bf) >= 2:
                notes.append(f"{name.split('.')[-1]}: bf16 hi+lo pair")
                continue
            if len(bf) == 1:
                risks.append(f"{name.split('.')[-1]} fp32 -> one bf16 tensor ({bf[0]}): lossy unless a lo part exists")
                continue
        lower = [(p, s.dtype) for p, s in st if s.numel() in (n, n // 2) and s.element_size() < t.element_size()]
        if lower:
            risks.append(f"{name.split('.')[-1]} {str(t.dtype)[6:]} -> {str(lower[0][1])[6:]} ({lower[0][0]})")
        else:
            notes.append(f"{name.split('.')[-1]}: repacked (no same-size tensor)")
    for name, t in w.items():
        if t.dtype != torch.uint8 or t.dim() < 3 or name + "_sf" not in w and not any(k.startswith(name + "_s") for k in w):
            continue
        per_expert = (t.numel() // t.shape[0]) * 2
        float_sizes = {x.numel() for x in w.values() if x.is_floating_point() and x.element_size() >= 2}
        dq = [p for p, s in st if s.is_floating_point() and s.element_size() >= 2 and s.numel() >= 4 * per_expert
              and s.numel() % per_expert == 0 and s.numel() not in float_sizes]
        if dq:
            risks.append(f"{name.split('.')[-1]}: float tensor(s) sized like dequantised experts: {', '.join(dq[:3])}")
    bytes_by = {}
    for _, s in st:
        k = str(s.dtype)[6:]
        bytes_by[k] = bytes_by.get(k, 0) + s.numel() * s.element_size()
    det = "state " + ", ".join(f"{k} {v / 2**20:.0f} MB" for k, v in sorted(bytes_by.items(), key=lambda kv: -kv[1])[:5])
    if risks:
        return [ComplyItem("lossless_weights", "RISK", "; ".join(risks[:4]) + f" | {det}")]
    return [ComplyItem("lossless_weights", "PASS", (("; ".join(notes[:3]) + " | ") if notes else "") + det)]


def _sync_warnings(fn: Any) -> list[str]:
    import warnings

    import torch
    msgs: list[str] = []
    prev = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("warn")
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            fn()
        msgs = sorted({str(r.message).splitlines()[0][:120] for r in rec if "synchroniz" in str(r.message)})
    finally:
        torch.cuda.set_sync_debug_mode(prev)
    return msgs


MATMUL_OPS = {"mm", "addmm", "bmm", "baddbmm", "matmul", "linear", "_scaled_mm", "einsum", "addmv", "mv", "dot"}


def op_census(fn: Any) -> dict[tuple[str, tuple[str, ...]], int]:
    """{(aten op, input dtypes): count} for one eager call (custom CUDA kernels do not show up)."""
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode

    seen: dict[tuple[str, tuple[str, ...]], int] = {}

    class Census(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            dts = []
            for a in list(args) + list(kwargs.values()):
                for x in (a if isinstance(a, (list, tuple)) else [a]):
                    if isinstance(x, torch.Tensor):
                        dts.append(str(x.dtype)[6:])
            key = (func.overloadpacket.__name__, tuple(dts))
            seen[key] = seen.get(key, 0) + 1
            return func(*args, **kwargs)

    with Census():
        fn()
    return seen


def _kernel_names(fn: Any) -> list[str]:
    import torch
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    names = []
    for e in prof.events():
        dt = getattr(getattr(e, "device_type", None), "name", "")
        if dt == "CUDA" and not re.match(r"(?i)memcpy|memset", e.name):
            names.append(e.name)
    return names


def _base_name(kernel: str) -> str:
    k = re.sub(r"^void\s+", "", kernel)
    k = re.split(r"[<(]", k, maxsplit=1)[0]
    return k.split("::")[-1].strip() or kernel.strip()[:40]


def dispatch_items(per_shape: dict[str, list[str]], defined: dict[str, int]) -> list[ComplyItem]:
    """Pure: active dispatch per scored shape vs kernels defined in the file."""
    items = []
    own_any: set[str] = set()
    lib_only = []
    desc = []
    for sid, names in per_shape.items():
        bases = [_base_name(n) for n in names]
        own = sorted({d for d in defined if any(d == b or d in n for b, n in zip(bases, names))})
        own_any |= set(own)
        if defined and not own:
            lib_only.append(sid)
        uniq = list(dict.fromkeys(bases))
        desc.append(f"{sid}: {len(names)} launches [{', '.join(uniq[:3])}{'...' if len(uniq) > 3 else ''}]")
    items.append(ComplyItem("active_dispatch", "PASS", " | ".join(desc[:6]) + (" ..." if len(desc) > 6 else "")))
    if lib_only:
        items.append(ComplyItem("active_dispatch_own_kernels", "RISK",
                                f"shapes launching none of the file's kernels: {', '.join(lib_only[:6])} (fallback path?)"))
    dead = sorted(set(defined) - own_any)
    if defined and dead:
        items.append(ComplyItem("dead_kernels", "RISK",
                                f"{len(dead)}/{len(defined)} defined kernels never launch on a scored shape: "
                                f"{', '.join(dead[:8])} (do not optimise these: LESSONS 7)"))
    return items


def routing_verdict(stats: dict[str, int], max_tokens: int) -> ComplyItem:
    """Pure: stats = {tokens, discriminating, match_fp32, match_lowp, mismatch_slots}."""
    d = stats.get("discriminating", 0)
    detail = (f"T<={max_tokens}: {stats.get('tokens', 0)} tokens, {d} discriminate fp32 vs bf16 routers; submission "
              f"matches fp32 on {stats.get('match_fp32', 0)}, bf16 on {stats.get('match_lowp', 0)}; "
              f"{stats.get('mismatch_slots', 0)} slots differ from the fp32 top-8")
    if d and stats.get("match_lowp", 0) > stats.get("match_fp32", 0):
        return ComplyItem("precision:router", "FAIL", "router looks lower than FP32 at small T -- " + detail)
    if d and stats.get("match_lowp", 0):
        return ComplyItem("precision:router", "RISK", detail)
    if not d:
        return ComplyItem("precision:router", "RISK" if stats.get("mismatch_slots") else "PASS",
                          "inconclusive (no discriminating tokens) -- " + detail)
    return ComplyItem("precision:router", "PASS", detail)


def precision_vs_baseline(rows: list[tuple[str, float | None, float | None, float | None]]) -> ComplyItem:
    """Pure: rows of (shape, sub_err, base_err, tol). RISK when the submission's error > 3x SGLang's and > 25% tol."""
    bad, desc = [], []
    for sid, se, be, tol in rows:
        if se is None or be is None:
            continue
        desc.append(f"{sid} {se:.2e} vs {be:.2e}")
        if se > 3 * max(be, 1e-12) and (tol is None or se > 0.25 * tol):
            bad.append(sid)
    if not desc:
        return ComplyItem("precision:vs_baseline", "RISK", "no comparable errors (baseline unavailable?)")
    st = "RISK" if bad else "PASS"
    return ComplyItem("precision:vs_baseline", st, ("error > 3x SGLang's on " + ", ".join(bad) + " (activations "
                                                    "below BF16?) | " if bad else "") + "; ".join(desc[:4]))


def _routing_probe(adapter: Any, module: Any, state: Any, cfg: dict[str, Any], seeds: list[int]) -> ComplyItem:
    import torch

    from kdev.checks.check import clone_output
    R = adapter.reference()
    W = adapter.weights()
    if not (hasattr(R, "ref_front") and hasattr(R, "route")):
        return ComplyItem("precision:router", "RISK", "reference lacks ref_front/route: probe not applicable")
    max_t = int(cfg.get("fp32_max_tokens", 8))
    idx = int(cfg.get("ids_output", 2))
    top_k = int(getattr(R, "TOP_K", 8))
    w_lo = dict(W)
    w_lo["router"] = W["router"].to(torch.bfloat16).float()
    stats = {"tokens": 0, "discriminating": 0, "match_fp32": 0, "match_lowp": 0, "mismatch_slots": 0}
    for sid in cfg.get("probe_shapes", []):
        shape = adapter.find_shape(sid)
        if shape is None:
            continue
        for seed in seeds:
            data = adapter.make_inputs(shape, seed)
            out = clone_output(adapter.call(module, state, data))
            torch.cuda.synchronize()
            ids = out[idx].long()
            if ids.shape[0] > max_t:
                break
            front = R.ref_front(data, W)
            h = front["h"].float()
            ref = front["biased"].topk(top_k, -1).indices
            low_a = R.route(h.to(torch.bfloat16).float(), W)[1].topk(top_k, -1).indices
            low_b = R.route(h, w_lo)[1].topk(top_k, -1).indices

            def same(a: Any, b: Any) -> Any:
                return (a.sort(-1).values == b.sort(-1).values).all(-1)
            sub_f = same(ids, ref)
            for low in (low_a, low_b):
                disc = ~same(low, ref)
                stats["discriminating"] += int(disc.sum())
                stats["match_fp32"] += int((disc & sub_f).sum())
                stats["match_lowp"] += int((disc & same(ids, low)).sum())
            stats["tokens"] += int(ids.shape[0])
            stats["mismatch_slots"] += int((~(ids[:, :, None] == ref[:, None, :]).any(-1)).sum())
            del data, out, front
    return routing_verdict(stats, max_t)


def _every_key_probe(adapter: Any, module: Any, state: Any, seed: int) -> ComplyItem:
    """Plant a band of large values at the window's oldest edge and in the middle of every request's attended range;
    the output must still pass the reference check (a kernel that skips keys/values diverges)."""
    import torch

    from kdev.checks.check import clone_output
    cands = [s for s in adapter.shapes() if s.case.get("batch", 1) == 1 and s.case.get("q_len", 1) == 1]
    cands = cands or adapter.shapes()[:2]
    msgs, ok_all = [], True
    for shape in cands[:2]:
        data = adapter.make_inputs(shape, seed)
        need = ("v_cache", "page_table", "seq_lens", "config")
        if not all(k in data for k in need):
            return ComplyItem("every_key", "RISK", "inputs lack v_cache/page_table/seq_lens: probe not applicable")
        cfg = data["config"]
        page, win = int(cfg["page_size"]), int(cfg.get("window", 0) or 0)
        v = data["v_cache"]
        flat = v.view(-1, v.shape[-2], v.shape[-1])
        scale = float(data["v_scale"]) if "v_scale" in data else 1.0
        pat = torch.tensor([1.0, -1.0], device=v.device).repeat(v.shape[-1] // 2) * min(64.0, 224.0 * scale) / scale
        for b in range(int(cfg.get("batch", 1))):
            seq = int(data["seq_lens"][b])
            start = max(0, seq - win + 1) if win else 0
            span = seq - start
            band = max(1, min(128, span // 8))
            for lo in (start, start + span // 2):
                pos = torch.arange(lo, min(lo + band, seq), device=v.device)
                rows = data["page_table"][b].long()[pos // page] * page + pos % page
                flat[rows] = pat.to(v.dtype)
        out = clone_output(adapter.call(module, state, data))
        torch.cuda.synchronize()
        r = adapter.check(data, out)
        ok_all &= r.ok
        msgs.append(f"{shape.shape_id}: {'ok' if r.ok else 'FAIL'} {r.message[:80]}")
        del data, out
    return ComplyItem("every_key", "PASS" if ok_all else "FAIL", " | ".join(msgs))


def _cross_call(adapter: Any, module: Any, state: Any, shape: Any, sa: int, sb: int) -> list[ComplyItem]:
    import torch

    from kdev.checks.check import capture_graph, clone_output, compare_outputs, copy_into, tensors_of
    items = []
    X, Y = adapter.make_inputs(shape, sa), adapter.make_inputs(shape, sb)
    out_y = clone_output(adapter.call(module, state, Y))
    torch.cuda.synchronize()
    out_x = adapter.call(module, state, X)
    torch.cuda.synchronize()
    rx = adapter.check(X, clone_output(out_x))
    st_ptrs = [(t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()) for _, t in _walk_tensors(state)]
    aliased = [t for t in tensors_of(out_x) if any(a <= t.data_ptr() < b for a, b in st_ptrs)]
    del Y
    Yc = adapter.make_inputs(shape, sb)
    if not copy_into(X, Yc):
        return [ComplyItem("cross_call_state", "RISK", "a second seed's inputs differ in structure: not tested")]
    del Yc
    out_xy = clone_output(adapter.call(module, state, X))
    torch.cuda.synchronize()
    rxy = adapter.check(X, out_xy)
    cmp = compare_outputs(out_y, out_xy)
    note = (f"; returns {len(aliased)} buffer(s) owned by the prepared state (fine when every call rewrites them)"
            if aliased else "")
    if not rx.ok or not rxy.ok:
        items.append(ComplyItem("cross_call_state", "FAIL",
                                f"{shape.shape_id}: wrong after a previous call (second input {'ok' if rx.ok else 'FAIL'}, "
                                f"same buffers refilled {'ok' if rxy.ok else 'FAIL'}): state cached across calls" + note))
    elif (cmp["rel_l2"] or 0.0) > 0.1 * (rxy.tol or 1e-2):
        items.append(ComplyItem("cross_call_state", "RISK",
                                f"{shape.shape_id}: refilled buffers differ from a fresh call by rel L2 {cmp['rel_l2']:.2e}"
                                f" (> 10% of tol {rxy.tol}; int diffs {cmp['int_diffs']})" + note))
    else:
        items.append(ComplyItem("cross_call_state", "PASS",
                                f"{shape.shape_id}: refilled buffers match a fresh call (rel L2 {cmp['rel_l2'] or 0:.1e}, "
                                f"bitwise {cmp['bitwise']}, int diffs {cmp['int_diffs']})" + note))
    # graph: capture on X, refill with a third seed, replay
    g, gout = capture_graph(adapter, module, state, X)
    Z = adapter.make_inputs(shape, sa ^ 0x2545F491)
    if copy_into(X, Z):
        del Z
        g.replay()
        torch.cuda.synchronize()
        r = adapter.check(X, clone_output(gout))
        items.append(ComplyItem("graph_fresh_inputs", "PASS" if r.ok else "FAIL",
                                f"{shape.shape_id}: graph replay on fresh inputs {'passes' if r.ok else 'FAILS'}: "
                                f"{r.message[:100]}"))
    del g, gout, X
    return items


def dynamic_checks(adapter: Any, path: str, source: str, rules: ComplyRules, params: dict[str, Any],
                   run_dir: Any = None) -> tuple[list[ComplyItem], dict[str, Any]]:
    """GPU part of comply (worker). Each probe is isolated: a crashing probe becomes a RISK/FAIL item."""
    import gc

    import torch

    from kdev.checks.check import capture_graph, clone_output, pick_seeds
    items: list[ComplyItem] = []
    info: dict[str, Any] = {}
    ex = dict(params.get("extra") or {})
    seeds = pick_seeds(int(ex.get("seeds", 2)) if ex.get("seeds") else 2, params.get("seed_list"))
    sa, sb = seeds[0], (seeds[1] if len(seeds) > 1 else seeds[0] + 1)
    module = adapter.load(path, "A")
    weights = adapter.weights() if hasattr(adapter, "weights") else None
    before = _checksums(weights) if weights is not None else {}
    state = adapter.prepare(module)
    prep = float(getattr(adapter, "last_prepare_s", 0.0) or 0.0)
    budget = float(adapter.protocol.prepare_budget_s)
    info["prepare_s"] = round(prep, 2)
    items.append(ComplyItem("prepare_budget", "FAIL" if prep > budget else ("RISK" if prep > 0.8 * budget else "PASS"),
                            f"prepare {prep:.1f}s (warm cache) vs {budget:.0f}s; cold compile: `kdev check --cold`"))
    if weights is not None:
        after = _checksums(weights)
        changed = [k for k in before if before[k] != after.get(k)]
        items.append(ComplyItem("weights_untouched", "FAIL" if changed else "PASS",
                                f"prepare modified weight tensors in place: {', '.join(changed[:4])}" if changed
                                else f"{len(before)} weight tensors unchanged by prepare"))
        if rules.lossless_weights:
            try:
                items += lossless_census(weights, state)
            except Exception as e:  # noqa: BLE001
                items.append(ComplyItem("lossless_weights", "RISK", f"census failed: {e}"))

    def guard(name: str, fn: Any) -> None:
        try:
            got = fn()
            if got:
                items.extend(got if isinstance(got, list) else [got])
        except Exception as e:  # noqa: BLE001
            items.append(ComplyItem(name, "RISK", f"probe crashed: {type(e).__name__}: {str(e)[:160]}"))
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    shapes = adapter.shapes()
    first = shapes[0]

    def host_sync() -> ComplyItem:
        data = adapter.make_inputs(first, sa)
        t0 = time.monotonic()
        msgs = _sync_warnings(lambda: adapter.call(module, state, data))
        torch.cuda.synchronize()
        info["first_call_s"] = round(time.monotonic() - t0, 3)
        return ComplyItem("no_host_sync", "RISK" if msgs else "PASS",
                          ("host syncs in kernel_fn: " + "; ".join(msgs[:3])) if msgs
                          else f"no syncs flagged by torch sync-debug ({first.shape_id})")
    guard("no_host_sync", host_sync)

    def census() -> ComplyItem:
        data = adapter.make_inputs(first, sa)
        seen = op_census(lambda: adapter.call(module, state, data))
        info["aten_ops"] = {f"{k[0]}({','.join(k[1])})": n for k, n in sorted(seen.items(), key=lambda kv: -kv[1])[:20]}
        fp8_ok = any("fp8" in v.lower() or "float8" in v.lower() for v in rules.precision.values())
        low = [f"{op}({','.join(d)})" for (op, d), _ in seen.items()
               if op in MATMUL_OPS and any(x.startswith(("float8", "int8")) for x in d)]
        if low and not fp8_ok:
            return ComplyItem("precision:aten_ops", "RISK", "matmuls on 8-bit inputs: " + ", ".join(low[:4]))
        n = sum(seen.values())
        return ComplyItem("precision:aten_ops", "PASS",
                          f"{n} aten ops in eager kernel_fn ({first.shape_id}); custom kernels not visible"
                          + (f"; 8-bit matmuls allowed here: {', '.join(low[:2])}" if low else ""))
    guard("precision:aten_ops", census)

    if rules.graph_capturable:
        def graphs() -> ComplyItem:
            bad = []
            for s in shapes:
                data = adapter.make_inputs(s, sa)
                try:
                    g, _ = capture_graph(adapter, module, state, data)
                    g.replay()
                    torch.cuda.synchronize()
                    del g
                except Exception as e:  # noqa: BLE001
                    bad.append(f"{s.shape_id}: {type(e).__name__}: {str(e)[:80]}")
                del data
                torch.cuda.empty_cache()
            return ComplyItem("graph_capturable", "FAIL" if bad else "PASS",
                              "; ".join(bad[:3]) if bad else f"{len(shapes)} scored shapes capture and replay")
        guard("graph_capturable", graphs)

    if rules.no_cross_call_state:
        probe = [first] + ([shapes[-1]] if len(shapes) > 1 else [])
        for s in probe:
            guard("cross_call_state", lambda s=s: _cross_call(adapter, module, state, s, sa, sb))

    extra = adapter.comply_extra() if hasattr(adapter, "comply_extra") else {}
    if extra.get("routing"):
        rseeds = pick_seeds(int(extra["routing"].get("seeds", 8)))
        guard("precision:router", lambda: _routing_probe(adapter, module, state, extra["routing"], rseeds))
    if extra.get("every_key", {}).get("enabled"):
        guard("every_key", lambda: _every_key_probe(adapter, module, state, sa))

    if not ex.get("no_baseline") and adapter.baseline_path():
        def vs_base() -> ComplyItem:
            base = adapter.load(adapter.baseline_path(), "base")
            bstate = adapter.prepare(base)
            rows = []
            for s in ([first, shapes[-1]] if len(shapes) > 1 else [first]):
                data = adapter.make_inputs(s, sb)
                rs = adapter.check(data, clone_output(adapter.call(module, state, data)))
                data = adapter.make_inputs(s, sb)
                rb = adapter.check(data, clone_output(adapter.call(base, bstate, data)))
                rows.append((s.shape_id, rs.max_rel_err, rb.max_rel_err, rs.tol))
                del data
            del base, bstate
            return precision_vs_baseline(rows)
        guard("precision:vs_baseline", vs_base)

    if not ex.get("no_dispatch"):
        def disp() -> list[ComplyItem]:
            per = {}
            for s in shapes:
                data = adapter.make_inputs(s, sa)
                per[s.shape_id] = _kernel_names(lambda: adapter.call(module, state, data))
                del data
            info["dispatch"] = {k: list(dict.fromkeys(_base_name(n) for n in v))[:12] for k, v in per.items()}
            return dispatch_items(per, static_kernels(source))
        guard("active_dispatch", disp)

    covered = {it.rule for it in items}
    for op, req in rules.precision.items():
        if f"precision:{op}" not in covered:
            items.append(ComplyItem(f"precision:{op}", "RISK", f"{req} -- no automatic probe; review the code path"))
    return items, info


def run_comply(spec: JobSpec, run_dir: Any) -> Result:
    """Worker entry for kind='comply' (static part + dynamic part unless params.extra.static_only). Never raises."""
    try:
        if not spec.files:
            raise KdevError(ErrorCode.INVALID_REQUEST, "comply needs a submission file")
        from kdev.adapters.registry import get_adapter
        path = spec.files[0].path
        adapter = get_adapter(problem=spec.problem, file=path, adapter=spec.adapter)
        items = run_static(adapter, path)
        ex = dict(spec.params.get("extra") or {})
        info: dict[str, Any] = {}
        mode = "static"
        if ex.get("static_only"):
            items += unverified_precision_items(adapter.rules())
        else:
            mode = "static+dynamic"
            source = Path(path).read_text(encoding="utf-8", errors="replace")
            dyn, info = dynamic_checks(adapter, path, source, adapter.rules(), dict(spec.params), run_dir)
            items += dyn
        return make_result(items, problem=adapter.problem, file=path, mode=mode, run_dir=run_dir,
                           job_id=spec.job_id or None, extra={"probe": info} if info else None)
    except KdevError as e:
        return Result.from_error("comply", e)


def to_json(items: list[ComplyItem]) -> str:
    return json.dumps([asdict(i) for i in items])
