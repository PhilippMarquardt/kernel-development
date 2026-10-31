"""Generic OKC adapter (the five muse_* problems). Owner: P7.

Problem dir layout (e.g. okc/muse_moe_w4a16/muse_moe_w4a16/): TASK.md, reference.py (generate_input,
[generate_weights + WEIGHTS_SEED], check_implementation, [collect], [case_fields], [speed_of_light_ns]),
checks.py, baseline.py, submission.py. Per-problem config: kdev/adapters/okc_problems/<problem>.toml
(scored shapes with ids/weights/loads from TASK.md 'Shapes' table, tests, extra shapes, input_variance,
kernel_fn arity, protocol constants, comply rules). Differences handled here:
- attention: ``kernel_fn(data)``, no prepare/weights; check via ``R.collect(data, out)`` then
  ``R.check_implementation(data, raw)``.
- moe/bf16: ``state = prepare(R.generate_weights(R.WEIGHTS_SEED))``; ``kernel_fn(data, state)``;
  ``R.check_implementation(data, out, weights)``.
- decode/prefill: as moe, plus ``R.case_fields(case, weights)`` (local_start) merged into every input and the
  check through ``R.collect`` (outputs + the new cache entries).
- shapes from a judge JSON: records[event == 'benchmark'].case minus load/repeats/weight/router_* (calib.py).
reference.py is imported from the problem dir with that dir first on sys.path (never from an extracted
archive's own directory), under a private module name so a submission's ``import reference`` is not satisfied.

Without a TOML (a new problem) the shapes and tests are parsed from TASK.md (:func:`parse_task_md`).
Everything except make_inputs/call/check/prepare/load/weights is pure and GPU-free.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from kdev.adapters.base import (CheckResult, ComplyRules, ExtSpec, Shape, TimingProtocol,
                                load_weighted_geomean)
from kdev.contract import ErrorCode, KdevError

try:  # Python >= 3.11
    import tomllib as _toml
except ModuleNotFoundError:  # pragma: no cover - 3.10
    import tomli as _toml  # type: ignore[no-redef]

PROBLEMS_DIR = Path(__file__).parent / "okc_problems"
CONFIG_SCHEMA = "kdev.okc_problem/1"
# judge-JSON case keys that are not generate_input arguments (calib.py's drop list, plus the seed)
JUDGE_DROP = ("load", "repeats", "weight", "router_load", "router_distinct", "seed")
DEFAULT_ROOTS = ("/root/okc", "~/okc")


# ----------------------------------------------------------------------------------------------- config

def known_problems() -> list[str]:
    """Problem names that ship a TOML."""
    return sorted(p.stem for p in PROBLEMS_DIR.glob("*.toml"))


def load_problem_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "rb") as f:
        cfg = _toml.load(f)
    if cfg.get("schema", CONFIG_SCHEMA) != CONFIG_SCHEMA:
        raise KdevError(ErrorCode.INVALID_REQUEST, f"{path}: schema {cfg.get('schema')!r} != {CONFIG_SCHEMA}")
    return cfg


def is_problem_dir(path: str | os.PathLike[str]) -> bool:
    p = Path(path)
    return p.is_dir() and (p / "reference.py").is_file() and (p / "TASK.md").is_file()


def find_problem_dir(problem: str, search_dirs: list[str] | None = None,
                     roots: list[str] | None = None) -> str | None:
    """Locate ``<root>/<p>/<p>`` (okc init layout) or ``<root>/<p>``. Search order: $KDEV_PROBLEMS_ROOT, the given
    roots (config.problems_root), explicit search_dirs (TOML), /root/okc, ~/okc."""
    cands: list[Path] = []
    for r in [os.environ.get("KDEV_PROBLEMS_ROOT", "")] + list(roots or []):
        if r:
            base = Path(os.path.expanduser(r))
            cands += [base / problem / problem, base / problem]
    cands += [Path(os.path.expanduser(d)) for d in (search_dirs or [])]
    for r in DEFAULT_ROOTS:
        base = Path(os.path.expanduser(r))
        cands += [base / problem / problem, base / problem]
    for c in cands:
        if is_problem_dir(c):
            return str(c)
    return None


def _num(s: str) -> Any:
    s = s.strip().replace(",", "")
    for conv in (int, float):
        try:
            return conv(s)
        except ValueError:
            pass
    return s


def parse_case_text(text: str) -> dict[str, Any]:
    """'batch 1, ctx 80000, heads tp8, layer sliding' -> {'batch': 1, 'ctx': 80000, ...} (TASK.md notation)."""
    case: dict[str, Any] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        key, _, val = part.partition(" ")
        case[key.strip()] = _num(val) if val else True
    return case


def _slug(load: str) -> str:
    words = load.split()
    if len(words) == 2 and words[1].isdigit():
        return words[0][0] + words[1]
    return "".join(w[0] for w in words) or "load"


def parse_task_md(text: str) -> tuple[list[Shape], list[Shape]]:
    """(scored shapes, tests) from a TASK.md: the '| load | shape | weight | floor |' table and the fenced list
    after 'must pass these N tests'. Shape ids: '<load slug>/<layer or index>' (made unique)."""
    shapes: list[Shape] = []
    used: dict[str, int] = {}
    in_table = False
    for line in text.splitlines():
        s = line.strip()
        cells = [c.strip() for c in s.strip("|").split("|")] if s.startswith("|") else []
        if cells[:2] == ["load", "shape"]:
            in_table = True
            continue
        if not in_table:
            continue
        if not cells:
            in_table = False
            continue
        if set(cells[0]) <= {"-", ":"}:
            continue
        load, case = cells[0], parse_case_text(cells[1])
        weight = float(_num(cells[2])) if len(cells) > 2 else 1.0
        tail = str(case.get("layer", "")) or str(len(shapes))
        sid = f"{_slug(load)}/{tail}"
        if sid in used:
            used[sid] += 1
            sid = f"{sid}-{used[sid]}"
        else:
            used[sid] = 0
        shapes.append(Shape(sid, case, weight=weight, load=load))
    tests: list[Shape] = []
    m = re.search(r"must pass these \d+ tests[^\n]*\n+```[^\n]*\n(.*?)```", text, re.S)
    if m:
        for i, line in enumerate(x for x in m.group(1).splitlines() if x.strip()):
            tests.append(Shape(f"t{i}", parse_case_text(line), scored=False))
    return shapes, tests


def _shape_from_cfg(d: dict[str, Any], scored: bool) -> Shape:
    return Shape(str(d["id"]), dict(d.get("case", {})), weight=float(d.get("weight", 1.0)),
                 load=str(d.get("load", "")), scored=scored, seed=d.get("seed"))


def case_key(case: dict[str, Any]) -> tuple:
    """Order-insensitive identity of a case (judge-only keys dropped)."""
    return tuple(sorted((k, str(v)) for k, v in case.items() if k not in JUDGE_DROP))


# ------------------------------------------------------------------------------------- module loading

def _purge_foreign(names: tuple[str, ...], folder: str) -> None:
    for name in names:
        mod = sys.modules.get(name)
        f = getattr(mod, "__file__", None) if mod is not None else None
        if mod is not None and (not f or os.path.dirname(os.path.abspath(f)) != folder):
            del sys.modules[name]


def import_reference(problem_dir: str, private_name: str) -> Any:
    """Import reference.py from the trusted problem dir (that dir first on sys.path so ``checks``/``nvfp4``
    resolve to its siblings)."""
    d = os.path.abspath(problem_dir)
    path = os.path.join(d, "reference.py")
    if not os.path.isfile(path):
        raise KdevError(ErrorCode.NOT_FOUND, f"no reference.py in {d}",
                        recovery="pass --problem <dir with TASK.md + reference.py> or set KDEV_PROBLEMS_ROOT")
    _purge_foreign(("checks", "nvfp4"), d)
    if d in sys.path:
        sys.path.remove(d)
    sys.path.insert(0, d)
    spec = importlib.util.spec_from_file_location(private_name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[private_name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_module(path: str, name: str) -> Any:
    """exec a submission file as module ``name`` (its own directory is NOT put on sys.path)."""
    p = os.path.abspath(path)
    if not os.path.isfile(p):
        raise KdevError(ErrorCode.NOT_FOUND, f"submission not found: {path}")
    spec = importlib.util.spec_from_file_location(name, p)
    if spec is None or spec.loader is None:
        raise KdevError(ErrorCode.INVALID_REQUEST, f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return mod


def apply_knobs(module: Any, knobs: dict[str, Any] | None) -> None:
    """Set module-level knob overrides (sweeps / ``--set``)."""
    for k, v in (knobs or {}).items():
        setattr(module, k, v)


def arity(fn: Any) -> int | None:
    """Number of positional parameters (None for *args or uninspectable callables)."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return None
    if any(p.kind == p.VAR_POSITIONAL for p in sig.parameters.values()):
        return None
    return len([p for p in sig.parameters.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)])


_F = r"([0-9.]+(?:e[+-]?\d+)?)"
_ERR_RE = re.compile(r"relative L2 error " + _F, re.I)
_BOUND_RE = re.compile(r"\[bound " + _F)
_ROW_RE = re.compile(r"(?:worst row|is off by) " + _F)


def parse_check_message(msg: str) -> dict[str, float]:
    """Numbers in a reference check message: rel_err (first 'relative L2 error'), bound, worst_row."""
    out: dict[str, float] = {}
    for key, rx in (("rel_err", _ERR_RE), ("bound", _BOUND_RE), ("worst_row", _ROW_RE)):
        m = rx.search(msg or "")
        if m:
            try:
                out[key] = float(m.group(1).rstrip("."))
            except ValueError:
                pass
    return out


def cuda_sync() -> None:
    """torch.cuda.synchronize() when torch + CUDA are present (no-op in GPU-free unit tests)."""
    torch = sys.modules.get("torch")
    if torch is None:
        try:
            import torch  # noqa: F811
        except ImportError:
            return
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:  # noqa: BLE001 - fake/blocked torch
        pass


def discover_extensions(path: str) -> list[ExtSpec]:
    """Compile units from ``load_inline(...)`` / ``load(...)`` calls whose arguments are literals or module-level
    string constants (best effort; dynamically built sources are skipped)."""
    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError):
        return []
    consts: dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                consts[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, SyntaxError, TypeError):
                pass

    def val(n: ast.AST) -> Any:
        if isinstance(n, ast.Name):
            return consts.get(n.id)
        if isinstance(n, (ast.List, ast.Tuple)):
            return [val(e) for e in n.elts]
        try:
            return ast.literal_eval(n)
        except (ValueError, SyntaxError, TypeError):
            return None

    def strs(v: Any) -> list[str]:
        if isinstance(v, str):
            return [v]
        if isinstance(v, (list, tuple)):
            return [x for x in v if isinstance(x, str)]
        return []

    specs: list[ExtSpec] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fname = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        if fname not in ("load_inline", "load"):
            continue
        kw = {k.arg: val(k.value) for k in node.keywords if k.arg}
        name = kw.get("name") or (val(node.args[0]) if node.args else None)
        if not isinstance(name, str):
            continue
        inline = fname == "load_inline"
        srcs = strs(kw.get("cuda_sources")) + strs(kw.get("cpp_sources")) if inline else strs(kw.get("sources"))
        specs.append(ExtSpec(name=name, sources=srcs, inline=inline, cflags=strs(kw.get("extra_cflags")),
                             cuda_cflags=strs(kw.get("extra_cuda_cflags")),
                             ldflags=strs(kw.get("extra_ldflags"))))
    return specs


# ------------------------------------------------------------------------------------------- adapter

class OkcAdapter:
    """Adapter for an OKC problem dir (see module docstring)."""

    name = "okc"

    def __init__(self, problem_dir: str, config_path: str | None = None, *, router_seed: int | None = None,
                 roots: list[str] | None = None) -> None:
        raw = os.path.expanduser(problem_dir or "")
        if raw and is_problem_dir(raw):
            pdir: str | None = os.path.abspath(raw)
            name = os.path.basename(pdir.rstrip("/\\"))
        else:
            name = os.path.basename(raw.rstrip("/\\")) if raw else ""
            pdir = None
        cfg_path = Path(config_path) if config_path else PROBLEMS_DIR / f"{name}.toml"
        if cfg_path.is_file():
            self.config: dict[str, Any] = load_problem_config(cfg_path)
            self.config_path: str | None = str(cfg_path)
        elif pdir:
            self.config, self.config_path = {"problem": name}, None
        else:
            raise KdevError(ErrorCode.INVALID_REQUEST, f"unknown OKC problem {problem_dir!r}",
                            details={"known": known_problems()},
                            recovery="pass a problem dir (TASK.md + reference.py) or one of: "
                                     + ", ".join(known_problems()))
        self.problem = str(self.config.get("problem") or name)
        if pdir is None:
            pdir = find_problem_dir(self.problem, self.config.get("search_dirs"), roots)
        self.problem_dir = pdir or ""
        self.gpu = str(self.config.get("gpu", "h200"))
        prot = dict(self.config.get("protocol", {}))
        known = set(TimingProtocol.__dataclass_fields__)  # type: ignore[attr-defined]
        self.protocol = TimingProtocol(**{k: v for k, v in prot.items() if k in known})
        self.budget_s = self.protocol.budget_s
        self.router_seed = router_seed if router_seed is not None else self.config.get("router_seed")
        self._shapes = [_shape_from_cfg(d, True) for d in self.config.get("shapes", [])]
        self._tests = [_shape_from_cfg(d, False) for d in self.config.get("tests", [])]
        self._extra = [_shape_from_cfg(d, False) for d in self.config.get("extra", [])]
        self._meta = {str(d["id"]): d for d in self.config.get("shapes", [])}
        if not self._shapes and self.problem_dir:
            task = Path(self.problem_dir) / "TASK.md"
            if task.is_file():
                self._shapes, tests = parse_task_md(task.read_text(encoding="utf-8", errors="replace"))
                self._tests = self._tests or tests
        self._R: Any = None
        self._weights: Any = None
        self.last_prepare_s: float | None = None

    # ---------------------------------------------------------------- pure (GPU-free) part
    def shapes(self) -> list[Shape]:
        return list(self._shapes)

    def tests(self) -> list[Shape]:
        return list(self._tests)

    def extra_shapes(self) -> list[Shape]:
        return list(self._extra)

    def input_variance(self) -> dict[str, bool]:
        return {k: bool(v) for k, v in dict(self.config.get("input_variance", {})).items()}

    def default_seeds(self) -> int:
        """Seeds per shape by default (DESIGN 6.5): 5 for routing/length-dependent inputs, else 1."""
        return 5 if any(self.input_variance().values()) else 1

    def judge_seed(self, shape: Shape) -> int | None:
        """The judge seed of a shape when known: set on shapes built from a judge JSON (judge_cases), else the
        task's nominal seed from the TOML (the API benchmark list)."""
        if shape.seed is not None:
            return int(shape.seed)
        meta = self._meta.get(shape.shape_id)
        return int(meta["seed"]) if meta and meta.get("seed") is not None else None

    def judge_repeats(self, shape_id: str) -> int:
        """Timed repetitions the judge runs for a scored shape (records[].runs)."""
        meta = self._meta.get(shape_id) or {}
        return int(meta.get("repeats", self.protocol.judge_reps))

    def floor_us(self, shape_id: str) -> float | None:
        """TASK.md floor (4.8 TB/s or 989 TFLOP/s) of a scored shape."""
        meta = self._meta.get(shape_id) or {}
        return float(meta["floor_us"]) if meta.get("floor_us") is not None else None

    def kernel_fn_args(self) -> int:
        return int(self.config.get("kernel_fn_args", 2))

    def baseline_path(self) -> str | None:
        if not self.problem_dir:
            return None
        p = os.path.join(self.problem_dir, "baseline.py")
        return p if os.path.isfile(p) else None

    def score(self, sub_us: dict[str, float], base_us: dict[str, float]) -> float:
        return load_weighted_geomean(self.shapes(), sub_us, base_us)

    def rules(self) -> ComplyRules:
        c = dict(self.config.get("comply", {}))
        return ComplyRules(precision={str(k): str(v) for k, v in dict(c.get("precision", {})).items()},
                           forbid_patterns=list(c.get("forbid_patterns", [])),
                           no_cross_call_state=bool(c.get("no_cross_call_state", True)),
                           graph_capturable=bool(c.get("graph_capturable", True)),
                           lossless_weights=bool(c.get("lossless_weights", True)),
                           notes=list(c.get("notes", [])))

    def comply_extra(self) -> dict[str, Any]:
        """Problem-specific dynamic probes: {'every_key': {...}, 'routing': {...}} from [comply.*]."""
        c = dict(self.config.get("comply", {}))
        return {k: v for k, v in c.items() if isinstance(v, dict) and k != "precision"}

    def find_shape(self, shape_id: str) -> Shape | None:
        for s in self._shapes + self._tests + self._extra:
            if s.shape_id == shape_id:
                return s
        return None

    def smallest_test(self) -> Shape:
        """The cheapest correctness case (for sanitize): minimal product of the numeric case values."""
        pool = self._tests or self._shapes
        if not pool:
            raise KdevError(ErrorCode.INVALID_REQUEST, f"{self.problem}: no shapes/tests configured")

        def size(s: Shape) -> float:
            v = 1.0
            for x in s.case.values():
                if isinstance(x, (int, float)) and not isinstance(x, bool) and x > 0:
                    v *= float(x)
            return v
        return min(pool, key=size)

    def judge_cases(self, judge_json: dict[str, Any]) -> list[Shape]:
        """Scored shapes (with the judge's seeds) from an ``okc status --json``: benchmark records' cases minus
        load/repeats/weight/router_*/seed; ids matched to the TOML shapes by case, else 'j<index>'."""
        return self._judge_records(judge_json, "benchmark")

    def judge_tests(self, judge_json: dict[str, Any]) -> list[Shape]:
        """The judge's test cases with their seeds (to reproduce a judge FAIL locally)."""
        return self._judge_records(judge_json, "test")

    def _judge_records(self, judge_json: dict[str, Any], event: str) -> list[Shape]:
        res = judge_json.get("result") or judge_json
        by_case = {case_key(s.case): s for s in (self._shapes if event == "benchmark" else self._tests)}
        out: list[Shape] = []
        for i, rec in enumerate(r for r in res.get("records", []) if r.get("event") == event):
            raw = dict(rec.get("case", {}))
            case = {k: v for k, v in raw.items() if k not in JUDGE_DROP}
            known = by_case.get(case_key(case))
            sid = known.shape_id if known else f"j{rec.get('index', i)}"
            out.append(Shape(sid, case, weight=float(raw.get("weight", known.weight if known else 1.0)),
                             load=str(raw.get("load", known.load if known else "")),
                             scored=event == "benchmark", seed=raw.get("seed")))
        return out

    def build_hook(self, path: str) -> list[ExtSpec]:
        return discover_extensions(path)

    # ------------------------------------------------------------ worker (GPU) part
    def reference(self) -> Any:
        if self._R is None:
            if not self.problem_dir:
                raise KdevError(ErrorCode.NOT_FOUND, f"problem dir for {self.problem} not found",
                                recovery="pass --problem <dir> or set KDEV_PROBLEMS_ROOT (layout <root>/<p>/<p>/)")
            self._R = import_reference(self.problem_dir, "_kdev_ref_" + re.sub(r"\W", "_", self.problem))
        return self._R

    def weights(self) -> Any:
        """The problem's weights (generated once per adapter, as the judge does once per run)."""
        if not self.config.get("weights", True):
            return None
        R = self.reference()
        if self._weights is None and hasattr(R, "generate_weights"):
            kw = {"router_seed": int(self.router_seed)} if self.router_seed is not None else {}
            self._weights = R.generate_weights(getattr(R, "WEIGHTS_SEED", 0), **kw)
        return self._weights

    def load(self, path: str, tag: str) -> Any:
        h = hashlib.md5(os.path.abspath(path).encode()).hexdigest()[:8]  # noqa: S324 - identity only
        mod = load_module(path, f"kdev_sub_{re.sub(r'[^0-9A-Za-z_]', '_', tag)}_{h}")
        if not callable(getattr(mod, "kernel_fn", None)):
            raise KdevError(ErrorCode.INVALID_REQUEST, f"{path} defines no kernel_fn")
        return mod

    def prepare(self, module: Any) -> Any:
        """Untimed state: ``prepare(weights)`` (weights dict shallow-copied), ``prepare()`` without weights, or
        None when the module has no prepare. Records ``last_prepare_s`` (synchronised wall time)."""
        t0 = time.monotonic()
        state = None
        fn = getattr(module, "prepare", None)
        if callable(fn):
            w = self.weights()
            if w is None:
                state = fn() if arity(fn) == 0 else fn(None)
            else:
                state = fn(dict(w) if isinstance(w, dict) else w)
        cuda_sync()
        self.last_prepare_s = time.monotonic() - t0
        return state

    def make_inputs(self, shape: Shape, seed: int) -> Any:
        """``reference.generate_input(**case, seed=seed)`` (+ case_fields). Fresh objects every call: the
        reference's functools caches are cleared so two calls never alias (decode/prefill cache inputs)."""
        R = self.reference()
        gen = R.generate_input
        clear = getattr(gen, "cache_clear", None)
        if clear:
            clear()
        case = dict(shape.case)
        case["seed"] = int(seed)
        if self.config.get("case_fields") and hasattr(R, "case_fields"):
            case.update(R.case_fields(dict(case), self.weights()))
        data = gen(**case)
        if clear:
            clear()
        return data

    def call(self, module: Any, state: Any, inputs: Any) -> Any:
        if self.kernel_fn_args() == 1:
            return module.kernel_fn(inputs)
        return module.kernel_fn(inputs, state)

    def check(self, inputs: Any, output: Any) -> CheckResult:
        """The judge's check: ``check_implementation`` (through ``collect`` when the problem has it)."""
        R = self.reference()
        try:
            got: Any = output
            style = self.config.get("check_style") or ("collect" if hasattr(R, "collect") else "direct")
            if style == "collect":
                try:
                    got = R.collect(inputs, output)
                except ValueError as e:
                    return CheckResult(False, f"malformed output: {e}")
            fn = R.check_implementation
            ok, msg = fn(inputs, got) if arity(fn) == 2 else fn(inputs, got, self.weights())
        except Exception as e:  # noqa: BLE001 - a crashing check is a failed check, with the reason
            return CheckResult(False, f"check raised {type(e).__name__}: {e}")
        nums = parse_check_message(str(msg))
        tol = nums.get("bound")
        if tol is None:
            ref_tol = getattr(R, "REL_L2_TOL", None)
            layer = (inputs.get("config") or {}).get("layer") if isinstance(inputs, dict) else None
            if layer is not None and layer in getattr(R, "FP32_LAYERS", ()):
                ref_tol = getattr(R, "ROUTER_TOL", ref_tol)
            tol = float(ref_tol) if isinstance(ref_tol, (int, float)) else None
        return CheckResult(bool(ok), str(msg), max_rel_err=nums.get("rel_err"), tol=tol,
                           details={k: v for k, v in nums.items() if k != "rel_err"})

    def speed_of_light_us(self, shape: Shape, hw: dict[str, float] | None = None) -> float | None:
        """reference.speed_of_light_ns(**case, hw) / 1000 (H200: 4.8 TB/s, 989 TFLOP/s), else the TOML floor."""
        hw = hw or {"hbm_bytes_per_s": 4.8e12, "bf16_dense_flops_per_s": 989e12}
        try:
            fn = getattr(self.reference(), "speed_of_light_ns", None)
            if fn:
                return float(fn(**shape.case, hw=hw)) / 1000.0
        except KdevError:
            pass
        return self.floor_us(shape.shape_id)

    def __repr__(self) -> str:  # pragma: no cover
        return f"OkcAdapter({self.problem!r}, dir={self.problem_dir!r})"


def geomean(xs: list[float]) -> float:
    xs = [x for x in xs if x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")
