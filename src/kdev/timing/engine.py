"""Timing engine: `time` and `ab` jobs under protocol kdev-tp/1 (docs/PLAN.md 3.7, docs/PROTOCOL.md).
Owner: P3.

The scheduling/statistics core is written against the :class:`Timer` protocol so it is unit-tested with
:class:`FakeTimer` (no GPU). :class:`TorchTimer` is the real implementation (graph capture + read flush).

Per shape: inputs per seed (judge seed first, calib_seeds rotation), every file captured as a CUDA graph
on the same inputs, warm-up replay, then ``reps`` repetitions per seed; in each repetition every file is
replayed once in ABBA (or rotating) order, each replay preceded by the read flush; consecutive repetitions
(2k, 2k+1) share a seed and have the order swapped, so the judge's pair geomean cancels the order. After
timing, each file's graph output is poisoned (NaN), replayed once more and checked with the adapter
(materialisation + correctness guard on the timed inputs).
"""
from __future__ import annotations

import math
import os
import time
import zlib
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from kdev.contract import ErrorCode, Fidelity, KdevError, KdevErrorInfo, Result
from kdev.models import FileShapeTiming, Interval, JobSpec, Sample, ShapeResult, TimingParams
from kdev.timing import schedule, stats
from kdev.timing.overhead import SENSITIVE_BELOW_US, Overhead

BASE_LABEL = "base"
DEFAULT_NOISE_FLOOR = 0.03
N_BOOT_RATIO = 10000
N_BOOT_FILE = 2000
N_BOOT_SCORE = 2000


class Timer(Protocol):
    """Prepared timing targets for one shape+seed."""

    def labels(self) -> list[str]: ...
    def replay_us(self, label: str) -> float:
        """Flush, replay the label's graph once, return raw elapsed us."""
        ...
    def overhead_us(self) -> float: ...


@dataclass
class FakeTimer:
    """Deterministic timer for tests: ``means[label]`` us + optional per-call noise sequence.

    ``replay_us`` returns means[label] + overhead + noise[i % len(noise)] where i is the call index, so the
    engine's overhead subtraction gives back means[label] + noise."""

    means: dict[str, float]
    noise: list[float] = field(default_factory=list)
    overhead: float = 4.0
    calls: list[str] = field(default_factory=list)

    def labels(self) -> list[str]:
        return list(self.means)

    def replay_us(self, label: str) -> float:
        i = len(self.calls)
        self.calls.append(label)
        n = self.noise[i % len(self.noise)] if self.noise else 0.0
        return self.means[label] + self.overhead + n

    def overhead_us(self) -> float:
        return self.overhead


def _tensors(obj: Any, depth: int = 0) -> list[Any]:
    """All torch tensors inside obj (dict/list/tuple/attrs, 3 levels)."""
    try:
        import torch
    except Exception:  # noqa: BLE001
        return []
    if isinstance(obj, torch.Tensor):
        return [obj]
    if depth > 3:
        return []
    if isinstance(obj, dict):
        return [t for v in obj.values() for t in _tensors(v, depth + 1)]
    if isinstance(obj, (list, tuple)):
        return [t for v in obj for t in _tensors(v, depth + 1)]
    d = getattr(obj, "__dict__", None)
    if isinstance(d, dict) and depth < 2:
        return [t for v in d.values() for t in _tensors(v, depth + 1)]
    return []


def _storage_ptrs(ts: list[Any]) -> set[int]:
    out = set()
    for t in ts:
        try:
            out.add(t.untyped_storage().data_ptr())
        except Exception:  # noqa: BLE001
            pass
    return out


class TorchTimer:
    """Real timer: adapter.load/prepare per file (once), make_inputs(shape, seed), graph.capture of
    adapter.call, Flusher before each replay, CUDA events. Verifies output materialisation (poisoned graph
    output, replay, adapter.check)."""

    def __init__(self, adapter: Any, modules: dict[str, Any], states: dict[str, Any], shape: Any, seed: int,
                 params: TimingParams, flusher: Any = None, overhead: Overhead | None = None,
                 select: Callable[[str], None] | None = None) -> None:
        from kdev.timing import graph as G
        from kdev.timing.flush import DEFAULT_FLUSH_BYTES, Flusher

        self.adapter = adapter
        self.params = params
        self.shape = shape
        self.seed = seed
        self._labels = list(modules)
        self._flush = flusher or Flusher(params.flush, params.flush_bytes or DEFAULT_FLUSH_BYTES)
        self._ovh = overhead or Overhead(mode=params.overhead)
        self.inputs = adapter.make_inputs(shape, seed)
        self._fns: dict[str, Callable[[], Any]] = {}
        self._graphs: dict[str, Any] = {}
        self._outs: dict[str, Any] = {}
        for lb in self._labels:
            mod, st = modules[lb], states.get(lb)
            if select is not None:      # lazy load_inline on first call must resolve to THIS label's variant
                select(lb)
            def fn(m: Any = mod, s: Any = st) -> Any:
                return adapter.call(m, s, self.inputs)

            self._fns[lb] = fn
            if params.graph:
                g, _eager, out = G.capture_with_output(fn)
                self._graphs[lb], self._outs[lb] = g, out
            else:
                self._outs[lb] = fn()

    def labels(self) -> list[str]:
        return list(self._labels)

    def replay_us(self, label: str) -> float:
        from kdev.timing import graph as G

        if self.params.graph:
            return G.time_replay(self._graphs[label], self._flush)
        return G.time_eager(self._fns[label], self._flush)

    def overhead_us(self) -> float:
        return self._ovh.subtracted

    def verify(self, label: str) -> dict[str, Any]:
        """Correctness on the timed inputs: replay once and adapter.check the graph output (authoritative).
        Then the materialisation probe: fill float outputs (not aliasing inputs) with NaN, replay, check
        again; a failure only there -> ``materialized: False`` (warning: the replay does not rewrite the
        whole output, e.g. packed outputs or stale-buffer reuse)."""
        import torch

        info: dict[str, Any] = {"label": label, "seed": self.seed}
        if self.params.graph:
            self._graphs[label].replay()
        else:
            self._outs[label] = self._fns[label]()
        torch.cuda.synchronize()
        info["ok"], info["message"] = self._check(self._outs[label])
        if self.params.graph and info["ok"]:
            in_ptrs = _storage_ptrs(_tensors(self.inputs))
            poisoned = 0
            for t in _tensors(self._outs[label]):
                if t.is_floating_point() and t.element_size() >= 2 and                         t.untyped_storage().data_ptr() not in in_ptrs:
                    t.fill_(float("nan"))
                    poisoned += 1
            if poisoned:
                self._graphs[label].replay()
                torch.cuda.synchronize()
                ok2, msg2 = self._check(self._outs[label])
                info["materialized"] = ok2
                if not ok2:
                    info["materialized_message"] = msg2
            info["poisoned"] = poisoned
        return info

    def _check(self, out: Any) -> tuple[bool | None, str]:
        try:
            cr = self.adapter.check(self.inputs, out)
            return bool(cr.ok), (cr.message or "")[:300]
        except NotImplementedError:
            return None, "adapter.check not implemented"
        except Exception as e:  # noqa: BLE001
            return False, f"check raised {type(e).__name__}: {e}"[:300]

    def close(self) -> None:
        self._graphs.clear()
        self._outs.clear()
        self._fns.clear()
        self.inputs = None


def _measure_interleaved(timers: list[tuple[int, Timer]], shape_id: str, params: TimingParams,
                         warmup: int = 1) -> list[Sample]:
    """Interleave several seeds' timers: rep r uses seed index (r // 2) % k, so each order-swapped pair
    (2j, 2j+1) shares its inputs; total reps = params.reps * k. Warm-up replays every graph once."""
    if not timers:
        return []
    labels = timers[0][1].labels()
    n = len(labels)
    k = len(timers)
    for w in range(warmup):
        for _, t in timers:
            for lb in labels:
                t.replay_us(lb)
    out: list[Sample] = []
    total = max(0, int(params.reps)) * k
    for r in range(total):
        seed, t = timers[(r // 2) % k]
        ovh = t.overhead_us()
        for pos, idx in enumerate(schedule.order_for_rep(n, r, params.order)):
            lb = labels[idx]
            raw = t.replay_us(lb)
            out.append(Sample(shape_id=shape_id, label=lb, seed=seed, rep=r, order_pos=pos, us_raw=raw,
                              us=raw - ovh))
    return out


def measure_shape(timer: Timer, shape_id: str, seed: int, params: TimingParams) -> list[Sample]:
    """Run the interleaved schedule for one shape/seed and return samples (overhead subtracted)."""
    return _measure_interleaved([(seed, timer)], shape_id, params)


# ------------------------------------------------------------------------------------------------ stats


def _sid(shape: Any) -> str:
    return shape["shape_id"] if isinstance(shape, dict) else shape.shape_id


def _sattr(shape: Any, k: str, default: Any) -> Any:
    if isinstance(shape, dict):
        return shape.get(k, default)
    return getattr(shape, k, default)


def _bseed(*parts: str) -> int:
    return zlib.crc32("|".join(parts).encode())


def _group(samples: list[Sample]) -> dict[str, dict[str, list[Sample]]]:
    g: dict[str, dict[str, list[Sample]]] = {}
    for s in samples:
        g.setdefault(s.shape_id, {}).setdefault(s.label, []).append(s)
    return g


def pair_speedups(by_label: dict[str, list[Sample]], num: str, ref: str) -> list[float]:
    """Judge-paired speedups of ``num`` over ``ref`` (ref_us/num_us per rep, geomean of order-swapped
    consecutive reps within the same seed)."""
    a = {(s.seed, s.rep): s for s in by_label.get(num, [])}
    b = {(s.seed, s.rep): s for s in by_label.get(ref, [])}
    keys = sorted(set(a) & set(b), key=lambda k: k[1])
    out: list[float] = []
    by_seed: dict[int, list[tuple[int, int]]] = {}
    for k in keys:
        by_seed.setdefault(k[0], []).append(k)
    for seed_keys in by_seed.values():
        nums = [a[k].us for k in seed_keys]
        dens = [b[k].us for k in seed_keys]
        firsts = [0 if b[k].order_pos < a[k].order_pos else 1 for k in seed_keys]
        out.extend(stats.paired_ratios(nums, dens, firsts))
    return out


def file_timing(ss: list[Sample], label: str, md5: str = "", n_boot: int = N_BOOT_FILE) -> FileShapeTiming:
    us = [s.us for s in ss]
    ci = stats.bootstrap_ci(us, n_boot=n_boot, seed=_bseed(label, ss[0].shape_id if ss else ""))
    seeds: list[int] = []
    for s in ss:
        if s.seed not in seeds:
            seeds.append(s.seed)
    per_seed = [stats.median([s.us for s in ss if s.seed == sd]) for sd in seeds]
    return FileShapeTiming(label=label, md5=md5, median_us=ci.median, ci_lo_us=ci.ci_lo, ci_hi_us=ci.ci_hi,
                           n=len(us), seeds=seeds, per_seed_median_us=per_seed)


def _fallback_score(shapes: list[Any]) -> Callable[[dict[str, float], dict[str, float]], float]:
    from kdev.timing._direct import load_weighted_geomean

    from kdev.adapters.base import Shape

    sh = [s if isinstance(s, Shape) else Shape(_sid(s), {}, weight=float(_sattr(s, "weight", 1.0)),
                                               load=str(_sattr(s, "load", ""))) for s in shapes]
    return lambda sub, base: load_weighted_geomean(sh, sub, base)


def score_block(groups: dict[str, dict[str, list[Sample]]], shapes: list[Any], ref: str, labels: list[str],
                score_fn: Callable[[dict[str, float], dict[str, float]], float], n_boot: int = N_BOOT_SCORE
                ) -> dict[str, Any]:
    """Judge-formula score of each label vs ``ref`` with a bootstrap CI over paired reps (per shape).
    Score = score_fn(sub_us = ref_median / pair_speedup, base_us = ref_median), as the judge derives the
    submission time from the baseline median and the paired speedup."""
    per_file: dict[str, float] = {}
    ci: dict[str, list[float]] = {}
    judge_ci: dict[str, list[float]] = {}
    sids = [_sid(s) for s in shapes if _sid(s) in groups and ref in groups[_sid(s)]]
    ref_med = {sid: stats.median([s.us for s in groups[sid][ref]]) for sid in sids}
    for lb in labels:
        if lb == ref:
            continue
        pairs = {sid: pair_speedups(groups[sid], lb, ref) for sid in sids if lb in groups[sid]}
        pairs = {k: v for k, v in pairs.items() if v}
        if not pairs:
            continue

        def fn(d: dict[str, float], _r: dict[str, float] = ref_med) -> float:
            return score_fn({s: _r[s] / d[s] for s in d}, {s: _r[s] for s in d})

        iv = stats.score_ci(pairs, fn, n_boot=n_boot, seed=_bseed("score", lb, ref))
        per_file[lb] = iv.median
        ci[lb] = [iv.ci_lo, iv.ci_hi]
        h = stats.JUDGE_MIN_CI_HALF
        judge_ci[lb] = [min(iv.ci_lo, iv.median * (1 - h)), max(iv.ci_hi, iv.median * (1 + h))]
    return {"ref": ref, "per_file": per_file, "ci": ci, "ci_per_file": ci, "judge_ci": judge_ci,
            "n_shapes": len(sids)}


def summarize(samples: list[Sample], shapes: list[Any], params: TimingParams, ref_label: str,
              score_fn: Any = None, *, md5s: dict[str, str] | None = None,
              checks: dict[tuple[str, str], bool | None] | None = None,
              floors_us: dict[str, float] | None = None, noise_floor: float | None = None
              ) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Per-shape FileShapeTiming + paired ratio CI + verdicts, and the score block (Result.shapes/.score).

    ``ratio`` per shape is the TIME ratio num/den (num = first non-reference label, den = ref; <1 = num
    faster) with ``speedup`` = den/num judge-paired and its CI; ``ratios`` lists every non-ref label.
    ``checks[(shape_id, label)]`` False -> verdict 'incorrect'; a median under ``floors_us[shape_id]`` ->
    'suspect'. The score block (when ``score_fn(sub_us, base_us)`` is given) has per-label scores vs ref
    and ``delta`` for the first non-ref label."""
    md5s = md5s or {}
    checks = checks or {}
    floors_us = floors_us or {}
    nf = DEFAULT_NOISE_FLOOR if noise_floor is None else noise_floor
    groups = _group(samples)
    order: list[str] = []
    for s in samples:
        if s.label not in order:
            order.append(s.label)
    others = [lb for lb in order if lb != ref_label]
    rows: list[dict[str, Any]] = []
    for sh in shapes:
        sid = _sid(sh)
        g = groups.get(sid)
        if not g:
            continue
        per_file = [file_timing(g[lb], lb, md5s.get(lb, "")).to_dict() for lb in order if lb in g]
        warnings: list[str] = []
        ratios = []
        verdicts = []
        for lb in others:
            if lb not in g or ref_label not in g:
                continue
            pairs = pair_speedups(g, lb, ref_label)
            if not pairs:
                continue
            sp = stats.bootstrap_ci(pairs, n_boot=N_BOOT_RATIO, seed=_bseed(sid, lb, ref_label))
            ref_med = stats.median([s.us for s in g[ref_label]])
            num_med = stats.median([s.us for s in g[lb]])
            bad = checks.get((sid, lb)) is False or checks.get((sid, ref_label)) is False
            floor = floors_us.get(sid)
            suspect = floor is not None and num_med < floor
            if bad:
                v = "incorrect"
            else:
                v = stats.verdict(sp, nf, stats.ratio_se(sp), suspect=suspect)
            if suspect:
                warnings.append(f"{lb} faster than the floor ({num_med:.2f} < {floor:.2f} us)")
            verdicts.append(v)
            ratios.append({"num": lb, "den": ref_label,
                           "median": 1.0 / sp.median if sp.median else float("nan"),
                           "ci_lo": 1.0 / sp.ci_hi if sp.ci_hi else float("nan"),
                           "ci_hi": 1.0 / sp.ci_lo if sp.ci_lo else float("nan"),
                           "speedup": sp.median, "speedup_ci": [sp.ci_lo, sp.ci_hi], "n_pairs": sp.n,
                           "num_judge_us": ref_med / sp.median if sp.median else None,
                           "abs_delta_us": num_med - ref_med, "verdict": v})
        ref_rows = [p for p in per_file if p["label"] == ref_label]
        if ref_rows and ref_rows[0]["median_us"] < SENSITIVE_BELOW_US:
            warnings.append(f"{sid} < {SENSITIVE_BELOW_US:.0f} us: overhead-sensitive")
        chk = None
        bad_labels = [lb for lb in order if checks.get((sid, lb)) is False]
        if any((sid, lb) in checks for lb in order):
            chk = "fail:" + ",".join(bad_labels) if bad_labels else "pass"
        sr = ShapeResult(shape_id=sid, weight=float(_sattr(sh, "weight", 1.0) or 1.0),
                         load=str(_sattr(sh, "load", "") or ""), per_file=per_file,
                         ratio=ratios[0] if ratios else None, verdict=verdicts[0] if verdicts else "",
                         check_status=chk, warnings=warnings).to_dict()
        sr["ratios"] = ratios
        rows.append(sr)
    score = None
    if score_fn is not None and others:
        score = score_block(groups, shapes, ref_label, order, score_fn)
        first = next((lb for lb in others if lb in score["per_file"]), None)
        if first:
            lo, hi = score["ci"][first]
            score["delta"] = {"num": first, "den": ref_label, "median": score["per_file"][first],
                              "ci_lo": lo, "ci_hi": hi}
    return rows, score


def overall_verdict(rows: list[dict[str, Any]], score: dict[str, Any] | None, noise_floor: float) -> str:
    vs = [r.get("verdict") for r in rows]
    if "incorrect" in vs:
        return "incorrect"
    if "suspect" in vs:
        return "suspect"
    if score and score.get("delta"):
        d = score["delta"]
        iv = Interval(d["median"], d["ci_lo"], d["ci_hi"])
        return stats.verdict(iv, noise_floor, stats.ratio_se(iv))
    if not vs:
        return ""
    if all(v == vs[0] for v in vs):
        return vs[0]
    return "mixed"


# ------------------------------------------------------------------------------------------- GPU session


@dataclass
class _File:
    label: str
    path: str
    role: str
    md5: str = ""
    judge_json: str | None = None
    knobs: dict[str, Any] = field(default_factory=dict)


def _md5(path: str) -> str:
    try:
        from kdev.util import md5_file

        return md5_file(path)
    except OSError:
        return ""


def find_problem_dir(spec: JobSpec) -> str | None:
    """spec.problem if it is a directory with reference.py, else the first file's directory that has one
    (walking up 2 levels), else params.extra.problem_dir."""
    extra = (spec.params or {}).get("extra") or {}
    cands = [extra.get("problem_dir"), spec.problem]
    for f in spec.files:
        d = os.path.dirname(os.path.abspath(f.path))
        cands += [d, os.path.dirname(d), os.path.dirname(os.path.dirname(d))]
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "reference.py")):
            return os.path.abspath(c)
    return None


def resolve_adapter(spec: JobSpec) -> tuple[Any, str | None, str]:
    """(adapter, problem_dir, source): P7's registry adapter, else the private direct OKC adapter."""
    extra = (spec.params or {}).get("extra") or {}
    pdir = find_problem_dir(spec)
    if extra.get("adapter") != "direct":
        try:
            from kdev.adapters.registry import get_adapter

            ad = get_adapter(problem=spec.problem, file=spec.files[0].path if spec.files else None,
                             adapter=spec.adapter)
            ad.shapes()          # probe: stubs raise NotImplementedError here
            return ad, getattr(ad, "problem_dir", pdir) or pdir, "registry"
        except Exception:  # noqa: BLE001 - fall back to the direct adapter
            pass
    if not pdir:
        raise KdevError(ErrorCode.INVALID_REQUEST, "cannot find the problem dir (reference.py)",
                        details={"problem": spec.problem},
                        recovery="pass --problem DIR (the directory with reference.py and baseline.py)")
    from kdev.timing._direct import DirectOkcAdapter

    return DirectOkcAdapter(pdir), pdir, "direct"


def _load_json(path: str) -> dict[str, Any]:
    import json

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def select_shapes(adapter: Any, spec: JobSpec, params: TimingParams) -> list[Any]:
    """Adapter's scored shapes; else cases from a judge JSON (files[].judge_json or extra.judge_json);
    filtered by params.shapes (substring of shape_id or the case dict)."""
    from kdev.adapters.base import Shape

    shapes: list[Any] = []
    try:
        shapes = [s for s in adapter.shapes() if getattr(s, "scored", True)]
    except Exception:  # noqa: BLE001
        shapes = []
    extra = params.extra or {}
    jpath = extra.get("judge_json") or next((f.judge_json for f in spec.files if f.judge_json), None)
    if not shapes and jpath:
        jj = _load_json(jpath)
        try:
            shapes = adapter.judge_cases(jj)
        except Exception:  # noqa: BLE001
            from kdev.timing.calib import parse_judge_json

            shapes = [Shape(js.shape_id, dict(js.case), weight=js.weight, load=js.load, seed=js.seed)
                      for js in parse_judge_json(jj).shapes]
    if not shapes and extra.get("cases"):
        shapes = [Shape(c.get("shape_id", f"case{i}"), dict(c.get("case", c)), weight=c.get("weight", 1.0),
                        load=c.get("load", ""), seed=c.get("seed")) for i, c in enumerate(extra["cases"])]
    if params.shapes:
        f = params.shapes
        shapes = [s for s in shapes if f in s.shape_id or f in str(s.case)]
    if hasattr(adapter, "set_shapes"):
        adapter.set_shapes(shapes)
    return shapes


@dataclass
class Session:
    """Everything a timing job sets up once: adapter, modules/states per label, flusher, overhead."""

    spec: JobSpec
    params: TimingParams
    run_dir: Any
    files: list[_File]
    adapter: Any = None
    problem_dir: str | None = None
    adapter_source: str = ""
    modules: dict[str, Any] = field(default_factory=dict)
    states: dict[str, Any] = field(default_factory=dict)
    prepare_s: dict[str, float] = field(default_factory=dict)
    flusher: Any = None
    overhead: Overhead | None = None
    overhead_end: Overhead | None = None
    optouts: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    clocks_before: dict[str, Any] = field(default_factory=dict)
    clocks_after: dict[str, Any] = field(default_factory=dict)
    foreign_start: list[int] | None = None
    foreign_mid: list[int] = field(default_factory=list)
    foreign_end: list[int] | None = None
    hook: Any = None                               # P5 load_inline hook (None: legacy ext dir / P5 absent)
    variants: dict[str, tuple[dict[str, Any], list[str]]] = field(default_factory=dict)  # label -> knobs, defines

    def select(self, label: str) -> None:
        """Point the build hook at ``label``'s knobs/defines (submissions often load_inline lazily, at the
        first call inside graph capture, not at import/prepare)."""
        if self.hook is not None and label in self.variants:
            self.hook.knobs = dict(self.variants[label][0])
            self.hook.defines = list(self.variants[label][1])
    toolchain: dict[str, Any] = field(default_factory=dict)
    seeds_used: dict[str, list[int]] = field(default_factory=dict)
    checks: dict[tuple[str, str], bool | None] = field(default_factory=dict)
    check_msgs: list[dict[str, Any]] = field(default_factory=list)
    samples: list[Sample] = field(default_factory=list)
    shapes: list[Any] = field(default_factory=list)
    error: KdevErrorInfo | None = None
    t0: float = field(default_factory=time.time)

    def log(self, msg: str) -> None:
        from kdev.timing._io import log

        log(self.run_dir, msg)


def _optouts(params: TimingParams) -> list[str]:
    o = []
    if params.flush != "read":
        o.append(f"flush={params.flush}")
    if not params.graph:
        o.append("eager")
    if params.overhead != "judge":
        o.append(f"overhead={params.overhead}")
    if params.allow_jit:
        o.append("allow_jit")
    if (params.extra or {}).get("legacy_ext"):
        o.append("legacy_ext")
    return o


def taint_optouts(tool: str, params: TimingParams) -> list[str]:
    """P6 profiler-taint guard: torch.profiler leaves CUPTI on (+2-5 us on every later timing in this
    process). Raises (invalid_request) by default; ``extra.allow_profiled`` returns the opt-out instead."""
    from kdev.observe.guard import check_clean

    tag = check_clean(tool, refuse=not (params.extra or {}).get("allow_profiled"))
    return [tag] if tag else []


def open_session(spec: JobSpec, run_dir: Any, files: list[_File]) -> Session:
    """Resolve the adapter, select shapes, load + prepare every file under the no-compile guard, allocate
    the flush buffer and measure the replay overhead. GPU (worker only)."""
    import torch

    from kdev.timing import _env
    from kdev.timing import overhead as OV
    from kdev.timing.flush import DEFAULT_FLUSH_BYTES, Flusher

    params = TimingParams.from_dict(spec.params or {})
    ss = Session(spec=spec, params=params, run_dir=run_dir, files=files)
    ss.optouts = _optouts(params)
    ss.optouts += taint_optouts(spec.kind, params)
    if not torch.cuda.is_available():
        raise KdevError(ErrorCode.CAPABILITY_UNAVAILABLE, "no CUDA device in the worker",
                        recovery="run on the GPU pod (kdev remote) or via the service")
    ss.adapter, ss.problem_dir, ss.adapter_source = resolve_adapter(spec)
    ss.shapes = select_shapes(ss.adapter, spec, params)
    if not ss.shapes:
        raise KdevError(ErrorCode.INVALID_REQUEST, "no shapes to time (adapter has none / filter matched none)",
                        details={"filter": params.shapes},
                        recovery="check --shapes, or give a judge JSON (params.extra.judge_json)")
    extra = params.extra
    if spec.requires_prebuilt and not params.allow_jit:
        try:
            from kdev.build.cache import is_prebuilt

            if is_prebuilt(spec) is False:
                # Not in kdev's build cache (e.g. a legacy TORCH_EXTENSIONS_DIR). The ninja guard below
                # still refuses any actual compile under the lock (needs_build), so only warn here.
                ss.warnings.append("files not in the kdev build cache; relying on the no-compile guard")
        except Exception:  # noqa: BLE001 - P5 not available: the ninja guard below still refuses compiles
            pass
    ss.foreign_start = _env.foreign_contexts()
    ss.hook = None if extra.get("legacy_ext") else _install_build_hook(ss, spec)
    with _env.no_compile_guard(enabled=not params.allow_jit):
        for f in files:
            t0 = time.time()
            ss.variants[f.label] = (dict(f.knobs or {}),
                                    list(next((r.defines for r in spec.files if os.path.abspath(r.path) == f.path),
                                              []) or []))
            ss.select(f.label)       # P5 prebuilt cache: this file's knobs/defines select the variant
            try:
                mod = ss.adapter.load(f.path, f.label)
                ss.modules[f.label] = mod
                ss.states[f.label] = ss.adapter.prepare(mod)
            except Exception as e:  # noqa: BLE001
                if type(e).__name__ == "NeedsBuild":
                    raise KdevError(ErrorCode.NEEDS_BUILD, f"{f.label} ({os.path.basename(f.path)}): {e}",
                                    recovery="run `kdev build FILE` (same knobs/defines) first, or --allow-jit")
                if type(e).__name__ == "ExtBuildFailed":
                    raise KdevError(ErrorCode.BUILD_FAILED, f"{f.label}: {e}", recovery="see the build log") from e
                raise
            mod = ss.modules[f.label]
            if f.knobs and hasattr(mod, "__dict__"):
                for k, v in f.knobs.items():
                    setattr(mod, k, v)
            torch.cuda.synchronize()
            ss.prepare_s[f.label] = round(time.time() - t0, 2)
            ss.log(f"prepare {f.label} ({os.path.basename(f.path)}) {ss.prepare_s[f.label]:.1f}s")
    ss.toolchain = _env.toolchain_fidelity(ss.problem_dir, extra)
    _hook_toolchain(ss)
    nbytes = params.flush_bytes or _profile_flush_bytes(ss.adapter) or DEFAULT_FLUSH_BYTES
    ss.flusher = Flusher(params.flush, nbytes, gate_cycles=int(extra.get("gate_cycles", 0) or 0))
    if params.overhead == "local":
        ss.overhead = OV.measure(ss.flusher, reps=int(extra.get("overhead_reps", 50)))
        ss.overhead.mode = params.overhead
    else:
        # measured at the end (close_session): the measurement graphs' memory pools allocated before the
        # inputs shift their placement, which moved a page-gather attention shape by ~2.5 us.
        ss.overhead = Overhead(mode=params.overhead)
    ss.clocks_before = _env.gpu_state()      # after prepare + overhead warm-up: not the idle clock
    if params.overhead == "local":
        ss.log(f"overhead: {ss.overhead.to_dict()}")
    return ss


def _hook_toolchain(ss: Session, final: bool = False) -> None:
    """Fold the build hook's records (which extensions loaded, from which cache entry, their .so toolchain
    notes) into ss.toolchain. Called after prepare and again at the end (``final``): lazy submissions only
    load their extension at the first call inside graph capture."""
    from kdev.timing import _env

    if ss.hook is None:
        ss.toolchain["build_cache"] = "legacy:" + os.environ.get("TORCH_EXTENSIONS_DIR", "")
    else:
        recs = list(getattr(ss.hook, "records", []))
        ss.toolchain["build_cache"] = "kdev"
        ss.toolchain["exts"] = [{k: r.get(k) for k in ("name", "key", "cache_hit", "status", "so_path")}
                                for r in recs][:40]
        ss.toolchain["n_ext_so"] = len({r.get("so_path") for r in recs if r.get("so_path")})
        sos = [r.get("so_path") for r in recs if r.get("so_path")]
        notes = sorted({v for p in sos for v in _env.so_notes(p)} | set(ss.toolchain.get("so_notes") or []))
        ss.toolchain["so_notes"] = notes
        pin = ss.toolchain.get("pin")
        if pin:
            ss.toolchain["match"] = all(v == pin for v in notes)
    if final and ss.hook is not None:
        leg = sorted({r.get("name", "?") for r in getattr(ss.hook, "records", []) if r.get("status") == "legacy"})
        if leg:
            ss.warnings.append(f"{len(leg)} extension(s) loaded from TORCH_EXTENSIONS_DIR, not the kdev cache "
                               f"(up to date, nothing compiled; toolchain checked via .so notes): "
                               + ", ".join(leg)[:160])
    if final and ss.toolchain.get("match") is False:
        ss.warnings.append(f"TOOLCHAIN MISMATCH: so_notes {ss.toolchain.get('so_notes')} nvcc "
                           f"{ss.toolchain.get('nvcc')} != pin {ss.toolchain.get('pin')}")


def _install_build_hook(ss: Session, spec: JobSpec) -> Any:
    """P5 load_inline hook (mode 'use'): submissions' load_inline/load resolve to the prebuilt kdev cache
    for the pinned toolchain; a missing variant raises NeedsBuild (-> needs_build). None if P5 is absent."""
    try:
        from kdev.build.cache import install_load_inline_hook
        from kdev.build.toolchain import pin_for
    except Exception:  # noqa: BLE001
        ss.warnings.append("P5 build hook unavailable: falling back to the ninja no-compile guard")
        return None
    f0 = spec.files[0] if spec.files else None
    tc = pin_for(ss.problem_dir or spec.problem, f0.path if f0 else None)
    if not ss.params.extra.get("toolchain_pin") and getattr(tc, "nvcc_version", ""):
        if getattr(tc, "source", "") != "detect":
            ss.params.extra.setdefault("toolchain_pin", tc.nvcc_version)
    return install_load_inline_hook(f0, tc, allow_jit=ss.params.allow_jit)


def _profile_flush_bytes(adapter: Any) -> int | None:
    try:
        from kdev.floor.gpus import load_profile

        return int(load_profile(getattr(adapter, "gpu", "h200")).flush_bytes)
    except Exception:  # noqa: BLE001
        return None


def seeds_for(ss: Session, shape: Any) -> list[int]:
    from kdev.timing.seeds import default_seed_count, seeds_for_shape

    p = ss.params
    if p.seed_list:
        return [int(s) for s in p.seed_list]
    try:
        n = p.seeds or default_seed_count(ss.adapter.input_variance())
    except Exception:  # noqa: BLE001
        n = p.seeds or 1
    try:
        base = ss.adapter.judge_seed(shape)
    except Exception:  # noqa: BLE001
        base = getattr(shape, "seed", None)
    return seeds_for_shape(base, n)


def time_shapes(ss: Session) -> None:
    """Measure every shape (all labels interleaved, all seeds), verify outputs, append raw.jsonl.
    KdevError (timeout/cancel) stops the loop; samples so far are kept and ss.error is set."""
    import torch

    from kdev.timing import _env
    from kdev.timing._io import append_jsonl

    try:
        for i, sh in enumerate(ss.shapes):
            seeds = seeds_for(ss, sh)
            ss.seeds_used[sh.shape_id] = seeds
            timers: list[tuple[int, TorchTimer]] = []
            try:
                for sd in seeds:
                    timers.append((sd, TorchTimer(ss.adapter, ss.modules, ss.states, sh, sd, ss.params,
                                                  flusher=ss.flusher, overhead=ss.overhead, select=ss.select)))
                smp = _measure_interleaved(timers, sh.shape_id, ss.params)
                for sd, t in timers:
                    for lb in t.labels():
                        info = t.verify(lb)
                        prev = ss.checks.get((sh.shape_id, lb))
                        ok = info.get("ok")
                        ss.checks[(sh.shape_id, lb)] = (False if (prev is False or ok is False)
                                                        else (ok if prev is None else prev))
                        if ok is False:
                            ss.check_msgs.append(dict(info, shape_id=sh.shape_id))
                        elif info.get("materialized") is False:
                            ss.warnings.append(f"materialisation probe: {sh.shape_id} {lb} output not fully "
                                               f"rewritten by replay ({info.get('materialized_message', '')[:80]})")
            finally:
                for _, t in timers:
                    t.close()
                del timers
                torch.cuda.empty_cache()
            ss.samples.extend(smp)
            append_jsonl(ss.run_dir, "raw.jsonl", [s.to_dict() for s in smp])
            med = {lb: stats.median([s.us for s in smp if s.label == lb]) for lb in ss.modules}
            ss.log(f"[{i + 1}/{len(ss.shapes)}] {sh.shape_id} seeds={len(seeds)} " +
                   " ".join(f"{lb}={v:.2f}us" for lb, v in med.items()))
            fc = _env.foreign_contexts()
            if fc:
                ss.foreign_mid.extend(fc)
                ss.log(f"WARNING foreign CUDA contexts during timing: {fc}")
    except KdevError as e:
        ss.error = e.info()
        ss.log(f"stopped: {e.code.value}: {e.message} (partial samples kept)")


def close_session(ss: Session) -> None:
    from kdev.timing import _env
    from kdev.timing import overhead as OV

    try:
        ss.overhead_end = OV.measure(ss.flusher, reps=50)
        if ss.overhead is not None and ss.overhead.one_kernel_graph is None:
            ss.overhead.empty_graph = ss.overhead_end.empty_graph
            ss.overhead.one_kernel_graph = ss.overhead_end.one_kernel_graph
            ss.overhead.two_kernel_graph = ss.overhead_end.two_kernel_graph
        ss.log(f"overhead (end): {ss.overhead_end.to_dict()}")
    except Exception:  # noqa: BLE001
        ss.overhead_end = None
    ss.clocks_after = _env.gpu_state()
    ss.foreign_end = _env.foreign_contexts()
    try:
        _hook_toolchain(ss, final=True)
    except Exception as e:  # noqa: BLE001
        ss.warnings.append(f"toolchain record failed: {e}")
    try:
        ss.flusher.release()
    except Exception:  # noqa: BLE001
        pass


def fidelity(ss: Session) -> Fidelity:
    from kdev.timing import _env

    excl, waited = _env.exclusivity(ss.foreign_start, ss.foreign_end, ss.foreign_mid)
    ovh = {k: v for k, v in (ss.overhead.to_dict() if ss.overhead else {}).items() if v is not None}
    if ss.overhead_end and ss.overhead_end.local_estimate is not None:
        ovh["local_estimate_end"] = ss.overhead_end.local_estimate
    first = next(iter(ss.seeds_used.values()), [])
    return Fidelity(toolchain=ss.toolchain, flush=ss.flusher.describe() if ss.flusher else {},
                    graph=ss.params.graph, exclusive=excl, waited_s=waited, seeds=list(first),
                    overhead_us=ovh, clocks={"before": ss.clocks_before, "after": ss.clocks_after},
                    optouts=list(ss.optouts))


def _score_fn(ss: Session) -> Callable[[dict[str, float], dict[str, float]], float]:
    fb = _fallback_score(ss.shapes)

    def fn(sub: dict[str, float], base: dict[str, float]) -> float:
        try:
            v = ss.adapter.score(sub, base)
            if v is not None and math.isfinite(v):
                return v
        except Exception:  # noqa: BLE001
            pass
        return fb(sub, base)

    return fn


def _fmt_us(x: float | None) -> str:
    if x is None or not math.isfinite(x):
        return "-"
    return f"{x:.1f}" if x >= 10 else f"{x:.2f}"


def fidelity_line(fid: Fidelity) -> str:
    tc = fid.toolchain or {}
    nv = tc.get("so_notes") or ([tc["nvcc"]] if tc.get("nvcc") else [])
    pin = tc.get("pin")
    tcs = f"nvcc {','.join(nv) or '?'}" + (f"={pin}" if pin and tc.get("match") else
                                           (f"!=PIN {pin}" if pin else " (no pin)"))
    fl = fid.flush or {}
    fls = f"{fl.get('method', '?')}-flush {int(fl.get('bytes', 0)) >> 20}MB"
    if fl.get("method") != "read":
        fls = fls.upper()
    sub = (fid.overhead_us or {}).get("subtracted")
    opt = ",".join(o.upper() for o in fid.optouts) or "none"
    return (f"fidelity: {tcs} {fls} {'graph' if fid.graph else 'EAGER'} excl={fid.exclusive} "
            f"seeds={len(fid.seeds)} overhead=-{sub if sub is not None else 0:.2f}us optouts={opt}")


def render_lines(tool: str, verdict: str, rows: list[dict[str, Any]], score: dict[str, Any] | None,
                 fid: Fidelity | None, labels: list[str], max_rows: int = 15) -> list[str]:
    head = f"{verdict.upper() or 'DONE'}  {tool}"
    if score and score.get("delta"):
        d = score["delta"]
        head += f"  {d['num']}/{d['den']} score x{d['median']:.3f} [{d['ci_lo']:.3f}, {d['ci_hi']:.3f}]"
    lines = [head]
    lines.append("shape".ljust(34) + "".join(f"{lb[:10]:>11}" for lb in labels) + "   speedup [95% CI]   verdict")
    for r in rows[:max_rows]:
        us = {p["label"]: p["median_us"] for p in r["per_file"]}
        rt = r.get("ratio") or {}
        sp = (f"x{rt['speedup']:.3f} [{rt['speedup_ci'][0]:.3f}, {rt['speedup_ci'][1]:.3f}]"
              if rt.get("speedup") else "-")
        lines.append(r["shape_id"][:33].ljust(34) + "".join(f"{_fmt_us(us.get(lb)):>11}" for lb in labels)
                     + f"   {sp:<22} {r.get('verdict', '')}")
    if len(rows) > max_rows:
        lines.append(f"... {len(rows) - max_rows} more rows in table.md")
    # the fidelity line is added by kdev.output.render (Result.fidelity), once
    return lines


def write_table(run_dir: Any, rows: list[dict[str, Any]], labels: list[str]) -> Any:
    from kdev.timing._io import write_text

    out = ["| shape | weight | " + " | ".join(f"{lb} us" for lb in labels) + " | speedup | CI | verdict |",
           "|---|---|" + "---|" * len(labels) + "---|---|---|"]
    for r in rows:
        us = {p["label"]: p["median_us"] for p in r["per_file"]}
        for rt in (r.get("ratios") or [None]):
            sp = f"x{rt['speedup']:.3f}" if rt else "-"
            ci = f"[{rt['speedup_ci'][0]:.3f}, {rt['speedup_ci'][1]:.3f}]" if rt else "-"
            v = rt["verdict"] if rt else r.get("verdict", "")
            nm = f"{r['shape_id']}" + (f" ({rt['num']}/{rt['den']})" if rt else "")
            out.append(f"| {nm} | {r.get('weight', '')} | " + " | ".join(_fmt_us(us.get(lb)) for lb in labels)
                       + f" | {sp} | {ci} | {v} |")
    return write_text(run_dir, "table.md", "\n".join(out) + "\n")


def _files_for(spec: JobSpec, kind: str, adapter_base: Callable[[], str | None]) -> list[_File]:
    """Labels in load/prepare/replay order: the files, then the baseline (calib.py order; load order shifts
    memory placement by up to ~1 us on placement-sensitive shapes, so it is part of the protocol).
    ``extra.base_first`` puts the baseline first."""
    files: list[_File] = []
    used: set[str] = set()
    for i, f in enumerate(spec.files):
        lb = f.label or chr(ord("A") + i)
        if lb in used:
            lb = f"{lb}{i}"
        used.add(lb)
        files.append(_File(lb, os.path.abspath(f.path), f.role or "sub", f.md5 or _md5(f.path), f.judge_json,
                           dict(f.knobs or {})))
    base = (spec.params or {}).get("base", "none" if kind == "ab" else "baseline")
    if base and base != "none":
        path = adapter_base() if base == "baseline" else base
        if not path or not os.path.exists(path):
            raise KdevError(ErrorCode.NOT_FOUND, f"baseline not found ({base!r})",
                            recovery="pass --base PATH or --base none")
        bf = _File(BASE_LABEL, os.path.abspath(path), "base", _md5(path))
        if ((spec.params or {}).get("extra") or {}).get("base_first"):
            files.insert(0, bf)
        else:
            files.append(bf)          # calib.py order (default): subs loaded/prepared first, base last
    return files


def _run(spec: JobSpec, run_dir: Any, kind: str) -> Result:
    from kdev.timing._io import run_id as _rid
    from kdev.timing._io import write_json

    if not spec.files:
        raise KdevError(ErrorCode.INVALID_REQUEST, "no files", recovery="give at least one file")
    if kind == "ab" and len(spec.files) < 2:
        raise KdevError(ErrorCode.INVALID_REQUEST, "ab needs two or more files", recovery="kdev ab A.py B.py")
    ad, _, _ = resolve_adapter(spec)
    files = _files_for(spec, kind, lambda: ad.baseline_path())
    ss = open_session(spec, run_dir, files)
    try:
        time_shapes(ss)
    finally:
        close_session(ss)
    disp, disp_warn = _isolated_dispatch(spec, run_dir, kind) if ss.error is None else (None, [])
    ss.warnings.extend(disp_warn[:5])
    labels = [f.label for f in files]
    ref = BASE_LABEL if kind == "time" and BASE_LABEL in labels else labels[0]
    if kind == "ab":
        ref = next(f.label for f in files if f.role != "base")
    nf = float((ss.params.extra or {}).get("noise_floor", DEFAULT_NOISE_FLOOR))
    floors = (ss.params.extra or {}).get("floors_us") or {}
    rows, score = summarize(ss.samples, ss.shapes, ss.params, ref, _score_fn(ss),
                            md5s={f.label: f.md5 for f in files}, checks=ss.checks, floors_us=floors,
                            noise_floor=nf)
    if kind == "ab" and BASE_LABEL in labels and score is not None:
        score["vs_base"] = score_block(_group(ss.samples), ss.shapes, BASE_LABEL, labels, _score_fn(ss))
    fid = fidelity(ss)
    verdict = overall_verdict(rows, score, nf)
    warnings = list(ss.warnings)
    if _clocks_moved(ss):
        warnings.append(f"SM clock moved >3%: {ss.clocks_before.get('sm_mhz')} -> {ss.clocks_after.get('sm_mhz')}")
    if fid.exclusive not in ("ok", "waited"):
        warnings.append(f"GPU exclusivity: {fid.exclusive.upper()} (result not comparable)")
    for r in rows:
        warnings.extend(r.get("warnings") or [])
    for m in ss.check_msgs[:5]:
        warnings.append(f"check FAILED {m['shape_id']} {m['label']} seed {m['seed']}: {m.get('message', '')}")
    tpath = write_table(run_dir, rows, labels)
    rp = os.path.join(str(getattr(run_dir, "path", run_dir)), "raw.jsonl")
    res = Result(tool=kind, status="ok", verdict=verdict, run_id=_rid(run_dir), job_id=spec.job_id or None,
                 shapes=rows, score=score, fidelity=fid,
                 files={"raw": rp, "table": str(tpath),
                        "log": os.path.join(str(getattr(run_dir, "path", run_dir)), "log.txt")},
                 warnings=warnings[:20], dispatch=disp or None)
    dj = os.path.join(str(getattr(run_dir, "path", run_dir)), "dispatch", "dispatch.json")
    if disp and os.path.isfile(dj):
        res.files["dispatch"] = dj
    res.summary = {"labels": labels, "ref": ref, "n_shapes": len(rows), "reps": ss.params.reps,
                   "order": ss.params.order, "adapter": ss.adapter_source, "problem_dir": ss.problem_dir,
                   "prepare_s": ss.prepare_s, "elapsed_s": round(time.time() - ss.t0, 1),
                   "verdicts": {v: sum(1 for r in rows if r.get("verdict") == v)
                                for v in sorted({r.get("verdict", "") for r in rows})},
                   "checks": "fail" if any(v is False for v in ss.checks.values()) else "pass"}
    prob = spec.problem or (os.path.basename(os.path.normpath(ss.problem_dir)) if ss.problem_dir else None)
    if prob:
        res.summary["problem"] = prob           # ledger rows need it when --problem was not given
    if score and score.get("delta"):
        res.summary["score_delta"] = score["delta"]
    res.lines = render_lines(kind, verdict, rows, score, fid, labels)
    if any(v is False for v in ss.checks.values()):
        res.status = "fail"
    if ss.error is not None:
        res.status, res.error = "error", ss.error
    elif fid.exclusive == "fail":
        res.status = "error"
        res.error = KdevErrorInfo(ErrorCode.NOT_EXCLUSIVE.value, "foreign CUDA contexts during timing",
                                  details={"pids": sorted(set((ss.foreign_start or []) + ss.foreign_mid +
                                                              (ss.foreign_end or [])))},
                                  recovery="re-run under the GPU lock with no other GPU processes")
    write_json(run_dir, "shapes.json", rows)
    return res


def _isolated_dispatch(spec: JobSpec, run_dir: Any, kind: str) -> tuple[dict[str, Any] | None, list[str]]:
    """Result.dispatch for time/ab via P6 ``probe_isolated`` (a child process: the profiler never runs in the
    timing process). Default on for `time`, off for `ab`; ``extra.dispatch`` = true/false overrides. Runs after
    all timing, still under the job's GPU lock."""
    want = ((spec.params or {}).get("extra") or {}).get("dispatch")
    if want is None:
        want = kind == "time"
    if not want:
        return None, []
    try:
        from kdev.observe.dispatch import probe_isolated
    except Exception as e:  # noqa: BLE001
        return None, [f"dispatch probe unavailable: {e}"]
    from dataclasses import replace

    child = replace(spec, kind="dispatch", files=[f for f in spec.files if f.role != "base"][:1])
    out = os.path.join(str(getattr(run_dir, "path", run_dir)), "dispatch")
    t0 = time.time()
    extra = (spec.params or {}).get("extra") or {}
    disp, warns = probe_isolated(child, out, timeout_s=float(extra.get("dispatch_timeout_s", 300)))
    try:
        from kdev.timing._io import log

        log(run_dir, f"dispatch probe (child process) {time.time() - t0:.1f}s, {len(disp)} shapes")
    except Exception:  # noqa: BLE001
        pass
    return disp, warns


def _clocks_moved(ss: Session) -> bool:
    from kdev.timing._env import clocks_moved

    return clocks_moved(ss.clocks_before, ss.clocks_after)


def run_time(spec: JobSpec, run_dir: Any) -> Result:
    """Worker entry for kind='time' (one file vs the baseline unless params.base == 'none')."""
    return _run(spec, run_dir, "time")


def run_ab(spec: JobSpec, run_dir: Any) -> Result:
    """Worker entry for kind='ab' (N files interleaved; decisions on absolute times, M6)."""
    return _run(spec, run_dir, "ab")
