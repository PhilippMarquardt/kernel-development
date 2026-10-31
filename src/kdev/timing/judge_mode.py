"""Judge mode (``--judge-mode``, protocol ``kdev-tp/1+jm``): the OKC judge's per-run loop, one submission per
process. Owner: P3. Semantics of rt2_'s ``rt2/jf.py`` (+ ``m1.sh`` ABBA processes, ``cmp.py`` deltas),
board #49; docs/PROTOCOL.md section 9.

time (one process, this module's :func:`time_shapes_judge`):
  - the SGLang baseline is loaded and prepared FIRST (judge ``setup.first = baseline``; ``extra.first =
    submission`` flips it), each prepare preceded by a random 8-32 MB pad (judge ``prepare_pad``, kept alive);
  - per shape: inputs from the judge seed (``--judge-json`` takes the per-case seeds of a judge record),
    a pool of ``fresh`` (8) inputs seed + 7919*j, graphs captured submission first, one warm-up replay each;
  - ``reps`` (60) repetitions; every pair of reps (2k, 2k+1) gets the next pool input copied into the
    captured input tensors; even reps replay the baseline first, odd reps the submission first;
  - every replay: read flush (256 MB) + gate (``torch.cuda._sleep``, 235 us ~ judge gate_ns 300 us incl.
    the flush), events around ``g.replay()``;
  - 1- and 2-kernel graphs are replayed every 4th rep; per-shape overhead = 2*median(one) - median(two)
    (judge ``replay_overhead``) is subtracted from every replay of that shape;
  - reported: median RAW us per side, the per-shape overhead, judge pair speedup and sub = base / speedup.

ab (:func:`run_ab_judge`, the worker never touches CUDA itself): one child ``time --judge-mode`` worker
process per file in ABBA order (A B B A per round, distinct pad seeds), then per-shape deltas of the
median RAW us (mean over each file's processes; also per half-block) and of the judge-formula us, and the
judge score of every process vs its own in-process baseline.
"""
from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
import time
from typing import Any, Callable

from kdev.contract import ErrorCode, Fidelity, KdevError, KdevErrorInfo, Result
from kdev.models import FileRef, Interval, JobSpec, Sample
from kdev.timing import stats

PROTOCOL_TAG = "kdev-tp/1+jm"
GATE_US = 235.0                 # + ~65 us read flush = judge replay_overhead.gate_ns (~299 us)
SM_MHZ = 1980                   # H200 max SM clock: torch.cuda._sleep cycles per us
FRESH = 8                       # judge: fresh inputs every pair; jf.py pool of 8 seeded inputs
FRESH_STRIDE = 7919
POOL_BUDGET_BYTES = 2 << 30
REPS = 60                       # judge runs per bf16/attention shape
OVH_EVERY = 4
N_BOOT = 1000
BASE = "base"


def enabled(params: Any) -> bool:
    extra = params.get("extra") if isinstance(params, dict) else getattr(params, "extra", None)
    return bool((extra or {}).get("judge_mode"))


def gate_cycles(gate_us: float) -> int:
    return int(float(gate_us) * SM_MHZ)


# ------------------------------------------------------------------------------------------- one process
def _median(xs: list[float]) -> float:
    return stats.median(xs) if xs else float("nan")


def time_shapes_judge(ss: Any) -> None:
    """Judge-loop timing of every shape of an open engine Session (labels: the submission(s) + ``base``).
    Fills ss.samples (us = raw - per-shape overhead), ss.checks and ss.shape_overhead; raw.jsonl per shape."""
    import torch

    from kdev.timing import engine as E
    from kdev.timing import graph as G
    from kdev.timing._io import append_jsonl

    extra = ss.params.extra or {}
    k_fresh = int(extra.get("fresh", FRESH))
    reps = int(ss.params.reps)
    labels = [f.label for f in ss.files]
    subs = [lb for lb in labels if lb != BASE]
    base = [lb for lb in labels if lb == BASE]
    cap_order = subs + base                                   # jf.py: sub captured first
    even = (base + subs) if extra.get("first", "baseline") == "baseline" else (subs + base)
    tiny = torch.zeros(1, device="cuda")

    def kn(n: int) -> Callable[[], None]:
        def f() -> None:
            for _ in range(n):
                tiny.add_(1)
        return f

    g1, g2 = G.capture(kn(1)), G.capture(kn(2))
    try:
        for i, sh in enumerate(ss.shapes):
            seed = int(E.seeds_for(ss, sh)[0])
            ss.seeds_used[sh.shape_id] = [seed]
            d = ss.adapter.make_inputs(sh, seed)
            dst = E._tensors(d)
            nbytes = sum(t.numel() * t.element_size() for t in dst)
            k = min(k_fresh, max(1, POOL_BUDGET_BYTES // max(1, nbytes))) if k_fresh > 1 else 0
            pool: list[list[Any]] = []
            for j in range(k):
                src = E._tensors(ss.adapter.make_inputs(sh, seed + FRESH_STRIDE * j))
                if len(src) != len(dst) or any(a.shape != b.shape or a.dtype != b.dtype for a, b in zip(src, dst)):
                    ss.warnings.append(f"{sh.shape_id}: fresh inputs change shape/dtype; fixed inputs used")
                    pool = []
                    break
                pool.append(src)
            modules = {lb: ss.modules[lb] for lb in cap_order}
            t = E.TorchTimer(ss.adapter, modules, ss.states, sh, seed, ss.params, flusher=ss.flusher,
                             overhead=ss.overhead, select=ss.select, inputs=d)
            try:
                for lb in cap_order:
                    t.replay_us(lb)
                rec: list[tuple[str, int, int, float]] = []
                o1: list[float] = []
                o2: list[float] = []
                for r in range(reps):
                    if pool and r % 2 == 0:
                        for a, b in zip(dst, pool[(r // 2) % len(pool)]):
                            a.copy_(b)
                    seq = even if r % 2 == 0 else even[::-1]
                    for pos, lb in enumerate(seq):
                        rec.append((lb, r, pos, t.replay_us(lb)))
                    if r % OVH_EVERY == 0:
                        o1.append(G.time_replay(g1, ss.flusher))
                        o2.append(G.time_replay(g2, ss.flusher))
                one, two = _median(o1), _median(o2)
                ovh = 2 * one - two
                smp = [Sample(shape_id=sh.shape_id, label=lb, seed=seed, rep=r, order_pos=pos, us_raw=raw,
                              us=raw - ovh) for lb, r, pos, raw in rec]
                if pool:                                       # verify on the judge-seed input
                    for a, b in zip(dst, pool[0]):
                        a.copy_(b)
                for lb in cap_order:
                    info = t.verify(lb)
                    ok = info.get("ok")
                    prev = ss.checks.get((sh.shape_id, lb))
                    ss.checks[(sh.shape_id, lb)] = (False if (prev is False or ok is False)
                                                    else (ok if prev is None else prev))
                    if ok is False:
                        ss.check_msgs.append(dict(info, shape_id=sh.shape_id))
            finally:
                t.close()
                del t, pool, dst, d
                torch.cuda.empty_cache()
            ss.shape_overhead[sh.shape_id] = {"one_us": one, "two_us": two, "overhead_us": ovh,
                                              "n": len(o1), "fresh": k}
            ss.samples.extend(smp)
            append_jsonl(ss.run_dir, "raw.jsonl", [s.to_dict() for s in smp])
            raw = {lb: _median([s.us_raw for s in smp if s.label == lb]) for lb in cap_order}
            ss.log(f"[{i + 1}/{len(ss.shapes)}] {sh.shape_id} ovh {ovh:.2f} (one {one:.2f}) raw "
                   + " ".join(f"{lb}={v:.2f}" for lb, v in raw.items()))
    except KdevError as e:
        ss.error = e.info()
        ss.log(f"stopped: {e.code.value}: {e.message} (partial samples kept)")


def annotate_time(ss: Any, rows: list[dict[str, Any]], fid: Fidelity) -> None:
    """Add median RAW us per file and the per-shape overhead to the rows; fidelity shows the median overhead."""
    raw: dict[tuple[str, str], list[float]] = {}
    for s in ss.samples:
        raw.setdefault((s.shape_id, s.label), []).append(s.us_raw)
    for r in rows:
        r["overhead"] = ss.shape_overhead.get(r["shape_id"])
        for p in r.get("per_file") or []:
            p["median_raw_us"] = _median(raw.get((r["shape_id"], p["label"]), []))
    ovs = [v["overhead_us"] for v in ss.shape_overhead.values() if v.get("overhead_us") is not None]
    if ovs:
        fid.overhead_us = dict(fid.overhead_us or {}, subtracted=_median(ovs), per_shape=True,
                               min=min(ovs), max=max(ovs))


def _f(x: Any, w: int = 7, nd: int = 2) -> str:
    if x is None or not isinstance(x, (int, float)) or not math.isfinite(x):
        return "-".rjust(w)
    return f"{x:{w}.{nd}f}"


def render_time(verdict: str, rows: list[dict[str, Any]], score: dict[str, Any] | None, max_rows: int = 30
                ) -> list[str]:
    head = f"{verdict.upper() or 'DONE'}  time --judge-mode"
    if score and score.get("delta"):
        d = score["delta"]
        head += f"  score x{d['median']:.4f} [{d['ci_lo']:.4f}, {d['ci_hi']:.4f}] vs SGLang"
    lines = [head, "shape".ljust(22) + "  subRAW baseRAW    ovh   sub us  base us   speedup [95% CI]       verdict"]
    for r in rows[:max_rows]:
        pf = {p["label"]: p for p in r.get("per_file") or []}
        sub = next((p for lb, p in pf.items() if lb != BASE), {})
        bp = pf.get(BASE, {})
        rt = r.get("ratio") or {}
        sp = f"x{rt['speedup']:.3f} [{rt['speedup_ci'][0]:.3f}, {rt['speedup_ci'][1]:.3f}]" if rt.get("speedup") else "-"
        lines.append(r["shape_id"][:22].ljust(22) + _f(sub.get("median_raw_us"), 8) + _f(bp.get("median_raw_us"), 8)
                     + _f((r.get("overhead") or {}).get("overhead_us"), 7) + _f(rt.get("num_judge_us"), 9)
                     + _f(bp.get("median_us"), 9) + f"   {sp:<24} {r.get('verdict', '')}")
    if len(rows) > max_rows:
        lines.append(f"... {len(rows) - max_rows} more rows in table.md")
    return lines


# ---------------------------------------------------------------------------------- ab: ABBA processes
def proc_sequence(labels: list[str], rounds: int = 1) -> list[str]:
    """A B B A per round (A B C C B A for three files)."""
    out: list[str] = []
    for _ in range(max(1, int(rounds))):
        out += list(labels) + list(labels)[::-1]
    return out


def _child_spec(spec: JobSpec, f: FileRef, i: int, timeout_s: int) -> JobSpec:
    params = json.loads(json.dumps(spec.params or {}))
    extra = dict(params.get("extra") or {})
    extra.update(judge_mode=True, pad_seed=int(extra.get("pad_seed", 0) or 0) + i + 1, dispatch=False,
                 ab_proc=i)
    extra["base_first"] = extra.get("first", "baseline") == "baseline"
    params["extra"] = extra
    params["base"] = "baseline"
    child = FileRef(path=f.path, label=f.label, md5=f.md5, knobs=dict(f.knobs or {}), defines=list(f.defines or []),
                    role="sub")
    return JobSpec(kind="time", files=[child], params=params, problem=spec.problem, adapter=spec.adapter,
                   job_id=f"{spec.job_id or 'ab'}-p{i}", lane="gpu", agent=spec.agent, priority=spec.priority,
                   timeout_s=timeout_s, max_hold_s=timeout_s, run_name=f"p{i}_{f.label}",
                   requires_prebuilt=spec.requires_prebuilt, parent_job=spec.job_id or None)


def _run_child(child: JobSpec, d: str, timeout_s: float, python: str | None = None) -> tuple[dict[str, Any], str]:
    os.makedirs(d, exist_ok=True)
    jp = os.path.join(d, "job.json")
    with open(jp, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(child.to_dict()))
    env = dict(os.environ)
    with open(os.path.join(d, "stderr.txt"), "w", encoding="utf-8") as err:
        p = subprocess.Popen([python or sys.executable, "-m", "kdev.worker", jp, "--run-dir", d],
                             stdout=subprocess.PIPE, stderr=err, text=True, env=env)
        try:
            out, _ = p.communicate(timeout=max(30.0, timeout_s))
        except BaseException:
            p.kill()
            p.wait()
            raise
    line = next((ln for ln in reversed((out or "").splitlines()) if ln.strip().startswith("{")), "")
    try:
        return json.loads(line), d
    except ValueError:
        return {"status": "error", "error": {"code": "internal", "message": f"child rc={p.returncode}, no Result",
                                             "recovery": f"see {d}/stderr.txt"}}, d


def _load_samples(d: str) -> list[Sample]:
    p = os.path.join(d, "raw.jsonl")
    out: list[Sample] = []
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as fh:
            for ln in fh:
                if ln.strip():
                    out.append(Sample.from_dict(json.loads(ln)))
    return out


def proc_shape_data(samples: list[Sample], label: str) -> dict[str, dict[str, Any]]:
    """Per shape for one process: judge pair speedups vs base, base net median, sub/base raw medians."""
    from kdev.timing import engine as E

    out: dict[str, dict[str, Any]] = {}
    for sid, g in E._group(samples).items():
        if label not in g or BASE not in g:
            continue
        pairs = E.pair_speedups(g, label, BASE)
        if not pairs:
            continue
        sub_raw = [s.us_raw for s in g[label]]
        out[sid] = {"pairs": pairs, "base_us": stats.median([s.us for s in g[BASE]]),
                    "sub_raw": stats.median(sub_raw), "base_raw": stats.median([s.us_raw for s in g[BASE]]),
                    "ovh": g[label][0].us_raw - g[label][0].us, "sub_raw_runs": sub_raw}
    for v in out.values():
        v["speedup"] = stats.median(v["pairs"])
        v["sub_us"] = v["base_us"] / v["speedup"]
    return out


def _mean(xs: list[float]) -> float:
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def aggregate(procs: list[tuple[str, dict[str, dict[str, Any]]]], labels: list[str], shapes: list[Any],
              score_fn: Callable[[dict[str, float], dict[str, float]], float], noise_floor: float,
              n_boot: int = N_BOOT, seed: int = 0) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    """ABBA processes [(label, per-shape data)] -> rows (per-shape deltas vs labels[0]), score block, verdict."""
    ref = labels[0]
    sids = [getattr(s, "shape_id", s) for s in shapes]
    sids = [s for s in sids if all(s in d for _, d in procs)]
    idx = {lb: [i for i, (l2, _) in enumerate(procs) if l2 == lb] for lb in labels}
    n = len(labels)
    blocks = [list(range(b, b + 2 * n)) for b in range(0, len(procs) - 2 * n + 1, 2 * n)]

    def halves(lb: str, val: Callable[[int], float]) -> list[float]:
        """Per half-block: val(X proc) - val(ref proc) (forward half, then backward half of each block)."""
        hs = []
        for blk in blocks:
            for half in (blk[:n], blk[n:]):
                xi = next(i for i in half if procs[i][0] == lb)
                ri = next(i for i in half if procs[i][0] == ref)
                hs.append(val(xi) - val(ri))
        return hs

    def scores(sub: dict[int, dict[str, float]]) -> dict[int, float]:
        return {i: score_fn({s: sub[i][s] for s in sids}, {s: procs[i][1][s]["base_us"] for s in sids})
                for i in range(len(procs))}

    point_sub = {i: {s: procs[i][1][s]["sub_us"] for s in sids} for i in range(len(procs))}
    point_sc = scores(point_sub)
    rng = random.Random(seed)
    boot_ratio: dict[str, dict[str, list[float]]] = {lb: {s: [] for s in sids} for lb in labels[1:]}
    boot_score: dict[str, list[float]] = {lb: [] for lb in labels[1:]}
    for _ in range(n_boot):
        sub = {}
        for i, (_, d) in enumerate(procs):
            sub[i] = {}
            for s in sids:
                pr = d[s]["pairs"]
                sub[i][s] = d[s]["base_us"] / stats.median(rng.choices(pr, k=len(pr)))
        sc = scores(sub)
        for lb in labels[1:]:
            for s in sids:
                boot_ratio[lb][s].append(_mean([sub[i][s] for i in idx[ref]]) / _mean([sub[i][s] for i in idx[lb]]))
            boot_score[lb].append(_mean([sc[i] for i in idx[lb]]) / _mean([sc[i] for i in idx[ref]]))

    rows: list[dict[str, Any]] = []
    sh_by = {getattr(s, "shape_id", s): s for s in shapes}
    for s in sids:
        per_file = []
        for lb in labels:
            ii = idx[lb]
            per_file.append({"label": lb, "median_us": _mean([point_sub[i][s] for i in ii]),
                             "median_raw_us": _mean([procs[i][1][s]["sub_raw"] for i in ii]),
                             "procs_raw_us": [procs[i][1][s]["sub_raw"] for i in ii],
                             "base_raw_us": _mean([procs[i][1][s]["base_raw"] for i in ii]),
                             "overhead_us": _mean([procs[i][1][s]["ovh"] for i in ii])})
        per_file.append({"label": BASE, "median_us": _mean([procs[i][1][s]["base_us"] for i in range(len(procs))]),
                         "median_raw_us": _mean([procs[i][1][s]["base_raw"] for i in range(len(procs))])})
        pf = {p["label"]: p for p in per_file}
        ratios = []
        for lb in labels[1:]:
            sp_pt = pf[ref]["median_us"] / pf[lb]["median_us"]
            bs = boot_ratio[lb][s]
            iv = Interval(sp_pt, stats.quantile(bs, 0.025), stats.quantile(bs, 0.975), len(procs))
            hraw = halves(lb, lambda i: procs[i][1][s]["sub_raw"])
            hsp = halves(lb, lambda i: -math.log(point_sub[i][s]))     # >0: lb faster in that half
            v = stats.verdict(iv, noise_floor, stats.ratio_se(iv))
            if v in ("faster", "slower") and not (all(h > 0 for h in hsp) or all(h < 0 for h in hsp)):
                v = "same"                                          # half-blocks disagree: placement noise
            ratios.append({"num": lb, "den": ref, "median": 1 / sp_pt, "ci_lo": 1 / iv.ci_hi, "ci_hi": 1 / iv.ci_lo,
                           "speedup": sp_pt, "speedup_ci": [iv.ci_lo, iv.ci_hi], "n_pairs": len(procs),
                           "num_judge_us": pf[lb]["median_us"], "abs_delta_us": pf[lb]["median_us"] - pf[ref]["median_us"],
                           "delta_raw_us": pf[lb]["median_raw_us"] - pf[ref]["median_raw_us"],
                           "delta_raw_halves_us": hraw, "verdict": v})
        sh = sh_by.get(s)
        rows.append({"shape_id": s, "weight": float(getattr(sh, "weight", 1.0) or 1.0),
                     "load": str(getattr(sh, "load", "") or ""), "per_file": per_file,
                     "ratio": ratios[0] if ratios else None, "ratios": ratios,
                     "verdict": ratios[0]["verdict"] if ratios else "", "check_status": None, "warnings": []})
    per_file_sc: dict[str, float] = {}
    ci: dict[str, list[float]] = {}
    half_sc: dict[str, list[float]] = {}
    for lb in labels[1:]:
        per_file_sc[lb] = _mean([point_sc[i] for i in idx[lb]]) / _mean([point_sc[i] for i in idx[ref]])
        ci[lb] = [stats.quantile(boot_score[lb], 0.025), stats.quantile(boot_score[lb], 0.975)]
        half_sc[lb] = [math.exp(h) for h in halves(lb, lambda i: math.log(point_sc[i]))]
    vs_base = {lb: _mean([point_sc[i] for i in idx[lb]]) for lb in labels}
    score: dict[str, Any] = {"ref": ref, "per_file": per_file_sc, "ci": ci, "ci_per_file": ci, "halves": half_sc,
                             "vs_base": {"ref": BASE, "per_file": vs_base,
                                         "per_proc": [{"label": procs[i][0], "score": point_sc[i]}
                                                      for i in range(len(procs))]},
                             "n_shapes": len(sids), "n_procs": len(procs)}
    verdict = ""
    if len(labels) > 1:
        lb = labels[1]
        score["delta"] = {"num": lb, "den": ref, "median": per_file_sc[lb], "ci_lo": ci[lb][0], "ci_hi": ci[lb][1]}
        iv = Interval(per_file_sc[lb], ci[lb][0], ci[lb][1])
        verdict = stats.verdict(iv, noise_floor, stats.ratio_se(iv))
        hs = half_sc[lb]
        if verdict in ("faster", "slower") and not (all(h > 1 for h in hs) or all(h < 1 for h in hs)):
            verdict = "same"
    return rows, score, verdict


def render_ab(verdict: str, rows: list[dict[str, Any]], score: dict[str, Any], labels: list[str], seq: list[str],
              max_rows: int = 30) -> list[str]:
    ref = labels[0]
    head = f"{verdict.upper() or 'DONE'}  ab --judge-mode ({' '.join(seq)} processes)"
    if score.get("delta"):
        d = score["delta"]
        hs = " ".join(f"x{h:.4f}" for h in score["halves"].get(d["num"], []))
        head += f"  {d['num']}/{d['den']} score x{d['median']:.4f} [{d['ci_lo']:.4f}, {d['ci_hi']:.4f}] halves {hs}"
    vb = score.get("vs_base", {}).get("per_file", {})
    lines = [head, "vs SGLang: " + "  ".join(f"{lb} x{vb[lb]:.4f}" for lb in labels if lb in vb)]
    lb = labels[1] if len(labels) > 1 else ref
    lines.append("shape".ljust(22) + f"{ref + ' RAW':>8}{lb + ' RAW':>8}   dRAW (halves)    "
                 f"{ref + ' us':>7}{lb + ' us':>8}  {lb}/{ref} speedup [95% CI]  verdict")
    for r in rows[:max_rows]:
        pf = {p["label"]: p for p in r["per_file"]}
        rt = r.get("ratio") or {}
        hv = ",".join(f"{h:+.2f}" for h in rt.get("delta_raw_halves_us", []))
        sp = f"x{rt['speedup']:.3f} [{rt['speedup_ci'][0]:.3f}, {rt['speedup_ci'][1]:.3f}]" if rt.get("speedup") else "-"
        lines.append(r["shape_id"][:22].ljust(22) + _f(pf[ref]["median_raw_us"], 8) + _f(pf[lb]["median_raw_us"], 8)
                     + f"  {rt.get('delta_raw_us', float('nan')):+5.2f} ({hv})".ljust(20)
                     + _f(pf[ref]["median_us"], 7) + _f(pf[lb]["median_us"], 8) + f"  {sp:<24} {r.get('verdict', '')}")
    if len(rows) > max_rows:
        lines.append(f"... {len(rows) - max_rows} more rows in table.md")
    return lines


def run_ab_judge(spec: JobSpec, run_dir: Any) -> Result:
    """Worker entry for ``ab --judge-mode``: ABBA child processes (each a judge-mode ``time`` of one file vs
    the in-process SGLang baseline), aggregated per shape. The parent never initialises CUDA."""
    from kdev.timing import engine as E
    from kdev.timing._io import log, run_id, run_path, write_json

    if len(spec.files) < 2:
        raise KdevError(ErrorCode.INVALID_REQUEST, "ab needs two or more files", recovery="kdev ab A.py B.py")
    t0 = time.time()
    extra = (spec.params or {}).get("extra") or {}
    labels = [f.label for f in spec.files]
    by_label = {f.label: f for f in spec.files}
    seq = proc_sequence(labels, int(extra.get("rounds", 1) or 1))
    rp = str(run_path(run_dir))
    procs: list[tuple[str, dict[str, dict[str, Any]]]] = []
    children: list[dict[str, Any]] = []
    fid: Fidelity | None = None
    warnings: list[str] = []
    excl = []
    deadline = t0 + float(spec.timeout_s or 1800)
    for i, lb in enumerate(seq):
        left = deadline - time.time()
        if left < 60:
            raise KdevError(ErrorCode.TIMEOUT, f"ab --judge-mode: no time left for process {i + 1}/{len(seq)}",
                            recovery="raise --timeout or time fewer shapes (--shapes)")
        child = _child_spec(spec, by_label[lb], i, int(left - 20))
        d = os.path.join(rp, f"p{i}_{lb}")
        log(run_dir, f"process {i + 1}/{len(seq)}: {lb} ({os.path.basename(by_label[lb].path)})")
        res, d = _run_child(child, d, left - 10)
        children.append({"proc": i, "label": lb, "dir": d, "run_id": res.get("run_id"), "status": res.get("status"),
                         "verdict": res.get("verdict"), "prepare_s": (res.get("summary") or {}).get("prepare_s"),
                         "pad_bytes": (res.get("summary") or {}).get("pad_bytes")})
        if res.get("error"):
            e = res["error"]
            r = Result(tool="ab", status="error", verdict="error", run_id=run_id(run_dir), job_id=spec.job_id or None,
                       error=KdevErrorInfo(e.get("code", "internal"), f"process {i} ({lb}): {e.get('message', '')}"[:400],
                                           details={"child": d}, recovery=e.get("recovery") or f"see {d}/log.txt"),
                       summary={"children": children})
            return r
        warnings += [f"p{i} {lb}: {w}" for w in (res.get("warnings") or [])[:3]]
        if res.get("fidelity"):
            f0 = res["fidelity"]
            excl.append(f0.get("exclusive", "unknown"))
            if fid is None:
                fid = Fidelity(**{k: v for k, v in f0.items() if k in Fidelity.__dataclass_fields__})
        data = proc_shape_data(_load_samples(d), lb)
        procs.append((lb, data))
        if res.get("status") == "fail":
            warnings.append(f"p{i} {lb}: check FAILED ({res.get('verdict')}), see {d}")
    shapes: list[Any] = []
    try:
        ad, _, _ = E.resolve_adapter(spec)
        shapes = [s for s in ad.shapes() if getattr(s, "scored", True)]
        score_fn = lambda sub, base: ad.score(sub, base)  # noqa: E731
    except Exception:  # noqa: BLE001
        ad = None
    common = [sid for sid in procs[0][1] if all(sid in d for _, d in procs)]
    known = {getattr(s, "shape_id", None) for s in shapes}
    if not shapes or not set(common) <= known:
        from kdev.adapters.base import Shape

        shapes = [Shape(sid, {}) for sid in common]
        score_fn = E._fallback_score(shapes)
    shapes = [s for s in shapes if s.shape_id in set(common)]
    nf = float(extra.get("noise_floor", E.DEFAULT_NOISE_FLOOR))
    rows, score, verdict = aggregate(procs, labels, shapes, score_fn, nf, n_boot=int(extra.get("n_boot", N_BOOT)))
    if any(c.get("status") == "fail" for c in children):
        verdict = "incorrect"
    if fid is not None:
        worst = next((x for x in ("fail", "shared", "unknown", "waited") if x in excl), "ok")
        fid.exclusive = worst
    write_json(run_dir, "shapes.json", rows)
    E.write_table(run_dir, rows, labels + [BASE])
    res = Result(tool="ab", status="fail" if verdict == "incorrect" else "ok", verdict=verdict,
                 run_id=run_id(run_dir), job_id=spec.job_id or None, shapes=rows, score=score, fidelity=fid,
                 files={"table": os.path.join(rp, "table.md"), "log": os.path.join(rp, "log.txt")},
                 warnings=warnings[:20])
    res.summary = {"labels": labels, "ref": labels[0], "mode": "judge", "protocol": PROTOCOL_TAG, "procs": seq,
                   "children": children, "n_shapes": len(rows), "reps": (spec.params or {}).get("reps"),
                   "elapsed_s": round(time.time() - t0, 1),
                   "verdicts": {v: sum(1 for r in rows if r.get("verdict") == v)
                                for v in sorted({r.get("verdict", "") for r in rows})}}
    prob = spec.problem or (getattr(ad, "problem", None) if ad is not None else None)
    if prob:
        res.summary["problem"] = prob
    if score.get("delta"):
        res.summary["score_delta"] = score["delta"]
    res.lines = render_ab(verdict, rows, score, labels, seq)
    return res
