"""`kdev judge pair A [--base B] [--allow-check]`: same-hour judge pair of a candidate and its unmodified
base (LESSONS 3). Owner: P4.

Without --allow-check: prepare both, print the two exact `okc check` commands, exit 0 (no network).
With --allow-check (CLI only, never MCP): refuse duplicates, run both checks back to back via
okc_cli._okc_check(_allow=_ALLOW_CHECK), record both, report the judged A/B ratio per shape.

Rounds (``rounds=2``, default): round 1 sends base then candidate, round 2 candidate then base, each file of
round 2 being a comment-changed copy (judge prepare --copy) so the judge does not answer ``duplicate``. A
file whose exact hash was judged before is replaced by a fresh copy as well, so no duplicate hash is ever
sent. Both files of a round are queued back to back (``--no-wait``) and then awaited with
``okc status --wait``. Pooled result: per shape and overall, geomean(candidate speedups) / geomean(base
speedups) over the rounds. Verdict ``faster``/``slower`` needs the pooled ratio outside the judge noise band
(+-2 %) and every round agreeing in direction; otherwise ``same``. A cluster mix (sfo3 vs sfo3c, 1-2 %
offset) is warned about and prevents a faster/slower verdict.
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Callable

from kdev.contract import ErrorCode, KdevError, Result
from kdev.judge._parse import JRun, parse_judge
from kdev.ledger._util import md5_file
from kdev.util import Runner, run

_ALLOW_CHECK = object()   # identity token passed to okc_cli._okc_check
NOISE = 0.02


def _gmean(xs: list[float]) -> float | None:
    xs = [x for x in xs if isinstance(x, (int, float)) and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def pool_pair(base_runs: list[JRun], cand_runs: list[JRun], noise: float = NOISE) -> dict[str, Any]:
    """Pure: pooled candidate/base ratio per shape and overall from judged runs (index i = round i)."""
    ok_b = [r for r in base_runs if r.status == "ok"]
    ok_c = [r for r in cand_runs if r.status == "ok"]
    out: dict[str, Any] = {"n_base": len(ok_b), "n_cand": len(ok_c), "warnings": []}
    failed = [r.judge_id for r in cand_runs if r.status == "fail"]
    if failed:
        out.update(verdict="incorrect", ratio=None, shapes=[], rounds=[])
        out["warnings"].append(f"candidate failed the judge tests: {', '.join(failed)}")
        return out
    if not ok_b or not ok_c:
        out.update(verdict="error", ratio=None, shapes=[], rounds=[])
        out["warnings"].append("need at least one ok base and one ok candidate run")
        return out
    b_sp, c_sp = _gmean([r.speedup for r in ok_b]), _gmean([r.speedup for r in ok_c])
    ratio = c_sp / b_sp if (b_sp and c_sp) else None
    rounds = []
    for i, (b, c) in enumerate(zip(base_runs, cand_runs)):
        if b.status == "ok" and c.status == "ok" and b.speedup and c.speedup:
            rounds.append({"round": i + 1, "base": b.judge_id, "cand": c.judge_id, "base_speedup": b.speedup,
                           "cand_speedup": c.speedup, "ratio": c.speedup / b.speedup,
                           "clusters": sorted({b.cluster or "?", c.cluster or "?"})})
    shapes = []
    ids = [s.shape_id for s in ok_c[0].shapes]
    for sid in ids:
        bs = [s.speedup for r in ok_b for s in r.shapes if s.shape_id == sid]
        cs = [s.speedup for r in ok_c for s in r.shapes if s.shape_id == sid]
        gb, gc = _gmean(bs), _gmean(cs)
        rr = gc / gb if (gb and gc) else None
        spread = None
        if len(cs) > 1 and len(bs) > 1:
            per = [c / b for c, b in zip(cs, bs) if b and c]
            spread = (min(per), max(per)) if per else None
        shapes.append({"shape_id": sid, "base_speedup": gb, "cand_speedup": gc, "ratio": rr,
                       "round_min": spread[0] if spread else None, "round_max": spread[1] if spread else None,
                       "verdict": ("faster" if rr and rr > 1 + noise else "slower" if rr and rr < 1 - noise
                                   else "same")})
    clusters = sorted({r.cluster or "?" for r in ok_b + ok_c})
    if len(clusters) > 1:
        out["warnings"].append(f"cluster mix {'/'.join(clusters)}: offsets of 1-2 % between clusters; "
                               "ratio not decisive")
    agree_up = all(r["ratio"] > 1 for r in rounds) if rounds else False
    agree_dn = all(r["ratio"] < 1 for r in rounds) if rounds else False
    verdict = "same"
    if ratio is not None and len(clusters) == 1:
        if ratio > 1 + noise and agree_up:
            verdict = "faster"
        elif ratio < 1 - noise and agree_dn:
            verdict = "slower"
    if ratio is not None and verdict == "same" and abs(ratio - 1) > noise:
        out["warnings"].append("pooled ratio outside the noise band but rounds disagree / clusters differ")
    out.update(verdict=verdict, ratio=ratio, base_speedup=b_sp, cand_speedup=c_sp, rounds=rounds,
               shapes=shapes, clusters=clusters, noise=noise)
    return out


def plan_pair(candidate: str, base: str, conn: sqlite3.Connection | None, rounds: int = 2
              ) -> list[list[tuple[str, str]]]:
    """[[(role, path), (role, path)], ...] per round; never the same hash twice, never a judged hash."""
    from kdev.judge.prepare import fresh_copy

    used: set[str] = set()

    def pick(path: str, first: bool) -> str:
        md5 = md5_file(path)
        judged = conn is not None and conn.execute(
            "SELECT 1 FROM judge_results WHERE file_md5=?", (md5,)).fetchone() is not None
        if first and not judged and md5 not in used:
            used.add(md5)
            return path
        cp = fresh_copy(path, conn, avoid=used)
        used.add(md5_file(cp))
        return cp

    plan = []
    for r in range(rounds):
        order = [("base", base), ("cand", candidate)] if r % 2 == 0 else [("cand", candidate), ("base", base)]
        plan.append([(role, pick(p, first=(r == 0))) for role, p in order])
    return plan


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def run_pair(candidate: str, base: str | None, problem: str | None, *, allow_check: bool = False,
             rounds: int = 2, conn: sqlite3.Connection | None = None, runner: Runner = run,
             log: Callable[[str], None] = _stderr, out_dir: str | None = None) -> Result:
    from kdev.judge import okc_cli
    from kdev.judge.prepare import judge_command, prepare
    from kdev.judge.record import record_parsed
    from kdev.util import new_id

    tool = "judge pair"
    if not base:
        raise KdevError(ErrorCode.INVALID_REQUEST, "judge pair needs --base (the unmodified file the candidate "
                        "was derived from)", details={"field": "base"}, recovery="pass --base BASE.py")
    for p in (candidate, base):
        if not os.path.isfile(p):
            raise KdevError(ErrorCode.NOT_FOUND, f"no such file: {p}", details={"path": p})
    if md5_file(candidate) == md5_file(base):
        raise KdevError(ErrorCode.INVALID_REQUEST, "candidate and base are identical",
                        details={"field": "candidate"})
    if rounds < 1 or rounds > 4:
        raise KdevError(ErrorCode.INVALID_REQUEST, "rounds must be 1..4", details={"field": "rounds"})
    own = conn is None
    if own:
        from kdev.ledger.db import open_default
        conn = open_default()
    try:
        reps = {role: prepare(p, problem, conn=conn) for role, p in (("cand", candidate), ("base", base))}
        blocking = [f"{role}: {b}" for role, r in reps.items() for b in r.blocking]
        if blocking:
            raise KdevError(ErrorCode.INVALID_REQUEST, "files not ready for the judge: " + "; ".join(blocking),
                            details={"findings": blocking}, recovery="fix the findings (kdev judge prepare FILE)")
        prob = problem or reps["cand"].problem
        if reps["base"].problem and prob and reps["base"].problem != prob:
            raise KdevError(ErrorCode.INVALID_REQUEST, f"base is for {reps['base'].problem}, candidate for {prob}")
        plan = plan_pair(os.path.abspath(candidate), os.path.abspath(base), conn, rounds)
        cmds = [judge_command(p) for rnd in plan for _role, p in rnd]
        n = sum(len(r) for r in plan)
        if not allow_check:
            lines = [f"plan: {n} judge checks ({rounds} round(s), ABBA order), NOT run (no --allow-check)"]
            lines += [f"  {c}" for c in cmds]
            lines.append("record each: kdev judge record chk_<stem>.json --file <stem>.py")
            return Result(tool=tool, status="ok", verdict="planned", lines=lines,
                          summary={"problem": prob, "plan": [[{"role": r, "path": p} for r, p in rnd]
                                                             for rnd in plan], "commands": cmds},
                          next=["kdev judge pair CAND --base BASE --allow-check   (CLI only; uses quota)"])
        pair_id = new_id("p")
        log(f"kdev judge pair {pair_id}: running {n} `okc check` (quota!) in this order:")
        for c in cmds:
            log(f"  {c}")
        odir = Path(out_dir or Path(candidate).resolve().parent / f"kdev_pair_{pair_id}")
        odir.mkdir(parents=True, exist_ok=True)
        base_runs: list[JRun] = []
        cand_runs: list[JRun] = []
        job_ids: list[str] = []
        warnings: list[str] = []
        for rnd in plan:
            queued = []
            for role, path in rnd:
                d = okc_cli._okc_check(path, prob or "", _allow=_ALLOW_CHECK, runner=runner, wait=False)
                jid = d.get("id") or d.get("job") or d.get("job_id")
                if not jid:
                    raise KdevError(ErrorCode.INTERNAL, f"`okc check --no-wait` returned no job id for {path}",
                                    details={"response": {k: d[k] for k in list(d)[:8]}, "job_ids": job_ids})
                job_ids.append(str(jid))
                log(f"  queued {role} {os.path.basename(path)} -> {jid}")
                queued.append((role, path, str(jid)))
            for role, path, jid in queued:
                d = okc_cli.status(jid, runner=runner, wait=True)
                raw = odir / f"chk_{Path(path).stem}.json"
                raw.write_text(json.dumps(d, indent=1), encoding="utf-8")
                jr = parse_judge(d)
                record_parsed(conn, jr, file_path=path, raw_path=str(raw), label=os.path.basename(path),
                              role=role, pair_id=pair_id)
                if jr.duplicate_of:
                    warnings.append(f"{jid} answered duplicate of {jr.duplicate_of}")
                (base_runs if role == "base" else cand_runs).append(jr)
                log(f"  done {role} {jid}: {jr.status} x{jr.speedup or 0:.4f} {jr.cluster}")
        pooled = pool_pair(base_runs, cand_runs)
        warnings += pooled.pop("warnings", [])
        lines = [f"pair {pair_id}: cand/base x{pooled['ratio']:.4f} (base x{pooled['base_speedup']:.4f}, "
                 f"cand x{pooled['cand_speedup']:.4f}) noise +-{NOISE:.0%}" if pooled.get("ratio")
                 else f"pair {pair_id}: no pooled ratio"]
        for r in pooled.get("rounds", []):
            lines.append(f"  round {r['round']}: {r['base']} x{r['base_speedup']:.4f} / {r['cand']} "
                         f"x{r['cand_speedup']:.4f} -> x{r['ratio']:.4f} {','.join(r['clusters'])}")
        for s in pooled.get("shapes", [])[:15]:
            lines.append(f"  {s['shape_id'][:48]:48s} x{(s['ratio'] or 0):.4f} {s['verdict']}")
        status = "fail" if pooled["verdict"] == "incorrect" else "ok"
        return Result(tool=tool, status=status, verdict=pooled["verdict"], lines=lines, warnings=warnings,
                      summary={"pair_id": pair_id, "problem": prob, "job_ids": job_ids, "raw_dir": str(odir),
                               **pooled},
                      files={"raw_dir": str(odir)})
    finally:
        if own:
            conn.close()
