"""CLI commands of the service: serve, job *, queue *, lock run. Owner: P2."""
from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

from kdev.contract import Effects, ErrorCode, KdevError, Result
from kdev.models import JobSpec, JobStatus
from kdev.registry import Arg, Context, Registry
from kdev.service._common import get_config, parse_duration

P = "P2"


def _job_row(j: JobStatus) -> str:
    pos = f"#{j.position}" if j.position else ""
    return (f"{j.job_id:24s} {j.lane:5s} {j.kind:9s} {j.state:9s}{pos:>4s} {j.agent[:12]:12s} "
            f"{(j.run_id or '')[:40]}")


def _jobs_lines(jobs: list[JobStatus], limit: int = 15) -> list[str]:
    lines = [f"{'job':24s} {'lane':5s} {'kind':9s} {'state':9s}{'':>4s} {'agent':12s} run"]
    lines += [_job_row(j) for j in jobs[:limit]]
    if len(jobs) > limit:
        lines.append(f"... {len(jobs) - limit} more (use --json or --limit)")
    return lines


# ---------------------------------------------------------------- serve

def cmd_serve(args: Namespace, ctx: Context) -> Result:
    from kdev.service import daemon, queue
    cfg = get_config(ctx)
    opts = daemon.DaemonOptions(build_workers=args.build_workers, foreground=args.foreground)
    action = args.action or "status"
    if action == "run" or (action == "start" and args.foreground):
        rc = daemon.serve(cfg, opts)
        return Result(tool="serve", status="ok" if rc == 0 else "error", verdict="stopped" if rc == 0 else "error",
                      summary={"exit": rc},
                      error=None if rc == 0 else KdevError(ErrorCode.CONFLICT, "daemon already running").info())
    if action == "start":
        pid = daemon.start_background(cfg, opts)
        alive = queue.daemon_alive(cfg)
        r = Result(tool="serve", verdict="running" if alive else "starting", summary={"pid": pid, "alive": alive},
                   lines=[f"kdev serve pid {pid} ({'alive' if alive else 'not yet heartbeating'}); "
                          f"log {Path(cfg.home) / 'serve.log'}"])
        if not alive:
            r.warnings.append("no heartbeat within 15 s: check serve.log")
        return r
    if action == "stop":
        ok = daemon.stop(cfg)
        if not ok:
            raise KdevError(ErrorCode.TIMEOUT, "daemon did not stop within 60 s",
                            recovery="it may be finishing a GPU job; check `kdev queue status`")
        return Result(tool="serve", verdict="stopped", lines=["kdev serve stopped"])
    hb = queue.heartbeat_row(cfg) or {}
    alive = queue.daemon_alive(cfg)
    return Result(tool="serve", verdict="running" if alive else "down", summary={"alive": alive, "heartbeat": hb},
                  lines=[f"daemon {'UP' if alive else 'DOWN'}  pid {hb.get('pid', '-')}  last heartbeat "
                         f"{hb.get('ts_utc', '-')}  gpu job {hb.get('gpu_lane_job') or '-'}  "
                         f"build workers {hb.get('build_workers', '-')}"],
                  next=[] if alive else ["kdev serve start"])


# ---------------------------------------------------------------- job *

def cmd_job_submit(args: Namespace, ctx: Context) -> Result:
    from kdev.service import client
    cfg = get_config(ctx)
    try:
        spec = JobSpec.from_dict(json.loads(Path(args.spec).read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError) as e:
        raise KdevError(ErrorCode.INVALID_REQUEST, f"cannot read job spec {args.spec}: {e}") from e
    spec.agent = spec.agent or ctx.agent
    st = client.submit(cfg, spec)
    return Result(tool="job submit", verdict="queued", job_id=st.job_id, summary=st.to_dict(),
                  lines=[f"job: {st.job_id}  lane {st.lane}  position {st.position}"],
                  next=[f"kdev job wait {st.job_id}"])


def cmd_job_status(args: Namespace, ctx: Context) -> Result:
    from kdev.service import queue
    st = queue.get(get_config(ctx), args.job_id)
    return Result(tool="job status", verdict=st.state, job_id=st.job_id, run_id=st.run_id, summary=st.to_dict(),
                  files={"dir": st.run_dir} if st.run_dir else {}, lines=_jobs_lines([st]))


def cmd_job_wait(args: Namespace, ctx: Context) -> Result:
    from kdev.service import client
    return client.wait(get_config(ctx), args.job_id, args.timeout)


def cmd_job_cancel(args: Namespace, ctx: Context) -> Result:
    from kdev.service import queue
    st = queue.cancel(get_config(ctx), args.job_id, ctx.agent)
    msg = {"cancelled": "cancelled", "running": "cancel requested (the daemon kills it)"}.get(st.state,
                                                                                            f"already {st.state}")
    return Result(tool="job cancel", verdict=st.state, job_id=st.job_id, summary=st.to_dict(),
                  lines=[f"job {st.job_id}: {msg}"])


def cmd_job_list(args: Namespace, ctx: Context) -> Result:
    from kdev.service import queue
    jobs = queue.list_jobs(get_config(ctx), args.state, args.limit)
    return Result(tool="job list", verdict=f"{len(jobs)} jobs", summary={"jobs": [j.to_dict() for j in jobs]},
                  lines=_jobs_lines(jobs))


def cmd_job_logs(args: Namespace, ctx: Context) -> Result:
    from kdev.service import queue
    st = queue.get(get_config(ctx), args.job_id)
    if not st.run_dir:
        return Result(tool="job logs", verdict=st.state, job_id=st.job_id, lines=[f"job {st.job_id} has no run dir yet ({st.state})"])
    log = Path(st.run_dir) / "log.txt"
    try:
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        lines = []
    n = max(1, min(int(args.tail or 40), 30))
    tail = lines[-n:]
    r = Result(tool="job logs", verdict=st.state, job_id=st.job_id, run_id=st.run_id,
               summary={"lines": len(lines), "tail": tail}, files={"log": str(log), "dir": st.run_dir},
               lines=([f"... {len(lines) - len(tail)} earlier lines in {log}"] if len(lines) > len(tail) else []) + tail)
    return r


# ---------------------------------------------------------------- queue *

def cmd_queue_status(args: Namespace, ctx: Context) -> Result:
    from kdev.service import queue
    snap = queue.snapshot(get_config(ctx))
    d, g = snap.daemon, snap.gpu
    lk = g.get("lock", {})
    if lk.get("held"):
        who = (f"{lk['pid']} {str(lk.get('cmdline', ''))[:60]}" if lk.get("pid") else
               "open by " + ", ".join(f"{o['pid']} {o['cmdline'][:40]}" for o in lk.get("openers", [])[:2]) or "?")
        lk_txt = "HELD by " + who
    else:
        lk_txt = "free"
    lines = [f"daemon {'UP' if d.get('alive') else 'DOWN'} pid {d.get('pid', '-')} hb {d.get('ts_utc', '-')}  "
             f"lock {lk_txt}"]
    st = g.get("state") or {}
    if st:
        lines.append(f"gpu {st.get('name', '?')} sm {st.get('sm_mhz')} MHz {st.get('power_w')} W {st.get('temp_c')} C "
                     f"util {st.get('util_pct')}%  foreign CUDA pids: "
                     + (", ".join(str(a['pid']) for a in g.get("foreign_apps", [])) or "none"))
    lines += [f"running {len(snap.running)}  queued {len(snap.queued)}  offenders(3h) {len(snap.offenders)}"]
    if snap.running or snap.queued:
        lines += _jobs_lines(snap.running + snap.queued, 12)
    for o in snap.offenders[:3]:
        lines.append(f"offender {o['ts_utc']} pid {o['pid']} {o['action']}: {(o.get('cmdline') or '')[:70]}")
    verdict = "busy" if snap.running else ("idle" if d.get("alive") else "down")
    return Result(tool="queue status", verdict=verdict, summary=snap.to_dict(), lines=lines,
                  next=[] if d.get("alive") else ["kdev serve start"])


def cmd_queue_offenders(args: Namespace, ctx: Context) -> Result:
    from kdev.service import queue
    since = parse_duration(args.since)
    offs = queue.offenders_since(get_config(ctx), since)
    lines = [f"{o['ts_utc']} pid {o['pid']:>7} {o['action']:8s} {o.get('gpu_mem_mib') or '-':>6} MiB "
             f"job {o.get('during_job') or '-'}: {(o.get('cmdline') or '')[:80]}" for o in offs[:15]]
    if len(offs) > 15:
        lines.append(f"... {len(offs) - 15} more (use --json)")
    return Result(tool="queue offenders", verdict=f"{len(offs)} offenders", summary={"offenders": offs,
                  "since_s": since}, lines=lines or ["no unlocked CUDA processes logged"])


# ---------------------------------------------------------------- lock run

def cmd_lock_run(args: Namespace, ctx: Context) -> Result:
    from kdev.service import lock
    cfg = get_config(ctx)
    argv = lock.strip_dashdash(args.argv)
    # with --json stdout must carry exactly one JSON object: the child's stdout goes to stderr
    rc = lock._run_locked(cfg, argv, args.timeout, child_stdout=sys.stderr if ctx.json else None)
    verdict = "ok" if rc == 0 else ("timeout" if rc == lock.EXIT_TIMEOUT else "fail")
    return Result(tool="lock run", status="ok" if rc == 0 else "fail", verdict=verdict,
                  summary={"returncode": rc, "argv": argv}, lines=[f"exit {rc}: {' '.join(argv)[:100]}"])


def register(reg: Registry) -> None:
    reg.command("serve", "GPU job service daemon: start|stop|status|run", cmd_serve,
                args=[Arg(("action",), "start|stop|status|run", choices=("start", "stop", "status", "run"),
                          nargs="?", default="status"),
                      Arg(("--build-workers",), "parallel compile workers (capped by CPU quota and 16)", type="int",
                          default=8),
                      Arg(("--foreground",), "do not daemonise", action="store_true")],
                effects=Effects(gpu="owns", writes=("queue", "run_dir", "ledger")), mcp=False, package=P)
    reg.command("job submit", "enqueue a JobSpec JSON file", cmd_job_submit,
                args=[Arg(("spec",), "path to kdev.job/1 JSON", type="path")],
                effects=Effects(gpu="queued", writes=("queue",)), package=P)
    reg.command("job status", "status of a job", cmd_job_status,
                args=[Arg(("job_id",), "job id")], package=P)
    reg.command("job wait", "wait for a job and print its Result", cmd_job_wait,
                args=[Arg(("job_id",), "job id"),
                      Arg(("--timeout",), "seconds (default: forever)", type="float", default=None)],
                long_running=True, package=P)
    reg.command("job cancel", "cancel a queued/running job", cmd_job_cancel,
                args=[Arg(("job_id",), "job id")], effects=Effects(writes=("queue",)), package=P)
    reg.command("job list", "list jobs", cmd_job_list,
                args=[Arg(("--state",), "queued|running|done|failed|cancelled|timeout", default=None),
                      Arg(("--limit",), "max rows", type="int", default=30)], package=P)
    reg.command("job logs", "tail a job's log.txt", cmd_job_logs,
                args=[Arg(("job_id",), "job id"), Arg(("--tail",), "lines (max 30)", type="int", default=30)],
                package=P)
    reg.command("queue status", "lock holder, queue, offenders, GPU state", cmd_queue_status, package=P)
    reg.command("queue offenders", "unlocked CUDA processes seen while waiting", cmd_queue_offenders,
                args=[Arg(("--since",), "window, e.g. 3h", default="3h")], package=P)
    reg.command("lock run", "run CMD under the exclusive GPU lock (gpu_run.sh semantics; deny-listed: okc)",
                cmd_lock_run,
                args=[Arg(("--timeout",), "seconds", type="int", default=300),
                      Arg(("argv",), "-- CMD ...", nargs="...")],
                effects=Effects(gpu="owns"), mcp=False, package=P)
