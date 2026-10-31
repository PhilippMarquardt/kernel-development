"""CLI commands of the service: serve, job *, queue *, lock run. Owner: P2."""
from __future__ import annotations

from kdev.contract import Effects
from kdev.registry import Arg, Registry, todo

P = "P2"


def register(reg: Registry) -> None:
    reg.command("serve", "GPU job service daemon: start|stop|status|run", todo("service.daemon.serve (P2)"),
                args=[Arg(("action",), "start|stop|status|run", choices=("start", "stop", "status", "run"),
                          nargs="?", default="status"),
                      Arg(("--build-workers",), "parallel compile workers", type="int", default=8),
                      Arg(("--foreground",), "do not daemonise", action="store_true")],
                effects=Effects(gpu="owns", writes=("queue", "run_dir", "ledger")), mcp=False, package=P)
    reg.command("job submit", "enqueue a JobSpec JSON file", todo("service.client.submit (P2)"),
                args=[Arg(("spec",), "path to kdev.job/1 JSON", type="path")],
                effects=Effects(gpu="queued", writes=("queue",)), package=P)
    reg.command("job status", "status of a job", todo("service.queue.get (P2)"),
                args=[Arg(("job_id",), "job id")], package=P)
    reg.command("job wait", "wait for a job and print its Result", todo("service.client.wait (P2)"),
                args=[Arg(("job_id",), "job id"),
                      Arg(("--timeout",), "seconds (default: forever)", type="float", default=None)],
                long_running=True, package=P)
    reg.command("job cancel", "cancel a queued/running job", todo("service.queue.cancel (P2)"),
                args=[Arg(("job_id",), "job id")], effects=Effects(writes=("queue",)), package=P)
    reg.command("job list", "list jobs", todo("service.queue.list_jobs (P2)"),
                args=[Arg(("--state",), "queued|running|done|failed|cancelled|timeout", default=None),
                      Arg(("--limit",), "max rows", type="int", default=30)], package=P)
    reg.command("job logs", "tail a job's log.txt", todo("service.client + runs.show (P2)"),
                args=[Arg(("job_id",), "job id"), Arg(("--tail",), "lines", type="int", default=40)],
                package=P)
    reg.command("queue status", "lock holder, queue, offenders, GPU state", todo("service.queue.snapshot (P2)"),
                package=P)
    reg.command("queue offenders", "unlocked CUDA processes seen while waiting",
                todo("service.queue.snapshot (P2)"),
                args=[Arg(("--since",), "window, e.g. 3h", default="3h")], package=P)
    reg.command("lock run", "run CMD under the exclusive GPU lock (gpu_run.sh semantics; deny-listed: okc)",
                todo("service.lock.run_locked (P2)"),
                args=[Arg(("--timeout",), "seconds", type="int", default=300),
                      Arg(("argv",), "-- CMD ...", nargs="...")],
                effects=Effects(gpu="owns"), mcp=False, package=P)
