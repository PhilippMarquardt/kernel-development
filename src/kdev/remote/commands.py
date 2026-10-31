"""CLI: sync, remote, remote setup. Owner: P2."""
from __future__ import annotations

import dataclasses
import json
import sys
from argparse import Namespace

from kdev.contract import Effects, ErrorCode, KdevError, Result
from kdev.registry import Arg, Context, Registry
from kdev.service._common import get_config, parse_duration

P = "P2"


def _remote_cfg(args: Namespace, ctx: Context):
    cfg = get_config(ctx)
    if getattr(args, "host", None):
        cfg.remote.host = args.host
    if getattr(args, "port", None):
        cfg.remote.port = int(args.port)
    return cfg


def cmd_sync(args: Namespace, ctx: Context) -> Result:
    from kdev.remote import sync
    cfg = get_config(ctx)
    rep, items = sync.sync_with(cfg, sync._transport(cfg), args.direction, parse_duration(args.since), args.dry_run)
    acts: dict[str, int] = {}
    for i in items:
        acts[i.action] = acts.get(i.action, 0) + 1
    lines = [f"{args.direction}{' (dry run)' if args.dry_run else ''}: " + ", ".join(f"{k} {v}" for k, v in sorted(acts.items()))]
    if not args.dry_run:
        lines.append(f"copied {rep.copied} verified {rep.verified} bytes {rep.bytes} conflicts {len(rep.conflicts)} "
                     f"failed {len(rep.failed)}")
    lines += [f"{i.action:14s} {i.rel_path}" for i in items if i.action != "skip_same"][:12]
    lines += [f"CONFLICT kept as .conflict-*: {c}" for c in rep.conflicts[:3]]
    status = "fail" if rep.failed else "ok"
    return Result(tool="sync", status=status, verdict="md5 FAILED" if rep.failed else ("planned" if args.dry_run else "synced"),
                  summary={"report": dataclasses.asdict(rep), "actions": acts,
                           "items": [dataclasses.asdict(i) for i in items if i.action != "skip_same"][:500]},
                  lines=lines, warnings=[f"md5 verification failed: {f}" for f in rep.failed[:5]])


def cmd_remote(args: Namespace, ctx: Context) -> Result:
    from kdev.remote import ssh
    from kdev.service.lock import strip_dashdash
    cfg = _remote_cfg(args, ctx)
    argv = strip_dashdash(args.argv)
    if not argv:
        raise KdevError(ErrorCode.INVALID_REQUEST, "usage: kdev remote [--host H --port P] -- kdev CMD ...")
    if argv[0] == "kdev":
        argv = argv[1:]
    from kdev.remote.bootstrap import remote_python
    argv = [remote_python(), "-m", "kdev", *argv]   # no reliance on the pod's PATH
    if "--json" not in argv:
        argv.append("--json")            # parse the remote Result and render it locally
    p = ssh.run_remote(cfg.remote, argv)
    print(p.stderr[-4000:], file=sys.stderr, end="")
    try:
        res = Result.from_dict(json.loads(p.stdout.strip().splitlines()[-1]))
    except (ValueError, IndexError, TypeError):
        tail = p.stdout.strip().splitlines()[-30:]
        if p.returncode == 255:
            raise KdevError(ErrorCode.CAPABILITY_UNAVAILABLE, f"ssh to {cfg.remote.host}:{cfg.remote.port} failed",
                            details={"stderr": p.stderr[-300:]}, recovery="check host/port/key (pod restarted?)")
        return Result(tool="remote", status="ok" if p.returncode == 0 else "fail", verdict=f"exit {p.returncode}",
                      summary={"returncode": p.returncode}, lines=tail)
    if args.pull and res.run_id:
        local = ssh.pull_run(cfg.remote, res.run_id, str(cfg.runs_dir))
        res.files["local_dir"] = local
        res.next = [f"kdev show {res.run_id}"] + res.next[:1]
    return res


def cmd_remote_setup(args: Namespace, ctx: Context) -> Result:
    from kdev.remote import bootstrap
    cfg = _remote_cfg(args, ctx)
    out = bootstrap.setup(cfg, force=args.force)
    failed = [k for k, v in out.items() if v == "failed"]
    lines = [f"{k:16s} {v}" for k, v in out.items()]
    return Result(tool="remote setup", status="fail" if failed else "ok", verdict="failed" if failed else "ready",
                  summary=out, lines=lines)


def register(reg: Registry) -> None:
    reg.command("sync", "md5-verified copy pod<->local (pull|push|both)", cmd_sync,
                args=[Arg(("direction",), "pull|push|both", choices=("pull", "push", "both"), nargs="?",
                          default="pull"),
                      Arg(("--since",), "only files changed in this window, e.g. 3h", default=None),
                      Arg(("--dry-run",), "plan only", action="store_true")],
                effects=Effects(writes=("files", "remote"), network="ssh"), mcp=False, package=P)
    reg.command("remote", "run a kdev command on the pod over ssh", cmd_remote,
                args=[Arg(("--host",), "pod host", default=None), Arg(("--port",), "ssh port", type="int",
                                                                    default=None),
                      Arg(("--pull",), "pull the run dir afterwards", action="store_true"),
                      Arg(("argv",), "-- kdev ...", nargs="...")],
                effects=Effects(network="ssh"), mcp=False, package=P)
    reg.command("remote setup", "idempotent pod bootstrap (toolchain pin, venv, self-tests)", cmd_remote_setup,
                args=[Arg(("--force",), "re-run done phases", action="store_true"),
                      Arg(("--host",), "pod host", default=None),
                      Arg(("--port",), "ssh port", type="int", default=None)],
                effects=Effects(writes=("remote",), network="ssh"), mcp=False, package=P)
