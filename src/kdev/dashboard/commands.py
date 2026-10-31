"""CLI: dashboard build|watch|serve. Owner: P8."""
from __future__ import annotations

import os
import time
from argparse import Namespace
from pathlib import Path

from kdev.config import Config
from kdev.contract import Effects, Result
from kdev.registry import Arg, Context, Registry

P = "P8"


def _config(ctx: Context) -> Config:
    if ctx.config is not None:
        return ctx.config
    try:
        from kdev.config import load_config
        return load_config(home=ctx.home)
    except NotImplementedError:          # P1 not landed yet: env/default config
        cfg = Config()
        if ctx.home:
            cfg.home = Path(ctx.home).expanduser()
        return cfg


def _out_dir(cfg: Config, out: str | None) -> str:
    return out or cfg.dashboard_out or str(cfg.home / "dashboard")


def _build(args: Namespace, ctx: Context) -> Result:
    from kdev.dashboard.collect import collect
    from kdev.dashboard.render import write

    cfg = _config(ctx)
    team = getattr(args, "team", None) or os.environ.get("KDEV_TEAM") or None
    extra = [p for p in ([ctx.problem] if ctx.problem else []) +
             [x.strip() for x in os.environ.get("KDEV_BOARD_PROBLEMS", "").split(",")] if p]
    data = collect(cfg, scrape=bool(args.scrape), recent=getattr(args, "recent", 30) or 30, team=team,
                   extra_problems=extra)
    files = write(data, _out_dir(cfg, args.out))
    q = data.queue or {}
    lines = [f"OK  dashboard  problems={len(data.problems)} runs={len(data.recent_runs)} ideas={len(data.ideas)} "
             f"queued={len(q.get('queued') or [])} running={len(q.get('running') or [])}"]
    for p in data.problems[:15]:
        bj = (p.get("best_judged") or {}).get("speedup")
        g = p.get("gap") or {}
        lines.append(f"  {p['problem']:24s} judged {('x%.3f' % bj) if bj else '-':>7s}  "
                     f"#1 {('x%.3f' % g['board1']) if g.get('board1') else '-':>7s}  "
                     f"gap {('%.1f%%' % g['gap_to_1_pct']) if g.get('gap_to_1_pct') is not None else '-':>6s}  "
                     f"rank {g.get('our_rank') or g.get('rank_kind') or '-'}")
    if data.unverified_limit_claims:
        lines.append(f"  WARN {len(data.unverified_limit_claims)} unverified 'at the limit' claims (need sass+sweep)")
    lines.append(f"html: {files['html']}")
    return Result(tool="dashboard build", verdict="ok", files=files, warnings=list(data.warnings[:10]),
                  summary={"problems": len(data.problems), "recent_runs": len(data.recent_runs),
                           "ideas": len(data.ideas), "unverified_limit_claims": len(data.unverified_limit_claims),
                           "board": data.board is not None,
                           "gaps": {p["problem"]: (p.get("gap") or {}).get("gap_to_1_pct") for p in data.problems}},
                  next=["kdev dashboard serve"], lines=lines)


def cmd_build(args: Namespace, ctx: Context) -> Result:
    return _build(args, ctx)


def cmd_watch(args: Namespace, ctx: Context) -> Result:
    """Rebuild every --every seconds until interrupted (board scrape at most every 3 h unless --every larger)."""
    every = max(10, int(args.every or 300))
    last_scrape = 0.0
    r = None
    try:
        while True:
            want = bool(args.scrape) and time.time() - last_scrape >= 3 * 3600
            ns = Namespace(**{**vars(args), "scrape": want})
            r = _build(ns, ctx)
            if want:
                last_scrape = time.time()
            print(r.lines[0] if r.lines else "dashboard rebuilt", flush=True)
            time.sleep(every)
    except KeyboardInterrupt:
        pass
    return r or Result(tool="dashboard watch", verdict="ok")


def cmd_serve(args: Namespace, ctx: Context) -> Result:
    """Build once, then serve the output dir over HTTP on 127.0.0.1 (read-only static files)."""
    import functools
    import http.server

    r = _build(args, ctx)
    out = str(Path(r.files["html"]).parent)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=out)
    httpd = http.server.ThreadingHTTPServer((args.bind, int(args.port)), handler)
    print(f"serving {out} at http://{args.bind}:{httpd.server_address[1]}/ (Ctrl-C to stop)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return r


def register(reg: Registry) -> None:
    out = Arg(("--out",), "output dir (default $KDEV_HOME/dashboard)", type="path", default=None)
    scrape = Arg(("--scrape",), "include the read-only public board (GET only)", action="store_true")
    team = Arg(("--team",), "our board team name (default $KDEV_TEAM)", default=None)
    recent = Arg(("--recent",), "recent runs to list", type="int", default=30)
    reg.command("dashboard build", "write the static HTML dashboard (+ dashboard.json)", cmd_build,
                args=[out, scrape, team, recent],
                effects=Effects(writes=("files",), network="okc_readonly"), package=P)
    reg.command("dashboard watch", "rebuild the dashboard every N seconds", cmd_watch,
                args=[Arg(("--every",), "seconds", type="int", default=300), out, scrape, team, recent],
                effects=Effects(writes=("files",), network="okc_readonly"), long_running=True, mcp=False,
                package=P)
    reg.command("dashboard serve", "build, then serve the dashboard on localhost", cmd_serve,
                args=[Arg(("--port",), "TCP port (0 = any)", type="int", default=8765),
                      Arg(("--bind",), "bind address", default="127.0.0.1"), out, scrape, team, recent],
                effects=Effects(writes=("files",), network="okc_readonly"), long_running=True, mcp=False,
                package=P)
