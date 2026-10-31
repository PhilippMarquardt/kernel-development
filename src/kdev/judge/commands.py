"""CLI: judge prepare|record|ledger|pair, okc status|leaderboard (read-only). Owner: P4."""
from __future__ import annotations

from kdev.contract import Effects
from kdev.registry import Arg, Registry, todo

P = "P4"


def register(reg: Registry) -> None:
    reg.command("judge prepare", "validate a file for judging, warn on duplicates, print the command",
                todo("judge.prepare.prepare (P4)"), args=[Arg(("file",), "file", type="path")], package=P)
    reg.command("judge record", "ingest an `okc status --json` result into the ledger",
                todo("judge.record.record (P4)"),
                args=[Arg(("job_json",), "okc status/check JSON", type="path"),
                      Arg(("--file",), "the exact submitted file (for md5)", type="path", default=None)],
                effects=Effects(writes=("ledger",)), package=P)
    reg.command("judge ledger", "judged bests and same-hour baselines", todo("judge.record.ledger (P4)"),
                package=P)
    reg.command("judge pair", "same-hour judge pair candidate+base (prints commands unless --allow-check)",
                todo("judge.pair.run_pair (P4)"),
                args=[Arg(("candidate",), "candidate file", type="path"),
                      Arg(("--base",), "unmodified base file", type="path", default=None),
                      Arg(("--allow-check",), "ACTUALLY run `okc check` twice (consumes quota)",
                          action="store_true", mcp=False)],
                effects=Effects(network="okc_check", quota="explicit_flag", writes=("ledger",)), mcp=False,
                package=P)
    reg.command("okc status", "read-only `okc status JOB --json`", todo("judge.okc_cli.status (P4)"),
                args=[Arg(("job_id",), "judge job id")], effects=Effects(network="okc_readonly"), package=P)
    reg.command("okc leaderboard", "read-only public board", todo("judge.okc_cli.leaderboard (P4)"),
                effects=Effects(network="okc_readonly"), package=P)
