"""CLI: ledger *, idea *, note. Owner: P4."""
from __future__ import annotations

from kdev.contract import Effects
from kdev.registry import Arg, Registry, todo

P = "P4"
W_LEDGER = Effects(writes=("ledger",))
W_IDEAS = Effects(writes=("ideas",))


def register(reg: Registry) -> None:
    reg.command("ledger query", "query recorded runs", todo("ledger.query.runs (P4)"),
                args=[Arg(("--file",), "file path (md5 lookup)", type="path", default=None),
                      Arg(("--md5",), "md5", default=None), Arg(("--tool",), "tool", default=None),
                      Arg(("--since",), "e.g. 6h or 2026-10-31", default=None),
                      Arg(("--limit",), "rows", type="int", default=30),
                      Arg(("--sql",), "read-only SELECT", default=None, mcp=False)], package=P)
    reg.command("ledger tried", "has this file / knob set / idea been tried?", todo("ledger.query.tried (P4)"),
                args=[Arg(("target",), "file path or free text"),
                      Arg(("--knobs",), "K=V (repeatable)", action="append", default=None)], package=P)
    reg.command("ledger best", "best judged + best local per problem", todo("ledger.query.best (P4)"),
                package=P)
    reg.command("ledger import", "ingest result.json files / run dirs", todo("ledger.record.ingest_result (P4)"),
                args=[Arg(("paths",), "result.json or run dirs", type="path", nargs="+")], effects=W_LEDGER,
                package=P)
    reg.command("ledger export", "export tables", todo("ledger.query.export (P4)"),
                args=[Arg(("--format",), "csv|jsonl", choices=("csv", "jsonl"), default="jsonl"),
                      Arg(("--table",), "table (repeatable)", action="append", default=None)], package=P)
    reg.command("idea add", "add a hypothesis to the board", todo("ledger.ideas.add (P4)"),
                args=[Arg(("title",), "short title"), Arg(("--hypothesis",), "what and why", default=""),
                      Arg(("--kind",), "idea|dead|limit|win", choices=("idea", "dead", "limit", "win"),
                          default="idea"),
                      Arg(("--gain",), "expected gain %", type="float", default=None),
                      Arg(("--tags",), "comma-separated", default=""),
                      Arg(("--variant",), "exact variant (required for dead)", default="")],
                effects=W_IDEAS, package=P)
    reg.command("idea list", "list ideas", todo("ledger.ideas.list_ideas (P4)"),
                args=[Arg(("--status",), "status", default=None), Arg(("--owner",), "owner", default=None),
                      Arg(("--kind",), "kind", default=None), Arg(("--limit",), "rows", type="int", default=30)],
                package=P)
    reg.command("idea show", "one idea with evidence", todo("ledger.ideas.get (P4)"),
                args=[Arg(("idea_id",), "id", type="int")], package=P)
    reg.command("idea claim", "claim an idea (owner = --agent)", todo("ledger.ideas.claim (P4)"),
                args=[Arg(("idea_id",), "id", type="int")], effects=W_IDEAS, package=P)
    reg.command("idea release", "release a claimed idea", todo("ledger.ideas.release (P4)"),
                args=[Arg(("idea_id",), "id", type="int")], effects=W_IDEAS, package=P)
    reg.command("idea evidence", "attach evidence (run/judge/file/url/text)", todo("ledger.ideas.add_evidence (P4)"),
                args=[Arg(("idea_id",), "id", type="int"), Arg(("--run",), "run id", default=None),
                      Arg(("--judge",), "judge job id", default=None),
                      Arg(("--file",), "file", type="path", default=None), Arg(("--url",), "url", default=None),
                      Arg(("--text",), "text", default=None),
                      Arg(("--supports",), "for|against|neutral", choices=("for", "against", "neutral"),
                          default="neutral")],
                effects=W_IDEAS, package=P)
    reg.command("idea resolve", "close an idea (gated for limit/dead)", todo("ledger.ideas.resolve (P4)"),
                args=[Arg(("idea_id",), "id", type="int"),
                      Arg(("--status",), "confirmed|refuted|abandoned",
                          choices=("confirmed", "refuted", "abandoned"), required=True),
                      Arg(("--resolution",), "one line", default="")],
                effects=W_IDEAS, package=P)
    reg.command("note", "record a claim (alias of idea add --kind; limit needs sass+sweep evidence)",
                todo("ledger.ideas.add (P4)"),
                args=[Arg(("claim",), "the claim"),
                      Arg(("--kind",), "dead|limit|win", choices=("dead", "limit", "win"), default="win"),
                      Arg(("--evidence",), "comma-separated run ids", default=""),
                      Arg(("--variant",), "exact variant", default="")],
                effects=W_IDEAS, package=P)
