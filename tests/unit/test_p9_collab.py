"""P9 collab: board (posts, claims, inbox, threads, concurrency), docs (import/export round trip, append-only,
re-import), checklist, backup, CLI + MCP exposure. No GPU."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from kdev.cli import main
from kdev.collab import board, db, docs
from kdev.collab.refs import parse_ref
from kdev.contract import ErrorCode, KdevError
from kdev.registry import build_registry

SRC = Path(__file__).resolve().parents[2] / "src"
OKC = Path(os.environ.get("KDEV_TEST_OKC", r"C:\Users\pmarq\source\repos\arcagi3\okc"))
REAL = [OKC / "muse_bf16_proj" / "muse_bf16_proj" / "NOTES.md", OKC / "LESSONS.md", OKC / "TRACKER.md"]


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "collab.db")
    yield c
    c.close()


def _cli(capsys, *argv: str, agent: str = "alice") -> dict:
    os.environ["KDEV_AGENT"] = agent
    main([*argv, "--json"])
    out = capsys.readouterr().out.strip().splitlines()[-1]
    return json.loads(out)


# ------------------------------------------------------------------------------------------------- db
def test_connect_wal_migrated_idempotent(tmp_path):
    p = tmp_path / "c.db"
    c = db.connect(p)
    assert c.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert db.schema_version(c) == db.SCHEMA_VERSION
    c.close()
    c = db.connect(p)
    assert db.migrate(c) == db.SCHEMA_VERSION
    c.close()


def test_commands_registered_and_mcp_safe():
    from kdev.mcp.schema import command_exposed
    from kdev.mcp.server import build_tools
    reg = build_registry()
    names = set(reg.names())
    for n in ("board post", "board read", "board tail", "board inbox", "board ack", "board search", "board thread",
              "board claims", "board release", "board import", "doc append", "doc show", "doc toc", "doc list",
              "doc search", "doc checklist", "doc import", "doc export", "collab backup"):
        assert n in names
    for hidden in ("doc import", "doc export", "collab backup", "board import"):
        assert not command_exposed(reg.get(hidden))
    tools = {t.name: t for t in build_tools(reg, "compact")}
    assert set(tools["kdev_board"].commands) == {"post", "read", "inbox", "ack", "search", "thread", "claims",
                                                 "release"}
    assert "append" in tools["kdev_doc"].commands and "export" not in tools["kdev_doc"].commands
    assert "follow" not in tools["kdev_board"].input_schema["properties"]


# ---------------------------------------------------------------------------------------------- board
def test_post_validation(conn):
    with pytest.raises(KdevError) as e:
        board.post(conn, agent="", kind="finding", body="x")
    assert e.value.code == ErrorCode.INVALID_REQUEST
    with pytest.raises(KdevError):
        board.post(conn, agent="a", kind="gossip", body="x")
    with pytest.raises(KdevError) as e:
        board.post(conn, agent="a", kind="answer", body="x", reply_to=99)
    assert e.value.code == ErrorCode.NOT_FOUND


def test_claim_conflicts_globs_shapes_problem(conn):
    a, c1, _ = board.post(conn, agent="alice", kind="claim", body="m128", problem="muse_bf16_proj",
                          files=["bm_*.py"], shapes=["m128"])
    assert c1 == [] and a.claim_open and a.scope == ["file:bm_*.py", "shape:m128"]
    _, conf, _ = board.post(conn, agent="bob", kind="claim", body="x", problem="muse_bf16_proj",
                            files=["/root/okc/muse_bf16_proj/bm/bm_7.py"])
    assert [(c.post_id, ov) for c, ov in conf] == [(a.post_id, ["file:bm_7.py"])]
    _, conf, _ = board.post(conn, agent="bob", kind="claim", body="y", problem="muse_bf16_proj", shapes=["M128"])
    assert conf and conf[0][1] == ["shape:m128"]
    _, conf, _ = board.post(conn, agent="bob", kind="claim", body="z", problem="muse_moe", files=["bm_7.py"])
    assert conf == []                                   # other problem
    _, conf, _ = board.post(conn, agent="alice", kind="claim", body="mine", problem="muse_bf16_proj",
                            files=["bm_7.py"])
    assert all(c.agent != "alice" for c, _ in conf)     # own claims never conflict
    with pytest.raises(KdevError) as e:
        board.post(conn, agent="carol", kind="claim", body="x", problem="muse_bf16_proj", files=["bm_1.py"],
                   exclusive=True)
    assert e.value.code == ErrorCode.CONFLICT
    # the conflicting owner sees the overlapping claim in the inbox
    assert any(p.agent == "bob" and p.kind == "claim" for p in board.inbox(conn, "alice"))


def test_done_closes_claims_and_release(conn):
    c1, _, _ = board.post(conn, agent="alice", kind="claim", body="a", problem="p", files=["a.py"])
    c2, _, _ = board.post(conn, agent="alice", kind="claim", body="b", problem="p", files=["b.py"])
    c3, _, _ = board.post(conn, agent="alice", kind="claim", body="c", problem="p", files=["c.py"])
    _, _, closed = board.post(conn, agent="alice", kind="done", body="a done", problem="p", reply_to=c1.post_id)
    assert closed == [c1.post_id]
    _, _, closed = board.post(conn, agent="alice", kind="done", body="b done", problem="p", files=["b.py"])
    assert closed == [c2.post_id]
    with pytest.raises(KdevError) as e:
        board.release(conn, c3.post_id, "bob")
    assert e.value.code == ErrorCode.CONFLICT
    assert board.release(conn, c3.post_id, "alice").claim_state == "released"
    assert board.open_claims(conn) == []
    # expired claims do not conflict
    board.post(conn, agent="dave", kind="claim", body="old", problem="p", files=["z.py"],
               created_utc="2020-01-01T00:00:00Z", ttl_s=3600)
    assert board.conflicts_for(conn, "erin", "p", ["file:z.py"]) == []


def test_inbox_mentions_replies_ack_thread(conn):
    q, _, _ = board.post(conn, agent="alice", kind="ask", body="who owns the router?")
    r, _, _ = board.post(conn, agent="bob", kind="answer", body="me", reply_to=q.post_id)
    board.post(conn, agent="carol", kind="finding", body="@alice see run", refs=[parse_ref("20261031-101010-ab-x")])
    board.post(conn, agent="dave", kind="warn", body="stop", to="alice")
    board.post(conn, agent="dave", kind="finding", body="unrelated")
    board.post(conn, agent="erin", kind="finding", body="mail me at a@alice.com")   # not a mention
    ib = board.inbox(conn, "alice")
    assert [p.agent for p in ib] == ["bob", "carol", "dave"]
    assert board.ack(conn, "alice", [p.post_id for p in ib[:2]]) == 2
    assert [p.agent for p in board.inbox(conn, "alice")] == ["dave"]
    assert len(board.inbox(conn, "alice", include_acked=True)) == 3
    th = board.thread(conn, r.post_id)
    assert [p.post_id for p in th] == [q.post_id, r.post_id] and th[1].thread_id == q.post_id


def test_read_filters_unread_cursor_search(conn):
    for i in range(5):
        board.post(conn, agent="bob" if i % 2 else "alice", kind="finding" if i < 4 else "warn",
                   body=f"note {i} sk2 split", problem="muse_moe" if i == 0 else "muse_bf16_proj")
    ps, total = board.read(conn, problem="muse_bf16_proj")
    assert total == 4
    assert len(board.read(conn, kind="warn")[0]) == 1
    assert {p.agent for p in board.read(conn, agent="bob")[0]} == {"bob"}
    ps, _ = board.read(conn, limit=2)
    assert [p.body for p in ps] == ["note 3 sk2 split", "note 4 sk2 split"]    # newest 2, chronological
    first, _ = board.read(conn, unread_for="alice")
    assert {p.agent for p in first} == {"bob"}
    assert board.read(conn, unread_for="alice")[0] == []
    board.post(conn, agent="bob", kind="finding", body="new one")
    assert [p.body for p in board.read(conn, unread_for="alice")[0]] == ["new one"]
    assert len(board.search(conn, "sk2 split")) == 5
    assert len(board.search(conn, "sk2", problem="muse_moe")) == 1
    assert board.read(conn, since="23:59")[1] >= 0


def test_legacy_board_import_idempotent(tmp_path, conn):
    f = tmp_path / "BOARD.md"
    f.write_text("- 10-31 16:44 UTC [coordinator] FINDING: Interim board; moving into kdev.\n"
                 "- 10-31 16:45 UTC [pp_] CLAIM: pp_: cold-prepare speedup of stk/stk3b.py\n"
                 "  continued line\n- 10-31 16:50 UTC [x_] NOTE: odd kind\n", encoding="utf-8")
    assert board.import_legacy(conn, f, year=2026) == (3, 0)
    assert board.import_legacy(conn, f, year=2026) == (0, 3)
    ps, _ = board.read(conn)
    assert ps[0].created_utc == "2026-10-31T16:44:00Z" and ps[0].source == "import:BOARD.md"
    assert ps[1].kind == "claim" and "continued line" in ps[1].body
    assert ps[2].kind == "finding" and ps[2].body.startswith("[note]")


_WORKER = r"""
import sys
sys.path.insert(0, sys.argv[1])
from kdev.collab import board, db
c = db.connect(sys.argv[2])
who = sys.argv[3]
for i in range(int(sys.argv[4])):
    board.post(c, agent=who, kind="finding", body=f"{who} {i}")
try:
    board.post(c, agent=who, kind="claim", body="race", problem="p", files=["race.py"], exclusive=True)
    print("CLAIMED")
except Exception as e:
    print("REFUSED", getattr(e, "code", e))
"""


def test_concurrent_processes_post_and_claim(tmp_path):
    path = tmp_path / "shared" / "collab.db"
    db.connect(path).close()
    n_proc, n_posts = 6, 25
    procs = [subprocess.Popen([sys.executable, "-c", _WORKER, str(SRC), str(path), f"agent{i}", str(n_posts)],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for i in range(n_proc)]
    outs = [p.communicate(timeout=180) for p in procs]
    assert all(p.returncode == 0 for p in procs), [o[1][-500:] for o in outs]
    c = db.connect(path)
    n = c.execute("SELECT count(*) FROM posts WHERE kind='finding'").fetchone()[0]
    ids = [r[0] for r in c.execute("SELECT post_id FROM posts")]
    assert n == n_proc * n_posts and len(set(ids)) == len(ids)
    for i in range(n_proc):
        bodies = [r[0] for r in c.execute("SELECT body FROM posts WHERE agent=? AND kind='finding' ORDER BY post_id",
                                          (f"agent{i}",))]
        assert bodies == [f"agent{i} {k}" for k in range(n_posts)]
    assert sum(o[0].count("CLAIMED") for o in outs) == 1          # exactly one exclusive claim wins the race
    assert c.execute("SELECT count(*) FROM posts WHERE kind='claim'").fetchone()[0] == 1
    c.close()


# ----------------------------------------------------------------------------------------------- docs
MD = """# demo notes

Judge env: H200.

## Method
- autotune
### sub
text

```
## not a heading (code)
```

## What did not work
- x

## Method
dup heading
"""


def test_parse_md_and_round_trip(conn, tmp_path):
    title, secs = docs.parse_md(MD)
    assert title == "demo notes"
    assert [s.key for s in secs] == ["", "Method", "What did not work", "Method (2)"]
    assert "## not a heading" in secs[1].body
    f = tmp_path / "NOTES.md"
    f.write_bytes(MD.replace("\n", "\r\n").encode("utf-8"))
    rep = docs.import_md(conn, f, problem="demo")
    assert rep.doc == "notes/demo" and rep.created and len(rep.added) == 4
    assert docs.render(conn, "notes/demo") == MD
    out = tmp_path / "out" / "NOTES.md"
    assert docs.export_md(conn, "notes/demo", out)["status"] == "written"
    assert out.read_bytes() == MD.encode("utf-8")
    again = docs.import_md(conn, f, problem="demo")
    assert again.unchanged == 4 and not (again.added or again.appended or again.superseded)


def test_titleless_and_no_preamble_round_trip(conn, tmp_path):
    for text in ("## A\nx\n## B\ny", "# T\n## A\nx\n", "plain text only\n", ""):
        f = tmp_path / "x.md"
        f.write_text(text, encoding="utf-8")
        name = f"design/t{abs(hash(text)) % 10_000}"
        docs.import_md(conn, f, doc=name)
        assert docs.render(conn, name) == text


def test_append_only_reimport_keeps_lost_sections(conn, tmp_path):
    f = tmp_path / "NOTES.md"
    f.write_text(MD, encoding="utf-8")
    docs.import_md(conn, f, problem="demo")
    e = docs.append(conn, "notes/demo", "what did", "tile 256 x0.97", author="bS_",
                    refs=[parse_ref("20261031-101010-ab-t256"), parse_ref("#12")])
    assert e.section == "What did not work"            # prefix / case-insensitive match
    txt = docs.render(conn, "notes/demo")
    assert "- **" in txt and "bS_**: tile 256 x0.97" in txt and "refs: run:20261031-101010-ab-t256, #12" in txt
    # an agent's sync overwrites NOTES.md: one section gone, one appended at the end, one rewritten
    f.write_text("# demo notes\n\nJudge env: H200.\n\n## Method\n- autotune v2\n\n## What did not work\n- x\n"
                 "- y (new)\n", encoding="utf-8")
    rep = docs.import_md(conn, f, problem="demo")
    assert rep.superseded == ["Method", "What did not work"]
    assert "Method (2)" in rep.kept_missing
    after = docs.render(conn, "notes/demo")
    assert "dup heading" in after and "- autotune v2" in after and "### sub" not in after
    assert "tile 256 x0.97" in after                   # appended entry not in the file is still shown
    hist = docs.entries(conn, "notes/demo", "Method", history=True)
    assert any(h.superseded_by for h in hist) and "### sub" in hist[0].body
    # supersede an entry explicitly
    e2 = docs.append(conn, "notes/demo", "What did not work", "tile 256 x0.98 (rerun)", author="bS_",
                     supersedes=e.entry_id)
    assert "x0.97" not in docs.render(conn, "notes/demo") and e2.entry_id > e.entry_id


def test_export_refuses_hand_edits(conn, tmp_path):
    docs.append(conn, "lessons", "Brief checklist", "1. measure with kdev", author="me")
    out = tmp_path / "LESSONS.md"
    docs.export_md(conn, "lessons", out)
    docs.append(conn, "lessons", "Brief checklist", "2. sass first", author="me")
    assert docs.export_md(conn, "lessons", out)["status"] == "written"   # file unchanged since our export
    out.write_text(out.read_text(encoding="utf-8") + "hand edit\n", encoding="utf-8")
    with pytest.raises(KdevError) as e:
        docs.export_md(conn, "lessons", out)
    assert e.value.code == ErrorCode.CONFLICT and "doc import" in e.value.recovery
    assert docs.export_md(conn, "lessons", out, force=True)["status"] == "written"


def test_tracker_log_newest_first(conn, tmp_path):
    f = tmp_path / "TRACKER.md"
    f.write_text("# tracker\n\n## Log (newest first)\n- 14:08 old b\n- 13:00 old a\n", encoding="utf-8")
    docs.import_md(conn, f)
    docs.append(conn, "tracker", "Log", "15:00 new", author="coord")
    txt = docs.render(conn, "tracker")
    assert txt.index("15:00 new") < txt.index("14:08 old b")
    f.write_text("# tracker\n\n## Log (newest first)\n- 16:00 hand\n" + txt.split("## Log (newest first)\n", 1)[1],
                 encoding="utf-8")
    rep = docs.import_md(conn, f)
    assert rep.appended == ["Log (newest first)"]
    assert docs.render(conn, "tracker") == f.read_text(encoding="utf-8")


def test_checklist(conn, tmp_path):
    f = tmp_path / "LESSONS.md"
    f.write_text("# L\n\n## 1. Limits\n- Mistake: m\n- Rule: read SASS\n  and sweep knobs.\n\n"
                 "## Brief checklist (paste)\n1. baseline first;\n   anchored\n2. sass before limit\n",
                 encoding="utf-8")
    docs.import_md(conn, f)
    docs.append(conn, "lessons", "Brief checklist", "measure only with kdev time/ab", author="coord")
    ck = docs.checklist(conn, "lessons", rules=True)
    assert ck["items"] == ["baseline first; anchored", "sass before limit", "measure only with kdev time/ab"]
    assert ck["rules"] == ["1. Limits: read SASS and sweep knobs."]


@pytest.mark.skipif(not all(p.is_file() for p in REAL), reason="OKC notes not on this machine")
def test_real_okc_files_round_trip(conn, tmp_path):
    for src in REAL:
        dst = tmp_path / src.parent.name / src.name           # read-only use: work on a copy
        dst.parent.mkdir(exist_ok=True)
        shutil.copyfile(src, dst)
        rep = docs.import_md(conn, dst, problem="muse_bf16_proj" if src.name == "NOTES.md" else None)
        original = dst.read_text(encoding="utf-8").replace("\r\n", "\n")
        out = tmp_path / "export" / src.name
        docs.export_md(conn, rep.doc, out)
        assert out.read_text(encoding="utf-8") == original, rep.doc
        assert docs.import_md(conn, dst).unchanged >= len(rep.added) - 1
    assert {d["doc"] for d in docs.docs_list(conn)} == {"notes/muse_bf16_proj", "lessons", "tracker"}
    ck = docs.checklist(conn, "lessons")
    assert len(ck["items"]) >= 5 and any("SASS" in i for i in ck["items"])
    assert docs.search(conn, "wgmma", doc="notes/muse_bf16_proj")


# ---------------------------------------------------------------------------------------- CLI / backup
def test_cli_end_to_end_and_backup(capsys, tmp_path):
    r = _cli(capsys, "board", "post", "claim", "m128 tiles", "--problem", "/root/okc/muse_bf16_proj/muse_bf16_proj",
             "--files", "bm_3.py", agent="bob")
    assert r["status"] == "ok" and r["summary"]["post"]["problem"] == "muse_bf16_proj"
    r = _cli(capsys, "board", "post", "claim", "bm all", "--problem", "muse_bf16_proj", "--files", "bm_*.py")
    assert r["verdict"] == "CONFLICT" and r["summary"]["conflict_posts"][0]["agent"] == "bob"
    r = _cli(capsys, "board", "post", "finding", "x0.98 at TILE=256", "--ref", "20261031-101010-ab-t", "--doc",
             "notes", "--section", "Round 11", "--problem", "muse_bf16_proj")
    assert r["summary"]["doc_entry"]
    assert _cli(capsys, "board", "inbox", agent="bob")["summary"]["n"] >= 1
    assert _cli(capsys, "board", "inbox", "--ack", agent="bob")["summary"]["acked"] >= 1
    assert _cli(capsys, "board", "inbox", agent="bob")["summary"]["n"] == 0
    r = _cli(capsys, "doc", "show", "notes", "--problem", "muse_bf16_proj")
    assert "TILE=256" in r["summary"]["text"]
    r = _cli(capsys, "doc", "search", "TILE", "--ref", "20261031")
    assert len(r["summary"]["entries"]) == 1
    assert _cli(capsys, "board", "claims")["verdict"] == "CONFLICT"
    r = _cli(capsys, "board", "post", "finding", "", agent="bob")
    assert r["status"] == "error" and r["error"]["code"] == "invalid_request"
    r = _cli(capsys, "collab", "backup", "--out", str(tmp_path / "bk"))
    bk = tmp_path / "bk"
    man = json.loads((bk / "manifest.json").read_text(encoding="utf-8"))
    assert {"collab.db", "board.jsonl", "docs/notes__muse_bf16_proj.md"} <= set(man["files"])
    assert len((bk / "board.jsonl").read_text(encoding="utf-8").splitlines()) == man["posts"] == 3
    snap = db.connect(bk / "collab.db")
    assert snap.execute("SELECT count(*) FROM posts").fetchone()[0] == 3
    snap.close()
    main(["board", "read"])
    text = capsys.readouterr().out
    assert len(text.splitlines()) <= 40 and "#1" in text


def test_mcp_dispatch_board_and_doc():
    from kdev.mcp.server import Dispatcher
    d = Dispatcher(build_registry(), profile="compact")
    r = d.call("kdev_board", {"action": "post", "kind": "finding", "text": "via mcp", "agent": "m1",
                              "ref": ["#3"]})
    assert r["status"] == "ok", r
    r = d.call("kdev_board", {"action": "read", "agent": "m2"})
    assert r["summary"]["n"] == 1
    r = d.call("kdev_doc", {"action": "append", "doc": "lessons", "section": "Brief checklist",
                            "text": "read the inbox first", "agent": "m1"})
    assert r["status"] == "ok", r
    r = d.call("kdev_doc", {"action": "checklist"})
    assert r["summary"]["items"] == ["read the inbox first"]
    r = d.call("kdev_run", {"command": "doc export", "argv": ["lessons", "--md", "x.md"]})
    assert r["status"] == "error"
