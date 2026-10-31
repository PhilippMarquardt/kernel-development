"""P4: ledger db/migrations, result ingest, queries (tried/best/history/export), idea board."""
from __future__ import annotations

import json
import sqlite3

import pytest

from kdev.contract import ErrorCode, Fidelity, KdevError, Result
from kdev.ledger import db, ideas, query, record
from kdev.ledger.db import SCHEMA_FILE
from kdev.registry import build_registry


def test_ledger_schema_is_valid_sql():
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA_FILE.read_text(encoding="utf-8"))
    tables = {r[0] for r in conn.execute("select name from sqlite_master where type='table'")}
    assert {"runs", "artifacts", "shape_results", "scores", "builds", "kernels", "judge_results",
            "judge_shapes", "calib", "ideas", "idea_evidence"} <= tables


def test_ledger_commands_registered():
    names = set(build_registry().names())
    assert {"ledger query", "ledger tried", "ledger best", "ledger history", "ledger import", "idea add",
            "idea claim", "idea resolve", "note", "judge record", "judge import"} <= names


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "ledger.db")
    yield c
    c.close()


def test_connect_migrates_and_is_idempotent(tmp_path):
    p = tmp_path / "l.db"
    c = db.connect(p)
    assert db.schema_version(c) == db.SCHEMA_VERSION
    cols = {r[1] for r in c.execute("PRAGMA table_info(judge_results)")}
    assert {"content_sha256", "team", "label", "role", "pair_id"} <= cols
    assert c.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    c.close()
    c = db.connect(p)          # second open: no error, same version
    assert db.migrate(c) == db.SCHEMA_VERSION
    c.close()


def test_migrate_v1_database(tmp_path):
    p = tmp_path / "old.db"
    raw = sqlite3.connect(p)
    raw.executescript(SCHEMA_FILE.read_text(encoding="utf-8"))
    raw.execute("INSERT INTO judge_results(judge_id, recorded_utc) VALUES ('j1', 'x')")
    raw.commit()
    raw.close()
    c = db.connect(p)
    assert db.schema_version(c) == 2
    assert c.execute("SELECT judge_id, pair_id FROM judge_results").fetchone()[0] == "j1"
    c.close()


def test_readonly_query_denies_writes(conn):
    assert db.readonly_query(conn, "SELECT count(*) AS n FROM runs") == [{"n": 0}]
    for bad in ("DELETE FROM runs", "SELECT 1; DELETE FROM runs", "UPDATE runs SET tool='x'",
                "WITH x AS (SELECT 1) DELETE FROM runs", "ATTACH 'x.db' AS y", "PRAGMA writable_schema=1"):
        with pytest.raises(KdevError) as e:
            db.readonly_query(conn, bad)
        assert e.value.code == ErrorCode.INVALID_REQUEST


def _ab_result(run_id="20261031-143012-ab-tile128", optouts=None, exclusive="ok", match=True, a_md5="aaa",
               b_md5="bbb"):
    fid = Fidelity(toolchain={"nvcc": "13.0.88", "pin": "13.0.88", "match": match}, exclusive=exclusive,
                   seeds=[11, 12], overhead_us={"judge_const": 4.0, "subtracted": 4.0},
                   clocks={"before": {"sm_mhz": 1980}, "after": {"sm_mhz": 1980}}, optouts=list(optouts or []))
    return Result(tool="ab", verdict="faster", run_id=run_id, job_id="j1", fidelity=fid,
                  shapes=[{"shape_id": "d1/sliding", "weight": 46, "load": "d1",
                           "per_file": [{"label": "A", "md5": a_md5, "median_us": 21.3, "ci_lo_us": 21.1,
                                         "ci_hi_us": 21.6, "n": 40, "seeds": [11, 12],
                                         "per_seed_median_us": [21.2, 21.4]},
                                        {"label": "B", "md5": b_md5, "median_us": 20.7, "ci_lo_us": 20.5,
                                         "ci_hi_us": 20.9, "n": 40}],
                           "ratio": {"num": "B", "den": "A", "median": 0.973, "ci_lo": 0.961, "ci_hi": 0.985},
                           "verdict": "faster"}],
                  score={"per_file": {"A": 2.593, "B": 2.640},
                         "delta": {"median": 1.018, "ci_lo": 1.004, "ci_hi": 1.031}},
                  summary={"builds": [{"label": "B", "ext_name": "vjc", "cache_hit": True, "seconds": 0.1,
                                       "so_notes": ["V13.0.88"]}],
                           "kernels": [{"label": "B", "name": "vjc_k<1>", "regs": 168,
                                        "warnings": ["ptxas C7515 wgmma serialized"]}],
                           "calib": [{"judge_id": "2bbe", "label": "B", "shape_id": "d1/sliding",
                                      "local_speedup": 3.0, "judge_speedup": 3.01, "local_over_judge": 0.997,
                                      "flag": "ok"}]})


def _meta(a_path=None, b_path=None, knobs=None):
    return {"job": {"kind": "ab", "problem": "muse_attention_decode", "adapter": "okc", "agent": "w1",
                    "params": {"reps": 40},
                    "files": [{"path": a_path, "label": "A", "md5": "aaa", "role": "base"},
                              {"path": b_path, "label": "B", "md5": "bbb", "knobs": knobs or {"TILE": 128},
                               "role": "variant"}]},
            "host": "pod1", "gpu": {"name": "NVIDIA H200", "uuid": "GPU-1"}}


def test_rows_from_result_pure():
    rows = record.rows_from_result(_ab_result(), _meta())
    run = rows["runs"][0]
    assert run["comparable"] == 1 and run["problem"] == "muse_attention_decode" and run["agent"] == "w1"
    assert run["created_utc"] == "2026-10-31T14:30:12Z" and run["overhead_us"] == 4.0 and run["reps"] == 40
    assert {a["label"]: a["role"] for a in rows["artifacts"]} == {"A": "base", "B": "variant"}
    pooled = [r for r in rows["shape_results"] if r["seed"] == -1]
    assert len(pooled) == 2
    b = next(r for r in pooled if r["label"] == "B")
    assert b["ratio_den_label"] == "A" and b["ratio_median"] == 0.973
    assert sorted(r["seed"] for r in rows["shape_results"] if r["seed"] != -1) == [11, 12]
    sc = {s["label"]: s for s in rows["scores"]}
    assert sc["B/A"]["score"] == 1.018 and sc["B"]["vs_label"] == "A"
    assert rows["builds"][0]["cache_hit"] == 1 and rows["builds"][0]["so_notes"] == "V13.0.88"
    assert "C7515" in rows["kernels"][0]["c751x"]
    assert rows["calib"][0]["flag"] == "ok"


@pytest.mark.parametrize("kw,why", [({"optouts": ["flush=write"]}, "optouts"), ({"exclusive": "fail"}, "excl"),
                                    ({"match": False}, "toolchain")])
def test_incomparable_rows(kw, why):
    rows = record.rows_from_result(_ab_result(**kw), _meta())
    assert rows["runs"][0]["comparable"] == 0, why


def test_ingest_result_idempotent_and_queries(conn, tmp_path):
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    a.write_text("# okc: problem=x\nx = 1  # one\n", encoding="utf-8")
    b.write_text("# okc: problem=x\nx = 2\n", encoding="utf-8")
    from kdev.ledger._util import md5_file
    rd = tmp_path / "20261031-143012-ab-tile128"
    rd.mkdir()
    res = _ab_result(a_md5=md5_file(a), b_md5=md5_file(b))
    (rd / "result.json").write_text(res.to_json(), encoding="utf-8")
    meta = _meta(str(a), str(b))
    meta["job"]["files"][0]["md5"] = md5_file(a)
    meta["job"]["files"][1]["md5"] = md5_file(b)
    (rd / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    assert record.ingest_result(rd, conn=conn) == res.run_id
    assert record.ingest_result(rd / "result.json", conn=conn) == res.run_id     # idempotent
    assert conn.execute("SELECT count(*) FROM shape_results").fetchone()[0] == 4
    assert conn.execute("SELECT count(*) FROM runs").fetchone()[0] == 1

    # exact md5
    rep = query.tried(conn, file=str(a))
    assert [h.match for h in rep.hits] == ["md5"] and rep.hits[0].run_id == res.run_id
    # comment-only change -> norm_md5 hit (dedup-proof)
    a2 = tmp_path / "a2.py"
    a2.write_text("# okc: problem=x\n# a new comment\nx = 1\n", encoding="utf-8")
    rep = query.tried(conn, file=str(a2))
    assert [h.match for h in rep.hits] == ["norm_md5"]
    # knob subset within the problem
    assert query.tried(conn, knobs={"TILE": 128}, problem="muse_attention_decode").hits[0].match == "knobs"
    assert not query.tried(conn, knobs={"TILE": 64}, problem="muse_attention_decode").hits
    assert not query.tried(conn, knobs={"TILE": 128}, problem="other").hits

    assert query.runs(conn, md5=md5_file(b))[0]["run_id"] == res.run_id
    assert query.runs(conn, tool="time") == []
    best = query.best(conn, "muse_attention_decode")
    assert best[0]["local"]["label"] == "B" and best[0]["local"]["score"] == 2.640
    hist = query.shape_history(conn, "muse_attention_decode", "sliding")
    assert {h["source"] for h in hist} == {"local"}
    out = query.export(conn, "jsonl", ("runs", "artifacts"))
    assert len(out.strip().splitlines()) == 3
    assert query.export(conn, "csv", ("runs",)).startswith("run_id,")
    with pytest.raises(KdevError):
        query.export(conn, "jsonl", ("sqlite_master",))


def test_ingest_rejects_non_result(conn, tmp_path, judge_json):
    p = tmp_path / "j.json"
    p.write_text(json.dumps(judge_json), encoding="utf-8")
    with pytest.raises(KdevError) as e:
        record.ingest_result(p, conn=conn)
    assert e.value.code == ErrorCode.INVALID_REQUEST
    with pytest.raises(KdevError) as e:
        record.ingest_result(tmp_path / "missing", conn=conn)
    assert e.value.code == ErrorCode.NOT_FOUND


def test_mark_protocol_incomparable(conn):
    record.write_rows(conn, record.rows_from_result(_ab_result(), _meta()))
    old = _ab_result(run_id="20261030-000000-ab-old")
    old.protocol = "kdev-tp/0"
    record.write_rows(conn, record.rows_from_result(old, _meta()))
    assert conn.execute("SELECT comparable FROM runs WHERE run_id='20261030-000000-ab-old'").fetchone()[0] == 0
    conn.execute("UPDATE runs SET comparable=1")
    assert record.mark_protocol_incomparable(conn, "kdev-tp/1") == 1


# ------------------------------------------------------------------ ideas
def test_idea_claim_conflict_release(conn):
    i = ideas.add(conn, ideas.Idea(None, "split-K for verify 32", problem="p", hypothesis="more CTAs"))
    assert i.status == "open" and i.idea_id
    assert ideas.claim(conn, i.idea_id, "w1").status == "claimed"
    assert ideas.claim(conn, i.idea_id, "w1").owner == "w1"          # re-claim by owner is a no-op
    with pytest.raises(KdevError) as e:
        ideas.claim(conn, i.idea_id, "w2")
    assert e.value.code == ErrorCode.CONFLICT
    with pytest.raises(KdevError) as e:
        ideas.release(conn, i.idea_id, "w2")
    assert e.value.code == ErrorCode.CONFLICT
    i = ideas.add_evidence(conn, i.idea_id, ideas.Evidence("run", "20261031-1-ab-x", supports="for", author="w1"))
    assert i.status == "testing" and len(i.evidence) == 1
    with pytest.raises(KdevError) as e:
        ideas.resolve(conn, i.idea_id, "refuted", "no", "w2")
    assert e.value.code == ErrorCode.CONFLICT
    assert ideas.release(conn, i.idea_id, "w1").status == "open"
    assert ideas.claim(conn, i.idea_id, "w2").owner == "w2"
    done = ideas.resolve(conn, i.idea_id, "confirmed", "x1.04 judged", "w2")
    assert done.status == "confirmed"
    with pytest.raises(KdevError) as e:
        ideas.claim(conn, i.idea_id, "w3")
    assert e.value.code == ErrorCode.CONFLICT
    with pytest.raises(KdevError) as e:
        ideas.get(conn, 999)
    assert e.value.code == ErrorCode.NOT_FOUND


def test_dead_requires_variant(conn):
    with pytest.raises(KdevError) as e:
        ideas.add(conn, ideas.Idea(None, "TMA loads are dead", kind="dead"))
    assert e.value.details["field"] == "variant"
    i = ideas.add(conn, ideas.Idea(None, "TMA loads are dead", kind="dead", variant="tma_v2.py STAGES=3"))
    assert ideas.resolve(conn, i.idea_id, "refuted", "slower x0.97", "w").status == "refuted"


def test_limit_claim_needs_sass_and_sweep(conn):
    i = ideas.add(conn, ideas.Idea(None, "moe kernel at the DRAM floor", kind="limit", problem="moe"))
    with pytest.raises(KdevError) as e:
        ideas.resolve(conn, i.idea_id, "confirmed", "", "w")
    assert "sass" in e.value.message and "sweep" in e.value.message
    ideas.add_evidence(conn, i.idea_id, ideas.Evidence("run", "20261031-101010-sass-moe", supports="for"))
    with pytest.raises(KdevError):
        ideas.resolve(conn, i.idea_id, "confirmed", "", "w")
    # a sweep run recorded in the ledger (tool from the runs table)
    res = _ab_result(run_id="20261031-111111-x-grid")
    res.tool = "sweep"
    record.write_rows(conn, record.rows_from_result(res, _meta()))
    ideas.add_evidence(conn, i.idea_id, ideas.Evidence("run", "20261031-111111-x-grid", supports="for"))
    assert ideas.resolve(conn, i.idea_id, "confirmed", "SASS + sweep done", "w").status == "confirmed"
    # refuting a limit claim needs no gate
    j = ideas.add(conn, ideas.Idea(None, "attention at floor", kind="limit"))
    assert ideas.resolve(conn, j.idea_id, "refuted", "found 5%", "w").status == "refuted"


def test_idea_search_and_tried_text(conn):
    ideas.add(conn, ideas.Idea(None, "KPERM3 scale-sharing re-layout", hypothesis="share scales across k",
                               problem="prefill", variant="kperm3_v1.py"))
    ideas.add(conn, ideas.Idea(None, "cuBLASLt algo pin", problem="proj"))
    assert [i.title for i in ideas.search(conn, "re-layout")] == ["KPERM3 scale-sharing re-layout"]
    assert ideas.search(conn, "kperm3_v1", problem="prefill")
    assert not ideas.search(conn, "kperm3", problem="proj")
    hits = query.tried(conn, text="cublaslt algorithm").hits
    assert hits and hits[0].match == "idea_text"
    lst = ideas.list_ideas(conn, status="open,claimed")
    assert len(lst) == 2


def test_idea_search_without_fts(tmp_path, monkeypatch):
    c = db.connect(tmp_path / "nofts.db")
    monkeypatch.setattr(db, "has_fts", lambda conn: False)
    ideas.add(c, ideas.Idea(None, "persistent kernel for decode"))
    assert ideas.search(c, "persistent")[0].title == "persistent kernel for decode"
    c.close()


def test_schema_without_fts_fallback():
    sql = db._schema_without_fts(SCHEMA_FILE.read_text(encoding="utf-8"))
    assert "VIRTUAL TABLE" not in sql.upper()
    c = sqlite3.connect(":memory:")
    c.executescript(sql)


def test_cli_idea_and_note_roundtrip(tmp_path, monkeypatch):
    from kdev.cli import run
    monkeypatch.setenv("KDEV_HOME", str(tmp_path / "h"))
    r = run(["idea", "add", "split-K verify", "--hypothesis", "more CTAs", "--gain", "5", "--problem", "p"])
    assert r.status == "ok" and r.verdict == "added"
    iid = r.summary["idea"]["idea_id"]
    r = run(["idea", "claim", str(iid), "--agent", "w1"])
    assert r.summary["idea"]["owner"] == "w1"
    r = run(["idea", "claim", str(iid), "--agent", "w2"])
    assert r.error is not None and r.error.code == "conflict" and r.exit_code == 11
    r = run(["note", "attention at the floor", "--kind", "limit", "--evidence", "20261031-000000-sass-a"])
    assert r.error is not None and r.error.code == "invalid_request"
    r = run(["note", "attention at the floor", "--kind", "limit", "--evidence",
             "20261031-000000-sass-a,20261031-000001-sweep-b", "--agent", "w1"])
    assert r.status == "ok" and r.summary["idea"]["status"] == "confirmed"
    r = run(["note", "fa3 swap", "--kind", "dead"])
    assert r.error is not None and r.error.code == "invalid_request"
    r = run(["idea", "list", "--json"])
    assert len(r.summary["ideas"]) == 2
    r = run(["ledger", "tried", "split-K"])
    assert r.verdict == "tried"
    r = run(["ledger", "query", "--sql", "DROP TABLE runs"])
    assert r.error is not None and r.error.code == "invalid_request"
    from kdev.contract import validate_result
    assert not validate_result(run(["ledger", "best"]).to_dict())
    assert len(run(["idea", "show", str(iid)]).lines) <= 40


def test_probe_never_raises(tmp_path):
    from kdev.ledger import probe

    class Ctx:
        home = str(tmp_path / "h")
    checks = probe(Ctx(), problem="muse_moe_w4a16")
    assert {c.name for c in checks} >= {"ledger.open", "ledger.schema", "ledger.rows"}
    assert all(c.status in ("PASS", "WARN") for c in checks)
