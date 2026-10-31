"""P4: judge record/ledger, prepare (duplicates, comment-changed copy), pair (plan + fake okc), okc read-only
CLI wrapper, leaderboard + OKC-tree importer. No network: every okc call goes through a fake runner."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from kdev.contract import ErrorCode, KdevError
from kdev.judge import _importer as imp
from kdev.judge import okc_cli, pair, prepare, record
from kdev.judge._parse import parse_judge
from kdev.ledger import db
from kdev.util import Proc


@pytest.fixture
def conn(tmp_path):
    c = db.connect(tmp_path / "ledger.db")
    yield c
    c.close()


def _fx(fixtures_dir: Path, name: str) -> dict:
    return json.loads((fixtures_dir / "judge" / name).read_text(encoding="utf-8"))


SUB = "# okc: problem=muse_bf16_proj\nimport torch\n\n\ndef kernel_fn(data, state):\n    return data  # {tag}\n"


def _sub(tmp_path: Path, name: str, tag: str) -> str:
    p = tmp_path / name
    p.write_text(SUB.format(tag=tag), encoding="utf-8")
    return str(p)


# ---------------------------------------------------------------- record / ledger
def test_parse_and_record_moe(conn, judge_json, tmp_path):
    jr = parse_judge(judge_json)
    assert jr.judge_id == "2bbe824884d643bb" and jr.problem == "muse_moe_w4a16" and jr.status == "ok"
    assert len(jr.shapes) == 6 and jr.cluster == "sfo3" and jr.node.startswith("n-")
    s0 = jr.shapes[0]
    assert s0.base_us == pytest.approx(77.104, rel=1e-4)
    assert s0.sub_us == pytest.approx(s0.base_us / s0.speedup)
    f = _sub(tmp_path, "alyg_moe.py", "x")
    jid = record.record(conn, judge_json, file_path=f, raw_path="/x/check_alyg.json")
    row = dict(conn.execute("SELECT * FROM judge_results WHERE judge_id=?", (jid,)).fetchone())
    assert row["speedup"] == pytest.approx(2.3544, rel=1e-4) and row["mode"] == "check"
    assert row["content_sha256"].startswith("6a67e497") and row["file_md5"] and row["norm_md5"]
    assert json.loads(row["speedup_ci_json"])[0] == pytest.approx(2.319, rel=1e-3)
    assert conn.execute("SELECT count(*) FROM judge_shapes WHERE judge_id=?", (jid,)).fetchone()[0] == 6
    # re-record without --file keeps the md5 learned before (idempotent upsert)
    record.record(conn, judge_json)
    assert conn.execute("SELECT file_md5 FROM judge_results").fetchone()[0] == row["file_md5"]


def test_duplicate_and_failed_status(conn, judge_json):
    d = dict(judge_json, id="dup1", duplicate_of="2bbe824884d643bb")
    assert parse_judge(d).status == "duplicate"
    f = json.loads(json.dumps(judge_json))
    f["id"] = "bad1"
    f["result"]["passed"] = False
    assert parse_judge(f).status == "fail"
    with pytest.raises(KdevError):
        parse_judge({"schema": "kdev.result/1", "tool": "ab"})


def test_ledger_same_hour_baseline(conn, fixtures_dir):
    for n, role in (("j_base1", "base"), ("j_cand1", "cand"), ("j_base2", "base"), ("j_cand2", "cand")):
        record.record(conn, _fx(fixtures_dir, f"pair_bf16_{n}.json"), label=f"{n}.py", role=role, pair_id="p1")
    rows = record.ledger(conn)
    assert len(rows) == 1
    r = rows[0]
    assert r["problem"] == "muse_bf16_proj" and r["judge_id"] == "9c9fe5532dfa4216"
    assert r["baseline"]["judge_id"] in ("beac83e3020e4ff2", "467cc8706eb44153")
    assert r["baseline"]["speedup"] == pytest.approx((1.55396 * 1.55440) ** 0.5, rel=1e-4)   # pooled base
    assert r["vs_baseline"] == pytest.approx(1.0085, abs=5e-4)
    assert not r["cluster_mismatch"]


# ---------------------------------------------------------------- pair pooling (real judge pair, bf16 rn/)
def test_pool_pair_real_abba(fixtures_dir):
    b = [parse_judge(_fx(fixtures_dir, f"pair_bf16_j_base{i}.json")) for i in (1, 2)]
    c = [parse_judge(_fx(fixtures_dir, f"pair_bf16_j_cand{i}.json")) for i in (1, 2)]
    out = pair.pool_pair(b, c)
    assert out["verdict"] == "same"             # +0.6 % pooled: inside the +-2 % judge noise
    assert out["ratio"] == pytest.approx(1.0058, abs=5e-4)
    assert [r["ratio"] for r in out["rounds"]] == [pytest.approx(1.0087, abs=2e-4), pytest.approx(1.0030, abs=2e-4)]
    assert len(out["shapes"]) == 30 and out["clusters"] == ["sfo3"]


def test_pool_pair_verdicts(fixtures_dir):
    b = parse_judge(_fx(fixtures_dir, "pair_bf16_j_base1.json"))
    c = parse_judge(_fx(fixtures_dir, "pair_bf16_j_cand1.json"))
    c.speedup = b.speedup * 1.05
    assert pair.pool_pair([b], [c])["verdict"] == "faster"
    c.speedup = b.speedup * 0.95
    assert pair.pool_pair([b], [c])["verdict"] == "slower"
    c.cluster = "sfo3c"
    out = pair.pool_pair([b], [c])
    assert out["verdict"] == "same" and any("cluster mix" in w for w in out["warnings"])
    c.status = "fail"
    assert pair.pool_pair([b], [c])["verdict"] == "incorrect"


# ---------------------------------------------------------------- prepare
def test_prepare_duplicate_and_copy(conn, tmp_path, judge_json):
    f = _sub(tmp_path, "cand.py", "v1")
    rep = prepare.prepare(f, conn=conn, check_rules=False)
    assert rep.problem == "muse_bf16_proj" and not rep.blocking and not rep.duplicate_of
    assert "okc check cand.py --json" in rep.command
    assert prepare.to_result(rep).verdict == "ready"
    # record a judge run of exactly this file -> duplicate warning
    record.record(conn, dict(judge_json, id="j_cand"), file_path=f)
    rep = prepare.prepare(f, conn=conn, check_rules=False)
    assert rep.duplicate_of == ["j_cand"] and prepare.to_result(rep).verdict == "duplicate"
    # a comment-changed copy: new hash, same norm_md5, header still on line 1
    rep2 = prepare.prepare(f, conn=conn, copy=True, check_rules=False)
    assert rep2.path.endswith("cand__kj1.py") and rep2.copy_of == str(Path(f).resolve())
    assert rep2.md5 != rep.md5 and rep2.norm_md5 == rep.norm_md5
    assert not rep2.duplicate_of and rep2.same_code == ["j_cand"]
    lines = Path(rep2.path).read_text(encoding="utf-8").split("\n")
    assert lines[0] == "# okc: problem=muse_bf16_proj" and lines[1].startswith("# kdev judge copy 1:")
    assert prepare.make_copy(f, 1) == rep2.path                      # deterministic per k
    # once copy 1 is judged, the next --copy picks k=2
    record.record(conn, dict(judge_json, id="j_copy1"), file_path=rep2.path)
    assert prepare.prepare(f, conn=conn, copy=True, check_rules=False).path.endswith("cand__kj2.py")


def test_prepare_blocking_findings(conn, tmp_path):
    p = tmp_path / "nohdr.py"
    p.write_text("def kernel(x):\n    return x\n", encoding="utf-8")
    rep = prepare.prepare(str(p), conn=conn, check_rules=False)
    assert len(rep.blocking) == 2 and prepare.to_result(rep).verdict == "blocked"
    f = _sub(tmp_path, "c.py", "x")
    rep = prepare.prepare(f, "muse_moe_w4a16", conn=conn, check_rules=False)
    assert any("header says" in b for b in rep.blocking)
    with pytest.raises(KdevError) as e:
        prepare.prepare(str(tmp_path / "missing.py"), conn=conn)
    assert e.value.code == ErrorCode.NOT_FOUND


def test_prepare_adapter_rules_unavailable_is_warning(conn, tmp_path):
    rep = prepare.prepare(_sub(tmp_path, "c.py", "x"), conn=conn)
    assert not rep.rule_findings or all("forbidden" in r for r in rep.rule_findings)


# ---------------------------------------------------------------- okc wrapper (fake runner only)
def test_okc_readonly_allowed_and_forbidden(fake_runner, judge_json):
    r = fake_runner({("okc", "status"): (0, json.dumps(judge_json), "")})
    d = okc_cli.status("2bbe824884d643bb", runner=r)
    assert d["id"] == "2bbe824884d643bb"
    assert r.calls == [["okc", "status", "2bbe824884d643bb", "--json"]]
    for sub in ("submit", "check", "withdraw", "problems"):
        with pytest.raises(KdevError) as e:
            okc_cli.okc_readonly(sub, [], runner=r)
        assert e.value.code == ErrorCode.QUOTA_FORBIDDEN
    with pytest.raises(KdevError) as e:
        okc_cli.okc_readonly("status", ["x", "submit"], runner=r)
    assert e.value.code == ErrorCode.QUOTA_FORBIDDEN
    assert len(r.calls) == 1                         # nothing forbidden reached the runner
    with pytest.raises(KdevError) as e:
        okc_cli.status("--all", runner=r)
    assert e.value.code == ErrorCode.INVALID_REQUEST


def test_okc_not_on_path(fake_runner):
    with pytest.raises(KdevError) as e:
        okc_cli.leaderboard("muse_moe_w4a16", runner=fake_runner({}))
    assert e.value.code == ErrorCode.CAPABILITY_UNAVAILABLE

    def missing(argv, **kw):
        raise FileNotFoundError(argv[0])
    with pytest.raises(KdevError) as e:
        okc_cli.jobs(runner=missing)
    assert e.value.code == ErrorCode.CAPABILITY_UNAVAILABLE


def test_okc_check_needs_identity_token(tmp_path, fake_runner):
    r = fake_runner({("okc", "check"): (0, "{}", "")})
    f = _sub(tmp_path, "c.py", "x")
    for token in (None, object(), True, "allow"):
        with pytest.raises(KdevError) as e:
            okc_cli._okc_check(f, "p", _allow=token, runner=r)
        assert e.value.code == ErrorCode.QUOTA_FORBIDDEN
    assert r.calls == []


# ---------------------------------------------------------------- pair end-to-end with a fake okc
def _fake_okc(fixtures_dir: Path):
    """Queues check jobs (--no-wait) and answers status --wait with the real bf16 pair JSONs: base files get
    j_base1/j_base2, candidate files j_cand1/j_cand2, in submission order."""
    answers = {"base": [_fx(fixtures_dir, "pair_bf16_j_base1.json"), _fx(fixtures_dir, "pair_bf16_j_base2.json")],
               "cand": [_fx(fixtures_dir, "pair_bf16_j_cand1.json"), _fx(fixtures_dir, "pair_bf16_j_cand2.json")]}
    jobs: dict[str, dict] = {}
    calls: list[list[str]] = []

    def runner(argv, **kw):
        calls.append(list(argv))
        assert argv[0] == "okc" and argv[1] in ("check", "status"), argv
        if argv[1] == "check":
            assert "--no-wait" in argv and kw.get("cwd")
            role = "base" if argv[2].startswith("base") else "cand"
            d = answers[role].pop(0)
            jobs[d["id"]] = d
            return Proc(list(argv), 0, json.dumps({"id": d["id"], "status": "queued"}), "")
        assert "--wait" in argv
        return Proc(list(argv), 0, json.dumps(jobs[argv[2]]), "")

    runner.calls = calls  # type: ignore[attr-defined]
    return runner


def test_run_pair_plan_only_never_calls_okc(conn, tmp_path, fake_runner):
    c, b = _sub(tmp_path, "cand.py", "c"), _sub(tmp_path, "base.py", "b")
    r = fake_runner({})
    res = pair.run_pair(c, b, None, conn=conn, runner=r)
    assert res.verdict == "planned" and r.calls == []
    plan = res.summary["plan"]
    assert [[x["role"] for x in rnd] for rnd in plan] == [["base", "cand"], ["cand", "base"]]   # ABBA
    paths = [x["path"] for rnd in plan for x in rnd]
    assert len(set(paths)) == 4 and paths[0].endswith("base.py") and paths[2].endswith("cand__kj1.py")
    assert all("okc check" in cmd for cmd in res.summary["commands"])
    with pytest.raises(KdevError):
        pair.run_pair(c, None, None, conn=conn, runner=r)
    with pytest.raises(KdevError):
        pair.run_pair(c, c, None, conn=conn, runner=r)


def test_run_pair_allow_check_records_and_pools(conn, tmp_path, fixtures_dir):
    c, b = _sub(tmp_path, "cand.py", "c"), _sub(tmp_path, "base.py", "b")
    runner = _fake_okc(fixtures_dir)
    logs: list[str] = []
    res = pair.run_pair(c, b, None, allow_check=True, conn=conn, runner=runner, log=logs.append,
                        out_dir=str(tmp_path / "raw"))
    assert res.verdict == "same" and res.summary["ratio"] == pytest.approx(1.0058, abs=5e-4)
    assert len(res.summary["job_ids"]) == 4
    assert sum(1 for cl in runner.calls if cl[1] == "check") == 4
    assert any("okc check" in m for m in logs[:5])                    # commands printed before running
    rows = conn.execute("SELECT role, pair_id, file_md5, label FROM judge_results ORDER BY created_utc").fetchall()
    assert len(rows) == 4 and len({r["pair_id"] for r in rows}) == 1 and all(r["file_md5"] for r in rows)
    assert sorted(r["role"] for r in rows) == ["base", "base", "cand", "cand"]
    assert len(list((tmp_path / "raw").glob("chk_*.json"))) == 4
    # a second plan never re-sends a judged hash
    plan = pair.plan_pair(c, b, conn, 2)
    sent = {r["file_md5"] for r in rows}
    from kdev.ledger._util import md5_file
    assert not ({md5_file(p) for rnd in plan for _r, p in rnd} & sent)


def test_cli_pair_allow_check_refused_over_mcp(tmp_path, monkeypatch):
    from argparse import Namespace

    from kdev.judge.commands import cmd_pair
    from kdev.registry import Context
    monkeypatch.setenv("KDEV_HOME", str(tmp_path / "h"))
    args = Namespace(candidate="a.py", base="b.py", allow_check=True, rounds=2)
    with pytest.raises(KdevError) as e:
        cmd_pair(args, Context(via="mcp"))
    assert e.value.code == ErrorCode.QUOTA_FORBIDDEN


# ---------------------------------------------------------------- boards + importer
def test_board_formats(conn, fixtures_dir):
    api = imp.board_rows_from(_fx(fixtures_dir, "board_moe_api.json"), "2026-10-31T00:00:00Z", "api")
    assert len(api) == 8 and api[0]["team"] == "dolanzhao" and api[0]["rank"] == 1
    assert api[0]["time_us"] == pytest.approx(4876.5, rel=1e-4) and api[0]["submitted_utc"].startswith("2026-")
    agg = imp.board_rows_from(_fx(fixtures_dir, "board_research_agg.json"), "2026-10-31T00:00:00Z", "agg")
    assert {r["problem"] for r in agg} == {"muse_bf16_proj", "muse_moe_w4a16"} and len(agg) == 10
    assert imp.board_rows_from({"board": "overall", "rows": [{"rank": 1, "team": "x", "solved": 4}]}, "t", "s") == []
    assert imp.write_board_rows(conn, api + agg) == 18


def test_okc_leaderboard_cmd_record(tmp_path, monkeypatch, fixtures_dir):
    from argparse import Namespace

    from kdev.judge import commands
    from kdev.registry import Context
    monkeypatch.setenv("KDEV_HOME", str(tmp_path / "h"))
    board = _fx(fixtures_dir, "board_moe_api.json")
    monkeypatch.setattr(okc_cli, "leaderboard", lambda problem=None, runner=None: board)
    res = commands.cmd_okc_leaderboard(Namespace(record=True), Context(problem="muse_moe_w4a16"))
    assert res.summary["n"] == 8 and any("recorded 8" in ln for ln in res.lines)


def test_import_okc_tree(conn, tmp_path, fixtures_dir):
    root = tmp_path / "okc"
    pd = root / "muse_bf16_proj" / "muse_bf16_proj"
    (pd / "rn").mkdir(parents=True)
    (root / "research").mkdir()
    for n in ("j_base1", "j_base2", "j_cand1", "j_cand2"):
        shutil.copy(fixtures_dir / "judge" / f"pair_bf16_{n}.json", pd / "rn" / f"{n}.json")
    moe = root / "muse_moe_w4a16" / "muse_moe_w4a16"
    moe.mkdir(parents=True)
    shutil.copy(fixtures_dir / "judge" / "judge_moe.json", moe / "check_alyg.json")
    (pd / "rn" / "notes.json").write_text('{"not": "a judge file"}', encoding="utf-8")
    shutil.copy(fixtures_dir / "judge" / "board_research_agg.json", root / "research" / "leaderboard_2026-10-31.json")
    s = imp.import_okc_tree(conn, str(root))
    assert s["judge_files"] == 5 and s["pairs"] == 1 and s["board_rows"] == 10
    assert any("notes.json" in x for x in s["skipped"])
    r = dict(conn.execute("SELECT label, role, pair_id FROM judge_results WHERE judge_id='beac83e3020e4ff2'")
             .fetchone())
    assert r["role"] == "base" and r["pair_id"].startswith("imp-") and r["label"].endswith(".py")
    again = imp.import_okc_tree(conn, str(root))                       # idempotent
    assert again["unique_judge_ids"] == 5
    assert conn.execute("SELECT count(*) FROM judge_results").fetchone()[0] == 5
    led = {x["problem"]: x for x in record.ledger(conn)}
    assert led["muse_bf16_proj"]["baseline"] is not None and led["muse_bf16_proj"]["board_top"]["team"]
    assert led["muse_moe_w4a16"]["baseline"] is None
    assert imp.detect_kind(str(pd / "rn" / "j_base1.json")) == "judge"
    assert imp.detect_kind(str(root / "research" / "leaderboard_2026-10-31.json")) == "board"


def test_import_matches_submitted_file_by_hash(conn, tmp_path, judge_json):
    pd = tmp_path / "okc" / "muse_moe_w4a16" / "muse_moe_w4a16"
    pd.mkdir(parents=True)
    sub = pd / "best.py"
    sub.write_bytes(b"# okc: problem=muse_moe_w4a16\ndef kernel_fn(d, s):\n    return d\n")
    import hashlib
    d = dict(judge_json, content_hash=hashlib.sha256(sub.read_bytes()).hexdigest())
    (pd / "check_best.json").write_text(json.dumps(d), encoding="utf-8")
    s = imp.import_okc_tree(conn, str(tmp_path / "okc"))
    assert s["files_matched_by_hash"] == 1
    assert conn.execute("SELECT file_md5 FROM judge_results").fetchone()[0]
