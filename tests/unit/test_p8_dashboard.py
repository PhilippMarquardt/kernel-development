"""P8: dashboard collect/render/board + skill file. No GPU, no network (board fetch is injected)."""
from __future__ import annotations

import json
import re
import sqlite3
from argparse import Namespace
from pathlib import Path

import pytest

import kdev
from kdev.config import Config
from kdev.dashboard import board as B
from kdev.dashboard.collect import collect
from kdev.dashboard.render import render_html, write
from kdev.registry import Context, build_registry

SRC = Path(kdev.__file__).parent
ROOT = SRC.parents[1]
NOW = "2026-10-31T12:00:00Z"


def _ledger(home: Path) -> None:
    home.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(home / "ledger.db")
    c.executescript((SRC / "ledger" / "schema.sql").read_text(encoding="utf-8"))
    ins = c.execute
    for rid, tool, prob, status, comp in (("r-ab1", "ab", "moe", "ok", 1), ("r-ab2", "ab", "moe", "ok", 0),
                                          ("r-sass", "sass", "moe", "ok", 1), ("r-sweep", "sweep", "moe", "ok", 1),
                                          ("r-att", "time", "attn", "ok", 1)):
        ins("INSERT INTO runs (run_id, tool, status, verdict, problem, agent, created_utc, protocol, comparable) "
            "VALUES (?,?,?,?,?,?,?,?,?)", (rid, tool, status, "faster", prob, "w1", NOW, "kdev-tp/1", comp))
    ins("INSERT INTO artifacts (run_id, label, role, path, md5) VALUES ('r-ab1','B','variant','/x/b.py','m5')")
    ins("INSERT INTO scores VALUES ('r-ab1','B',2.41,2.39,2.43,'A')")
    ins("INSERT INTO scores VALUES ('r-ab2','B',9.99,9.0,10.0,'A')")          # not comparable: ignored
    ins("INSERT INTO shape_results (run_id,label,shape_id,weight,seed,median_us,ratio_median) "
        "VALUES ('r-ab1','B','d1',60,-1,21.3,0.97)")
    for jid, sp, st, dup in (("J1", 2.35, "ok", None), ("J2", 2.9, "failed", None), ("J3", 3.0, "ok", "J1"),
                             ("J4", 2.10, "ok", None)):
        ins("INSERT INTO judge_results (judge_id, problem, file_md5, mode, status, speedup, created_utc, "
            "duplicate_of, recorded_utc) VALUES (?,?,?,?,?,?,?,?,?)", (jid, "moe", "m5", "check", st, sp, NOW,
                                                                       dup, NOW))
    ins("INSERT INTO judge_shapes VALUES ('J1','d1','{\"weight\": 60}', 23.4, 77.1, 3.29)")
    ins("INSERT INTO judge_shapes VALUES ('J1','p1','{\"weight\": 1}', 900.0, 2000.0, 2.22)")
    ins("INSERT INTO calib VALUES ('r-ab1','J1','B','d1',2.5,2.3,1.087,'DISAGREE')")
    ins("INSERT INTO ideas (problem,title,kind,status,owner,created_utc,updated_utc,variant) VALUES "
        "('moe','gemv at the HBM floor','limit','confirmed','w1',?,?, '')", (NOW, NOW))
    ins("INSERT INTO ideas (problem,title,kind,status,owner,created_utc,updated_utc,variant) VALUES "
        "('moe','<script>alert(1)</script> {{DATA_JSON}}','idea','open',NULL,?,?,'TILE=128')", (NOW, NOW))
    ins("INSERT INTO ideas (problem,title,kind,status,created_utc,updated_utc) VALUES "
        "('moe','verified limit','limit','confirmed',?,?)", (NOW, NOW))
    ins("INSERT INTO idea_evidence VALUES (1,'run','r-sass','sass read','for','w1',?)", (NOW,))
    ins("INSERT INTO idea_evidence VALUES (3,'run','r-sass','','for','w1',?)", (NOW,))
    ins("INSERT INTO idea_evidence VALUES (3,'run','r-sweep','','for','w1',?)", (NOW,))
    c.commit()
    c.close()


def _queue(home: Path) -> None:
    c = sqlite3.connect(home / "queue.db")
    c.executescript((SRC / "service" / "schema.sql").read_text(encoding="utf-8"))
    c.execute("INSERT INTO jobs (job_id,lane,kind,state,agent,spec_json,submitted_utc,started_utc) VALUES "
              "('j1','gpu','ab','running','w1','{}',?,?)", (NOW, NOW))
    c.execute("INSERT INTO jobs (job_id,lane,kind,state,agent,spec_json,submitted_utc,priority) VALUES "
              "('j2','gpu','time','queued','w2','{}',?,0)", (NOW,))
    c.execute("INSERT INTO jobs (job_id,lane,kind,state,agent,spec_json,submitted_utc,priority) VALUES "
              "('j3','gpu','time','queued','w3','{}',?,5)", (NOW,))
    c.execute("INSERT INTO heartbeat VALUES (1, 42, 'pod', '2000-01-01T00:00:00Z', '0.1.0', 'j1', 8)")
    c.commit()
    c.close()


BOARD_P = {"board": "moe", "sglang_per": "decode step", "rows": [
    {"job": f"j{i}", "team": f"t{i}", "speedup": 2.8 - i * 0.05, "value": 4800 + i, "attempts": 3,
     "submitted_at": 1793342085.0} for i in range(30)] + [
    {"job": "ours", "team": "Us", "speedup": 2.32, "value": 5722, "submitted_at": 1793342085.0}]}
BOARD_ALL = {"board": "overall", "rows": [{"rank": 1, "team": "t0", "solved": 4}, {"rank": 7, "team": "Us"}]}


def _fetch(url: str, timeout: float):
    assert url.startswith(B.BOARD_API)
    if url == B.BOARD_API:
        return BOARD_ALL
    if url.endswith("/moe"):
        return BOARD_P
    raise OSError("offline")


@pytest.fixture
def cfg(tmp_path) -> Config:
    home = tmp_path / "home"
    _ledger(home)
    _queue(home)
    return Config(home=home)


def test_collect_bests_ideas_calib_queue(cfg):
    d = collect(cfg)
    probs = {p["problem"]: p for p in d.problems}
    assert set(probs) == {"moe", "attn"}
    moe = probs["moe"]
    assert moe["best_judged"]["judge_id"] == "J1"          # failed J2 and duplicate J3 skipped
    assert moe["best_local"]["run_id"] == "r-ab1"          # non-comparable r-ab2 skipped
    assert moe["best_local"]["shapes"][0]["shape_id"] == "d1"
    assert {s["shape_id"] for s in moe["judge_shapes"]} == {"d1", "p1"}
    assert moe["disagree"] == 1 and moe["open_ideas"] == 1
    assert [c["idea_id"] for c in d.unverified_limit_claims] == [1]
    assert d.unverified_limit_claims[0]["missing"] == ["sweep"]
    assert d.calibration[0]["flag"] == "DISAGREE"
    q = d.queue
    assert [j["job_id"] for j in q["running"]] == ["j1"]
    if q.get("source", "").startswith("queue.db"):           # fallback reader (P2 snapshot not landed)
        assert [j["job_id"] for j in q["queued"]] == ["j3", "j2"]
        assert q["daemon"]["alive"] is False
    assert len(d.recent_runs) == 5
    json.dumps(d.to_dict())


def test_collect_never_writes(cfg):
    # WAL readers may create -shm/-wal side files; the databases themselves must be untouched
    before = {p.name: p.stat().st_mtime_ns for p in cfg.home.glob("*.db")}
    collect(cfg)
    assert {p.name: p.stat().st_mtime_ns for p in cfg.home.glob("*.db")} == before


def test_collect_empty_home(tmp_path):
    d = collect(Config(home=tmp_path / "nothing"), extra_problems=["moe"])
    assert any("no ledger" in w for w in d.warnings)
    assert [p["problem"] for p in d.problems] == ["moe"]
    html = render_html(d)
    assert "no runs recorded" in html


def test_board_normalize_and_gaps():
    nb = B.normalize_problem_board(BOARD_P, "moe", team="us")
    assert nb["n"] == 31 and len(nb["rows"]) == 10
    assert nb["first"]["team"] == "t0" and nb["tenth"]["team"] == "t9"
    assert nb["our_row"]["team"] == "Us" and nb["our_row"]["rank"] == 11
    board = {"problems": {"moe": nb}}
    ours = [{"problem": "moe", "best_judged": {"speedup": 2.35},
             "judge_shapes": [{"shape_id": "d1", "sub_us": 23.4, "case_json": '{"weight": 60}'},
                              {"shape_id": "p1", "sub_us": 900.0, "weight": 1}]}]
    g = B.gaps(board, ours)[0]
    assert g["board1"] == pytest.approx(2.8) and g["gap_to_1_pct"] == pytest.approx((2.8 / 2.35 - 1) * 100, abs=0.01)
    assert g["rank_kind"] == "board"
    assert [s["shape_id"] for s in g["shapes"]] == ["d1", "p1"]     # 60*23.4 > 900
    assert sum(s["time_share"] for s in g["shapes"]) == pytest.approx(1.0, abs=1e-3)
    assert g["shapes"][0]["cut_to_1_us"] == pytest.approx(23.4 * (1 - 2.35 / 2.8), abs=1e-3)
    # without our team row: would-be rank from our judged speedup
    nb2 = B.normalize_problem_board(BOARD_P, "moe", team=None)
    g2 = B.gaps({"problems": {"moe": nb2}}, ours)[0]
    assert g2["rank_kind"] in ("would-be",) or g2["rank_kind"].startswith(">")


def test_scrape_with_fake_fetch_and_errors(monkeypatch):
    import kdev.judge.okc_cli as okc

    def no_okc(*a, **k):
        raise OSError("no okc")

    monkeypatch.setattr(okc, "leaderboard", no_okc)
    s = B.scrape(problems=["moe", "attn"], team="Us", fetch=_fetch)
    assert s["problems"]["moe"]["our_row"]["team"] == "Us"
    assert "attn" in s["errors"] and "offline" in s["errors"]["attn"]
    assert s["overall"]["our_row"]["rank"] == 7


def test_http_get_refuses_other_hosts():
    with pytest.raises(ValueError):
        B.http_get_json("https://example.com/api")


def test_render_self_contained_and_escaped(cfg):
    d = collect(cfg, scrape=True, team="Us", board_fetch=_fetch)
    html = render_html(d)
    assert "<script>alert(1)</script>" not in html                    # escaped in body and JSON blob
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert html.count('id="kdev-data"') == 1
    assert not re.search(r"""(src|href)\s*=\s*["']?(https?:)?//""", html)   # no external resources
    assert "@import" not in html and "prefers-color-scheme: dark" in html
    blob = re.search(r'<script id="kdev-data" type="application/json">(.*?)</script>', html, re.S).group(1)
    data = json.loads(blob)
    assert data["problems"][0]["problem"] in ("attn", "moe")
    for word in ("Problems", "Per-shape gap", "GPU queue", "Recent runs", "Idea board", "Calibration", "DISAGREE",
                 "unverified limit", "needs sweep"):
        assert word in html, word


def test_write_and_cli_build(cfg, tmp_path):
    out = tmp_path / "dash"
    files = write(collect(cfg), str(out))
    assert Path(files["html"]).exists() and json.loads(Path(files["json"]).read_text(encoding="utf-8"))
    reg = build_registry()
    cmd = reg.get("dashboard build")
    r = cmd.handler(Namespace(out=str(tmp_path / "d2"), scrape=False, team=None, recent=10),
                    Context(config=cfg))
    assert r.status == "ok" and Path(r.files["html"]).exists() and r.summary["problems"] == 2
    assert {"dashboard build", "dashboard watch", "dashboard serve"} <= set(reg.names())
    assert not reg.get("dashboard serve").mcp and not reg.get("dashboard watch").mcp


# ---------------------------------------------------------------------------------------------- skill

def test_skill_file_workflow():
    text = (ROOT / "skills" / "kdev" / "SKILL.md").read_text(encoding="utf-8")
    assert text.startswith("---\nname: kdev\ndescription: ")
    for needle in ("kdev ledger tried", "kdev ab", "kdev time", "kdev sass", "kdev sweep", "kdev calib",
                   "at the limit", "kdev idea", "kdev show", "kdev check", "kdev comply"):
        assert needle in text, needle
    reg = build_registry()
    for m in re.finditer(r"`kdev ([a-z]+(?: [a-z]+)?)", text):
        words = m.group(1).split()
        cmd, _ = reg.resolve(words)
        assert cmd is not None, m.group(0)
