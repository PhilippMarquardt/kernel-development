"""P4: safety rule 8.1 - kdev never submits; `okc check` only through judge.pair with --allow-check.

- AST scan: only ``judge/okc_cli.py`` may build an argv for the ``okc`` binary (a list/tuple literal whose first
  element is "okc", or the bare constant "okc" anywhere outside docstrings);
- ``okc_cli`` has no argv containing "submit"; ``_okc_check`` is only referenced by ``judge/pair.py``;
- regex: no "okc submit" outside the deny-list/doc files;
- registry: only ``judge pair`` declares ``network=okc_check``, it is hidden from MCP, its ``--allow-check``
  flag is ``mcp=False``, and no command anywhere is a judge submit.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import kdev
from kdev.registry import build_registry

SRC = Path(kdev.__file__).parent
# files allowed to *mention* "okc submit" (deny-lists, docs); none of them may build an okc argv (AST test)
ALLOWED_OKC_FILES = {"judge/okc_cli.py", "judge/pair.py", "judge/__init__.py", "judge/commands.py",
                     "config.py", "service/lock.py", "dashboard/board.py", "registry.py", "worker.py"}
ALLOWED_OKC_DIRS = ("service/",)
OKC_CALLER = "judge/okc_cli.py"


def _py_files():
    for p in sorted(SRC.rglob("*.py")):
        yield p.relative_to(SRC).as_posix(), p.read_text(encoding="utf-8")


def test_no_submit_code_path():
    pat = re.compile(r"""["']okc["']\s*,\s*["']submit["']|okc\s+submit""")
    offenders = []
    for rel, text in _py_files():
        if rel in ALLOWED_OKC_FILES or rel.startswith(ALLOWED_OKC_DIRS):
            continue
        if pat.search(text):
            offenders.append(rel)
    assert not offenders


def test_only_okc_cli_builds_okc_argv():
    offenders = []
    for rel, text in _py_files():
        if rel == OKC_CALLER:
            continue
        tree = ast.parse(text)
        okc_names = {t.id for n in ast.walk(tree) if isinstance(n, ast.Assign)
                     and isinstance(n.value, ast.Constant) and n.value.value == "okc"
                     for t in n.targets if isinstance(t, ast.Name)}

        def is_okc(e: ast.AST, cmdline: bool = False) -> bool:
            if isinstance(e, ast.Name):
                return e.id in okc_names
            if not (isinstance(e, ast.Constant) and isinstance(e.value, str)):
                return False
            v = e.value.strip()
            return v == "okc" or (cmdline and v.startswith("okc "))

        for node in ast.walk(tree):
            # argv literal ["okc", ...]; deny-list entries such as "okc submit" are single strings (allowed)
            if isinstance(node, ast.List) and len(node.elts) > 1 and is_okc(node.elts[0]):
                offenders.append(f"{rel}:{node.lineno} argv literal starting with 'okc'")
            if isinstance(node, ast.Call) and node.args:
                fn = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
                a0 = node.args[0]
                if fn in ("which", "run", "Popen", "call", "check_call", "check_output", "system", "runner") \
                        and (is_okc(a0, cmdline=True) or (isinstance(a0, (ast.List, ast.Tuple)) and a0.elts
                                            and is_okc(a0.elts[0]))):
                    offenders.append(f"{rel}:{node.lineno} {fn}() of okc")
    assert not offenders, "\n".join(offenders)


def test_okc_cli_never_builds_submit():
    tree = ast.parse((SRC / OKC_CALLER).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.List, ast.Tuple)):
            consts = [e.value for e in node.elts if isinstance(e, ast.Constant)]
            if "submit" in consts or "withdraw" in consts:
                # the only tuple allowed to name them is the deny-list itself
                assert consts == ["submit", "withdraw"], ast.dump(node)


def test_okc_check_only_referenced_by_pair():
    users = set()
    for rel, text in _py_files():
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Attribute) and node.attr == "_okc_check":
                users.add(rel)
            if isinstance(node, ast.ImportFrom) and any(a.name == "_okc_check" for a in node.names):
                users.add(rel)
    assert users == {"judge/pair.py"}
    assert "_ALLOW_CHECK" in (SRC / "judge" / "pair.py").read_text(encoding="utf-8")


def test_allow_check_not_exposed_over_mcp():
    cmd = build_registry().get("judge pair")
    assert cmd.mcp is False
    assert all(a.mcp is False for a in cmd.args if "--allow-check" in a.flags)


def test_only_judge_pair_can_spend_quota():
    reg = build_registry()
    spenders = [c.name for c in reg.all() if c.effects.network == "okc_check" or c.effects.quota != "never"]
    assert spenders == ["judge pair"]
    assert not [c.name for c in reg.all() if re.search(r"\b(judge|okc)\s+(submit|check|withdraw)\b", c.name)]
    for c in reg.all():
        if c.name.startswith("okc "):
            assert c.effects.network == "okc_readonly" and c.effects.quota == "never"


def test_readonly_wrapper_rejects_quota_subcommands(fake_runner):
    import pytest

    from kdev.contract import ErrorCode, KdevError
    from kdev.judge import okc_cli
    r = fake_runner({})
    for sub in ("submit", "check", "withdraw"):
        with pytest.raises(KdevError) as e:
            okc_cli.okc_readonly(sub, ["x.py"], runner=r)
        assert e.value.code == ErrorCode.QUOTA_FORBIDDEN
    assert r.calls == []
