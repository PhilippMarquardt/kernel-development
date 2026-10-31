"""Adapter lookup. Owner: P7.

Resolution of ``--problem P`` / a file path: (1) explicit adapter:problem ('okc:muse_moe_w4a16'); (2) a
problem dir containing reference.py + TASK.md -> okc; (3) a .cu with main() -> cu_main; (4) a file
defining Model/ModelNew -> kernelbench; (5) the directory of the submission file walking upwards;
(6) config default_problem. Unknown -> KdevError(invalid_request).

A bare problem name with a shipped TOML (``muse_moe_w4a16``) resolves to the okc adapter; its problem dir is
searched under $KDEV_PROBLEMS_ROOT, config.problems_root, the TOML's search_dirs, /root/okc and ~/okc. The
adapter still works without the dir for the pure methods (shapes, rules, score).
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path

from kdev.adapters.base import Adapter
from kdev.contract import ErrorCode, KdevError

ADAPTERS = {
    "okc": "kdev.adapters.okc:OkcAdapter",
    "cu_main": "kdev.adapters.cu_main:CuMainAdapter",
    "kernelbench": "kdev.adapters.kernelbench:KernelBenchAdapter",
}


def _cls(name: str):
    if name not in ADAPTERS:
        raise KdevError(ErrorCode.INVALID_REQUEST, f"unknown adapter {name!r}", details={"known": sorted(ADAPTERS)})
    mod, _, attr = ADAPTERS[name].partition(":")
    return getattr(importlib.import_module(mod), attr)


def _config_value(key: str) -> str:
    try:
        from kdev.config import load_config
        return str(getattr(load_config(), key, "") or "")
    except Exception:  # noqa: BLE001 - config is P1's; absent/unimplemented config = no default
        return ""


def _roots() -> list[str]:
    r = _config_value("problems_root")
    return [r] if r else []


def detect(file: str) -> tuple[str, str] | None:
    """(adapter name, problem dir) for a submission path, or None."""
    from kdev.adapters import cu_main, kernelbench, okc

    p = Path(file).expanduser()
    if p.is_dir() and okc.is_problem_dir(p):
        return "okc", str(p.resolve())
    if p.suffix in (".cu", ".cuh") and cu_main.has_main(str(p)):
        return "cu_main", str(p.resolve())
    if p.is_file():
        for d in [p.resolve().parent, *p.resolve().parent.parents]:
            if okc.is_problem_dir(d):
                return "okc", str(d)
        if p.suffix == ".py" and kernelbench.looks_like_kernelbench(str(p)):
            return "kernelbench", str(p.resolve())
    return None


def _okc(problem: str):
    from kdev.adapters.okc import OkcAdapter
    return OkcAdapter(problem, roots=_roots())


def get_adapter(problem: str | None = None, file: str | None = None, adapter: str | None = None) -> Adapter:
    """Resolve an adapter (see module docstring). ``problem`` may be 'adapter:problem', a dir, a name or a file."""
    from kdev.adapters import okc

    if problem and ":" in problem and problem.split(":", 1)[0] in ADAPTERS and not os.path.exists(problem):
        adapter, problem = problem.split(":", 1)
    if adapter:
        target = problem or file
        if not target:
            raise KdevError(ErrorCode.INVALID_REQUEST, f"adapter {adapter!r} needs a problem or file")
        if adapter == "okc":
            return _okc(target)
        return _cls(adapter)(target)
    if problem:
        pp = Path(problem).expanduser()
        if pp.is_dir() and okc.is_problem_dir(pp):
            return _okc(str(pp))
        if Path(problem).name in okc.known_problems() or problem in okc.known_problems():
            return _okc(problem)
        if pp.exists():
            hit = detect(str(pp))
            if hit:
                return _okc(hit[1]) if hit[0] == "okc" else _cls(hit[0])(hit[1])
        found = okc.find_problem_dir(problem, None, _roots())
        if found:
            return _okc(found)
        raise KdevError(ErrorCode.INVALID_REQUEST, f"unknown problem {problem!r}",
                        details={"known": okc.known_problems()},
                        recovery="pass adapter:problem, a problem dir (TASK.md + reference.py) or a known name")
    if file:
        hit = detect(file)
        if hit:
            return _okc(hit[1]) if hit[0] == "okc" else _cls(hit[0])(hit[1])
    default = _config_value("default_problem") or os.environ.get("KDEV_PROBLEM", "")
    if default:
        return get_adapter(problem=default)
    raise KdevError(ErrorCode.INVALID_REQUEST, f"cannot tell the problem of {file or '(no file)'}",
                    recovery="pass --problem NAME|DIR (e.g. muse_moe_w4a16) or put the file in its problem dir")
