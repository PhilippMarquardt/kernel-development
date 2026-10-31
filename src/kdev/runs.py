"""Run directories and paged reads (`kdev show`). Owner: P1. See docs/PLAN.md sections 3.3 and 6.

Layout: $KDEV_HOME/runs/<yyyymmdd-hhmmss>-<tool>-<slug>/ with meta.json, result.json, raw.jsonl, log.txt and
tool artifacts.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kdev.config import Config
from kdev.contract import ErrorCode, KdevError, Result
from kdev.util import atomic_write_json, atomic_write_text, slugify, utc_now

#: Built-in sections of every run (besides the keys of ``result.files``).
BUILTIN_SECTIONS = {"meta": "meta.json", "result": "result.json", "log": "log.txt", "raw": "raw.jsonl"}
MAX_LINE_CHARS = 400   # longer lines are clipped with an explicit marker


def result_doc(result: Result) -> dict[str, Any]:
    """The result.json document: the ``kdev.result/1`` envelope plus ``lines`` (the tool's own rendering,
    which ``Result.to_dict`` leaves out of the stdout envelope). ``Result.from_dict`` restores ``lines``."""
    d = result.to_dict()
    if result.lines:
        d["lines"] = list(result.lines)
    return d


def write_result_files(run_dir: str | os.PathLike[str], result: Result) -> Path:
    """Write result.json (:func:`result_doc`) and summary.txt (what ``kdev show RUN`` prints) into
    ``run_dir``. Every writer of result.json (CLI, worker, service, build lane) goes through this, so a run
    always has both and they agree."""
    from kdev.output import render

    d = Path(run_dir)
    try:
        atomic_write_text(d / "summary.txt", render(result, quiet=False) + "\n")
    except Exception:  # noqa: BLE001 - the summary is a convenience; never lose result.json over it
        pass
    return atomic_write_json(d / "result.json", result_doc(result))


@dataclass
class RunDir:
    run_id: str
    path: Path

    def file(self, name: str) -> Path:
        return self.path / name

    def write_json(self, name: str, obj: Any) -> Path:
        """Atomic write (tmp + os.replace) of a JSON document."""
        return atomic_write_json(self.file(name), obj)

    def append_jsonl(self, name: str, obj: Any) -> None:
        with open(self.file(name), "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(obj, separators=(",", ":"), default=str) + "\n")

    def log(self, message: str) -> None:
        """Append a timestamped line to log.txt."""
        with open(self.file("log.txt"), "a", encoding="utf-8", newline="\n") as f:
            f.write(f"{utc_now()} {message.rstrip()}\n")

    def write_result(self, result: Result) -> Path:
        """Write result.json; fills result.run_id, result.files['result'] and result.files['dir']."""
        result.run_id = self.run_id
        p = self.file("result.json")
        result.files["result"] = str(p)
        result.files["dir"] = str(self.path)
        for name, key in (("raw.jsonl", "raw"), ("log.txt", "log")):
            if key not in result.files and self.file(name).exists():
                result.files[key] = str(self.file(name))
        return write_result_files(self.path, result)

    def read_result(self) -> Result | None:
        p = self.file("result.json")
        if not p.is_file():
            return None
        try:
            return Result.from_dict(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            return None


@dataclass
class Page:
    """One page of a run section."""

    run_id: str
    section: str
    offset: int
    limit: int
    lines: list[str] = field(default_factory=list)
    total: int | None = None
    next_offset: int | None = None   # None = end
    path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"run_id": self.run_id, "section": self.section, "offset": self.offset, "limit": self.limit,
                "total": self.total, "next_offset": self.next_offset, "path": self.path, "lines": self.lines}


def _stamp() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d-%H%M%S")


def new_run(cfg: Config, tool: str, slug: str = "") -> RunDir:
    """Create a new run dir (id = <yyyymmdd-hhmmss>-<tool>[-<slug>], uniquified)."""
    base = f"{_stamp()}-{slugify(tool) or 'run'}"
    s = slugify(slug)
    if s:
        base += f"-{s}"
    cfg.runs_dir.mkdir(parents=True, exist_ok=True)
    for i in range(1, 1000):
        run_id = base if i == 1 else f"{base}-{i}"
        path = cfg.runs_dir / run_id
        try:
            path.mkdir()
        except FileExistsError:
            continue
        return RunDir(run_id, path)
    raise KdevError(ErrorCode.INTERNAL, f"cannot create a unique run dir for {base}")


def _is_run_dir(p: Path) -> bool:
    return p.is_dir() and any((p / n).exists() for n in ("result.json", "meta.json", "log.txt"))


def open_run(cfg: Config, run_id: str) -> RunDir:
    """Find an existing run by id or unique prefix; KdevError(not_found) otherwise.

    Also accepts a run-dir path. An ambiguous prefix is ``invalid_request`` listing the candidates."""
    rid = run_id.strip().rstrip("/\\")
    if not rid:
        raise KdevError(ErrorCode.INVALID_REQUEST, "empty run id", details={"field": "run_id"},
                        recovery="pass a run id from `kdev runs list`")
    as_path = Path(rid).expanduser()
    if ("/" in rid or "\\" in rid) and _is_run_dir(as_path):
        return RunDir(as_path.name, as_path)
    root = cfg.runs_dir
    exact = root / rid
    if exact.is_dir():
        return RunDir(rid, exact)
    matches = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.startswith(rid)) \
        if root.is_dir() else []
    if not matches:
        # also allow the trailing part of the id (e.g. "ab-tile128" or a slug)
        matches = sorted(p.name for p in root.iterdir() if p.is_dir() and rid in p.name) if root.is_dir() else []
    if len(matches) == 1:
        return RunDir(matches[0], root / matches[0])
    if not matches:
        raise KdevError(ErrorCode.NOT_FOUND, f"run {rid!r} not found under {root}",
                        details={"run_id": rid, "runs_dir": str(root)},
                        recovery="list runs with `kdev runs list`")
    raise KdevError(ErrorCode.INVALID_REQUEST, f"run id {rid!r} is ambiguous ({len(matches)} matches)",
                    details={"run_id": rid, "matches": matches[-10:]},
                    recovery="use a longer prefix or the full id")


def _tool_of(run_id: str) -> str:
    parts = run_id.split("-")
    return parts[2] if len(parts) >= 3 else ""


def list_runs(cfg: Config, tool: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
    """Newest first: ``{run_id, tool, status, verdict, created_utc, dir}``. ``tool`` filters on the result's
    tool or the id's tool component (spaces/hyphens equivalent)."""
    root = cfg.runs_dir
    if not root.is_dir():
        return []
    want = slugify(tool) if tool else None
    out: list[dict[str, Any]] = []
    for p in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name, reverse=True):
        rd = RunDir(p.name, p)
        res = rd.read_result()
        rtool = res.tool if res else _tool_of(p.name)
        tail = p.name.split("-", 2)[-1]
        if want and slugify(rtool) != want and not (tail == want or tail.startswith(want + "-")):
            continue
        created = ""
        if len(p.name) >= 15:
            try:
                created = _dt.datetime.strptime(p.name[:15], "%Y%m%d-%H%M%S").strftime("%Y-%m-%dT%H:%M:%SZ")
            except ValueError:
                created = ""
        out.append({"run_id": p.name, "tool": rtool, "status": res.status if res else "incomplete",
                    "verdict": (res.verdict if res else "") or "", "created_utc": created, "dir": str(p)})
        if len(out) >= max(0, limit):
            break
    return out


def section_names(rd: RunDir) -> list[str]:
    """Sections available for a run: built-ins that exist + ``result.files`` keys + 'summary'."""
    names = ["summary"]
    for sec, fname in BUILTIN_SECTIONS.items():
        if rd.file(fname).exists():
            names.append(sec)
    res = rd.read_result()
    if res:
        for k, v in res.files.items():
            if k not in names and k != "dir" and v and Path(v).is_file():
                names.append(k)
    return names


def _resolve_section(rd: RunDir, section: str) -> tuple[Path | None, list[str] | None]:
    """(file path, None) or (None, generated lines) for a section name."""
    res = rd.read_result()
    if section == "summary":
        if rd.file("summary.txt").is_file():
            return rd.file("summary.txt"), None
        if res is None:
            raise KdevError(ErrorCode.NOT_FOUND, f"run {rd.run_id} has no result.json yet",
                            details={"run_id": rd.run_id, "sections": section_names(rd)},
                            recovery=f"try `kdev show {rd.run_id} --section log`")
        from kdev.output import render
        return None, render(res).splitlines()
    if section in BUILTIN_SECTIONS:
        p = rd.file(BUILTIN_SECTIONS[section])
        if section in ("meta", "result") and p.is_file():
            try:
                return None, json.dumps(json.loads(p.read_text(encoding="utf-8")), indent=2,
                                        default=str).splitlines()
            except ValueError:
                return p, None
        if p.is_file():
            return p, None
    if res is not None and section in res.files and section != "dir":
        p = Path(res.files[section])
        if not p.is_absolute():
            p = rd.path / p
        if p.is_file():
            return p, None
    # any file inside the run dir by name (no escaping the dir)
    cand = (rd.path / section).resolve()
    try:
        cand.relative_to(rd.path.resolve())
    except ValueError:
        cand = None  # type: ignore[assignment]
    if cand is not None and cand.is_file():
        return cand, None
    raise KdevError(ErrorCode.NOT_FOUND, f"section {section!r} not found in run {rd.run_id}",
                    details={"run_id": rd.run_id, "sections": section_names(rd)},
                    recovery=f"use one of: {', '.join(section_names(rd))}")


def _clip(line: str) -> str:
    line = line.rstrip("\r\n")
    if len(line) > MAX_LINE_CHARS:
        return line[:MAX_LINE_CHARS] + f" [... +{len(line) - MAX_LINE_CHARS} chars]"
    return line


def show(cfg: Config, run_id: str, section: str = "summary", offset: int = 0, limit: int = 60) -> Page:
    """Paged read of a section: any key of result.files, or summary/meta/log/raw."""
    if offset < 0 or limit <= 0:
        raise KdevError(ErrorCode.INVALID_REQUEST, f"bad page offset={offset} limit={limit}",
                        details={"field": "offset" if offset < 0 else "limit"},
                        recovery="use --offset >= 0 and --limit >= 1")
    rd = open_run(cfg, run_id)
    path, gen = _resolve_section(rd, section)
    page = Page(rd.run_id, section, offset, limit, path=str(path) if path else str(rd.path))
    if gen is not None:
        page.total = len(gen)
        page.lines = [_clip(x) for x in gen[offset:offset + limit]]
    else:
        assert path is not None
        total = 0
        with open(path, encoding="utf-8", errors="replace", newline="") as f:
            for i, line in enumerate(f):
                if offset <= i < offset + limit:
                    page.lines.append(_clip(line))
                total = i + 1
        page.total = total
    end = offset + len(page.lines)
    page.next_offset = end if page.total is not None and end < page.total else None
    return page


def run_dir_size(rd: RunDir) -> int:
    """Total bytes of a run dir (dashboard/doctor)."""
    total = 0
    for dp, _dn, fns in os.walk(rd.path):
        for fn in fns:
            try:
                total += os.path.getsize(os.path.join(dp, fn))
            except OSError:
                pass
    return total
