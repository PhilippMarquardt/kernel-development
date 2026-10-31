"""Source instrumentation for timelines. Owner: P6.

Finds ``// @kdev:mark NAME`` annotations in CUDA sources (standalone .cu or load_inline strings inside a .py),
assigns ids, inserts ``KD_MARK(id);`` at those lines, adds the marks.cuh include + the kd_marks_buf
definition + a host ``kd_marks_init`` binding, and returns the instrumented copy + id->name map.

Annotation forms (one per line):
  ``// @kdev:mark NAME``            own line: ``KD_MARK(id);`` replaces it (lane 0 of every warpgroup records)
  ``stmt; // @kdev:mark NAME``      trailing: the mark goes after the statement (line must end in ; { or })
  ``// @kdev:mark NAME cta``        only thread 0 of the CTA records (``KD_MARK_CTA``)
  ``// @kdev:mark NAME thread``     the calling thread records (``KD_MARK_T``; you guard it yourself)
  ``// @kdev:mark sm``              record the SM id of the CTA (``KD_MARK_SM``)
The same NAME used twice gets the same id (e.g. the two branches of a route). Explicit ``KD_MARK(3); // NAME``
calls (``--marks header``) are kept; their name comes from the trailing comment. ``#include "kdev/marks.cuh"``
lines are replaced by the inlined header, so no include path is needed. The inlined header starts with
``#define KDEV_MARKS 1`` (unless ``KDEV_MARKS_OFF``), so the copy is instrumented even if a build path drops
the ``-DKDEV_MARKS`` define. Python files: only string literals that contain marks are changed (located with
``tokenize``); strings that look like ``str.format`` templates (contain ``{{``) get their braces doubled.
"""
from __future__ import annotations

import io
import json
import re
import tokenize
from dataclasses import dataclass, field
from pathlib import Path

from kdev.contract import ErrorCode, KdevError

MARKS_CUH = Path(__file__).resolve().parent / "marks.cuh"
DEFAULT_MAX_MARKS = 32

_ANN = re.compile(r"//\s*@kdev:mark\s+([A-Za-z_][A-Za-z0-9_.:/+-]*)(?:\s+(wg|cta|thread))?\s*$")
_INC = re.compile(r'^\s*#\s*include\s*[<"](?:kdev/)?marks\.cuh[>"]\s*$')
_EXPLICIT = re.compile(r"\bKD_MARK(?:_CTA|_T)?\s*\(\s*(\d+)\s*\)\s*;?\s*(?://\s*(?!@kdev)([A-Za-z_][\w.:/+-]*))?")
_ANY_MARK = re.compile(r"\bKD_MARK(?:_CTA|_T|_SM)?\s*\(")
_MACRO = {"wg": "KD_MARK", "cta": "KD_MARK_CTA", "thread": "KD_MARK_T"}


@dataclass
class Instrumented:
    path: str                       # instrumented copy (in the run dir)
    marks: dict[int, str] = field(default_factory=dict)
    defines: list[str] = field(default_factory=lambda: ["-DKDEV_MARKS"])
    warnings: list[str] = field(default_factory=list)
    source: str = ""                # original path
    marks_json: str = ""            # id -> name map written next to the copy


def find_annotations(source: str) -> list[tuple[int, str]]:
    """[(line_no, name)] of '// @kdev:mark NAME' (pure; 1-based line numbers)."""
    out = []
    for i, line in enumerate(source.splitlines(), 1):
        m = _ANN.search(line)
        if m:
            out.append((i, m.group(1)))
    return out


def prelude(n_marks: int = 0, brace_double: bool = False) -> str:
    """The inlined marks.cuh (comments and #pragma once dropped) behind an auto ``#define KDEV_MARKS 1``."""
    body = [ln for ln in MARKS_CUH.read_text(encoding="utf-8").splitlines()
            if not ln.lstrip().startswith("//") and ln.strip() != "#pragma once"]
    head = ["// ---- kdev timeline marks (inlined marks.cuh) ----", "#ifndef KDEV_MARKS_OFF", "#ifndef KDEV_MARKS",
            "#define KDEV_MARKS 1", "#endif", "#endif"]
    if n_marks > DEFAULT_MAX_MARKS:
        head += ["#ifndef KD_MAX_MARKS", f"#define KD_MAX_MARKS {min(254, n_marks)}", "#endif"]
    text = "\n".join(head + body + ["// ---- end kdev marks ----", ""])
    if brace_double:
        text = text.replace("{", "{{").replace("}", "}}")
    return text


class _Ids:
    def __init__(self, reserved: set[int]) -> None:
        self.by_name: dict[str, int] = {}
        self.names: dict[int, str] = {}
        self.reserved = set(reserved)
        self.next = 0

    def get(self, name: str) -> int:
        if name in self.by_name:
            return self.by_name[name]
        while self.next in self.reserved or self.next in self.names:
            self.next += 1
        i = self.next
        self.by_name[name] = i
        self.names[i] = name
        return i


def _placeable(code: str) -> bool:
    """A mark can follow ``code`` on the same line (empty, or a complete statement / block brace)."""
    c = code.rstrip()
    return c == "" or c.endswith((";", "{", "}"))


def _register_ids(text: str, ids: _Ids) -> None:
    for line in text.split("\n"):
        m = _ANN.search(line)
        if m and m.group(1) != "sm" and _placeable(line[: m.start()]):
            ids.get(m.group(1))


def _transform_cuda(text: str, ids: _Ids, warnings: list[str], where: str, brace_double: bool,
                    n_marks_hint: int) -> tuple[str, bool]:
    """Transform one CUDA text; returns (text, changed)."""
    lines = text.split("\n")
    changed = False
    has_prelude = False
    for i, line in enumerate(lines):
        if _INC.match(line):
            lines[i] = prelude(n_marks_hint, brace_double).rstrip("\n")
            has_prelude = changed = True
            continue
        m = _ANN.search(line)
        if not m:
            continue
        name, mode = m.group(1), m.group(2) or "wg"
        code = line[: m.start()]
        ann = line[m.start():].strip()
        if not _placeable(code):
            warnings.append(f"{where}:{i + 1}: trailing @kdev:mark {name} after a line not ending in ; {{ or }} "
                            "-- skipped (put the annotation on its own line)")
            continue
        stmt = "KD_MARK_SM();" if name == "sm" else f"{_MACRO[mode]}({ids.get(name)});"
        if code.strip() == "":
            lines[i] = f"{code}{stmt} {ann}"
        else:
            lines[i] = f"{code.rstrip()} {stmt} {ann}"
        changed = True
    out = "\n".join(lines)
    if (changed or _ANY_MARK.search(out)) and not has_prelude:
        out = prelude(n_marks_hint, brace_double) + out
        changed = True
    return out, changed


def _explicit_ids(text: str) -> dict[int, str]:
    found: dict[int, str] = {}
    for m in _EXPLICIT.finditer(text):
        i = int(m.group(1))
        if m.group(2):
            found[i] = m.group(2)
        else:
            found.setdefault(i, f"m{i}")
    return found


def instrument_source(source: str) -> tuple[str, dict[int, str]]:
    """Pure transform of one CUDA source string."""
    warnings: list[str] = []
    explicit = _explicit_ids(source)
    ids = _Ids(set(explicit))
    n_hint = len(find_annotations(source)) + len(explicit)
    out, _ = _transform_cuda(source, ids, warnings, "<source>", False, max(n_hint, max(explicit, default=-1) + 1))
    marks = dict(explicit)
    marks.update(ids.names)
    return out, dict(sorted(marks.items()))


def _string_body(tok: str) -> tuple[str, str, str, str] | None:
    """(prefix, quote, body, suffix) of a Python string literal token."""
    m = re.match(r"(?is)^([rbuf]*)('''|\"\"\"|'|\")", tok)
    if not m:
        return None
    prefix, q = m.group(1), m.group(2)
    if not tok.endswith(q) or len(tok) < len(prefix) + 2 * len(q):
        return None
    return prefix, q, tok[len(prefix) + len(q): len(tok) - len(q)], q


def instrument_python(text: str, where: str = "<py>") -> tuple[str, dict[int, str], list[str]]:
    """Instrument CUDA string literals inside a Python module (pure)."""
    warnings: list[str] = []
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, SyntaxError) as e:
        raise KdevError(ErrorCode.INVALID_REQUEST, f"{where}: cannot tokenize: {e}") from e
    line_starts = [0]
    for ln in text.splitlines(keepends=True):
        line_starts.append(line_starts[-1] + len(ln))

    def off(rc: tuple[int, int]) -> int:
        return line_starts[rc[0] - 1] + rc[1]

    cands = []
    for t in toks:
        if t.type != tokenize.STRING:
            continue
        s = t.string
        if "@kdev:mark" in s or _ANY_MARK.search(s) or "marks.cuh" in s:
            cands.append(t)
    # (f-strings tokenize as FSTRING_* on 3.12+ and are never STRING tokens: they are not rewritten)
    explicit: dict[int, str] = {}
    for t in cands:
        explicit.update(_explicit_ids(t.string))
    ids = _Ids(set(explicit))
    for t in cands:                       # ids in source order (strings are rewritten back to front below)
        _register_ids(t.string, ids)
    n_hint = sum(len(find_annotations(t.string)) for t in cands) + len(explicit)
    n_hint = max(n_hint, max(explicit, default=-1) + 1)
    out = text
    for t in sorted(cands, key=lambda t: t.start, reverse=True):
        parts = _string_body(t.string)
        if parts is None:
            warnings.append(f"{where}:{t.start[0]}: cannot parse string literal; skipped")
            continue
        prefix, q, body, suf = parts
        if "f" in prefix.lower():
            warnings.append(f"{where}:{t.start[0]}: f-string CUDA source not instrumented (use a plain string)")
            continue
        if len(q) == 1 and "@kdev:mark" in body:
            warnings.append(f"{where}:{t.start[0]}: annotation inside a single-quoted string; use triple quotes")
            continue
        brace = "{{" in body or "}}" in body
        new_body, changed = _transform_cuda(body, ids, warnings, f"{where}:{t.start[0]}", brace, n_hint)
        if not changed:
            continue
        a, b = off(t.start), off(t.end)
        out = out[:a] + prefix + q + new_body + suf + out[b:]
    marks = dict(explicit)
    marks.update(ids.names)
    return out, dict(sorted(marks.items())), warnings


def instrument_file(path: str, out_dir: str) -> Instrumented:
    """Write an instrumented copy (same basename) into ``out_dir`` plus ``marks.json``."""
    src = Path(path)
    if not src.is_file():
        raise KdevError(ErrorCode.NOT_FOUND, f"no such file {path}")
    od = Path(out_dir)
    if od.resolve() == src.parent.resolve():
        raise KdevError(ErrorCode.INVALID_REQUEST, "out_dir must differ from the source directory",
                        recovery="instrumented copies go into the run dir")
    od.mkdir(parents=True, exist_ok=True)
    text = src.read_text(encoding="utf-8")
    warnings: list[str] = []
    if src.suffix == ".py":
        new, marks, warnings = instrument_python(text, src.name)
    else:
        new, marks = instrument_source(text)
    if not marks:
        warnings.append("no // @kdev:mark annotations or KD_MARK(n) calls found")
    dst = od / src.name
    dst.write_text(new, encoding="utf-8", newline="\n")
    mj = od / "marks.json"
    mj.write_text(json.dumps({"source": str(src), "copy": str(dst), "marks": {str(k): v for k, v in marks.items()}},
                             indent=1), encoding="utf-8")
    return Instrumented(path=str(dst), marks=marks, warnings=warnings, source=str(src), marks_json=str(mj))
