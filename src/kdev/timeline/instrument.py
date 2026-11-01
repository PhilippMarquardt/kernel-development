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
  ``// @kdev:acc NAME decl|begin|end|flush``   per-thread clock accumulator (KD_ACC_*); ``flush`` writes the
                                    accumulated cycles as a value record named NAME (call it from one thread per
                                    role, e.g. the producer lane and consumer thread 0). Names containing ``wait``
                                    count as waits in the per-role accounting.
  ``// @kdev:names A=x B=y`` / ``// @kdev:roles 0-1=consumer 2=producer`` / ``// @kdev:t0 NAME``   decode hints
                                    (also as ``#`` comments in the Python file).
The same NAME used twice gets the same id (e.g. the two branches of a route). Explicit ``KD_MARK(3); // NAME``
calls (``--marks header``) are kept; their name comes from the trailing comment. ``#include "kdev/marks.cuh"``
lines are replaced by the inlined header, so no include path is needed. The inlined header starts with
``#define KDEV_MARKS 1`` (unless ``KDEV_MARKS_OFF``), so the copy is instrumented even if a build path drops
the ``-DKDEV_MARKS`` define.

Agent mark styles (``styles``): one-line ``#define NAME(k) ...`` macros that store a timer
(``gtimer()`` / ``%globaltimer`` / ``clock64()``) into a debug buffer -- e.g. ``DBG(k)`` ->
``p.dbg[blockIdx.x * 8 + k] = gtimer()``, ``MWT(code, info)`` (per-CTA log with an info word), and two-argument
value macros on the same buffer (``DBGV(k, v)``) -- are rewritten to kdev marks: ``KD_MARK_CTA(base + (k))``
(thread 0, when the body tests ``threadIdx.x == 0`` / ``tid == 0``), else ``KD_MARK_T``; ``KD_MARK_CTA_I`` for
two-argument timed macros, ``KD_VAL_*`` for value macros. Mark names are ``NAME<k>`` (``DBG6``, ``MWT30``).
The macro's own gates (``OKC_DEBUG &&``, ``p.dbg`` non-null) go away with the body, so the file needs no extra
buffer argument: the kdev buffer comes in through ``kd_marks_init``. Only families with call sites are rewritten.
``styles="none"`` disables this, a comma list keeps only those macro names.

Layout (``layout``): ``slots`` (one record per CTA/warpgroup/mark, no atomics; loops keep the last iteration),
``ctalog`` (per CTA/warpgroup log: every iteration), ``log`` (global append log). ``auto`` picks ``ctalog`` when a
two-argument timed macro (a segment log like ``MWT``) or a ``@kdev:acc`` is present, else ``slots``.

Python files: string literals that contain marks or mark macros are changed (located with ``tokenize``);
triple-quoted strings are edited in place, single-quoted ones are decoded, edited and re-emitted with ``repr``;
strings that look like ``str.format`` templates (contain ``{{``) get their braces doubled.
"""
from __future__ import annotations

import ast
import io
import json
import re
import tokenize
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kdev.contract import ErrorCode, KdevError

MARKS_CUH = Path(__file__).resolve().parent / "marks.cuh"
DEFAULT_MAX_MARKS = 32
MAX_IDS = 254
LAYOUTS = ("auto", "slots", "ctalog", "log")

_ANN = re.compile(r"//\s*@kdev:mark\s+([A-Za-z_][A-Za-z0-9_.:/+-]*)(?:\s+(wg|cta|thread))?\s*$")
_ACC = re.compile(r"//\s*@kdev:acc\s+([A-Za-z_][A-Za-z0-9_]*)\s+(decl|begin|end|flush)\s*$")
_HINT = re.compile(r"@kdev:(names|roles|t0)\s+(.+?)\s*(?:\*/)?$", re.M)
_INC = re.compile(r'^\s*#\s*include\s*[<"](?:kdev/)?marks\.cuh[>"]\s*$')
_EXPLICIT = re.compile(r"\bKD_MARK(?:_CTA|_T)?\s*\(\s*(\d+)\s*\)\s*;?\s*(?://\s*(?!@kdev)([A-Za-z_][\w.:/+-]*))?")
_ANY_MARK = re.compile(r"\bKD_(?:MARK(?:_CTA|_T|_SM)?(?:_I)?|VAL_T|VAL_CTA|ACC_\w+)\s*\(")
_MACRO = {"wg": "KD_MARK", "cta": "KD_MARK_CTA", "thread": "KD_MARK_T"}
_DEF = re.compile(r"^(\s*)#\s*define\s+([A-Za-z_]\w*)\(\s*([A-Za-z_]\w*)\s*(?:,\s*([A-Za-z_]\w*)\s*)?\)\s+(.*)$")
_TIMER = re.compile(r"gtimer\s*\(|%+globaltimer|globaltimer\s*\(|clock64\s*\(|__globaltimer\s*\(")
_STORE = re.compile(r"([A-Za-z_]\w*(?:(?:\.|->)[A-Za-z_]\w*)*)\s*\[[^\]]*\]\s*=(?!=)")
_GUARD0 = re.compile(r"\b(?:threadIdx\.x|tid)\s*==\s*0\b")
_CG_POLL = re.compile(r"ld\.global\.cg")


@dataclass
class Family:
    """One agent mark macro (``DBG(k)``, ``MWT(c, inf)``, ``DBGV(k, v)``) rewritten to kdev marks."""
    name: str
    kind: str                    # timed | info | value
    thread0: bool
    buffer: str
    range: int = 64
    base: int = 0
    calls: int = 0


@dataclass
class Instrumented:
    path: str                       # instrumented copy (in the run dir)
    marks: dict[int, str] = field(default_factory=dict)
    defines: list[str] = field(default_factory=lambda: ["-DKDEV_MARKS"])
    warnings: list[str] = field(default_factory=list)
    source: str = ""                # original path
    marks_json: str = ""            # id -> name map written next to the copy
    families: list[dict[str, Any]] = field(default_factory=list)
    layout: str = "slots"
    hints: dict[str, Any] = field(default_factory=dict)   # names / roles / t0 from @kdev: comments


def find_annotations(source: str) -> list[tuple[int, str]]:
    """[(line_no, name)] of '// @kdev:mark NAME' (pure; 1-based line numbers)."""
    out = []
    for i, line in enumerate(source.splitlines(), 1):
        m = _ANN.search(line)
        if m:
            out.append((i, m.group(1)))
    return out


def parse_kv(text: str) -> dict[str, str]:
    """'A=x, B=y C=z' -> {A: x, B: y, C: z}."""
    out: dict[str, str] = {}
    for part in re.split(r"[,\s]+", text or ""):
        k, sep, v = part.partition("=")
        if sep and k and v:
            out[k] = v
    return out


def parse_roles(text: str | dict[str, str]) -> dict[int, str]:
    """'0-1=consumer 2=producer' -> {0: consumer, 1: consumer, 2: producer} (warpgroup -> role)."""
    kv = parse_kv(text) if isinstance(text, str) else dict(text)
    out: dict[int, str] = {}
    for k, v in kv.items():
        a, _, b = str(k).partition("-")
        try:
            lo, hi = int(a), int(b or a)
        except ValueError:
            continue
        for w in range(lo, hi + 1):
            out[w] = v
    return out


def hints(text: str) -> dict[str, Any]:
    """``@kdev:names`` / ``@kdev:roles`` / ``@kdev:t0`` comments anywhere in a file (pure)."""
    names: dict[str, str] = {}
    roles: dict[str, str] = {}
    t0 = None
    for m in _HINT.finditer(text):
        kind, val = m.group(1), m.group(2).strip().strip("'\"")
        if kind == "names":
            names.update(parse_kv(val))
        elif kind == "roles":
            roles.update(parse_kv(val))
        else:
            t0 = val.split()[0]
    out: dict[str, Any] = {}
    if names:
        out["names"] = names
    if roles:
        out["roles"] = roles
    if t0:
        out["t0"] = t0
    return out


def prelude(n_marks: int = 0, brace_double: bool = False, layout: str = "slots") -> str:
    """The inlined marks.cuh (comments and #pragma once dropped) behind an auto ``#define KDEV_MARKS 1``."""
    body = [ln for ln in MARKS_CUH.read_text(encoding="utf-8").splitlines()
            if not ln.lstrip().startswith("//") and ln.strip() != "#pragma once"]
    head = ["// ---- kdev timeline marks (inlined marks.cuh) ----", "#ifndef KDEV_MARKS_OFF", "#ifndef KDEV_MARKS",
            "#define KDEV_MARKS 1", "#endif", "#endif"]
    if n_marks > DEFAULT_MAX_MARKS:
        head += ["#ifndef KD_MAX_MARKS", f"#define KD_MAX_MARKS {min(MAX_IDS, n_marks)}", "#endif"]
    if layout == "ctalog":
        head += ["#ifndef KD_MARKS_CTALOG", "#define KD_MARKS_CTALOG 1", "#endif"]
    elif layout == "log":
        head += ["#ifndef KD_MARKS_LOG", "#define KD_MARKS_LOG 1", "#endif"]
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
        a = _ACC.search(line)
        if a and a.group(2) == "flush":
            ids.get(a.group(1))


# ------------------------------------------------------------------------------------ agent mark macros

def detect_families(text: str) -> list[Family]:
    """Agent mark macros defined in ``text`` (pure). Timed: the body stores a timer read into a buffer; value:
    a two-argument macro storing its 2nd argument into the buffer of a timed family."""
    timed: list[Family] = []
    vals: list[tuple[str, str, bool, str]] = []
    for line in text.split("\n"):
        m = _DEF.match(line)
        if not m or m.group(2).startswith(("KD_", "kd_")):
            continue
        name, a1, a2, body = m.group(2), m.group(3), m.group(4), m.group(5)
        sm = _STORE.search(body)
        if not sm:
            continue
        buf = sm.group(1)
        thread0 = bool(_GUARD0.search(body))
        if _TIMER.search(body):
            st = re.search(r"\*\s*(\d+)\s*\+\s*\(?\s*" + re.escape(a1) + r"\b", body)
            rng = max(32, int(st.group(1))) if st and 0 < int(st.group(1)) <= 128 else 64
            kind = "info" if a2 and re.search(r"\b" + re.escape(a2) + r"\b", body) else "timed"
            timed.append(Family(name, kind, thread0, buf, rng))
        elif a2 and re.search(r"\]\s*=\s*\(?\s*" + re.escape(a2) + r"\b", body):
            vals.append((name, buf, thread0, a1))
    out = list(timed)
    tbufs = {f.buffer for f in timed}
    for name, buf, thread0, _ in vals:
        if buf and buf in tbufs:
            out.append(Family(name, "value", thread0, buf, 64))
    return out


def _count_calls(texts: list[str], fams: list[Family]) -> None:
    for f in fams:
        pat = re.compile(r"\b" + re.escape(f.name) + r"\s*\(")
        n = 0
        for t in texts:
            for line in t.split("\n"):
                d = _DEF.match(line)
                if d and d.group(2) == f.name:
                    continue
                n += len(pat.findall(line))
        f.calls = n


def _family_line(m: re.Match[str], f: Family) -> str:
    ind, a1, a2 = m.group(1), m.group(3), m.group(4)
    idx = f"{f.base} + ({a1})"
    if f.kind == "timed":
        mac = "KD_MARK_CTA" if f.thread0 else "KD_MARK_T"
        return f"{ind}#define {f.name}({a1}) {mac}({idx})"
    if f.kind == "info":
        mac = "KD_MARK_CTA_I" if f.thread0 else "KD_MARK_T_I"
        return f"{ind}#define {f.name}({a1}, {a2}) {mac}({idx}, ({a2}))"
    mac = "KD_VAL_CTA" if f.thread0 else "KD_VAL_T"
    return f"{ind}#define {f.name}({a1}, {a2}) {mac}({idx}, ({a2}))"


def select_families(fams: list[Family], styles: str | None) -> list[Family]:
    if styles in ("none", "off", ""):
        return []
    keep = [f for f in fams if f.calls > 0]
    if styles and styles != "auto":
        want = {s.strip() for s in styles.split(",") if s.strip()}
        keep = [f for f in keep if f.name in want]
    return keep


def _assign_bases(fams: list[Family], first_free: int, warnings: list[str]) -> int:
    """Give each family an id range after the annotation ids; returns the id count (KD_MAX_MARKS)."""
    nxt = first_free
    for f in fams:
        if nxt + f.range > MAX_IDS:
            room = MAX_IDS - nxt
            if room < 8:
                warnings.append(f"mark macro {f.name}: no id range left (>{MAX_IDS} ids); not instrumented")
                f.range = 0
                continue
            warnings.append(f"mark macro {f.name}: id range cut to {room} (codes >= {room} are dropped)")
            f.range = room
        f.base = nxt
        nxt += f.range
    return nxt


def _lint(text: str, where: str, warnings: list[str]) -> None:
    """mtn_ lesson (board #179): ld.global.cg polled in an asm volatile loop is hoisted by ptxas -> endless spin."""
    lines = text.split("\n")
    for i, ln in enumerate(lines):
        if _CG_POLL.search(ln) and "asm" in ln:
            ctx = "\n".join(lines[max(0, i - 4): i + 2])
            if re.search(r"\b(while|for|do)\b", ctx):
                warnings.append(f"{where}:{i + 1}: ld.global.cg polled in a loop: ptxas may hoist the weak load "
                                "(endless spin, board #179); poll with ld.relaxed.gpu / ld.acquire.gpu (kd_wait_geq)")
                return


# ------------------------------------------------------------------------------------ transform

@dataclass
class _Plan:
    ids: _Ids
    fams: list[Family]
    n_marks: int
    layout: str


def _transform_cuda(text: str, plan: _Plan, warnings: list[str], where: str, brace_double: bool
                    ) -> tuple[str, bool]:
    """Transform one CUDA text; returns (text, changed)."""
    ids = plan.ids
    fam_by = {f.name: f for f in plan.fams if f.range}
    lines = text.split("\n")
    changed = False
    has_prelude = False
    for i, line in enumerate(lines):
        if _INC.match(line):
            lines[i] = prelude(plan.n_marks, brace_double, plan.layout).rstrip("\n")
            has_prelude = changed = True
            continue
        d = _DEF.match(line)
        if d and d.group(2) in fam_by:
            lines[i] = _family_line(d, fam_by[d.group(2)])
            changed = True
            continue
        a = _ACC.search(line)
        if a:
            name, op = a.group(1), a.group(2)
            code = line[: a.start()]
            if not _placeable(code):
                warnings.append(f"{where}:{i + 1}: @kdev:acc {name} after a line not ending in ; {{ or }} -- skipped")
                continue
            stmt = {"decl": f"KD_ACC_DECL({name});", "begin": f"KD_ACC_BEGIN({name});",
                    "end": f"KD_ACC_END({name});"}.get(op) or f"KD_ACC_FLUSH({name}, {ids.get(name)});"
            lines[i] = (f"{code}{stmt} " if code.strip() == "" else f"{code.rstrip()} {stmt} ") + line[a.start():].strip()
            changed = True
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
        out = prelude(plan.n_marks, brace_double, plan.layout) + out
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


def _make_plan(texts: list[str], styles: str | None, layout: str, warnings: list[str],
               call_texts: list[str] | None = None) -> tuple[_Plan, dict[int, str]]:
    explicit: dict[int, str] = {}
    for t in texts:
        explicit.update(_explicit_ids(t))
    ids = _Ids(set(explicit))
    for t in texts:                                  # ids in source order
        _register_ids(t, ids)
    fams_all: list[Family] = []
    seen: set[str] = set()
    for t in texts:
        for f in detect_families(t):
            if f.name not in seen:
                seen.add(f.name)
                fams_all.append(f)
    _count_calls(call_texts if call_texts is not None else texts, fams_all)
    fams = select_families(fams_all, styles)
    first_free = max([*ids.names, *explicit, -1]) + 1
    n_ids = _assign_bases(fams, first_free, warnings)
    for f in fams:
        for k in range(f.range):
            ids.names.setdefault(f.base + k, f"{f.name}{k}")
    if layout not in LAYOUTS:
        raise KdevError(ErrorCode.INVALID_REQUEST, f"layout must be one of {', '.join(LAYOUTS)}, got {layout!r}")
    if layout == "auto":
        segment_log = any(f.kind == "info" for f in fams) or any(_ACC.search(t) for t in texts)
        layout = "ctalog" if segment_log else "slots"
    n_hint = max(n_ids, sum(len(find_annotations(t)) for t in texts) + len(explicit), max(explicit, default=-1) + 1)
    marks = dict(explicit)
    marks.update(ids.names)
    return _Plan(ids, fams, n_hint, layout), dict(sorted(marks.items()))


def instrument_source(source: str, *, styles: str | None = "auto", layout: str = "auto",
                      warnings: list[str] | None = None) -> tuple[str, dict[int, str]]:
    """Pure transform of one CUDA source string."""
    w: list[str] = [] if warnings is None else warnings
    plan, marks = _make_plan([source], styles, layout, w)
    _lint(source, "<source>", w)
    out, _ = _transform_cuda(source, plan, w, "<source>", False)
    _LAST.update(families=[f.__dict__.copy() for f in plan.fams], layout=plan.layout)
    return out, marks


_LAST: dict[str, Any] = {}


def _string_body(tok: str) -> tuple[str, str, str, str] | None:
    """(prefix, quote, body, suffix) of a Python string literal token."""
    m = re.match(r"(?is)^([rbuf]*)('''|\"\"\"|'|\")", tok)
    if not m:
        return None
    prefix, q = m.group(1), m.group(2)
    if not tok.endswith(q) or len(tok) < len(prefix) + 2 * len(q):
        return None
    return prefix, q, tok[len(prefix) + len(q): len(tok) - len(q)], q


def _decoded(tok: str) -> str | None:
    try:
        v = ast.literal_eval(tok)
    except (ValueError, SyntaxError, MemoryError):
        return None
    return v if isinstance(v, str) else None


def _is_candidate(s: str) -> bool:
    return ("@kdev:" in s or bool(_ANY_MARK.search(s)) or "marks.cuh" in s
            or (("#define" in s or "# define" in s) and bool(_TIMER.search(s))))


def instrument_python(text: str, where: str = "<py>", *, styles: str | None = "auto", layout: str = "auto"
                      ) -> tuple[str, dict[int, str], list[str]]:
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

    # (f-strings tokenize as FSTRING_* on 3.12+ and are never STRING tokens: they are not rewritten)
    cands: list[tuple[Any, str]] = []        # (token, CUDA text as the compiler sees it / edit view)
    for t in toks:
        if t.type != tokenize.STRING:
            continue
        s = t.string
        parts = _string_body(s)
        if parts is None:
            continue
        prefix, q, body, _ = parts
        if len(q) == 1:
            if not _is_candidate(s):
                continue
            dec = _decoded(s) if "b" not in prefix.lower() and "f" not in prefix.lower() else None
            if dec is None or not _is_candidate(dec):
                continue
            cands.append((t, dec))
        elif _is_candidate(body):
            cands.append((t, body))
    plan, marks = _make_plan([c for _, c in cands], styles, layout, warnings)
    out = text
    for t, view in sorted(cands, key=lambda c: c[0].start, reverse=True):
        prefix, q, body, suf = _string_body(t.string)  # type: ignore[misc]
        if "f" in prefix.lower():
            warnings.append(f"{where}:{t.start[0]}: f-string CUDA source not instrumented (use a plain string)")
            continue
        _lint(view, f"{where}:{t.start[0]}", warnings)
        brace = "{{" in view or "}}" in view
        new_view, changed = _transform_cuda(view, plan, warnings, f"{where}:{t.start[0]}", brace)
        if not changed:
            continue
        if len(q) == 1:
            new_tok = repr(new_view)
        else:
            new_tok = prefix + q + new_view + suf
        a, b = off(t.start), off(t.end)
        out = out[:a] + new_tok + out[b:]
    _LAST.update(families=[f.__dict__.copy() for f in plan.fams], layout=plan.layout)
    return out, marks, warnings


def _candidate_texts(text: str, where: str) -> list[str]:
    """Every string literal of a Python module that could hold CUDA marks or mark macros, as the compiler would
    see it (decoded), in source order; patch fragments (``s.replace(OLD, NEW)``) are included on purpose."""
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, SyntaxError) as e:
        raise KdevError(ErrorCode.INVALID_REQUEST, f"{where}: cannot tokenize: {e}") from e
    out: list[str] = []
    for t in toks:
        if t.type != tokenize.STRING or not _is_candidate(t.string):
            continue
        parts = _string_body(t.string)
        if parts is None or "b" in parts[0].lower() or "f" in parts[0].lower():
            continue
        dec = _decoded(t.string)
        if dec is not None and _is_candidate(dec):
            out.append(dec)
    return out


FOOTER_TAG = "# ---- kdev timeline instrumentation"


def plan_doc(plan: _Plan) -> dict[str, Any]:
    return {"ids": {str(k): v for k, v in plan.ids.names.items()}, "by_name": dict(plan.ids.by_name),
            "families": [f.__dict__.copy() for f in plan.fams], "n_marks": plan.n_marks, "layout": plan.layout}


def plan_from_doc(doc: dict[str, Any]) -> _Plan:
    names = {int(k): v for k, v in (doc.get("ids") or {}).items()}
    ids = _Ids(set(names))
    ids.names = dict(names)
    ids.by_name = {str(k): int(v) for k, v in (doc.get("by_name") or {}).items()}
    fams = [Family(**{k: v for k, v in f.items() if k in Family.__dataclass_fields__}) for f in doc.get("families") or []]
    return _Plan(ids, fams, int(doc.get("n_marks") or 0), str(doc.get("layout") or "slots"))


def runtime_transform(src: Any, doc: dict[str, Any]) -> Any:
    """Rewrite load_inline ``cuda_sources`` (str or list of str) with a fixed plan (called by the footer of an
    instrumented .py at load_inline time, so import-time source patches are seen). Idempotent; a source without
    marks or mark macros is returned unchanged (same build cache entry as the plain file)."""
    if isinstance(src, (list, tuple)):
        return type(src)(runtime_transform(s, doc) for s in src)
    if not isinstance(src, str) or "kdev timeline marks (inlined" in src:
        return src
    out, _ = _transform_cuda(src, plan_from_doc(doc), [], "<load_inline>", False)
    return out


def footer(plan: _Plan) -> str:
    """Python appended to the instrumented copy: wraps the module's ``load_inline`` so every CUDA source is
    transformed at build time with the static plan (ids/names stay those of marks.json)."""
    import kdev
    kroot = str(Path(kdev.__file__).resolve().parent.parent)
    doc = json.dumps(plan_doc(plan), sort_keys=True)
    return f"""

{FOOTER_TAG} (appended by `kdev timeline`; CUDA sources are rewritten at load_inline time) ----
def _kdev_tl_install():
    import sys as _s
    try:
        from kdev.timeline.instrument import runtime_transform as _tf
    except ImportError:
        _s.path.insert(0, {kroot!r})
        from kdev.timeline.instrument import runtime_transform as _tf
    import json as _j
    _doc = _j.loads({doc!r})
    _g = globals()
    _f = _g.get("load_inline")
    if _f is None:
        import torch.utils.cpp_extension as _ce
        _f = _ce.load_inline
    if getattr(_f, "_kdev_tl", False):
        return

    def load_inline(*a, **k):
        if "cuda_sources" in k:
            k["cuda_sources"] = _tf(k["cuda_sources"], _doc)
        elif len(a) >= 3:
            a = (a[0], a[1], _tf(a[2], _doc)) + tuple(a[3:])
        return _f(*a, **k)

    load_inline._kdev_tl = True
    _g["load_inline"] = load_inline


_kdev_tl_install()
"""


def instrument_python_runtime(text: str, where: str = "<py>", *, styles: str | None = "auto",
                              layout: str = "auto") -> tuple[str, dict[int, str], list[str], _Plan]:
    """Instrumented copy of a Python module = original text + footer (pure apart from reading marks.cuh)."""
    warnings: list[str] = []
    texts = _candidate_texts(text, where)
    plan, marks = _make_plan(texts, styles, layout, warnings, call_texts=[text])
    for i, t in enumerate(texts):
        _lint(t, f"{where}:str{i}", warnings)
    if "load_inline" not in text:
        warnings.append(f"{where}: no load_inline in the file: runtime instrumentation needs load_inline")
    return text.rstrip("\n") + "\n" + footer(plan), marks, warnings, plan


def instrument_file(path: str, out_dir: str, *, styles: str | None = "auto", layout: str = "auto") -> Instrumented:
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
    _LAST.clear()
    if src.suffix == ".py":
        new, marks, warnings, plan = instrument_python_runtime(text, src.name, styles=styles, layout=layout)
        _LAST.update(families=[f.__dict__.copy() for f in plan.fams], layout=plan.layout)
    else:
        new, marks = instrument_source(text, styles=styles, layout=layout, warnings=warnings)
    fams = list(_LAST.get("families") or [])
    lay = str(_LAST.get("layout") or "slots")
    if not marks:
        warnings.append("no // @kdev:mark annotations, KD_MARK(n) calls or agent mark macros found")
    dst = od / src.name
    dst.write_text(new, encoding="utf-8", newline="\n")
    hn = hints(text)
    mj = od / "marks.json"
    mj.write_text(json.dumps({"source": str(src), "copy": str(dst), "marks": {str(k): v for k, v in marks.items()},
                              "families": fams, "layout": lay, "hints": hn}, indent=1), encoding="utf-8")
    return Instrumented(path=str(dst), marks=marks, warnings=warnings, source=str(src), marks_json=str(mj),
                        families=fams, layout=lay, hints=hn)
