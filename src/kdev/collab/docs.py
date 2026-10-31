"""Documentation store (``kdev doc``): append-only notes/lessons/tracker/design docs. Owner: P9.

Replaces hand-edited NOTES.md / LESSONS.md / TRACKER.md, which agents overwrote on each other (a sync
overwrote a NOTES.md and lost sections). Nothing is ever deleted: an entry can only be *superseded* by a newer
one (``--supersedes ID``; the old one stays in ``doc show --history``).

Docs: name ``notes/<problem>`` (kind notes, per problem), ``lessons`` (global rules), ``tracker`` (the
orchestrator's log), ``design/<topic>``; the kind is the first path component (else ``--kind``, default notes).

Markdown model (``import`` / ``export``; CRLF is normalised to LF):
- ``# Title`` on the first line -> doc title; the text up to the first ``## `` heading -> preamble (section
  ``""``); each ``## heading`` -> a section (``### `` sub-headings stay inside their section's text; headings
  inside code fences are ignored). Duplicate headings get the key ``name (2)`` but render unchanged.
- Imported text is stored verbatim (``raw`` entries), so ``export(import(file)) == file``.
- Appended entries render as ``- **YYYY-MM-DD HH:MM UTC author**: text`` (+ indented continuation lines and a
  ``refs:`` line), after the section's last line; in newest-first sections (tracker ``Log*``) at the top.
- Re-import of a changed file: identical section -> skipped; text appended at the end (newest-first: at the
  top) -> only the delta is added; otherwise the new text supersedes the section (history kept). Sections in
  the store but missing from the file are KEPT and reported (the overwrite incident cannot lose them).
- ``export --md PATH`` refuses to overwrite a file that changed since kdev last wrote it (hand edits) unless
  ``force``: import it first.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from kdev.collab.board import utc_now
from kdev.collab.db import has_fts, write_tx
from kdev.contract import ErrorCode, KdevError

KINDS = ("notes", "lessons", "tracker", "design")
_FENCE = re.compile(r"^\s{0,3}(```|~~~)")


def _bad(msg: str, fld: str, recovery: str = "") -> KdevError:
    return KdevError(ErrorCode.INVALID_REQUEST, msg, details={"field": fld}, recovery=recovery)


@dataclass
class Entry:
    entry_id: int
    doc: str
    section: str
    created_utc: str
    author: str
    body: str
    refs: list[dict[str, Any]] = field(default_factory=list)
    raw: bool = False
    source: str = "cli"
    superseded_by: int | None = None
    tags: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def render(self) -> str:
        """Markdown of one entry (raw = verbatim)."""
        if self.raw:
            return self.body
        from kdev.collab.refs import short
        lines = self.body.strip("\n").split("\n")
        t = self.created_utc[:16].replace("T", " ")
        out = [f"- **{t} UTC {self.author or '-'}**: {lines[0]}"] + [("  " + x) if x else "" for x in lines[1:]]
        if self.refs:
            out.append("  refs: " + ", ".join(short(r) for r in self.refs))
        return "\n".join(out) + "\n"


def _entry(r: sqlite3.Row) -> Entry:
    return Entry(entry_id=r["entry_id"], doc=r["doc"], section=r["section"], created_utc=r["created_utc"],
                 author=r["author"] or "", body=r["body"], refs=json.loads(r["refs"] or "[]"), raw=bool(r["raw"]),
                 source=r["source"] or "cli", superseded_by=r["superseded_by"], tags=r["tags"] or "")


# ---------------------------------------------------------------------------------------------- naming
def doc_name(doc: str, problem: str | None = None) -> str:
    """'notes' + problem -> 'notes/<problem basename>'; 'design' + problem -> 'design/<problem>'; names are
    lower-cased path-like slugs (``NOTES.md`` -> notes)."""
    d = (doc or "").strip().strip("/")
    if d.lower().endswith(".md"):
        d = d[:-3]
    if not d:
        raise _bad("doc name required", "doc")
    low = d.lower()
    if low in ("notes", "design"):
        if not problem:
            raise _bad(f"doc {d!r} is per problem", "problem", f"pass --problem P (-> {low}/P) or a full name")
        return f"{low}/{_slug(problem)}"
    return "/".join(_slug(p) if i == 0 else p.strip() for i, p in enumerate(d.split("/")))


def _slug(problem: str) -> str:
    b = os.path.basename(str(problem).replace("\\", "/").rstrip("/")) or str(problem)
    return b.lower()


def kind_of(name: str, kind: str | None = None) -> str:
    first = name.split("/")[0]
    if first in KINDS:
        return first
    k = (kind or "notes").lower()
    if k not in KINDS:
        raise _bad(f"unknown doc kind {k!r}", "kind", "use " + "|".join(KINDS))
    return k


def default_doc_for_file(path: str | Path, problem: str | None = None) -> str:
    """NOTES.md -> notes/<problem or parent dir>, LESSONS.md -> lessons, TRACKER.md -> tracker,
    DESIGN.md -> design/<parent>, other -> the file stem."""
    p = Path(path)
    stem = p.stem.lower()
    parent = p.resolve().parent.name
    if stem == "notes":
        return f"notes/{_slug(problem or parent)}"
    if stem in ("lessons", "tracker"):
        return stem
    if stem == "design":
        return f"design/{_slug(problem or parent)}"
    return stem


# ------------------------------------------------------------------------------------------- markdown
@dataclass
class MdSection:
    key: str
    heading: str       # rendered heading text after '## ' ('' for the preamble)
    body: str


def parse_md(text: str) -> tuple[str, list[MdSection]]:
    """Markdown -> (title, sections). Section 0 is always the preamble (key ''). See module docstring."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.splitlines(keepends=True)
    title = ""
    i = 0
    if lines and lines[0].startswith("# "):
        title = lines[0][2:].rstrip("\n")
        i = 1
    secs = [MdSection("", "", "")]
    seen: dict[str, int] = {}
    fence = False
    for line in lines[i:]:
        if _FENCE.match(line):
            fence = not fence
        if not fence and line.startswith("## ") and line[3:].strip():
            heading = line[3:].rstrip("\n")
            name = heading.strip()
            seen[name] = seen.get(name, 0) + 1
            key = name if seen[name] == 1 else f"{name} ({seen[name]})"
            secs.append(MdSection(key, heading, ""))
            continue
        secs[-1].body += line
    return title, secs


def _combine(entries: list[Entry], newest_first: bool) -> str:
    acc = ""
    for e in entries:
        t = e.render()
        if e.raw:
            acc = t + acc if newest_first else acc + t
            continue
        if not acc.strip("\n"):
            acc = "\n" + t + "\n"
        elif newest_first:
            lead = acc[: len(acc) - len(acc.lstrip("\n"))] or "\n"
            acc = lead + t + acc.lstrip("\n")
        else:
            core = acc.rstrip("\n")
            tail = acc[len(core):]
            acc = core + "\n" + t + tail[1:]
    return acc


# ---------------------------------------------------------------------------------------------- store
def _ensure_doc(conn: sqlite3.Connection, name: str, kind: str | None, problem: str | None,
                title: str | None = None, source: str = "cli") -> None:
    if conn.execute("SELECT 1 FROM docs WHERE doc=?", (name,)).fetchone():
        return
    conn.execute("INSERT INTO docs(doc, kind, problem, title, created_utc, source) VALUES (?,?,?,?,?,?)",
                 (name, kind_of(name, kind), problem, name if title is None else title, utc_now(), source))


def _ensure_section(conn: sqlite3.Connection, doc: str, key: str, heading: str | None = None,
                    newest_first: bool | None = None) -> None:
    if conn.execute("SELECT 1 FROM doc_sections WHERE doc=? AND section=?", (doc, key)).fetchone():
        return
    ord_ = conn.execute("SELECT coalesce(max(ord), -1) + 1 FROM doc_sections WHERE doc=?", (doc,)).fetchone()[0]
    if key == "":
        ord_ = -1
    if newest_first is None:
        kind = conn.execute("SELECT kind FROM docs WHERE doc=?", (doc,)).fetchone()[0]
        newest_first = kind == "tracker" and key.lower().startswith("log")
    conn.execute("INSERT INTO doc_sections(doc, section, heading, ord, newest_first, created_utc) "
                 "VALUES (?,?,?,?,?,?)", (doc, key, key if heading is None else heading, ord_,
                                          int(bool(newest_first)), utc_now()))


def _insert(conn: sqlite3.Connection, doc: str, section: str, body: str, *, author: str = "", raw: bool = False,
            refs: list[dict[str, Any]] | None = None, source: str = "cli", tags: str = "",
            created_utc: str | None = None) -> int:
    cur = conn.execute("INSERT INTO doc_entries(doc, section, created_utc, author, body, refs, raw, source, tags) "
                       "VALUES (?,?,?,?,?,?,?,?,?)", (doc, section, created_utc or utc_now(), author, body,
                                                      json.dumps(refs or []), int(raw), source, tags))
    return int(cur.lastrowid)


def docs_list(conn: sqlite3.Connection, kind: str | None = None) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT d.doc, d.kind, d.problem, d.title, d.created_utc, "
        "(SELECT count(*) FROM doc_sections s WHERE s.doc=d.doc AND s.section<>'') AS sections, "
        "(SELECT count(*) FROM doc_entries e WHERE e.doc=d.doc AND e.superseded_by IS NULL) AS entries, "
        "(SELECT max(created_utc) FROM doc_entries e WHERE e.doc=d.doc) AS updated_utc FROM docs d "
        + ("WHERE d.kind=? " if kind else "") + "ORDER BY d.doc", (kind,) if kind else ()).fetchall()
    return [dict(r) for r in rows]


def get_doc(conn: sqlite3.Connection, name: str) -> dict[str, Any]:
    r = conn.execute("SELECT * FROM docs WHERE doc=?", (name,)).fetchone()
    if r is None:
        near = [x[0] for x in conn.execute("SELECT doc FROM docs WHERE doc LIKE ?", (f"%{name.split('/')[-1]}%",))]
        raise KdevError(ErrorCode.NOT_FOUND, f"no doc {name!r}", details={"doc": name, "similar": near[:5]},
                        recovery="kdev doc list (or kdev doc import FILE.md)" + (f"; similar: {near[0]}" if near else ""))
    return dict(r)


def sections(conn: sqlite3.Connection, doc: str) -> list[dict[str, Any]]:
    return [dict(r) for r in conn.execute("SELECT * FROM doc_sections WHERE doc=? ORDER BY ord, created_utc",
                                          (doc,))]


def resolve_section(conn: sqlite3.Connection, doc: str, section: str, create: bool = False) -> str:
    """Exact key, else case-insensitive, else unique prefix / substring match; ``create`` -> new key."""
    keys = [s["section"] for s in sections(conn, doc)]
    if section in keys:
        return section
    low = section.strip().lower()
    for cand in ([k for k in keys if k.lower() == low], [k for k in keys if k.lower().startswith(low)],
                 [k for k in keys if low and low in k.lower()]):
        if len(cand) == 1:
            return cand[0]
        if len(cand) > 1 and not create:
            raise KdevError(ErrorCode.INVALID_REQUEST, f"section {section!r} is ambiguous in {doc}",
                            details={"field": "section", "candidates": cand[:8]},
                            recovery="pass the full heading (kdev doc toc " + doc + ")")
    if create:
        return section.strip()
    raise KdevError(ErrorCode.NOT_FOUND, f"no section {section!r} in {doc}", details={"sections": keys[:20]},
                    recovery=f"kdev doc toc {doc}")


def entries(conn: sqlite3.Connection, doc: str, section: str | None = None, history: bool = False) -> list[Entry]:
    w, p = "doc=?", [doc]
    if section is not None:
        w += " AND section=?"
        p.append(section)
    if not history:
        w += " AND superseded_by IS NULL"
    return [_entry(r) for r in conn.execute(f"SELECT * FROM doc_entries WHERE {w} ORDER BY entry_id", tuple(p))]


def get_entry(conn: sqlite3.Connection, entry_id: int) -> Entry:
    r = conn.execute("SELECT * FROM doc_entries WHERE entry_id=?", (int(entry_id),)).fetchone()
    if r is None:
        raise KdevError(ErrorCode.NOT_FOUND, f"no doc entry {entry_id}", recovery="kdev doc show DOC --history")
    return _entry(r)


def render_section(conn: sqlite3.Connection, doc: str, sec: dict[str, Any]) -> str:
    body = _combine(entries(conn, doc, sec["section"]), bool(sec["newest_first"]))
    return body if sec["section"] == "" else f"## {sec['heading']}\n{body}"


def render(conn: sqlite3.Connection, doc: str) -> str:
    """The whole doc as markdown (deterministic)."""
    d = get_doc(conn, doc)
    out = f"# {d['title']}\n" if d["title"] else ""
    secs = sections(conn, doc)
    if d["title"] and secs and not any(s["section"] == "" for s in secs):
        out += "\n"
    for s in secs:
        out += render_section(conn, doc, s)
    return out


def append(conn: sqlite3.Connection, doc: str, section: str, body: str, *, author: str, kind: str | None = None,
           problem: str | None = None, refs: list[dict[str, Any]] | None = None, supersedes: int | None = None,
           newest_first: bool | None = None, tags: str = "") -> Entry:
    """Append one entry (creates the doc / section on first use). ``supersedes`` hides that entry (same doc)."""
    if not body.strip():
        raise _bad("empty entry", "text")
    if not author:
        raise _bad("doc append needs an author", "agent", "pass --agent NAME or set KDEV_AGENT")
    with write_tx(conn):
        _ensure_doc(conn, doc, kind, problem)
        key = resolve_section(conn, doc, section or "", create=True) if section else ""
        _ensure_section(conn, doc, key, newest_first=newest_first)
        if supersedes is not None:
            old = conn.execute("SELECT doc, superseded_by FROM doc_entries WHERE entry_id=?",
                               (int(supersedes),)).fetchone()
            if old is None or old["doc"] != doc:
                raise KdevError(ErrorCode.NOT_FOUND, f"--supersedes {supersedes}: no such entry in {doc}")
        eid = _insert(conn, doc, key, body.strip("\n"), author=author, refs=refs, tags=tags)
        if supersedes is not None:
            conn.execute("UPDATE doc_entries SET superseded_by=? WHERE entry_id=?", (eid, int(supersedes)))
    return get_entry(conn, eid)


# --------------------------------------------------------------------------------------------- import
@dataclass
class ImportReport:
    doc: str
    path: str
    created: bool = False
    added: list[str] = field(default_factory=list)       # new sections
    appended: list[str] = field(default_factory=list)    # delta added to an existing section
    superseded: list[str] = field(default_factory=list)  # section text replaced (history kept)
    unchanged: int = 0
    kept_missing: list[str] = field(default_factory=list)  # in the store, not in the file: kept
    title_changed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def import_md(conn: sqlite3.Connection, path: str | Path, *, doc: str | None = None, kind: str | None = None,
              problem: str | None = None, author: str = "import") -> ImportReport:
    """Ingest a markdown file by headings (see module docstring). Idempotent."""
    p = Path(path)
    if not p.is_file():
        raise KdevError(ErrorCode.NOT_FOUND, f"no such file: {p}", details={"path": str(p)})
    name = doc or default_doc_for_file(p, problem)
    title, secs = parse_md(p.read_text(encoding="utf-8", errors="replace"))
    src = f"import:{p.name}"
    rep = ImportReport(doc=name, path=str(p.resolve()))
    with write_tx(conn):
        existing = conn.execute("SELECT title FROM docs WHERE doc=?", (name,)).fetchone()
        rep.created = existing is None
        _ensure_doc(conn, name, kind, problem or (name.split("/", 1)[1] if "/" in name else None), title, src)
        if existing is not None and title and existing["title"] != title:
            conn.execute("UPDATE docs SET title=? WHERE doc=?", (title, name))
            rep.title_changed = True
        in_file = set()
        for s in secs:
            in_file.add(s.key)
            had = conn.execute("SELECT * FROM doc_sections WHERE doc=? AND section=?", (name, s.key)).fetchone()
            if had is None:
                _ensure_section(conn, name, s.key, s.heading)
                _insert(conn, name, s.key, s.body, author=author, raw=True, source=src)
                rep.added.append(s.key)
                continue
            cur = _combine(entries(conn, name, s.key), bool(had["newest_first"]))
            nf = bool(had["newest_first"])
            if s.body == cur:
                rep.unchanged += 1
            elif cur and not nf and s.body.startswith(cur):
                _insert(conn, name, s.key, s.body[len(cur):], author=author, raw=True, source=src)
                rep.appended.append(s.key)
            elif cur and nf and s.body.endswith(cur):
                _insert(conn, name, s.key, s.body[: len(s.body) - len(cur)], author=author, raw=True, source=src)
                rep.appended.append(s.key)
            else:
                # replaced text: supersede the imported text and every appended entry the file contains;
                # appended entries the file does not contain stay visible (nothing is lost)
                old = entries(conn, name, s.key)
                eid = _insert(conn, name, s.key, s.body, author=author, raw=True, source=src)
                for e in old:
                    if e.raw or e.render().strip() in s.body:
                        conn.execute("UPDATE doc_entries SET superseded_by=? WHERE entry_id=?", (eid, e.entry_id))
                rep.superseded.append(s.key)
        for row in conn.execute("SELECT section FROM doc_sections WHERE doc=? ORDER BY ord", (name,)).fetchall():
            if row[0] not in in_file and row[0] != "":
                rep.kept_missing.append(row[0])
    return rep


# --------------------------------------------------------------------------------------------- export
def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()  # noqa: S324 - change detection, not security


def export_md(conn: sqlite3.Connection, doc: str, path: str | Path, force: bool = False) -> dict[str, Any]:
    """Write the rendered doc to ``path`` atomically. Refuses (conflict) when the file exists, differs from
    the rendering and is not the file kdev last exported there (hand edits would be lost)."""
    from kdev.util import atomic_write_text
    text = render(conn, doc)
    p = Path(path).expanduser()
    key = str(p.resolve())
    status = "written"
    if p.is_file():
        old = p.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
        if old == text:
            status = "unchanged"
        else:
            row = conn.execute("SELECT md5 FROM doc_exports WHERE path=?", (key,)).fetchone()
            if not force and (row is None or row["md5"] != _md5(old)):
                raise KdevError(ErrorCode.CONFLICT, f"{p} has content kdev did not write (hand edits?)",
                                details={"path": str(p), "doc": doc},
                                recovery=f"kdev doc import {p} --doc {doc} first (keeps both), or --force")
    if status == "written":
        atomic_write_text(p, text)
    with write_tx(conn):
        conn.execute("INSERT OR REPLACE INTO doc_exports(path, doc, md5, utc) VALUES (?,?,?,?)",
                     (key, doc, _md5(text), utc_now()))
    return {"path": str(p), "doc": doc, "status": status, "bytes": len(text.encode("utf-8")), "md5": _md5(text)}


# --------------------------------------------------------------------------------------------- search
def search(conn: sqlite3.Connection, query: str, *, doc: str | None = None, ref: str | None = None,
           limit: int = 30) -> list[Entry]:
    q = (query or "").strip()
    ids: list[int] | None = None
    if q and has_fts(conn):
        try:
            fq = " ".join('"' + t.replace('"', '""') + '"' for t in q.split())
            ids = [r[0] for r in conn.execute("SELECT rowid FROM entries_fts WHERE entries_fts MATCH ? "
                                              "ORDER BY rowid DESC LIMIT 2000", (fq,))]
        except sqlite3.OperationalError:
            ids = None
    w, p = ["superseded_by IS NULL"], []
    if q and ids is None:
        for t in q.split():
            w.append("(body LIKE ? OR section LIKE ? OR refs LIKE ?)")
            p += [f"%{t}%"] * 3
    elif q:
        if not ids:
            return []
        w.append(f"entry_id IN ({','.join(str(i) for i in ids)})")
    if doc:
        w.append("(doc=? OR doc LIKE ?)")
        p += [doc, doc + "/%"]
    if ref:
        w.append("refs LIKE ?")
        p.append(f"%{ref}%")
    rows = conn.execute(f"SELECT * FROM doc_entries WHERE {' AND '.join(w)} ORDER BY entry_id DESC LIMIT ?",
                        (*p, int(limit))).fetchall()
    return [_entry(r) for r in rows]


def snippet(text: str, query: str, width: int = 110) -> str:
    flat = " ".join(text.split())
    i = -1
    for t in (query or "").split():
        i = flat.lower().find(t.lower())
        if i >= 0:
            break
    start = max(0, i - 30) if i >= 0 else 0
    s = flat[start:start + width]
    return ("~" if start else "") + s + ("~" if start + width < len(flat) else "")


# ------------------------------------------------------------------------------------------- checklist
_ITEM = re.compile(r"^\s{0,3}(\d+[.)]|[-*])\s+(.*)$")


def checklist(conn: sqlite3.Connection, doc: str = "lessons", rules: bool = False) -> dict[str, Any]:
    """Brief checklist items (list items of every section whose heading contains 'checklist'), and with
    ``rules`` one line per other section: heading + its first 'Rule:' sentence."""
    secs = sections(conn, get_doc(conn, doc)["doc"])
    items: list[str] = []
    rule_lines: list[str] = []
    for s in secs:
        body = _combine(entries(conn, doc, s["section"]), bool(s["newest_first"]))
        if "checklist" in s["section"].lower():
            cur: list[str] = []
            for line in body.splitlines():
                m = _ITEM.match(line)
                if m:
                    if cur:
                        items.append(" ".join(cur))
                    txt = re.sub(r"^\*\*\d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC [^*]*\*\*: ", "", m.group(2))
                    cur = [txt.strip()]
                elif cur and line.strip() and line.startswith((" ", "\t")):
                    cur.append(line.strip())
                elif not line.strip() and cur:
                    items.append(" ".join(cur))
                    cur = []
            if cur:
                items.append(" ".join(cur))
        elif rules and s["section"]:
            m = re.search(r"Rule:\s*(.+?)(?:\n\s*\n|\n- |\Z)", body, re.S)
            if m:
                rule_lines.append(f"{s['heading'].strip()}: {' '.join(m.group(1).split())}")
    items = [i for i in items if not i.startswith("refs:")]
    return {"doc": doc, "items": items, "rules": rule_lines}
