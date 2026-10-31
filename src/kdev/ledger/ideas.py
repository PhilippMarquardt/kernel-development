"""Idea board: hypotheses with status, evidence and owner (also backs `kdev note`). Owner: P4.

Status machine: open -> claimed (owner set) -> testing -> confirmed|refuted|abandoned; any -> open on
release. claim() of an idea owned by someone else -> KdevError(conflict). kind='limit' (an "at the limit"
claim, LESSONS 1) can only be confirmed with evidence of a `sass` run AND a `sweep` run; kind='dead'
requires ``variant`` (the exact variant tested, LESSONS 6).
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

STATUSES = ("open", "claimed", "testing", "confirmed", "refuted", "abandoned")
KINDS = ("idea", "dead", "limit", "win")


@dataclass
class Evidence:
    ref_kind: str          # run | judge | file | url | text
    ref: str
    note: str = ""
    supports: str = "neutral"
    author: str = ""
    added_utc: str = ""


@dataclass
class Idea:
    idea_id: int | None
    title: str
    problem: str | None = None
    hypothesis: str = ""
    kind: str = "idea"
    status: str = "open"
    owner: str | None = None
    expected_gain_pct: float | None = None
    tags: list[str] = field(default_factory=list)
    variant: str = ""
    created_utc: str = ""
    updated_utc: str = ""
    resolution: str = ""
    evidence: list[Evidence] = field(default_factory=list)


def add(conn: sqlite3.Connection, idea: Idea) -> Idea:
    raise NotImplementedError("ideas.add (P4)")


def get(conn: sqlite3.Connection, idea_id: int) -> Idea:
    raise NotImplementedError("ideas.get (P4)")


def list_ideas(conn: sqlite3.Connection, *, problem: str | None = None, status: str | None = None,
               owner: str | None = None, kind: str | None = None, limit: int = 50) -> list[Idea]:
    raise NotImplementedError("ideas.list_ideas (P4)")


def claim(conn: sqlite3.Connection, idea_id: int, owner: str) -> Idea:
    raise NotImplementedError("ideas.claim (P4)")


def release(conn: sqlite3.Connection, idea_id: int, owner: str) -> Idea:
    raise NotImplementedError("ideas.release (P4)")


def add_evidence(conn: sqlite3.Connection, idea_id: int, ev: Evidence) -> Idea:
    raise NotImplementedError("ideas.add_evidence (P4)")


def resolve(conn: sqlite3.Connection, idea_id: int, status: str, resolution: str, author: str) -> Idea:
    """Validate the gating rules above, then set the final status."""
    raise NotImplementedError("ideas.resolve (P4)")


def search(conn: sqlite3.Connection, text: str, problem: str | None = None, limit: int = 20) -> list[Idea]:
    raise NotImplementedError("ideas.search (P4)")
