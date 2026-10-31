"""Private helpers for the ledger/judge packages (P4): hashing, time conversion, json columns."""
from __future__ import annotations

import datetime as _dt
import hashlib
import io
import json
import re
import tokenize
from pathlib import Path
from typing import Any


def md5_file(path: str | Path) -> str:
    from kdev.util import md5_file as _m
    return _m(str(path))


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


_C_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)


def _strip_python(src: str) -> str:
    """Drop '#' comments with tokenize (keeps '#' inside strings, e.g. embedded CUDA); whitespace removed."""
    out = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
                            tokenize.DEDENT, tokenize.ENCODING):
                continue
            s = tok.string
            if tok.type == tokenize.STRING:
                s = _C_COMMENT.sub("", s)   # embedded CUDA/C++ sources
            out.append(s)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        src = re.sub(r"(?m)^\s*#[^\n]*$", "", src)
        return re.sub(r"\s+", "", _C_COMMENT.sub("", src))
    return re.sub(r"\s+", "", "".join(out))


def norm_text(src: str, suffix: str = ".py") -> str:
    if suffix == ".py":
        return _strip_python(src)
    return re.sub(r"\s+", "", _C_COMMENT.sub("", src))


def norm_md5(path: str | Path) -> str:
    """Comment/whitespace-insensitive md5. Uses kdev.util.norm_md5 (P1) when implemented, else our own
    (Python '#' via tokenize, C '//' and '/* */' inside strings and in .cu/.cuh files)."""
    try:
        from kdev.util import norm_md5 as _n
        return _n(str(path))
    except NotImplementedError:
        pass
    p = Path(path)
    src = p.read_text(encoding="utf-8", errors="replace")
    return hashlib.md5(norm_text(src, p.suffix.lower()).encode("utf-8")).hexdigest()  # noqa: S324


def utc_now() -> str:
    from kdev.util import utc_now as _u
    return _u()


def epoch_to_utc(x: Any) -> str | None:
    if x is None or x == "":
        return None
    try:
        return _dt.datetime.fromtimestamp(float(x), _dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def utc_to_epoch(s: str | None) -> float | None:
    if not s:
        return None
    try:
        t = s.replace("Z", "+00:00")
        d = _dt.datetime.fromisoformat(t)
        if d.tzinfo is None:
            d = d.replace(tzinfo=_dt.timezone.utc)
        return d.timestamp()
    except ValueError:
        return None


_REL = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([smhdw])\s*$", re.I)


def since_to_utc(since: str | None) -> str | None:
    """'6h' / '2d' / '30m' -> absolute UTC; ISO date/time passed through (normalised)."""
    if not since:
        return None
    m = _REL.match(since)
    if m:
        mult = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}[m.group(2).lower()]
        t = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(seconds=float(m.group(1)) * mult)
        return t.strftime("%Y-%m-%dT%H:%M:%SZ")
    e = utc_to_epoch(since if "T" in since else since + "T00:00:00")
    if e is None:
        from kdev.contract import ErrorCode, KdevError
        raise KdevError(ErrorCode.INVALID_REQUEST, f"bad --since {since!r}", details={"field": "since"},
                        recovery="use e.g. 6h, 2d or 2026-10-31")
    return epoch_to_utc(e)


def jdump(x: Any) -> str | None:
    if x is None:
        return None
    return json.dumps(x, separators=(",", ":"), sort_keys=True, default=str)


def jload(s: str | None, default: Any = None) -> Any:
    if not s:
        return default
    try:
        return json.loads(s)
    except (TypeError, ValueError):
        return default


def parse_kv(items: list[str] | None) -> dict[str, Any]:
    """['TILE=128', 'X=a'] -> {'TILE': 128, 'X': 'a'} (ints/floats/bools coerced)."""
    out: dict[str, Any] = {}
    for it in items or []:
        for part in str(it).split(","):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                from kdev.contract import ErrorCode, KdevError
                raise KdevError(ErrorCode.INVALID_REQUEST, f"bad knob {part!r} (want K=V)",
                                details={"field": "knobs"}, recovery="pass --knobs K=V")
            k, v = part.split("=", 1)
            out[k.strip()] = coerce(v.strip())
    return out


def coerce(v: str) -> Any:
    lv = v.lower()
    if lv in ("true", "false"):
        return lv == "true"
    for t in (int, float):
        try:
            return t(v)
        except ValueError:
            pass
    return v


def rows(cur_rows: Any) -> list[dict[str, Any]]:
    return [dict(r) for r in cur_rows]
