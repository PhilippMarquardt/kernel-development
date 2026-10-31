"""Result envelope, error codes and effects (FROZEN after Phase 0; see docs/PLAN.md sections 3.2 and 7).

Every command returns a :class:`Result`. The CLI prints it as a <=40-line summary (``kdev.output``) or, with
``--json``, as exactly one JSON object (``Result.to_dict()``). Errors are Results with ``status="error"``,
never bare tracebacks.
"""
from __future__ import annotations

import enum
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from kdev import __version__

RESULT_SCHEMA = "kdev.result/1"
PROTOCOL = "kdev-tp/1"


class ErrorCode(str, enum.Enum):
    """Stable error codes. The value is the wire string; :data:`EXIT_CODES` maps to process exit codes."""

    CHECK_FAILED = "check_failed"
    INVALID_REQUEST = "invalid_request"
    QUOTA_FORBIDDEN = "quota_forbidden"
    NEEDS_BUILD = "needs_build"
    TOOLCHAIN_MISMATCH = "toolchain_mismatch"
    GPU_BUSY_TIMEOUT = "gpu_busy_timeout"
    CAPABILITY_UNAVAILABLE = "capability_unavailable"
    BUILD_FAILED = "build_failed"
    TIMEOUT = "timeout"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    CANCELLED = "cancelled"
    NOT_EXCLUSIVE = "not_exclusive"
    NOT_IMPLEMENTED = "not_implemented"
    INTERNAL = "internal"


EXIT_CODES: dict[ErrorCode, int] = {
    ErrorCode.CHECK_FAILED: 1,
    ErrorCode.INVALID_REQUEST: 2,
    ErrorCode.QUOTA_FORBIDDEN: 3,
    ErrorCode.NEEDS_BUILD: 4,
    ErrorCode.TOOLCHAIN_MISMATCH: 5,
    ErrorCode.GPU_BUSY_TIMEOUT: 6,
    ErrorCode.CAPABILITY_UNAVAILABLE: 7,
    ErrorCode.BUILD_FAILED: 8,
    ErrorCode.TIMEOUT: 9,
    ErrorCode.NOT_FOUND: 10,
    ErrorCode.CONFLICT: 11,
    ErrorCode.CANCELLED: 12,
    ErrorCode.NOT_EXCLUSIVE: 13,
    ErrorCode.NOT_IMPLEMENTED: 14,
    ErrorCode.INTERNAL: 15,
}


@dataclass
class KdevErrorInfo:
    """The ``error`` member of a Result."""

    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    recovery: str = ""


class KdevError(Exception):
    """Raise anywhere in the library; the CLI/MCP layer converts it into an error Result."""

    def __init__(self, code: ErrorCode, message: str, *, details: dict[str, Any] | None = None,
                 recovery: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}
        self.recovery = recovery

    def info(self) -> KdevErrorInfo:
        return KdevErrorInfo(self.code.value, self.message, self.details, self.recovery)


@dataclass
class Effects:
    """Declared side effects of a command (REA 'truthful effects')."""

    gpu: str = "none"            # none | queued | owns
    writes: tuple[str, ...] = ()  # run_dir | cache | ledger | queue | ideas | files | remote
    network: str = "none"        # none | ssh | okc_readonly | okc_check
    quota: str = "never"         # never | explicit_flag (only judge pair --allow-check)


@dataclass
class Fidelity:
    """How a timing was taken. Any deviation from the judge protocol appears in ``optouts``."""

    toolchain: dict[str, Any] = field(default_factory=dict)  # {nvcc, pin, so_notes, match}
    flush: dict[str, Any] = field(default_factory=lambda: {"method": "read", "bytes": 256 << 20})
    graph: bool = True
    exclusive: str = "unknown"   # ok | waited | fail | shared | unknown
    waited_s: float = 0.0
    seeds: list[int] = field(default_factory=list)
    overhead_us: dict[str, float] = field(default_factory=dict)
    clocks: dict[str, Any] = field(default_factory=dict)       # {before:{...}, after:{...}}
    optouts: list[str] = field(default_factory=list)          # e.g. ["flush=write", "eager", "allow_jit"]

    @property
    def comparable(self) -> bool:
        """False when the result must not be mixed with judge-protocol rows in the ledger."""
        return not self.optouts and self.exclusive in ("ok", "waited") and self.toolchain.get("match", True)


@dataclass
class Result:
    """The single output contract of every kdev command (``kdev.result/1``)."""

    tool: str
    status: str = "ok"                 # ok | fail | error
    verdict: str = ""
    run_id: str | None = None
    job_id: str | None = None
    summary: dict[str, Any] = field(default_factory=dict)
    shapes: list[dict[str, Any]] = field(default_factory=list)
    score: dict[str, Any] | None = None
    fidelity: Fidelity | None = None
    dispatch: dict[str, Any] | None = None
    files: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    next: list[str] = field(default_factory=list)
    error: KdevErrorInfo | None = None
    lines: list[str] = field(default_factory=list)   # pre-rendered summary lines (optional; output.py may use)
    version: str = __version__
    protocol: str = PROTOCOL
    schema: str = RESULT_SCHEMA

    @property
    def exit_code(self) -> int:
        if self.error is not None:
            try:
                return EXIT_CODES[ErrorCode(self.error.code)]
            except ValueError:
                return EXIT_CODES[ErrorCode.INTERNAL]
        return 1 if self.status == "fail" else 0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["exit_code"] = self.exit_code
        d.pop("lines", None)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=None, separators=(",", ":"), default=str)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Result":
        d = dict(d)
        d.pop("exit_code", None)
        fid = d.pop("fidelity", None)
        err = d.pop("error", None)
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        r = cls(**{k: v for k, v in d.items() if k in known})
        r.fidelity = Fidelity(**fid) if isinstance(fid, dict) else None
        r.error = KdevErrorInfo(**err) if isinstance(err, dict) else None
        return r

    @classmethod
    def from_error(cls, tool: str, exc: KdevError) -> "Result":
        return cls(tool=tool, status="error", verdict="error", error=exc.info())


def not_implemented(tool: str, what: str = "") -> Result:
    """Result returned for Phase-0 stubs (exit code 14)."""
    return Result(tool=tool, status="error", verdict="error",
                  error=KdevErrorInfo(ErrorCode.NOT_IMPLEMENTED.value,
                                      f"{what or tool} is not implemented yet (Phase-0 stub)",
                                      recovery="see docs/PLAN.md section 10 for the owning package"))


def validate_result(d: dict[str, Any]) -> list[str]:
    """Return a list of problems with a Result dict (empty = valid). Used by tests and the MCP layer."""
    problems = []
    for key in ("schema", "tool", "status", "version", "protocol"):
        if key not in d:
            problems.append(f"missing {key}")
    if d.get("status") not in ("ok", "fail", "error"):
        problems.append(f"bad status {d.get('status')!r}")
    if d.get("status") == "error" and not d.get("error"):
        problems.append("status=error without error{}")
    if d.get("error") and d["error"].get("code") not in {c.value for c in ErrorCode}:
        problems.append(f"unknown error code {d['error'].get('code')!r}")
    return problems
