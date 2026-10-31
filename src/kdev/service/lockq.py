"""GPU lock queue view, ETA and kdev-side priority (P2).

The lock is a plain flock on ``cfg.lock_path`` shared with /root/gpu_run.sh, so the kernel's wait queue is
the real queue. It is reconstructed from /proc: every process with the lock file open is a holder or a
waiter; a waiter has a task sleeping in the flock wait (``wchan`` *lock_inode_wait*); gpu_run.sh waits in a
``flock 9`` child whose parent (the gpu_run.sh shell) is grouped with it. Waiters are ordered by the start
time of the waiting process (gpu_run.sh's ``flock 9`` starts when its wait starts; for kdev the registry
entry time is used). Linux wakes flock waiters roughly in that order, so the position is an estimate.

ETA = (waiters ahead + 1/2 for the running holder) x the median observed hold. Holds are observed by every
waiting kdev process (holder set changes) and appended to ``$KDEV_HOME/lock_holds.jsonl``.

Priority (``--priority N``) is honoured among kdev waiters only, without touching mutual exclusion: a kdev
job registers in ``$KDEV_HOME/lockq/<pid>.json`` and does not enter the flock wait while a live kdev
waiter with a higher priority is registered and has not been granted the lock yet. gpu_run.sh jobs are
not reordered (their queue position is the kernel's).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from kdev.config import Config

DEFAULT_HOLD_S = 60.0
HOLDS_KEEP = 200
_WAIT_WCHAN = ("lock_inode_wait", "flock_lock", "locks_lock", "__flock")


@dataclass
class QEntry:
    pid: int                  # the waiting process (flock child, or the kdev process)
    owner_pid: int            # the job process (gpu_run.sh shell / kdev)
    cmdline: str
    started: float            # epoch seconds the wait started (approx.)
    kind: str                 # gpu_run | kdev | other
    priority: int = 0
    agent: str = ""


@dataclass
class QueueView:
    holders: list[dict[str, Any]] = field(default_factory=list)
    waiters: list[QEntry] = field(default_factory=list)
    median_hold_s: float = DEFAULT_HOLD_S
    n_holds: int = 0

    def position(self, pid: int) -> int | None:
        for n, w in enumerate(self.waiters, 1):
            if pid in (w.pid, w.owner_pid):
                return n
        return None

    def eta_s(self, position: int | None) -> float:
        ahead = (position - 1) if position else len(self.waiters)
        return round((ahead + (0.5 if self.holders else 0.0)) * self.median_hold_s, 0)

    def to_dict(self) -> dict[str, Any]:
        return {"holders": self.holders, "waiters": [asdict(w) for w in self.waiters],
                "median_hold_s": self.median_hold_s, "n_holds": self.n_holds}


# ------------------------------------------------------------------------------------------ /proc helpers
def _read(p: str) -> str:
    try:
        with open(p, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def _cmdline(pid: int, n: int = 160) -> str:
    return _read(f"/proc/{pid}/cmdline").replace("\0", " ").strip()[:n]


def _waiting(pid: int) -> bool:
    try:
        tasks = os.listdir(f"/proc/{pid}/task")
    except OSError:
        tasks = []
    for t in tasks or [str(pid)]:
        w = _read(f"/proc/{pid}/task/{t}/wchan")
        if any(k in w for k in _WAIT_WCHAN):
            return True
    return False


def _ppid(pid: int) -> int:
    st = _read(f"/proc/{pid}/stat")
    try:
        return int(st.rsplit(")", 1)[1].split()[1])
    except (IndexError, ValueError):
        return 0


_BOOT: float | None = None


def _start_epoch(pid: int) -> float:
    """Process start time (epoch s) from /proc/<pid>/stat field 22."""
    global _BOOT
    st = _read(f"/proc/{pid}/stat")
    try:
        ticks = int(st.rsplit(")", 1)[1].split()[19])
    except (IndexError, ValueError):
        return time.time()
    if _BOOT is None:
        for ln in _read("/proc/stat").splitlines():
            if ln.startswith("btime"):
                _BOOT = float(ln.split()[1])
        _BOOT = _BOOT or 0.0
    return _BOOT + ticks / (os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100)


def openers(path: str) -> list[int]:
    """Pids with ``path`` open, this process included (lock._openers skips the caller)."""
    from kdev.service.lock import _openers  # noqa: PLC0415

    out = set(_openers(path))
    try:
        real = os.path.realpath(path)
        for fd in os.listdir("/proc/self/fd"):
            try:
                if os.readlink(f"/proc/self/fd/{fd}") == real:
                    out.add(os.getpid())
                    break
            except OSError:
                continue
    except OSError:
        pass
    return sorted(out)


# ------------------------------------------------------------------------------------------ registry
def _qdir(cfg: Config) -> Path:
    return Path(cfg.home) / "lockq"


def register(cfg: Config, *, priority: int, job_id: str | None, agent: str) -> Path | None:
    try:
        d = _qdir(cfg)
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{os.getpid()}.json"
        p.write_text(json.dumps({"pid": os.getpid(), "priority": int(priority or 0), "t": time.time(),
                                 "job_id": job_id, "agent": agent, "state": "waiting"}), encoding="utf-8")
        return p
    except OSError:
        return None


def mark(entry: Path | None, state: str) -> None:
    if entry is None:
        return
    try:
        d = json.loads(entry.read_text(encoding="utf-8"))
        d["state"] = state
        entry.write_text(json.dumps(d), encoding="utf-8")
    except (OSError, ValueError):
        pass


def unregister(entry: Path | None) -> None:
    if entry is not None:
        try:
            entry.unlink()
        except OSError:
            pass


def _alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":                 # os.kill(pid, 0) would send CTRL_C_EVENT; the registry is POSIX-only
        return pid == os.getpid()
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    st = _read(f"/proc/{pid}/stat")
    return not (st and st.rsplit(")", 1)[-1].split()[:1] in (["Z"], ["X"]))


def registered(cfg: Config) -> list[dict[str, Any]]:
    out = []
    try:
        files = list(_qdir(cfg).glob("*.json"))
    except OSError:
        return out
    for f in files:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not _alive(int(d.get("pid", 0))):
            try:
                f.unlink()
            except OSError:
                pass
            continue
        out.append(d)
    return out


def higher_priority_waiting(cfg: Config, priority: int, me: int | None = None) -> list[dict[str, Any]]:
    me = os.getpid() if me is None else me
    return [d for d in registered(cfg) if d.get("pid") != me and d.get("state") == "waiting"
            and int(d.get("priority", 0)) > int(priority or 0)]


# ------------------------------------------------------------------------------------------ holds
def _holds_path(cfg: Config) -> Path:
    return Path(cfg.home) / "lock_holds.jsonl"


def record_hold(cfg: Config, key: str, seconds: float) -> None:
    if seconds <= 0.5:
        return
    p = _holds_path(cfg)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": round(time.time(), 1), "key": key, "s": round(seconds, 1)}) + "\n")
        if p.stat().st_size > 64 * 1024:
            lines = p.read_text(encoding="utf-8").splitlines()[-HOLDS_KEEP:]
            p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError:
        pass


def median_hold(cfg: Config, window_s: float = 3 * 3600) -> tuple[float, int]:
    try:
        lines = _holds_path(cfg).read_text(encoding="utf-8").splitlines()
    except OSError:
        return DEFAULT_HOLD_S, 0
    now = time.time()
    seen: dict[str, float] = {}
    for ln in lines[-HOLDS_KEEP:]:
        try:
            d = json.loads(ln)
        except ValueError:
            continue
        if now - float(d.get("t", 0)) <= window_s:
            seen[str(d.get("key"))] = max(seen.get(str(d.get("key")), 0.0), float(d.get("s", 0)))
    v = sorted(seen.values())
    if not v:
        return DEFAULT_HOLD_S, 0
    return v[len(v) // 2], len(v)


# ------------------------------------------------------------------------------------------ view
def view(cfg: Config, *, opener_fn: Callable[[str], list[int]] | None = None,
         waiting_fn: Callable[[int], bool] | None = None) -> QueueView:
    """Holders and waiters of the GPU lock (POSIX /proc; empty elsewhere)."""
    qv = QueueView()
    qv.median_hold_s, qv.n_holds = median_hold(cfg)
    if not os.path.isdir("/proc") and opener_fn is None:
        return qv
    ops = (opener_fn or openers)(cfg.lock_path)
    wfn = waiting_fn or _waiting
    reg = {int(d["pid"]): d for d in registered(cfg)}
    waiting = {p for p in ops if wfn(p)}
    parents = {_ppid(p): p for p in waiting}
    for p in sorted(waiting):
        owner = _ppid(p) if _ppid(p) in ops and _ppid(p) not in waiting else p
        cmd = _cmdline(owner)
        r = reg.get(p) or reg.get(owner)
        kind = "kdev" if r or "kdev" in cmd else ("gpu_run" if "gpu_run" in cmd else "other")
        started = float(r["t"]) if r else _start_epoch(p)
        qv.waiters.append(QEntry(pid=p, owner_pid=owner, cmdline=cmd, started=started, kind=kind,
                                 priority=int((r or {}).get("priority", 0)), agent=str((r or {}).get("agent", ""))))
    qv.waiters.sort(key=lambda w: (w.started, w.pid))
    for p in ops:
        if p in waiting or p in parents:
            continue
        # holder tree: report the topmost opener (its children inherit fd 9)
        if _ppid(p) in ops and _ppid(p) not in waiting:
            continue
        qv.holders.append({"pid": p, "cmdline": _cmdline(p), "since": round(_start_epoch(p), 1)})
    return qv


def holder_key(qv: QueueView) -> str:
    return ",".join(str(h["pid"]) for h in sorted(qv.holders, key=lambda h: h["pid"]))


def describe(qv: QueueView, me: int | None = None) -> str:
    me = os.getpid() if me is None else me
    pos = qv.position(me)
    n = len(qv.waiters)
    ahead = qv.waiters[: (pos - 1) if pos else n]
    kinds = {k: sum(1 for w in ahead if w.kind == k) for k in ("gpu_run", "kdev", "other")}
    who = qv.holders[0]["cmdline"][:60] if qv.holders else "-"
    eta = qv.eta_s(pos)
    src = f"median hold {qv.median_hold_s:.0f}s" + (f" of {qv.n_holds}" if qv.n_holds else " (default)")
    return (f"GPU lock queue: position {pos or '?'}/{n} (ahead: {kinds['gpu_run']} gpu_run.sh, {kinds['kdev']} kdev"
            + (f", {kinds['other']} other" if kinds["other"] else "") + f"), ETA ~{eta / 60:.1f} min ({src}); "
            f"holder: {who}")
