"""Phased GPU lock: prepare under a SHARED flock, timed sections EXCLUSIVE (board: lock only the timed part).

The GPU lock stays the plain flock on ``cfg.lock_path`` that /root/gpu_run.sh takes with ``flock 9``
(exclusive). A phased kdev worker (``time --judge-mode``, every child process of a judge-mode ``ab``):

1. ``prepare()``: takes a prepare *slot* (``$KDEV_HOME/phase/slots/<k>``, N = ``$KDEV_PHASE_SLOTS``, default 6:
   bounds the number of processes holding prepared GPU state), waits its turn behind EX waiters that queued
   before it (FIFO with gpu_run.sh / non-phased kdev jobs) and behind phased processes that are ready to time
   (writer preference: timed sections run back to back, prepares batch), then takes the flock SHARED. Only
   then may it create its CUDA context. Other prepares may overlap; a timed section (EX) never does.
2. ``park()``: after ``torch.cuda.synchronize()`` the process is idle (no queued GPU work). It registers as
   ``parked`` and waits for its go token (``$KDEV_PHASE_GO``: the ab parent gives it when the previous ABBA
   process has finished, which keeps the A B B A timing order).
3. ``exclusive()``: state ``ready``, drop SH, take EX (kernel flock queue). The fd is moved to a high number
   and kept until the process exits, so verification, overhead graphs, the dispatch probe and the CUDA
   teardown at exit are still inside the exclusive section. Then the usual foreign-context wait: no other
   live CUDA context except *parked/ready* kdev processes (idle: synchronized, no work queued, blocked in a
   flock wait).

gpu_run.sh does not know about parked processes and waits for every CUDA context to go away. A parked process
therefore never blocks it: if the lock is held for ``BAIL_S`` with no kdev holder registered (= gpu_run.sh or
another foreign flock user), the parked process gives up (``Requeue``), exits, and the ab parent restarts it
(a fresh prepare). Non-phased kdev holders (``lock.exclusive``) register as ``holding`` and ignore parked pids.

Registry: ``$KDEV_HOME/phase/procs/<pid>.json`` {pid, start (proc start ticks: pid-reuse guard), state, job,
agent, t, t_wait}; states: wait_sh | preparing | parked | ready | timing | holding. Stale entries (dead pid /
other start time) are ignored and removed. Opt-out: ``KDEV_PHASED=0`` or ``extra.phased = false``.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from kdev.contract import ErrorCode, KdevError

IDLE_STATES = ("parked", "ready")                 # synchronized, no queued GPU work: ignored as foreign contexts
KDEV_HOLD_STATES = ("preparing", "timing", "holding")
DEFAULT_SLOTS = 6
BAIL_S = 6.0                                      # lock held by a non-kdev process this long -> parked gives up
HIGH_FD = 900
REQUEUE_FLAG = "phase_requeue"

Stop = Callable[[], None]


def _noop() -> None:
    return None


# ------------------------------------------------------------------------------------------- registry
def home_dir(home: str | os.PathLike[str] | None = None) -> Path:
    return Path(home or os.environ.get("KDEV_HOME") or os.path.expanduser("~/.kdev")) / "phase"


def proc_start(pid: int) -> int:
    """/proc/<pid>/stat field 22 (start time in clock ticks); -1 if gone/unknown."""
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8", errors="replace") as f:
            rest = f.read().rsplit(")", 1)[1].split()
        if rest[0] in ("Z", "X"):
            return -1
        return int(rest[19])
    except (OSError, IndexError, ValueError):
        return -1


def _write_atomic(p: Path, doc: dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(doc), encoding="utf-8")
    os.replace(tmp, p)


class Entry:
    """This process's registry entry."""

    def __init__(self, home: Any = None, pid: int | None = None, **info: Any) -> None:
        self.pid = int(pid or os.getpid())
        self.path = home_dir(home) / "procs" / f"{self.pid}.json"
        self.doc: dict[str, Any] = {"pid": self.pid, "start": proc_start(self.pid), "t": time.time(), **info}

    def set(self, state: str, **kw: Any) -> None:
        self.doc.update(state=state, t_state=time.time(), **kw)
        try:
            _write_atomic(self.path, self.doc)
        except OSError:
            pass

    def remove(self) -> None:
        try:
            self.path.unlink()
        except OSError:
            pass


def entries(home: Any = None, *, start_fn: Callable[[int], int] = proc_start, clean: bool = True
            ) -> list[dict[str, Any]]:
    """Live registry entries (dead pids / pid reuse -> dropped, and removed when ``clean``)."""
    d = home_dir(home) / "procs"
    out: list[dict[str, Any]] = []
    try:
        names = os.listdir(d)
    except OSError:
        return out
    for n in names:
        if not n.endswith(".json") or n.startswith("."):
            continue
        p = d / n
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
            pid = int(doc["pid"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
        st = start_fn(pid)
        if st < 0 or (doc.get("start", -1) not in (-1, None) and int(doc["start"]) != st):
            if clean:
                try:
                    p.unlink()
                except OSError:
                    pass
            continue
        out.append(doc)
    return out


def idle_pids(home: Any = None, **kw: Any) -> set[int]:
    """Pids of parked/ready phased processes: their CUDA contexts are idle and not 'foreign' to a timer."""
    return {int(e["pid"]) for e in entries(home, **kw) if e.get("state") in IDLE_STATES}


def kdev_holding(home: Any = None, **kw: Any) -> bool:
    return any(e.get("state") in KDEV_HOLD_STATES for e in entries(home, **kw))


# ------------------------------------------------------------------------------------------- policy
SHARED_KINDS = ("discover", "check")              # untimed GPU work: the whole job runs under the SHARED flock


def mode(spec: Any) -> str | None:
    """'phased' (judge-mode time/ab: prepare shared, timed section exclusive), 'shared' (untimed GPU jobs:
    discover, check without --cold: shared for the whole job) or None (exclusive for the whole job).
    POSIX only; ``KDEV_PHASED=0`` or ``extra.phased = false`` opt out."""
    if sys.platform == "win32" or os.environ.get("KDEV_PHASED", "1") in ("0", "held"):
        return None
    kind = getattr(spec, "kind", None)
    params = getattr(spec, "params", None) or {}
    extra = params.get("extra") or {}
    if extra.get("phased") is False:
        return None
    if kind in SHARED_KINDS:
        return None if extra.get("cold") else "shared"
    if kind in ("time", "ab") and extra.get("judge_mode") and not extra.get("fast"):
        return "phased"
    return None


def enabled(spec: Any) -> bool:
    return mode(spec) is not None


def n_slots() -> int:
    try:
        return max(1, int(os.environ.get("KDEV_PHASE_SLOTS", DEFAULT_SLOTS)))
    except ValueError:
        return DEFAULT_SLOTS


def must_defer(me_t: float, waiters: list[Any], reg: list[dict[str, Any]], me: int | None = None) -> bool:
    """Pure: should a phased preparer that started waiting at ``me_t`` hold back its SH request?
    Yes while a phased process is ready to time (writer preference), or while an EX waiter that is not a
    phased process queued before it (FIFO with gpu_run.sh / non-phased kdev jobs)."""
    me = os.getpid() if me is None else me
    if any(e.get("state") == "ready" and int(e["pid"]) != me for e in reg):
        return True
    phased = {int(e["pid"]) for e in reg}
    for w in waiters:
        if w.pid == me or w.owner_pid == me or w.pid in phased or w.owner_pid in phased:
            continue
        if w.started < me_t:
            return True
    return False


# ------------------------------------------------------------------------------------------- flock helpers
def _flock_wait(fd: int, op: int, stop: Stop, poll_s: float = 0.2) -> None:
    """Blocking flock(op) in a helper thread (kernel FIFO queue), ``stop()`` polled; on abandon the thread
    drops the lock as soon as it gets it. Same scheme as lock._FileLock.acquire_waiting."""
    import fcntl

    try:
        fcntl.flock(fd, op | fcntl.LOCK_NB)
        return
    except BlockingIOError:
        pass
    st = {"s": "waiting", "err": None}
    mtx = threading.Lock()
    done = threading.Event()

    def waiter() -> None:
        try:
            fcntl.flock(fd, op)
        except OSError as e:
            st["err"] = e
        with mtx:
            if st["s"] == "abandoned" and st["err"] is None:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except OSError:
                    pass
            elif st["err"] is None:
                st["s"] = "acquired"
        done.set()

    threading.Thread(target=waiter, name="kdev-phase-flock", daemon=True).start()
    while not done.wait(poll_s):
        try:
            stop()
        except BaseException:
            with mtx:
                if st["s"] == "waiting":
                    st["s"] = "abandoned"
                    raise
            break
    done.wait()
    if st["err"] is not None:
        raise st["err"]


def try_slot(home: Any = None, n: int | None = None) -> int | None:
    """A free prepare slot fd (flock EX on slots/<k>), or None."""
    if sys.platform == "win32":
        return None
    import fcntl

    d = home_dir(home) / "slots"
    d.mkdir(parents=True, exist_ok=True)
    for k in range(n or n_slots()):
        fd = os.open(str(d / str(k)), os.O_WRONLY | os.O_CREAT, 0o666)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except BlockingIOError:
            os.close(fd)
    return None


def wait_slot(home: Any = None, stop: Stop = _noop, poll_s: float = 0.5, n: int | None = None) -> int:
    while True:
        fd = try_slot(home, n)
        if fd is not None:
            return fd
        stop()
        time.sleep(poll_s)


def _high(fd: int) -> int:
    import fcntl

    try:
        hi = fcntl.fcntl(fd, fcntl.F_DUPFD_CLOEXEC, HIGH_FD)
    except OSError:
        return fd
    os.close(fd)
    return hi


# ------------------------------------------------------------------------------------------- the worker side
@dataclass
class PhaseFacts:
    slot_wait_s: float = 0.0
    sh_wait_s: float = 0.0
    prepare_s: float = 0.0
    parked_s: float = 0.0
    ex_wait_s: float = 0.0
    foreign_wait_s: float = 0.0
    deferred: int = 0
    exclusive: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


class PhasedLock:
    """Worker side of the phased lock (see module doc). ``stop`` raises to abort a wait (timeout/cancel)."""

    def __init__(self, lock_path: str, home: Any = None, *, job_id: str = "", agent: str = "",
                 go_path: str | None = None, slot_fd: int | None = None, stop: Stop = _noop,
                 foreign_wait_s: float = 600.0, apps_fn: Callable[[], list[Any]] | None = None,
                 log: Callable[[str], None] | None = None) -> None:
        self.lock_path, self.home, self.go_path, self.stop = lock_path, home, go_path, stop
        self.slot_fd = slot_fd
        self.fd: int | None = None
        self.entry = Entry(home, job=job_id, agent=agent)
        self.facts = PhaseFacts()
        self.foreign_wait_s = foreign_wait_s
        self.apps_fn = apps_fn
        self.log = log or (lambda m: None)
        import atexit

        atexit.register(self.entry.remove)

    @classmethod
    def from_env(cls, **kw: Any) -> PhasedLock | None:
        if os.environ.get("KDEV_PHASED") != "1" or sys.platform == "win32":
            return None
        slot = os.environ.get("KDEV_PHASE_SLOT_FD")
        return cls(os.environ.get("KDEV_LOCK_PATH") or "/root/gpu.lock", os.environ.get("KDEV_HOME"),
                   job_id=os.environ.get("KDEV_JOB_ID", ""), agent=os.environ.get("KDEV_AGENT", ""),
                   go_path=os.environ.get("KDEV_PHASE_GO") or None,
                   slot_fd=int(slot) if slot and slot.isdigit() else None, **kw)

    # -- 1. prepare (shared)
    def prepare(self) -> None:
        import fcntl

        from kdev.config import load_config
        from kdev.service import lockq

        t0 = time.monotonic()
        if self.slot_fd is None:
            self.entry.set("wait_slot")
            self.slot_fd = wait_slot(self.home, self.stop)
        self.facts.slot_wait_s = time.monotonic() - t0
        t1 = time.time()
        self.entry.set("wait_sh", t_wait=t1)
        try:
            cfg = load_config()
            cfg.lock_path = self.lock_path
        except Exception:  # noqa: BLE001
            cfg = None
        while True:
            self.stop()
            try:
                qv = lockq.view(cfg) if cfg is not None else None
                defer = must_defer(t1, qv.waiters if qv else [], entries(self.home))
            except Exception:  # noqa: BLE001
                defer = False
            if not defer:
                break
            self.facts.deferred += 1
            time.sleep(1.0)
        Path(self.lock_path).parent.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(self.lock_path, os.O_WRONLY | os.O_CREAT, 0o666)
        _flock_wait(self.fd, fcntl.LOCK_SH, self.stop)
        self.entry.set("preparing")
        # descendants (dispatch probe, ...) run inside our lock: they must not queue for it again
        os.environ["KDEV_PHASED"] = "held"
        os.environ.pop("KDEV_PHASE_SLOT_FD", None)
        self.facts.sh_wait_s = time.monotonic() - t0 - self.facts.slot_wait_s
        self._t_prep = time.monotonic()

    # -- 2. park (idle, shared released later)
    def park(self) -> None:
        """Prepare done (caller synchronized): drop the SHARED lock at once (a parked process holds NO lock,
        else the head of the ABBA chain could never get EX), then wait for the go token."""
        import fcntl

        self.facts.prepare_s = time.monotonic() - getattr(self, "_t_prep", time.monotonic())
        self.entry.set("parked")
        if self.fd is not None:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
        t0 = time.monotonic()
        give_way = _every(self._give_way_check(), 1.0)
        while self.go_path and not os.path.exists(self.go_path):
            self.stop()
            give_way()
            time.sleep(0.1)
        self.facts.parked_s = time.monotonic() - t0

    def _give_way_check(self) -> Stop:
        """A stop() that raises Requeue once the lock has been held for BAIL_S by a non-kdev process
        (gpu_run.sh waits for every CUDA context, our idle one included)."""
        held_since: list[float | None] = [None]

        def check() -> None:
            if kdev_holding(self.home) or _lock_free(self.lock_path):
                held_since[0] = None
                return
            now = time.monotonic()
            held_since[0] = held_since[0] or now
            if now - held_since[0] >= BAIL_S:
                raise KdevError(ErrorCode.GPU_BUSY_TIMEOUT, "phased: the GPU lock is held by a non-kdev process "
                                "(gpu_run.sh?) that waits for our idle context: giving way",
                                details={REQUEUE_FLAG: True}, recovery="the ab parent restarts this process")
        return check

    # -- 3. exclusive (held until exit)
    def exclusive(self) -> PhaseFacts:
        import fcntl

        assert self.fd is not None, "prepare() first"
        self.entry.set("ready")
        fcntl.flock(self.fd, fcntl.LOCK_UN)            # no-op after park(); drops SH if park() was skipped
        t0 = time.monotonic()
        give_way = self._give_way_check()        # a non-kdev holder (gpu_run.sh) waits for every context

        def stop() -> None:
            self.stop()
            give_way()

        _flock_wait(self.fd, fcntl.LOCK_EX, _every(stop, 1.0))
        self.fd = _high(self.fd)                  # kept until exit: CUDA teardown stays inside EX
        self.entry.set("timing")
        self.facts.ex_wait_s = time.monotonic() - t0
        self.facts.exclusive = self._foreign_wait()
        os.environ["KDEV_EXCLUSIVE"] = self.facts.exclusive
        os.environ["KDEV_WAITED_S"] = f"{self.facts.slot_wait_s + self.facts.sh_wait_s + self.facts.ex_wait_s:.1f}"
        self.log("phase: " + " ".join(f"{k}={v}" for k, v in self.facts.to_dict().items()))
        return self.facts

    def _foreign_wait(self) -> str:
        from kdev.service import lock as L

        apps = self.apps_fn or L.compute_apps
        t0 = time.monotonic()
        waited = False
        while True:
            own = {os.getpid()} | _ancestors() | idle_pids(self.home)
            foreign = L.foreign_apps(apps(), own)
            if not foreign:
                break
            waited = True
            if time.monotonic() - t0 > self.foreign_wait_s:
                raise KdevError(ErrorCode.GPU_BUSY_TIMEOUT, "phased: unlocked CUDA process(es) did not exit: "
                                + ", ".join(f"{a.pid} {a.cmdline[:60]}" for a in foreign),
                                recovery="ask the owner to stop the unlocked GPU process (`kdev queue offenders`)")
            self.stop()
            time.sleep(1.0)
        self.facts.foreign_wait_s = time.monotonic() - t0
        return "waited" if waited else "ok"


def _every(fn: Stop, period_s: float) -> Stop:
    last = [0.0]

    def f() -> None:
        now = time.monotonic()
        if now - last[0] >= period_s:
            last[0] = now
            fn()
    return f


def _lock_free(path: str) -> bool:
    import fcntl

    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return True
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except OSError:
            return False
        fcntl.flock(fd, fcntl.LOCK_UN)
        return True
    finally:
        os.close(fd)


def _ancestors() -> set[int]:
    out, p = set(), os.getpid()
    for _ in range(64):
        try:
            with open(f"/proc/{p}/stat", encoding="utf-8") as f:
                p = int(f.read().rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            break
        if p <= 1:
            break
        out.add(p)
    return out
