"""Exclusive-GPU lock with foreign-context detection (port of okc/tools/gpu_run.sh). Owner: P2.

Semantics: take an flock on ``cfg.lock_path`` (default /root/gpu.lock, shared with the legacy gpu_run.sh so
legacy scripts and kdev exclude each other), then wait until no OTHER live process holds a CUDA context
(nvidia-smi --query-compute-apps=pid; zombies 'Z*' ignored; own pid and children ignored). After 10 s of
waiting, log each offender once (queue.db offenders + legacy offender_log). Abort after
``cfg.foreign_wait_s`` (600 s) with gpu_busy_timeout. On Windows (no fcntl) use msvcrt locking or a
lock-file fallback so unit tests run.

Lessons kept (okc/LESSONS.md): never compile inside the lock (callers gate on requires_prebuilt); an unlocked
CUDA process stalls the whole queue, so it is named in the offender log; pids are matched exactly from
nvidia-smi (no ``pgrep -f`` pattern, which matches its own shell).
"""
from __future__ import annotations

import os
import re
import shlex
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Callable, Iterator

from kdev.config import Config
from kdev.contract import ErrorCode, KdevError
from kdev.service._common import descendants, proc_cmdline, proc_state
from kdev.util import Runner, run

OFFENDER_LOG_AFTER_S = 10.0     # gpu_run.sh logs offenders after 10 s of waiting
EXIT_TIMEOUT = 124              # like coreutils timeout(1)
_runner: Runner = run           # monkeypatched in tests


@dataclass
class ComputeApp:
    pid: int
    used_mib: int | None = None
    cmdline: str = ""
    state: str = ""          # ps stat ('Z' = zombie)


@dataclass
class LockGrant:
    """What the holder learns when the lock is granted."""

    exclusive: str            # ok | waited | fail | shared
    waited_s: float
    offenders: list[ComputeApp] = field(default_factory=list)
    lock_path: str = ""
    queue: dict[str, Any] = field(default_factory=dict)   # position/ETA at the start of the wait (lockq)


ComputeAppsFn = Callable[[], list[ComputeApp]]


def parse_compute_apps_csv(text: str) -> list[tuple[int, int | None]]:
    """Pure: 'pid, used_mib' lines of nvidia-smi --query-compute-apps -> [(pid, mib)]."""
    out: list[tuple[int, int | None]] = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if not parts or not parts[0].isdigit():
            continue
        mib = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
        out.append((int(parts[0]), mib))
    return out


def compute_apps() -> list[ComputeApp]:
    """Live CUDA processes from nvidia-smi + /proc (cmdline, stat). Empty list if nvidia-smi is missing."""
    try:
        p = _runner(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                    timeout=20)
    except Exception:  # noqa: BLE001
        return []
    if p.returncode != 0:
        return []
    have_proc = Path("/proc/self/stat").exists()
    apps = []
    for pid, mib in parse_compute_apps_csv(p.stdout):
        st = proc_state(pid) if have_proc else ""
        if have_proc and not st:
            continue      # not visible in this pid namespace / already gone (gpu_run.sh: empty ps stat)
        apps.append(ComputeApp(pid=pid, used_mib=mib, cmdline=proc_cmdline(pid), state=st))
    return apps


_COMPUTE_APPS_DEF = compute_apps


def foreign_apps(apps: list[ComputeApp], own_pids: set[int]) -> list[ComputeApp]:
    """Filter out own pids and zombies (pure; unit-testable)."""
    return [a for a in apps if a.pid not in own_pids and not a.state.upper().startswith(("Z", "X"))]


# ---------------------------------------------------------------- file lock primitive (flock / msvcrt)

class _FileLock:
    def __init__(self, path: str) -> None:
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(path, os.O_WRONLY | os.O_CREAT, 0o666)
        self.held = False

    def try_acquire(self) -> bool:
        if sys.platform == "win32":
            import msvcrt
            try:
                os.lseek(self.fd, 0, os.SEEK_SET)
                msvcrt.locking(self.fd, msvcrt.LK_NBLCK, 1)
            except OSError:
                return False
        else:
            import fcntl
            try:
                fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
        self.held = True
        return True

    def acquire_waiting(self, stop: Callable[[], None], poll_s: float = 0.2) -> None:
        """Block until the lock is ours. ``stop()`` is called every ``poll_s`` and may raise (timeout/cancel).

        POSIX: a blocking flock in a helper thread, so we queue in the kernel like gpu_run.sh's ``flock 9``.
        Polling with LOCK_NB starves behind blocked waiters: on release the kernel hands the lock to one of
        them, and a poller only wins in the rare gap with no queued waiter (seen on the pod: 10+ min waits
        while gpu_run.sh jobs kept getting the lock). The thread flocks a dup of our fd (same open file
        description = same lock); if we stop waiting, the thread releases the lock as soon as it gets it."""
        if sys.platform == "win32":
            while not self.try_acquire():
                stop()
                time.sleep(poll_s)
            return
        import fcntl
        import threading

        if self.try_acquire():
            return
        fd2 = os.dup(self.fd)
        mtx = threading.Lock()
        state = {"s": "waiting", "err": None}
        done = threading.Event()

        def waiter() -> None:
            try:
                fcntl.flock(fd2, fcntl.LOCK_EX)
            except OSError as e:
                state["err"] = e
            with mtx:
                if state["s"] == "abandoned" and state["err"] is None:
                    try:
                        fcntl.flock(fd2, fcntl.LOCK_UN)
                    except OSError:
                        pass
                elif state["err"] is None:
                    state["s"] = "acquired"
            try:
                os.close(fd2)
            except OSError:
                pass
            done.set()

        threading.Thread(target=waiter, name="kdev-flock", daemon=True).start()
        while not done.wait(poll_s):
            try:
                stop()
            except BaseException:
                with mtx:
                    if state["s"] == "waiting":
                        state["s"] = "abandoned"
                        raise
                break                       # acquired in the same instant: keep it
        done.wait()
        if state["err"] is not None:
            raise state["err"]
        self.held = True

    def release(self) -> None:
        try:
            if self.held:
                if sys.platform == "win32":
                    import msvcrt
                    os.lseek(self.fd, 0, os.SEEK_SET)
                    msvcrt.locking(self.fd, msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.fd, fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            self.held = False
            try:
                os.close(self.fd)
            except OSError:
                pass


def holder(cfg: Config) -> dict[str, Any]:
    """Who holds the lock: {held: bool|None, pid, cmdline} (pid from /proc/locks on Linux)."""
    path = cfg.lock_path
    if not os.path.exists(path):
        return {"held": False, "path": path}
    try:
        lk = _FileLock(path)
    except OSError as e:
        return {"held": None, "path": path, "error": str(e)}
    free = lk.try_acquire()
    lk.release()
    info: dict[str, Any] = {"held": not free, "path": path}
    if not free:
        try:
            ino = os.stat(path).st_ino
            for line in Path("/proc/locks").read_text().splitlines():
                parts = line.split()
                if len(parts) >= 6 and parts[1] == "FLOCK" and parts[5].split(":")[-1] == str(ino):
                    pid = int(parts[4])
                    info.update(pid=pid, cmdline=proc_cmdline(pid, 200))
                    break
        except (OSError, ValueError):
            pass
        if "pid" not in info:                      # /proc/locks is often empty inside containers
            openers = _openers(path)
            if openers:
                info["openers"] = [{"pid": p, "cmdline": proc_cmdline(p, 120)} for p in openers[:5]]
    return info


def _openers(path: str) -> list[int]:
    """Pids with ``path`` open (holder and waiters; gpu_run.sh keeps fd 9 open in its command), via /proc."""
    out = []
    try:
        real = os.path.realpath(path)
        for d in Path("/proc").iterdir():
            if not d.name.isdigit() or int(d.name) == os.getpid():
                continue
            try:
                for fd in (d / "fd").iterdir():
                    if os.readlink(fd) == real:
                        out.append(int(d.name))
                        break
            except OSError:
                continue
    except OSError:
        pass
    return sorted(out)


# ---------------------------------------------------------------- offenders

def _record_offender(cfg: Config, app: ComputeApp, job_id: str | None, action: str) -> None:
    """queue.db offenders row + legacy gpu_offenders.log line (gpu_run.sh format, read by the watcher)."""
    try:
        from kdev.service import queue
        conn = queue.connect(cfg)
        try:
            queue.log_offender(conn, app.pid, app.cmdline, gpu_mem_mib=app.used_mib, during_job=job_id,
                               action=action)
        finally:
            conn.close()
    except Exception:  # noqa: BLE001  (never let logging break the lock)
        pass
    try:
        if cfg.offender_log and Path(cfg.offender_log).parent.is_dir():   # never create /root on a laptop
            with open(cfg.offender_log, "a", encoding="utf-8") as f:
                verb = "waiting on" if action == "waited" else f"{action}:"
                f.write(f"{time.strftime('%H:%M:%S', time.gmtime())} {verb} unlocked pid {app.pid}: "
                        f"{app.cmdline[:200]}{' (job ' + job_id + ')' if job_id else ''} [kdev]\n")
    except OSError:
        pass


# ---------------------------------------------------------------- exclusive()

def idle_pids(cfg: Config) -> set[int]:
    """Parked/ready phased kdev workers (service.phase): idle CUDA contexts, not foreign to a lock holder."""
    if sys.platform == "win32":
        return set()
    try:
        from kdev.service import phase  # noqa: PLC0415

        return phase.idle_pids(cfg.home)
    except Exception:  # noqa: BLE001
        return set()


def _phase_hold_entry(cfg: Config, job_id: str | None, agent: str) -> Any:
    """Register this (non-phased) EX holder as a kdev holder, so parked phased workers do not give way."""
    if sys.platform == "win32":
        return None
    try:
        from kdev.service import phase  # noqa: PLC0415

        e = phase.Entry(cfg.home, job=job_id or "", agent=agent)
        e.set("holding")
        return e
    except Exception:  # noqa: BLE001
        return None


def _own_pids() -> set[int]:
    me = os.getpid()
    return {me} | descendants(me)


class _Waiter:
    """Queue bookkeeping while waiting for the flock: position/ETA reports (``progress``), holder-change
    observations (hold durations for the ETA) and the kdev priority gate. Never raises."""

    REPORT_S = 30.0
    SCAN_S = 5.0

    def __init__(self, cfg: Config, priority: int, entry: Any, progress: Callable[[str], None] | None,
                 qinfo: dict[str, Any]) -> None:
        self.cfg, self.priority, self.entry, self.progress, self.qinfo = cfg, int(priority or 0), entry, progress, qinfo
        self.t_scan = self.t_report = 0.0
        self.key: str | None = None
        self.key_t = 0.0
        self.first_key = True

    def scan(self, report: bool = False) -> None:
        from kdev.service import lockq  # noqa: PLC0415

        try:
            qv = lockq.view(self.cfg)
        except Exception:  # noqa: BLE001
            return
        now = time.monotonic()
        k = lockq.holder_key(qv)
        if k != self.key:
            # a hold is recorded only when we saw it start (not the holder found at our first scan)
            if self.key and not self.first_key:
                lockq.record_hold(self.cfg, self.key, now - self.key_t)
            self.first_key = self.key is None
            self.key, self.key_t = k, now
        pos = qv.position(os.getpid())
        if not self.qinfo:
            self.qinfo.update(position=pos, queue_len=len(qv.waiters), eta_s=qv.eta_s(pos),
                              median_hold_s=qv.median_hold_s, holders=[h["cmdline"][:80] for h in qv.holders])
        self.qinfo.update(last_position=pos, last_eta_s=qv.eta_s(pos))
        if report and self.progress is not None:
            try:
                self.progress(lockq.describe(qv))
            except Exception:  # noqa: BLE001
                pass

    def tick(self) -> None:
        now = time.monotonic()
        if now - self.t_scan >= self.SCAN_S:
            rep = now - self.t_report >= self.REPORT_S
            self.t_scan = now
            if rep:
                self.t_report = now
            self.scan(report=rep)

    def blocked_by_priority(self) -> bool:
        from kdev.service import lockq  # noqa: PLC0415

        hp = lockq.higher_priority_waiting(self.cfg, self.priority)
        if hp:
            self.qinfo.setdefault("deferred_to", sorted({int(d["pid"]) for d in hp}))
            self.tick()
        return bool(hp)

    def done(self) -> None:
        from kdev.service import lockq  # noqa: PLC0415

        lockq.mark(self.entry, "granted")


def _queue_waiter(cfg: Config, priority: int | None, job_id: str | None, agent: str,
                  progress: Callable[[str], None] | None, qinfo: dict[str, Any]) -> tuple[Any, _Waiter]:
    entry = None
    if sys.platform != "win32":
        from kdev.service import lockq  # noqa: PLC0415

        entry = lockq.register(cfg, priority=int(priority or 0), job_id=job_id, agent=agent)
    w = _Waiter(cfg, int(priority or 0), entry, progress, qinfo)
    # first scan/report ~1 s into the wait: by then our flock waiter thread is queued (counted as a waiter)
    now = time.monotonic()
    w.t_scan, w.t_report = now - w.SCAN_S + 1.0, now - w.REPORT_S + 1.0
    return entry, w


def _stderr_progress(msg: str) -> None:
    print(f"kdev: {msg}", file=sys.stderr, flush=True)


def _exclusive_gen(cfg: Config, *, timeout_s: float | None = None, job_id: str | None = None,
                   apps_fn: ComputeAppsFn = compute_apps, poll_s: float = 2.0,
                   abort: Callable[[], bool] | None = None,
                   own_pids_fn: Callable[[], set[int]] | None = None, priority: int | None = None,
                   agent: str = "", progress: Callable[[str], None] | None = _stderr_progress
                   ) -> Iterator[LockGrant]:
    own_fn = own_pids_fn or _own_pids
    if apps_fn is _COMPUTE_APPS_DEF:          # default: resolve at call time (monkeypatchable module attr)
        apps_fn = lambda: globals()["compute_apps"]()  # noqa: E731
    t0 = time.monotonic()
    lk = _FileLock(cfg.lock_path)
    qinfo: dict[str, Any] = {}
    entry = None
    hold_entry = None
    try:
        def _stop() -> None:
            if timeout_s is not None and time.monotonic() - t0 > timeout_s:
                h = holder(cfg)
                raise KdevError(ErrorCode.GPU_BUSY_TIMEOUT,
                                f"GPU lock {cfg.lock_path} still held after {timeout_s:.0f} s "
                                f"(pid {h.get('pid', '?')}: {h.get('cmdline', '')[:80]})",
                                details={"holder": h}, recovery="retry later or check `kdev queue status`")
            if abort is not None and abort():
                raise KdevError(ErrorCode.CANCELLED, "cancelled while waiting for the GPU lock")

        if lk.try_acquire():
            qinfo.update(position=0, queue_len=0, eta_s=0)
        else:
            entry, waiter = _queue_waiter(cfg, priority, job_id, agent, progress, qinfo)
            while sys.platform != "win32" and priority is not None and waiter.blocked_by_priority():
                _stop()
                time.sleep(1.0)
            lk.acquire_waiting(lambda: (_stop(), waiter.tick()), min(0.2, poll_s))
            waiter.done()
        t_lock = time.monotonic()
        hold_entry = _phase_hold_entry(cfg, job_id, agent)
        seen: dict[int, ComputeApp] = {}       # logged offenders
        everseen: dict[int, ComputeApp] = {}   # every foreign app that delayed us
        waited = False
        while True:
            foreign = foreign_apps(apps_fn(), own_fn() | idle_pids(cfg))
            if not foreign:
                break
            waited = True
            for a in foreign:
                everseen.setdefault(a.pid, a)
            el = time.monotonic() - t_lock
            if el >= OFFENDER_LOG_AFTER_S:
                for a in foreign:
                    if a.pid not in seen:
                        seen[a.pid] = a
                        _record_offender(cfg, a, job_id, "waited")
            if el > cfg.foreign_wait_s:
                for a in foreign:
                    _record_offender(cfg, a, job_id, "aborted")
                print("GPU_EXCLUSIVE=FAIL other GPU processes: " + " ".join(str(a.pid) for a in foreign),
                      file=sys.stderr)
                raise KdevError(ErrorCode.GPU_BUSY_TIMEOUT,
                                f"{len(foreign)} unlocked CUDA process(es) did not exit within "
                                f"{cfg.foreign_wait_s} s: " + ", ".join(f"{a.pid} {a.cmdline[:60]}" for a in foreign),
                                details={"offenders": [a.__dict__ for a in foreign]},
                                recovery="ask the owner to stop the unlocked GPU process (`kdev queue offenders`)")
            if abort is not None and abort():
                raise KdevError(ErrorCode.CANCELLED, "cancelled while waiting for foreign CUDA contexts")
            time.sleep(poll_s)
        grant = LockGrant(exclusive="waited" if waited else "ok", waited_s=round(time.monotonic() - t0, 1),
                          offenders=list(everseen.values()), lock_path=cfg.lock_path, queue=dict(qinfo))
        grant._fd = lk.fd  # type: ignore[attr-defined]  # handed to the child (pass_fds) like gpu_run.sh fd 9
        yield grant
    finally:
        lk.release()
        if hold_entry is not None:
            hold_entry.remove()
        if entry is not None:
            from kdev.service import lockq  # noqa: PLC0415

            lockq.unregister(entry)


@contextmanager
def exclusive(cfg: Config, *, timeout_s: float | None = None, job_id: str | None = None,
              apps_fn: ComputeAppsFn = compute_apps, poll_s: float = 2.0) -> Iterator[LockGrant]:
    """flock + wait for no foreign CUDA contexts; yields a LockGrant; KdevError(gpu_busy_timeout) on abort.
    The holder must not compile (enforced by callers via requires_prebuilt). ``timeout_s`` bounds the wait
    for the flock itself (None = forever, like gpu_run.sh); the foreign-context wait is cfg.foreign_wait_s."""
    yield from _exclusive_gen(cfg, timeout_s=timeout_s, job_id=job_id, apps_fn=apps_fn, poll_s=poll_s)


exclusive_abortable = contextmanager(_exclusive_gen)   # daemon: same, plus abort=callable (cancel/stop)


# ---------------------------------------------------------------- deny-list + lock run

_SEP = re.compile(r"\s*(?:;|&&|\|\||\||&|\n|\(|\)|`|\$\()\s*")


def _segments(argv: list[str]) -> list[list[str]]:
    """Command segments of argv, also looking inside `bash -c "..."`-style string arguments."""
    segs: list[list[str]] = [[]]
    for arg in argv:
        pieces = _SEP.split(arg)
        for i, piece in enumerate(pieces):
            if i > 0:
                segs.append([])
            try:
                toks = shlex.split(piece) if (" " in piece or "'" in piece or '"' in piece) else [piece]
            except ValueError:
                toks = piece.split()
            segs[-1].extend(t for t in toks if t)
    return [s for s in segs if s]


def _norm(tok: str) -> str:
    b = os.path.basename(tok.replace("\\", "/")).lower()
    for ext in (".exe", ".py", ".sh"):
        if b.endswith(ext):
            b = b[: -len(ext)]
    return b


def is_denied(cfg: Config, argv: list[str]) -> bool:
    """True if argv matches cfg.deny_commands (word-prefix match after basename of argv[0]).

    Also catches wrapped forms (``timeout 30 <deny cmd>``, ``bash -c "cd x && <deny cmd> f"``,
    ``python -m <deny cmd>``): any segment containing the deny command's first word followed (later in the
    same segment) by its remaining words."""
    deny = [d.split() for d in (cfg.deny_commands or []) if d.strip()]
    for seg in _segments(list(argv)):
        words = [_norm(t) for t in seg]
        for d in deny:
            first, rest = d[0].lower(), [w.lower() for w in d[1:]]
            for i, w in enumerate(words):
                if w == first and all(r in words[i + 1:] for r in rest):
                    return True
    return False


def strip_dashdash(argv: list[str] | None) -> list[str]:
    argv = list(argv or [])
    while argv and argv[0] == "--":
        argv = argv[1:]
    return argv


def _kill_tree(proc: subprocess.Popen, grace_s: float = 5.0) -> None:
    """SIGTERM the child's process group, then SIGKILL after grace_s."""
    if proc.poll() is not None:
        return
    try:
        if sys.platform != "win32":
            os.killpg(proc.pid, signal.SIGTERM)
        else:
            proc.terminate()
        proc.wait(grace_s)
    except (subprocess.TimeoutExpired, ProcessLookupError, PermissionError, OSError):
        try:
            if sys.platform != "win32":
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
        except (ProcessLookupError, PermissionError, OSError):
            pass
        try:
            proc.wait(5)
        except subprocess.TimeoutExpired:
            pass


def spawn(argv: list[str], *, lock_fd: int | None = None, stdout: Any = None, stderr: Any = None,
          cwd: str | None = None, env: dict[str, str] | None = None) -> subprocess.Popen:
    """Start a child in its own session; on POSIX the lock fd is inherited (gpu_run.sh: fd 9 stays held by
    the command even if the parent dies, so the GPU never looks free while it runs)."""
    kw: dict[str, Any] = {"stdout": stdout, "stderr": stderr, "cwd": cwd, "env": env, "stdin": subprocess.DEVNULL}
    if sys.platform != "win32":
        kw["start_new_session"] = True
        if lock_fd is not None:
            kw["pass_fds"] = (lock_fd,)
    else:
        kw["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    return subprocess.Popen(argv, **kw)


def _run_locked(cfg: Config, argv: list[str], timeout_s: float = 300, child_stdout: IO[str] | int | None = None,
                apps_fn: ComputeAppsFn | None = None) -> int:
    argv = strip_dashdash(argv)
    if not argv:
        raise KdevError(ErrorCode.INVALID_REQUEST, "lock run needs a command: kdev lock run -- CMD ...",
                        recovery="put the command after `--`")
    if is_denied(cfg, argv):
        raise KdevError(ErrorCode.QUOTA_FORBIDDEN, f"deny-listed command: {' '.join(argv)[:120]}",
                        details={"deny_commands": cfg.deny_commands},
                        recovery="judge checks go through `kdev judge pair --allow-check`; submit is never run")
    with exclusive(cfg, apps_fn=apps_fn or _COMPUTE_APPS_DEF) as g:
        print(f"GPU_EXCLUSIVE=OK waited={int(g.waited_s)}s", file=sys.stderr, flush=True)
        try:
            proc = spawn(argv, lock_fd=getattr(g, "_fd", None), stdout=child_stdout)
        except FileNotFoundError:
            print(f"kdev lock run: command not found: {argv[0]}", file=sys.stderr)
            return 127
        try:
            return proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            print(f"kdev lock run: timeout after {timeout_s} s, killing pid {proc.pid}", file=sys.stderr)
            _kill_tree(proc)
            return EXIT_TIMEOUT
        except BaseException:
            _kill_tree(proc)
            raise


def run_locked(cfg: Config, argv: list[str], timeout_s: float = 300) -> int:
    """`kdev lock run -- CMD`: deny-list check (cfg.deny_commands -> quota_forbidden), then exclusive() and
    exec with timeout; prints GPU_EXCLUSIVE=OK waited=Ns to stderr like gpu_run.sh."""
    return _run_locked(cfg, argv, timeout_s)
