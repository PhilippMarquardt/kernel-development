"""Pod transport: ssh runner, md5-verified sync, pod bootstrap. Owner: P2."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from kdev.models import Check

NETWORK_FS = ("nfs", "nfs4", "cifs", "smbfs", "fuse", "fuse.mfs", "mfs", "moosefs", "glusterfs", "ceph",
              "fuse.sshfs", "9p", "lustre")


def _mount_fstype(path: str, mounts_file: str = "/proc/mounts") -> tuple[str, str]:
    """(mountpoint, fstype) of the longest mount prefix of path; ('', '') if unknown."""
    try:
        lines = Path(mounts_file).read_text().splitlines()
    except OSError:
        return "", ""
    best = ("", "")
    for line in lines:
        parts = line.split()
        if len(parts) < 3:
            continue
        mp = parts[1].replace("\\040", " ")
        if (path == mp or path.startswith(mp.rstrip("/") + "/") or mp == "/") and len(mp) >= len(best[0]):
            best = (mp, parts[2])
    return best


def probe(ctx: object, problem: str | None = None) -> list[Check]:
    """Doctor checks: KDEV_HOME not on a network/mfs path (e.g. /workspace), free disk, ssh key readable
    when remote is configured. Never raises."""
    checks: list[Check] = []
    try:
        from kdev.service._common import get_config
        cfg = get_config(ctx)
        home = str(Path(cfg.home).expanduser().resolve())
        mp, fstype = _mount_fstype(home)
        if home.startswith("/workspace") or fstype.lower() in NETWORK_FS or fstype.lower().startswith("fuse."):
            checks.append(Check("kdev home storage", "WARN", f"{home} is on {mp or '/workspace'} ({fstype or 'network'})",
                                hint="network/mfs storage hung before (M17): set KDEV_HOME to a local disk, "
                                     "back up with `kdev sync`"))
        else:
            checks.append(Check("kdev home storage", "PASS", f"{home} ({fstype or 'local'})"))
        probe_dir = home
        while probe_dir and not os.path.exists(probe_dir):
            probe_dir = os.path.dirname(probe_dir)
        free_gb = shutil.disk_usage(probe_dir or ".").free / 2**30
        checks.append(Check("free disk", "PASS" if free_gb >= 5 else "WARN", f"{free_gb:.1f} GB free at {probe_dir}",
                            hint="" if free_gb >= 5 else "builds/caches need several GB"))
        rc = cfg.remote
        if rc.host:
            key = os.path.expanduser(rc.key) if rc.key else ""
            if key and not os.access(key, os.R_OK):
                checks.append(Check("ssh key", "FAIL", f"{key} not readable", hint="fix remote.key in kdev.toml"))
            else:
                checks.append(Check("ssh key", "PASS", f"{key or 'ssh default'} for {rc.user}@{rc.host}:{rc.port}"))
            hb = Path(home) / "sync_heartbeat.json"
            if hb.exists():
                import json
                import time
                age_h = (time.time() - hb.stat().st_mtime) / 3600
                d = json.loads(hb.read_text(encoding="utf-8"))
                checks.append(Check("last sync", "PASS" if age_h <= 3 else "WARN",
                                    f"{age_h:.1f} h ago ({d.get('direction')}, copied {d.get('report', {}).get('copied')})",
                                    hint="" if age_h <= 3 else "pod storage is never persistent: run `kdev sync pull`"))
    except Exception as e:  # noqa: BLE001
        checks.append(Check("remote", "FAIL", f"probe error: {type(e).__name__}: {e}"))
    return checks
