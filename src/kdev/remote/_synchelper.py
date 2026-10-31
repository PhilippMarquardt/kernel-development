"""Private sync helper (P2). Self-contained and stdlib-only: imported locally AND sent verbatim to the pod as
``python3 -c <this source> <action> ...`` so both sides index/pack/unpack with identical code.

Actions (remote CLI): ``index HOME PATHS_JSON SINCE_S`` -> JSON {rel: [md5, mtime]};
``pack HOME LIST_JSON_FILE OUT_TGZ`` -> md5 of the tgz;
``unpack HOME TGZ MANIFEST_JSON_FILE TS`` -> JSON report {verified, failed, copied, conflicts, bytes}.
SQLite files are indexed/packed from a consistent ``sqlite3`` backup snapshot (WAL-safe).
"""
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import tarfile
import time

SKIP_DIRS = {"__pycache__", ".sync_snap", ".sync_staging", "bootstrap_tmp"}
SKIP_SUFFIX = (".part", "-wal", "-shm", ".lock", ".pyc", ".tmp", "serve.stop")
SNAP = ".sync_snap"


def md5_file(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _skip(name):
    return name.endswith(SKIP_SUFFIX) or ".conflict-" in name or name.startswith(".") and name.endswith(".tmp")


def _db_mtime(path):
    m = os.path.getmtime(path)
    if os.path.exists(path + "-wal"):
        m = max(m, os.path.getmtime(path + "-wal"))
    return m


def snapshot_db(home, rel):
    """Consistent copy of a sqlite db under HOME/.sync_snap/rel (WAL included); returns its path."""
    src = os.path.join(home, rel)
    dst = os.path.join(home, SNAP, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
    s = sqlite3.connect(src, timeout=30)
    d = sqlite3.connect(tmp)
    try:
        s.backup(d)
    finally:
        d.close()
        s.close()
    os.replace(tmp, dst)
    m = _db_mtime(src)
    os.utime(dst, (m, m))
    return dst


def source_path(home, rel):
    """Path whose bytes represent rel (the snapshot for sqlite dbs)."""
    snap = os.path.join(home, SNAP, rel)
    if rel.endswith(".db") and os.path.exists(snap):
        return snap
    return os.path.join(home, rel)


def index(home, paths, since_s=None):
    out = {}
    cut = time.time() - since_s if since_s else None
    for top in paths:
        top = top.strip("/")
        full = os.path.join(home, top)
        if os.path.isfile(full):
            files = [top]
        elif os.path.isdir(full):
            files = []
            for root, dirs, names in os.walk(full):
                dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
                for n in names:
                    if not _skip(n):
                        files.append(os.path.relpath(os.path.join(root, n), home).replace(os.sep, "/"))
        else:
            continue
        for rel in files:
            p = os.path.join(home, rel)
            try:
                if rel.endswith(".db"):
                    mt = _db_mtime(p)
                    if cut and mt < cut:
                        continue
                    out[rel] = [md5_file(snapshot_db(home, rel)), mt]
                else:
                    mt = os.path.getmtime(p)
                    if cut and mt < cut:
                        continue
                    out[rel] = [md5_file(p), mt]
            except (OSError, sqlite3.Error):
                continue
    return out


def pack(home, rels, out_tgz):
    with tarfile.open(out_tgz, "w:gz") as tf:
        for rel in rels:
            tf.add(source_path(home, rel), arcname=rel, recursive=False)
    return md5_file(out_tgz)


def unpack(home, tgz, manifest, ts):
    """manifest: {rel: [md5, action, mtime]}; action copy|conflict. Verifies every file's md5 before moving it
    into place; never overwrites on 'conflict' (writes rel.conflict-<ts>)."""
    rep = {"verified": 0, "failed": [], "copied": 0, "conflicts": [], "bytes": 0}
    stage = os.path.join(home, ".sync_staging", ts)
    os.makedirs(stage, exist_ok=True)
    real_stage = os.path.realpath(stage)
    with tarfile.open(tgz, "r:gz") as tf:
        for m in tf.getmembers():
            if not m.isfile() or m.name not in manifest:
                continue
            tgt = os.path.realpath(os.path.join(stage, m.name))
            if not tgt.startswith(real_stage + os.sep):
                rep["failed"].append(m.name)
                continue
            os.makedirs(os.path.dirname(tgt), exist_ok=True)
            with tf.extractfile(m) as src, open(tgt, "wb") as dst:
                shutil.copyfileobj(src, dst)
    for rel, (want, action, mtime) in manifest.items():
        staged = os.path.join(stage, rel)
        if not os.path.exists(staged) or md5_file(staged) != want:
            rep["failed"].append(rel)
            continue
        rep["verified"] += 1
        dst = os.path.join(home, rel)
        if action == "conflict":
            dst = dst + ".conflict-" + ts
            rep["conflicts"].append(rel)
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        if rel.endswith(".db") and action != "conflict":
            for side in ("-wal", "-shm"):     # replacing a WAL db: stale side files would corrupt it
                if os.path.exists(dst + side):
                    os.remove(dst + side)
        os.replace(staged, dst)
        os.utime(dst, (mtime, mtime))
        rep["copied"] += 1
        rep["bytes"] += os.path.getsize(dst)
    shutil.rmtree(stage, ignore_errors=True)
    return rep


def main(argv):
    act = argv[0]
    if act == "index":
        since = float(argv[3]) if len(argv) > 3 and argv[3] not in ("", "None", "0") else None
        print(json.dumps(index(argv[1], json.loads(argv[2]), since)))
    elif act == "pack":
        with open(argv[2]) as f:
            rels = json.load(f)
        print(pack(argv[1], rels, argv[3]))
    elif act == "unpack":
        with open(argv[3]) as f:
            manifest = json.load(f)
        print(json.dumps(unpack(argv[1], argv[2], manifest, argv[4])))
    else:
        raise SystemExit("unknown action " + act)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
