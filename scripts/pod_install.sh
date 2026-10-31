#!/usr/bin/env bash
# Install kdev on the GPU pod from this repo (the repo is the source of truth: pod storage is wiped).
#
#   scripts/pod_install.sh [HOST] [PORT]            # run locally (Git Bash / Linux / macOS)
#
# What it does (idempotent; re-run after every pod restart or repo change):
#   1. tars the repo's tracked + untracked-not-ignored files, copies it to the pod, md5-verifies the copy;
#   2. extracts to $DEST (/root/kdev, replaced), `uv pip install --no-deps -e $DEST` into $PY's env
#      (/root/vsgl: the judge-like env); no existing package may change (pip freeze before/after diff);
#   3. optional MCP SDK (WITH_MCP=1, default): installs "mcp>=1.2,<2" only if the dry run adds packages and
#      changes none (else skipped with a warning; `kdev mcp` then reports capability_unavailable);
#   4. writes $KHOME/kdev.toml (lock, okc_root, python, toolchain fallback) and $KHOME/toolchain.lock
#      (judge toolchain CUDA 13.0 at $CUDA) unless they exist (FORCE_CONFIG=1 overwrites);
#   5. links /usr/local/bin/kdev and prints `kdev version`.
# Never touches /workspace, never runs GPU work, never calls okc.
#
# env: KEY (~/.ssh/runpod_ed25519), PY (/root/vsgl/bin/python), DEST (/root/kdev), KHOME (/root/.kdev),
#      CUDA (/root/cuda130w), OKC (/root/okc), LOCK (/root/gpu.lock), WITH_MCP (1), FORCE_CONFIG (0)
set -euo pipefail

HOST=${1:-103.196.86.128}
PORT=${2:-20860}
KEY=${KEY:-$HOME/.ssh/runpod_ed25519}
PY=${PY:-/root/vsgl/bin/python}
DEST=${DEST:-/root/kdev}
KHOME=${KHOME:-/root/.kdev}
CUDA=${CUDA:-/root/cuda130w}
OKC=${OKC:-/root/okc}
LOCK=${LOCK:-/root/gpu.lock}
WITH_MCP=${WITH_MCP:-1}
FORCE_CONFIG=${FORCE_CONFIG:-0}

case "$DEST$KHOME" in */workspace*) echo "refusing a /workspace path (network volume)" >&2; exit 2 ;; esac

REPO=$(cd "$(dirname "$0")/.." && pwd)
SSH=(ssh -i "$KEY" -p "$PORT" -o ConnectTimeout=20 "root@$HOST")
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
TGZ="$TMP/kdev_src.tgz"

echo "== pack $REPO"
(cd "$REPO" && git ls-files -co --exclude-standard -z | grep -zv -E '(__pycache__|\.pyc$|\.egg-info)' \
  | tar --null -T - -czf "$TGZ")
LOCAL_MD5=$(md5sum "$TGZ" | cut -d' ' -f1)

echo "== copy -> $HOST:/tmp/kdev_src.tgz"
scp -q -i "$KEY" -P "$PORT" "$TGZ" "root@$HOST:/tmp/kdev_src.tgz"
REMOTE_MD5=$("${SSH[@]}" md5sum /tmp/kdev_src.tgz | cut -d' ' -f1)
[ "$LOCAL_MD5" = "$REMOTE_MD5" ] || { echo "md5 mismatch $LOCAL_MD5 != $REMOTE_MD5" >&2; exit 1; }

echo "== install on the pod"
"${SSH[@]}" PY="$PY" DEST="$DEST" KHOME="$KHOME" CUDA="$CUDA" OKC="$OKC" LOCK="$LOCK" WITH_MCP="$WITH_MCP" \
  FORCE_CONFIG="$FORCE_CONFIG" bash -s <<'REMOTE'
set -euo pipefail
command -v uv >/dev/null || { echo "uv not found on the pod (the env has no pip)" >&2; exit 1; }
[ -x "$PY" ] || { echo "python $PY not found" >&2; exit 1; }
freeze() { uv pip freeze --python "$PY" 2>/dev/null | grep -v -i -E '^(-e |kdev[ =@])' | sort; }
freeze > /tmp/kdev_freeze_before.txt
TORCH_BEFORE=$("$PY" -c 'import torch; print(torch.__version__)' 2>/dev/null || echo none)

rm -rf "$DEST" && mkdir -p "$DEST"
tar --no-same-owner -xzf /tmp/kdev_src.tgz -C "$DEST"
uv pip install -q --python "$PY" --no-deps -e "$DEST"

if [ "$WITH_MCP" = 1 ]; then
  if "$PY" -c 'import mcp' 2>/dev/null; then
    echo "mcp: already present"
  else
    DRY=$(uv pip install --python "$PY" --dry-run "mcp>=1.2,<2" 2>&1 || true)
    if echo "$DRY" | grep -q -E '^ - '; then
      echo "WARNING mcp: install would change existing packages; skipped:"; echo "$DRY" | grep -E '^ [-+] '
    else
      uv pip install -q --python "$PY" "mcp>=1.2,<2" && echo "mcp: installed (only new packages)"
    fi
  fi
fi

freeze > /tmp/kdev_freeze_after.txt
CHANGED=$(comm -23 /tmp/kdev_freeze_before.txt /tmp/kdev_freeze_after.txt || true)
if [ -n "$CHANGED" ]; then
  echo "ERROR existing packages changed or removed:" >&2; echo "$CHANGED" >&2; exit 3
fi
ADDED=$(comm -13 /tmp/kdev_freeze_before.txt /tmp/kdev_freeze_after.txt | tr '\n' ' ')
TORCH_AFTER=$("$PY" -c 'import torch; print(torch.__version__)' 2>/dev/null || echo none)
[ "$TORCH_BEFORE" = "$TORCH_AFTER" ] || { echo "ERROR torch changed $TORCH_BEFORE -> $TORCH_AFTER" >&2; exit 3; }
echo "packages unchanged; added: ${ADDED:-none}; torch $TORCH_AFTER"

mkdir -p "$KHOME"
NVCC_V=$("$CUDA/bin/nvcc" --version | sed -n 's/.*, V\([0-9.]*\).*/\1/p' | tail -1)
[ -n "$NVCC_V" ] || { echo "cannot read nvcc version under $CUDA" >&2; exit 1; }
if [ "$FORCE_CONFIG" = 1 ] || [ ! -f "$KHOME/kdev.toml" ]; then
  cat > "$KHOME/kdev.toml" <<EOF
# written by kernel-development/scripts/pod_install.sh
home = "$KHOME"                     # never on /workspace
lock_path = "$LOCK"                 # shared with /root/gpu_run.sh
offender_log = "/root/gpu_offenders.log"
okc_root = "$OKC"
problems_root = "$OKC"
python = "$PY"                      # worker interpreter (judge-like env)
gpu = "h200"

[toolchain]
nvcc = "$NVCC_V"
cuda_home = "$CUDA"
lock = "$KHOME/toolchain.lock"
EOF
  echo "wrote $KHOME/kdev.toml"
fi
if [ "$FORCE_CONFIG" = 1 ] || [ ! -f "$KHOME/toolchain.lock" ]; then
  TRITON=$("$PY" -c 'import triton; print(triton.__version__)' 2>/dev/null || true)
  cat > "$KHOME/toolchain.lock" <<EOF
schema = "kdev.toolchain/1"
nvcc_version = "$NVCC_V"
cuda_home = "$CUDA"
nvcc = "$CUDA/bin/nvcc"
ptxas = "$CUDA/bin/ptxas"
cuobjdump = "$CUDA/bin/cuobjdump"
nvdisasm = "$CUDA/bin/nvdisasm"
torch = "$TORCH_AFTER"
triton = "$TRITON"
python = "$PY"
source = "pod_install.sh (judge toolchain CUDA 13.0)"

[env]
NVCC_APPEND_FLAGS = "-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
EOF
  echo "wrote $KHOME/toolchain.lock (V$NVCC_V)"
fi
ln -sf "$(dirname "$PY")/kdev" /usr/local/bin/kdev
kdev version
REMOTE
echo "== done: ssh -p $PORT root@$HOST kdev doctor"
