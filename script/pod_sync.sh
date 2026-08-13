#!/usr/bin/env bash
# Pull the pod's out/ tree into the local repo. Runs LOCALLY.
#
#   script/pod_sync.sh [ssh-host]     # default host alias: runpod
#
# Uses tar over ssh rather than rsync (rsync is not installed on the local
# machine). Extracting over existing files is safe: run directories are
# append-only JSONL written by resumable stages, and the pod's copy is the
# superset during a generation phase.
set -euo pipefail

HOST="${1:-runpod}"
REMOTE_DIR="/workspace/idt-organism"

cd "$(dirname "$0")/.."

echo "==> remote runs under $HOST:$REMOTE_DIR/out"
ssh "$HOST" "cd $REMOTE_DIR && du -sh out/* 2>/dev/null" || {
    echo "FAIL: cannot list $HOST:$REMOTE_DIR/out (pod down? ssh config stale?)" >&2
    exit 1
}

echo "==> syncing"
ssh "$HOST" "cd $REMOTE_DIR && tar czf - out" | tar xzf -

echo "==> local out/ now holds"
du -sh out/* 2>/dev/null
echo "DONE"
