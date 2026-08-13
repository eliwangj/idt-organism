#!/usr/bin/env bash
# Idempotent RunPod bootstrap for idt-organism generation. Runs ON the pod.
#
#   ssh runpod 'bash -s' < script/pod_setup.sh      # from the local repo, or
#   bash script/pod_setup.sh                        # on the pod after a manual clone
#
# Safe to re-run: every step is skipped when already satisfied. Ends with a
# hard CUDA sanity gate — a pod whose preinstalled torch cannot drive the GPU
# (e.g. Blackwell sm_120 needs the cu128 wheels) fails here, loudly, before
# any generation time is wasted.
set -euo pipefail

REPO_URL="https://github.com/eliwangj/idt-organism.git"
REPO_DIR="/workspace/idt-organism"
UV="$HOME/.local/bin/uv"

step() { printf '\n==> %s\n' "$*"; }

step "uv"
if [ -x "$UV" ]; then
    echo "already installed: $("$UV" --version)"
else
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

step "repo"
if [ -d "$REPO_DIR/.git" ]; then
    git -C "$REPO_DIR" pull --ff-only
else
    git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"

step "venv (python 3.12)"
if [ -x .venv/bin/python ] && .venv/bin/python -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12))'; then
    echo "already present: $(.venv/bin/python --version)"
else
    "$UV" venv --python 3.12
fi

step "dependencies (torch cu128)"
# The venv copy can take ~7 minutes on the pod filesystem; uv's
# "Failed to hardlink ... falling back to full copy" warning is normal there.
if ! "$UV" pip install -e ".[local]" --torch-backend=cu128; then
    echo "--torch-backend unsupported by this uv; using explicit cu128 index"
    "$UV" pip install -e ".[local]" \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        --index-strategy unsafe-best-match
fi

step "CUDA sanity gate"
.venv/bin/python - <<'PY'
import sys
import torch

print(f"torch {torch.__version__}")
if not torch.cuda.is_available():
    sys.exit("FAIL: torch.cuda.is_available() is False")
print(f"gpu   {torch.cuda.get_device_name(0)}")
try:
    value = torch.zeros(2).cuda().sum().item()
except Exception as exc:  # noqa: BLE001 - any CUDA failure fails the gate
    sys.exit(f"FAIL: CUDA tensor op raised: {exc} "
             "(wrong wheel for this GPU arch? Blackwell sm_120 needs cu128)")
print(f"cuda tensor op ok (sum={value})")
PY

step "READY"
echo "next: uv run python script/generate_responses.py --smoke --run-name <phase>-smoke ..."
