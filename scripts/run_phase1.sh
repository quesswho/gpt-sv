#!/usr/bin/env bash
# Phase 1: 4x RTX 3060, DDP over PCIe.
set -euo pipefail
cd "$(dirname "$0")/.."
TORCHRUN=${TORCHRUN:-$([ -x .venv/bin/torchrun ] && echo .venv/bin/torchrun || echo torchrun)}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# GeForce cards have no working P2P; forcing it off avoids NCCL spending a long
# time probing for a path that does not exist.
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=WARN
exec "$TORCHRUN" --standalone --nproc_per_node=4 -m gptsv.train \
    --config configs/phase1_430m.toml "$@"
