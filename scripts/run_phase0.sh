#!/usr/bin/env bash
# Phase 0: single RTX 3060.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-$([ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec "$PYTHON" -m gptsv.train --config configs/phase0_130m.toml "$@"
