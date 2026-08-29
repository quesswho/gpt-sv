#!/usr/bin/env bash
# Full offline end-to-end check: data -> model -> optimizer -> ckpt -> eval.
# No downloads, no GPU. Override the interpreter with PYTHON=... if needed.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-$([ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)}

"$PYTHON" scripts/make_debug_data.py
"$PYTHON" -m gptsv.train --config configs/debug.toml

echo
echo "smoke test passed"
