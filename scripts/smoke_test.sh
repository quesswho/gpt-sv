#!/usr/bin/env bash
# Offline end-to-end check: data -> model -> optimizer -> checkpoint -> eval.
# No downloads, no GPU. Override the interpreter with PYTHON=... if needed.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-$([ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)}

rm -rf out/debug
"$PYTHON" scripts/make_debug_data.py
"$PYTHON" -m gptsv.train --config configs/debug.toml
# Resume from the step 60 checkpoint to exercise optimizer state loading.
"$PYTHON" -m gptsv.train --config configs/debug.toml --resume --set train.max_steps=70

echo
echo "smoke test passed"
