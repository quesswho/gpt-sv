#!/usr/bin/env bash
# Run scripts/train.sh and restart it from the latest checkpoint whenever it
# exits with an error. Meant to run inside tmux on the training machine.
#
set -uo pipefail
cd "$(dirname "$0")/.."

OUT=${OUT:-out/430m}
DELAY=${DELAY:-60}
attempt=0

while true; do
    attempt=$((attempt + 1))
    # Plain --resume picks the latest checkpoint in out_dir, so only pass it
    # once one exists.
    if compgen -G "$OUT/step_*.pt" > /dev/null; then
        echo "=== attempt $attempt: resuming from latest checkpoint ($(date -Is)) ==="
        scripts/train.sh --resume
    else
        echo "=== attempt $attempt: fresh start ($(date -Is)) ==="
        scripts/train.sh
    fi
    status=$?

    if [ $status -eq 0 ]; then
        echo "=== training finished cleanly ($(date -Is)) ==="
        break
    fi
    echo "=== exited with status $status, retrying in ${DELAY}s ($(date -Is)) ==="
    sleep "$DELAY"
done
