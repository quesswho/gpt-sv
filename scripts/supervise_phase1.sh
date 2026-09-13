#!/usr/bin/env bash
# Restart loop for phase 1. Runs ON the training host, inside tmux.
#
# A cheap host is not a reliable one: the container can be bounced, a GPU can
# fall off the bus, NCCL can wedge. Anything that is not a clean finish is worth
# retrying from the newest checkpoint, because configs/phase1_430m.toml writes
# one every 200 steps and the alternative is losing days of a week-long run.
#
# Nothing here depends on the workstation or on an open ssh session: tmux owns
# the process, and this loop owns the restarts.
#
set -uo pipefail
cd "$(dirname "$0")/.."

OUT=${OUT:-out/phase1_430m}
DELAY=${DELAY:-60}
attempt=0

while true; do
    attempt=$((attempt + 1))
    # --resume with no argument means "latest in out_dir"; only pass it once a
    # checkpoint exists, so the very first launch starts clean.
    if compgen -G "$OUT/step_*.pt" > /dev/null; then
        echo "=== attempt $attempt: resuming from latest checkpoint ($(date -Is)) ==="
        scripts/run_phase1.sh --resume
    else
        echo "=== attempt $attempt: fresh start ($(date -Is)) ==="
        scripts/run_phase1.sh
    fi
    status=$?

    if [ $status -eq 0 ]; then
        echo "=== training finished cleanly ($(date -Is)) ==="
        break
    fi
    echo "=== exited with status $status, retrying in ${DELAY}s ($(date -Is)) ==="
    sleep "$DELAY"
done
