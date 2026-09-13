#!/usr/bin/env bash
# Copy checkpoints off a training host, on a loop. Run in tmux on the workstation.
#
#   scripts/pull_checkpoints.sh root@1.2.3.4:22
#   RESOLVE='provider-cli ssh-url 12345' scripts/pull_checkpoints.sh
#
# Rented and preemptible hosts take their disk with them when they go. Phase 1 is
# ~10.5B tokens; losing it on day six because the checkpoints only ever existed
# on someone else's machine would be the single most expensive mistake available
# here.
#
# Keeps whatever is already local, so an interrupted copy resumes rather than
# restarting.
#
# The address is re-resolved on every pass when RESOLVE is set, never once up
# front: providers reassign host and port when an instance stops and starts, and
# a repair that moves the box would otherwise leave this pulling from a dead
# address for the rest of the run. RESOLVE prints `[user@]host[:port]` and exits
# non-zero (or prints nothing) once the host is gone for good, which ends the
# loop. Without it the address given on the command line is used unchanged.
#
set -uo pipefail
cd "$(dirname "$0")/.."
REMOTE=${REMOTE:-/workspace/gpt-sv}
LOCAL=${LOCAL:-out/phase1_430m}
INTERVAL=${INTERVAL:-1800}   # 30 min; ckpt_interval=200 writes more often than this

if [ -z "${RESOLVE:-}" ] && [ $# -eq 0 ]; then
    echo "usage: $0 [user@]host[:port]   (or set RESOLVE to a command printing one)" >&2
    exit 1
fi

mkdir -p "$LOCAL"
while true; do
    if [ -n "${RESOLVE:-}" ]; then
        TARGET=$(eval "$RESOLVE" 2>/dev/null || true)
        if [ -z "$TARGET" ]; then
            echo "[$(date -Is)] host is gone; stopping"
            exit 1
        fi
    else
        TARGET=$1
    fi
    HOST=${TARGET%:*}
    PORT=${TARGET##*:}
    [ "$PORT" = "$TARGET" ] && PORT=22
    [[ "$HOST" == *@* ]] || HOST="root@$HOST"

    # --partial-dir, never bare --partial: an interrupted transfer must not leave
    # a truncated file at the real filename. Paired with --ignore-existing that
    # silently poisons the backup - 16 of 64 checkpoints were unusable before
    # this, and torch.load only finds out at restore time.
    # No --ignore-existing either: rsync's size/mtime check then re-fetches
    # anything short. The remote keeps 3 files, so this compares 3 each pass.
    if rsync -az --partial-dir=.rsync-partial \
        -e "ssh -p $PORT -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15" \
        "$HOST:$REMOTE/$LOCAL/" "$LOCAL/" 2>/dev/null; then
        echo "[$(date -Is)] pulled: $(ls -1 "$LOCAL"/step_*.pt 2>/dev/null | wc -l) checkpoints local, latest $(ls -1t "$LOCAL"/step_*.pt 2>/dev/null | head -1 | xargs -r basename)"
    else
        echo "[$(date -Is)] pull failed (host busy or down); retrying in ${INTERVAL}s"
    fi
    sleep "$INTERVAL"
done
