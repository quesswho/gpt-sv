#!/usr/bin/env bash
# Periodically copy checkpoints from a remote training machine. Run in tmux on
# the local machine.
#
#   scripts/pull_checkpoints.sh root@1.2.3.4:22
#   RESOLVE='provider-cli ssh-url 12345' scripts/pull_checkpoints.sh
#
# Files already copied are kept, so an interrupted copy picks up where it left
# off.
#
# If RESOLVE is set, it is run before every pass to get the current address,
# since rented machines can change host and port after a restart. It should
# print `[user@]host[:port]`, and print nothing or exit non-zero once the
# machine is gone, which stops the loop. Without RESOLVE the address given on
# the command line is used.
#
set -uo pipefail
cd "$(dirname "$0")/.."
REMOTE=${REMOTE:-/workspace/gpt-sv}
LOCAL=${LOCAL:-out/430m}
INTERVAL=${INTERVAL:-1800}   # 30 min

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

    # --partial-dir keeps interrupted transfers out of the real filename, and
    # without --ignore-existing, rsync re-fetches any file whose size or mtime
    # differs.
    if rsync -az --partial-dir=.rsync-partial \
        -e "ssh -p $PORT -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15" \
        "$HOST:$REMOTE/$LOCAL/" "$LOCAL/" 2>/dev/null; then
        echo "[$(date -Is)] pulled: $(ls -1 "$LOCAL"/step_*.pt 2>/dev/null | wc -l) checkpoints local, latest $(ls -1t "$LOCAL"/step_*.pt 2>/dev/null | head -1 | xargs -r basename)"
    else
        echo "[$(date -Is)] pull failed (host busy or down); retrying in ${INTERVAL}s"
    fi
    sleep "$INTERVAL"
done
