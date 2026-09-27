#!/usr/bin/env bash
# Upload the sv64k-v2 shards to a private Hugging Face dataset repo, so training
# machines can download them with scripts/fetch_shards_hf.sh.
#
# Re-running is safe: files already on the Hub are skipped.
#
#   hf auth login                     # once, needs a write token
#   HF_DATASET=<user>/gptsv-sv64k-v2 scripts/upload_shards_hf.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."
HF=${HF:-$([ -x .venv/bin/hf ] && echo .venv/bin/hf || echo hf)}
SHARDS=${SHARDS:-data/shards/sv64k-v2}

if [ -z "${HF_DATASET:-}" ]; then
    echo "set HF_DATASET=<hf-user>/<repo-name>" >&2
    exit 1
fi

"$HF" auth whoami >/dev/null 2>&1 || { echo "run: $HF auth login" >&2; exit 1; }

# --private only applies when the repo is created, so create it first.
"$HF" repos create "$HF_DATASET" --type dataset --private 2>/dev/null || true

for split in train val; do
    "$HF" upload "$HF_DATASET" "$SHARDS/$split" "$split" \
        --type dataset --private \
        --commit-message "sv64k-v2 $split shards"
done

echo "uploaded to https://huggingface.co/datasets/$HF_DATASET"
