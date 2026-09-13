#!/usr/bin/env bash
# Push the sv64k-v2 shards to a private HF dataset repo. Run once, from the
# workstation.
#
# Training hosts pull the shards from the Hub at datacenter speed instead of
# through this machine's uplink, which matters because phase 1 is benchmarked on
# several hosts before the real run - the 21GB would otherwise be re-uploaded
# from home every time.
#
# Re-running is safe and resumes: files already on the Hub are skipped.
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

# --private only applies when the repo is created, so create it explicitly
# first: a public repo here would publish 11B tokens of scraped text.
"$HF" repos create "$HF_DATASET" --type dataset --private 2>/dev/null || true

# train/ is 111 x 200MB, val/ is one 36MB file; meta.json rides along with each
# and is what gptsv.data.loader reads to size the memmaps.
for split in train val; do
    "$HF" upload "$HF_DATASET" "$SHARDS/$split" "$split" \
        --type dataset --private \
        --commit-message "sv64k-v2 $split shards"
done

echo "uploaded to https://huggingface.co/datasets/$HF_DATASET"
