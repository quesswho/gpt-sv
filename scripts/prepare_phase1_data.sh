#!/usr/bin/env bash
# FineWeb-2 Swedish, sv64k-v2 tokenizer, uint16 token shards.
#
# sv64k-v2 is sv64k with its 59 rarest merges swapped for chat and reserved
# special tokens (gptsv.tokenizer.reserve). Every other token keeps its sv64k
# ID, so phase 0 results carry over.
#
# configs/phase1_430m.toml consumes ~10.5B tokens (40k steps x 262k). Phase 0's
# two files held ~4.5B, so six should hold ~13B; train is capped at 11B so no
# block repeats within the run. FineWeb-2's own test split is the val set.
#
# Shards get their own directory: phase 0 reads data/shards/sv, and its exact
# resume depends on that directory never changing.
#
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-$([ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)}

RAW=data/raw/fineweb2-swe
BASE=https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/resolve/main/data/swe_Latn
BASE_TOK=tokenizers/sv64k
TOK=tokenizers/sv64k-v2
SHARDS=data/shards/sv64k-v2
TRAIN=(
    train/000_00000.parquet train/000_00001.parquet train/000_00002.parquet
    train/000_00003.parquet train/001_00000.parquet train/001_00001.parquet
)

fetch() {
    local f="$RAW/$1"
    [ -f "$f" ] && return
    mkdir -p "$(dirname "$f")"
    curl -sSL --fail --retry 5 -C - -o "$f.part" "$BASE/$1"
    mv "$f.part" "$f"
}

for f in "${TRAIN[@]}" test/000_00000.parquet; do
    fetch "$f"
done

if [ ! -f "$BASE_TOK/tokenizer.json" ]; then
    echo "$BASE_TOK is missing; run scripts/prepare_phase0_data.sh first" >&2
    exit 1
fi

[ -f "$TOK/tokenizer.json" ] || "$PYTHON" -m gptsv.tokenizer.reserve \
    --base "$BASE_TOK" --out "$TOK" --total-specials 64

[ -f "$SHARDS/train/meta.json" ] || "$PYTHON" -m gptsv.data.prepare tokenize \
    --files "${TRAIN[@]/#/$RAW/}" --tokenizer "$TOK" --out "$SHARDS/train" --max-tokens 11e9

[ -f "$SHARDS/val/meta.json" ] || "$PYTHON" -m gptsv.data.prepare tokenize \
    --files "$RAW"/test/*.parquet --tokenizer "$TOK" --out "$SHARDS/val"
