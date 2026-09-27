#!/usr/bin/env bash
# Build the training data: FineWeb-2 Swedish, sv64k-v2 tokenizer, uint16 shards.
#
# sv64k is a 65536-token byte-level BPE trained on 1M documents. sv64k-v2 swaps
# its 59 rarest merges for chat and reserved special tokens
# (gptsv.tokenizer.reserve); every other token keeps its ID.
#
# configs/430m.toml consumes ~10.5B tokens (40k steps x 262k). The six train
# files hold ~13B, so train is capped at 11B and no block repeats within the
# run. FineWeb-2's own test split is the val set.
#
# Each step is skipped if its output already exists.
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

[ -f "$BASE_TOK/tokenizer.json" ] || "$PYTHON" -m gptsv.tokenizer.train \
    --files "$RAW/train/000_00000.parquet" --vocab-size 65536 --n-docs 1000000 --out "$BASE_TOK"

[ -f "$TOK/tokenizer.json" ] || "$PYTHON" -m gptsv.tokenizer.reserve \
    --base "$BASE_TOK" --out "$TOK" --total-specials 64

[ -f "$SHARDS/train/meta.json" ] || "$PYTHON" -m gptsv.data.prepare tokenize \
    --files "${TRAIN[@]/#/$RAW/}" --tokenizer "$TOK" --out "$SHARDS/train" --max-tokens 11e9

[ -f "$SHARDS/val/meta.json" ] || "$PYTHON" -m gptsv.data.prepare tokenize \
    --files "$RAW"/test/*.parquet --tokenizer "$TOK" --out "$SHARDS/val"
