#!/usr/bin/env bash
# FineWeb-2 Swedish, sv64k tokenizer, uint16 token shards.
#
# configs/phase0_130m.toml consumes ~2.6B tokens (20k steps x 131k). Two
# FineWeb-2 swe_Latn files hold ~4.5B, so train is capped at 3B. FineWeb-2's
# own test split is the val set, so val is never drawn from training text.
#
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-$([ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)}

RAW=data/raw/fineweb2-swe
BASE=https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/resolve/main/data/swe_Latn
TOK=tokenizers/sv64k
SHARDS=data/shards/sv

fetch() {
    local f="$RAW/$1"
    [ -f "$f" ] && return
    mkdir -p "$(dirname "$f")"
    curl -sSL --fail --retry 5 -C - -o "$f.part" "$BASE/$1"
    mv "$f.part" "$f"
}

for f in train/000_00000.parquet train/000_00001.parquet test/000_00000.parquet; do
    fetch "$f"
done

[ -f "$TOK/tokenizer.json" ] || "$PYTHON" -m gptsv.tokenizer.train \
    --files "$RAW/train/000_00000.parquet" --vocab-size 65536 --n-docs 1000000 --out "$TOK"

[ -f "$SHARDS/train/meta.json" ] || "$PYTHON" -m gptsv.data.prepare tokenize \
    --files "$RAW"/train/*.parquet --tokenizer "$TOK" --out "$SHARDS/train" --max-tokens 3e9

[ -f "$SHARDS/val/meta.json" ] || "$PYTHON" -m gptsv.data.prepare tokenize \
    --files "$RAW"/test/*.parquet --tokenizer "$TOK" --out "$SHARDS/val"
