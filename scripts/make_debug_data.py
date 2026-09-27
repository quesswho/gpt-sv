#!/usr/bin/env python3
"""Generate random token shards so the pipeline can be smoke-tested offline.

The data is a repeating motif with noise, enough to check that the loader,
model, optimizer, checkpointing and eval all work without any download.

    python scripts/make_debug_data.py
    gptsv-train --config configs/debug.toml

Loss should fall well below ln(vocab_size) within a few dozen steps. If it
does not, something in the model/optimizer/loader path is broken.
"""

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/debug")
    ap.add_argument("--vocab-size", type=int, default=512)
    ap.add_argument("--train-tokens", type=int, default=2_000_000)
    ap.add_argument("--val-tokens", type=int, default=200_000)
    ap.add_argument("--shards", type=int, default=3)
    ap.add_argument("--period", type=int, default=64, help="motif length; must be < seq_len")
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    # Train and val share the motif, so val loss falls too.
    motif = rng.integers(0, args.vocab_size, size=args.period, dtype=np.uint16)

    for split, total in (("train", args.train_tokens), ("val", args.val_tokens)):
        d = Path(args.out) / split
        d.mkdir(parents=True, exist_ok=True)
        per = total // args.shards
        for i in range(args.shards):
            # The period is shorter than seq_len, so the model can learn it.
            arr = np.tile(motif, per // args.period + 1)[:per].astype(np.uint16)
            noise = rng.random(arr.shape) < args.noise
            arr[noise] = rng.integers(0, args.vocab_size, size=int(noise.sum()), dtype=np.uint16)
            arr.tofile(d / f"shard_{i:05d}.bin")
        print(f"{split}: {args.shards} shards, {per * args.shards:,} tokens -> {d}")


if __name__ == "__main__":
    main()
