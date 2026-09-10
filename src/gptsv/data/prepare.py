"""Turn raw text into the token shards `gptsv-train` reads.

    # look at a source's fields before committing to it
    gptsv-data probe --files data/raw/fineweb2-swe/train/000_00000.parquet

    # tokenize into flat uint16 shards, <|endoftext|> after every document
    gptsv-data tokenize --files data/raw/fineweb2-swe/train/*.parquet \\
        --tokenizer tokenizers/sv64k --out data/shards/sv/train --max-tokens 3e9
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable, Iterator
from itertools import islice
from pathlib import Path

import numpy as np

from ..tokenizer import SPECIAL_TOKENS
from .sources import add_source_args, iter_texts, open_stream

EOT = SPECIAL_TOKENS[0]


class ShardWriter:
    """Streams token ids into fixed-size shard files.

    Each shard is written to a temp name and renamed once complete, so an
    interrupted run never leaves a truncated shard that looks valid.
    """

    def __init__(self, out_dir: Path, shard_tokens: int, dtype: np.dtype):
        self.out_dir = out_dir
        self.buf = np.empty(shard_tokens, dtype=dtype)
        self.fill = 0
        self.n_shards = 0
        self.n_tokens = 0

    def write(self, ids: np.ndarray) -> None:
        while len(ids):
            take = min(len(ids), len(self.buf) - self.fill)
            self.buf[self.fill : self.fill + take] = ids[:take]
            self.fill += take
            self.n_tokens += take
            ids = ids[take:]
            if self.fill == len(self.buf):
                self._flush()

    def close(self) -> None:
        self._flush()

    def _flush(self) -> None:
        if self.fill == 0:
            return
        path = self.out_dir / f"shard_{self.n_shards:05d}.bin"
        tmp = path.with_suffix(".bin.tmp")
        self.buf[: self.fill].tofile(tmp)
        tmp.rename(path)
        self.n_shards += 1
        self.fill = 0


def _batched(it: Iterable[str], n: int) -> Iterator[list[str]]:
    it = iter(it)
    while batch := list(islice(it, n)):
        yield batch


def cmd_probe(args) -> int:
    for i, row in enumerate(islice(open_stream(args), args.n)):
        print(f"--- row {i}")
        for key, value in row.items():
            s = repr(value)
            print(f"  {key}: {s[: args.width]}{'...' if len(s) > args.width else ''}")
    return 0


def cmd_tokenize(args) -> int:
    from tokenizers import Tokenizer
    from tqdm import tqdm

    tok = Tokenizer.from_file(str(Path(args.tokenizer) / "tokenizer.json"))
    dtype = np.dtype(args.dtype)
    if tok.get_vocab_size() - 1 > np.iinfo(dtype).max:
        raise SystemExit(f"vocab {tok.get_vocab_size()} does not fit in {dtype}")
    eot = tok.token_to_id(EOT)
    if eot is None:
        raise SystemExit(f"tokenizer has no {EOT} token")

    out = Path(args.out)
    if any(out.glob("*.bin")):
        raise SystemExit(f"{out} already contains shards; refusing to mix two runs")
    out.mkdir(parents=True, exist_ok=True)

    max_tokens = int(args.max_tokens) if args.max_tokens else None
    writer = ShardWriter(out, int(args.shard_tokens), dtype)
    n_docs = 0
    t0 = time.perf_counter()

    with tqdm(total=max_tokens, unit="tok", unit_scale=True, smoothing=0.05) as bar:
        for batch in _batched(iter_texts(args), args.batch_size):
            encs = tok.encode_batch(batch, add_special_tokens=False)
            lengths = np.fromiter((len(e.ids) + 1 for e in encs), dtype=np.int64, count=len(encs))
            ids = np.empty(int(lengths.sum()), dtype=dtype)
            ends = np.cumsum(lengths)
            for e, end, n in zip(encs, ends, lengths, strict=True):
                ids[end - n : end - 1] = e.ids
            ids[ends - 1] = eot

            writer.write(ids)
            n_docs += len(batch)
            bar.update(len(ids))
            if max_tokens is not None and writer.n_tokens >= max_tokens:
                break
    writer.close()

    meta = {
        "tokenizer": str(Path(args.tokenizer).resolve()),
        "vocab_size": tok.get_vocab_size(),
        "eot_id": eot,
        "dtype": str(dtype),
        "n_docs": n_docs,
        "n_tokens": writer.n_tokens,
        "n_shards": writer.n_shards,
        "source": args.files or args.dataset,
        "split": None if args.files else args.split,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    dt = time.perf_counter() - t0
    print(
        f"{n_docs:,} docs -> {writer.n_tokens:,} tokens in {writer.n_shards} shards "
        f"({writer.n_tokens / max(1, n_docs):.0f} tok/doc, {writer.n_tokens / dt / 1e6:.1f}M tok/s) -> {out}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="gptsv-data", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    probe = sub.add_parser("probe", help="print the first rows of a source")
    add_source_args(probe)
    probe.add_argument("--n", type=int, default=2, help="rows to print")
    probe.add_argument("--width", type=int, default=200, help="truncate each field to this many chars")

    tok = sub.add_parser("tokenize", help="tokenize a source into .bin shards")
    add_source_args(tok)
    tok.add_argument("--tokenizer", required=True, help="dir containing tokenizer.json")
    tok.add_argument("--out", required=True)
    tok.add_argument("--max-tokens", type=float, default=None, help="stop after this many tokens")
    tok.add_argument("--shard-tokens", type=float, default=1e8, help="tokens per shard file")
    tok.add_argument("--dtype", default="uint16")
    tok.add_argument("--batch-size", type=int, default=1024, help="documents per encode_batch call")

    args = p.parse_args(argv)
    return {"probe": cmd_probe, "tokenize": cmd_tokenize}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
