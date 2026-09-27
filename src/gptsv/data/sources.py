"""Where raw documents come from: local files or a HuggingFace dataset.

Shared by `gptsv-tokenizer` and `gptsv-data`, so the tokenizer is trained on the
same text stream the shards are built from.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path


def add_source_args(p: argparse.ArgumentParser) -> None:
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", help="HuggingFace dataset id")
    src.add_argument("--files", nargs="+", help="local jsonl/parquet/txt files")
    p.add_argument("--config", default=None)
    p.add_argument("--split", default="train")
    p.add_argument("--text-field", default="text")
    p.add_argument("--lang-field", default=None)
    p.add_argument("--langs", nargs="*", default=None)


def open_stream(args):
    """A streaming `datasets` iterable over raw rows."""
    from datasets import load_dataset

    if args.files:
        ext = Path(args.files[0]).suffix.lstrip(".")
        fmt = {"jsonl": "json", "json": "json", "parquet": "parquet", "txt": "text"}.get(ext, ext)
        return load_dataset(fmt, data_files=args.files, split="train", streaming=True)
    return load_dataset(args.dataset, args.config, split=args.split, streaming=True)


def iter_texts(args, limit: int | None = None) -> Iterator[str]:
    """Non-empty document texts, language-filtered if --langs is given."""
    langs = set(args.langs) if args.langs else None
    if langs is not None and args.lang_field is None:
        raise SystemExit("--langs requires --lang-field (run `gptsv-data probe`)")

    n = 0
    for row in open_stream(args):
        if langs is not None and row.get(args.lang_field) not in langs:
            continue
        text = row.get(args.text_field)
        if not text:
            continue
        yield text
        n += 1
        if limit is not None and n >= limit:
            return
