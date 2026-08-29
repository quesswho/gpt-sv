"""Train a byte-level BPE tokenizer tuned for Swedish.

This is the highest-leverage Swedish-specific decision in the project. Existing
options are all compromises: GPT-SW3's tokenizer predates the current recipe,
and the big multilingual tokenizers (Llama, Qwen, Gemma) spend most of their
vocab elsewhere and fragment Swedish morphology badly. Swedish is heavily
compounding - `arbetsmarknadsutbildning` - so fertility gains here are large
and they convert directly into training compute saved for the life of the
project.

    gptsv-tokenizer --dataset AI-Sweden-Models/SWEb --text-field text \
        --langs sv --vocab-size 65536 --n-docs 2000000 --out tokenizers/sv64k

Then measure it: `gptsv-fertility --tokenizers tokenizers/sv64k ...`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import SPECIAL_TOKENS


def _doc_iterator(args, limit: int):
    from datasets import load_dataset

    if args.files:
        ext = Path(args.files[0]).suffix.lstrip(".")
        fmt = {"jsonl": "json", "json": "json", "parquet": "parquet", "txt": "text"}.get(ext, ext)
        ds = load_dataset(fmt, data_files=args.files, split="train", streaming=True)
    else:
        ds = load_dataset(args.dataset, args.config, split=args.split, streaming=True)

    langs = set(args.langs) if args.langs else None
    n = 0
    for row in ds:
        if langs is not None:
            if args.lang_field is None:
                raise SystemExit("--langs requires --lang-field (run `gptsv-data probe`)")
            if row.get(args.lang_field) not in langs:
                continue
        text = row.get(args.text_field)
        if not text:
            continue
        yield text
        n += 1
        if n >= limit:
            return


def main(argv: list[str] | None = None) -> int:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    p = argparse.ArgumentParser(prog="gptsv-tokenizer", description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", help="HuggingFace dataset id")
    src.add_argument("--files", nargs="+", help="local jsonl/parquet/txt files")
    p.add_argument("--config", default=None)
    p.add_argument("--split", default="train")
    p.add_argument("--text-field", default="text")
    p.add_argument("--lang-field", default=None)
    p.add_argument("--langs", nargs="*", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--vocab-size", type=int, default=65536)
    p.add_argument("--min-frequency", type=int, default=2)
    p.add_argument("--n-docs", type=int, default=2_000_000, help="training sample size")
    p.add_argument(
        "--no-split-digits",
        dest="split_digits",
        action="store_false",
        help="keep multi-digit tokens (not recommended - hurts arithmetic)",
    )
    p.set_defaults(split_digits=True)
    args = p.parse_args(argv)

    if args.vocab_size > 65536:
        print(
            f"warning: vocab {args.vocab_size} > 65536 means token shards need uint32, "
            "doubling data size on disk and in the page cache."
        )

    tok = Tokenizer(models.BPE(unk_token=None, byte_fallback=False))

    # Byte level means no UNK and no <unk> handling for aa/ae/oe or any other
    # Unicode; digits split individually so numbers stay compositional.
    stages = []
    if args.split_digits:
        stages.append(pre_tokenizers.Digits(individual_digits=True))
    stages.append(pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True))
    tok.pre_tokenizer = pre_tokenizers.Sequence(stages)
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    print(f"training BPE (vocab={args.vocab_size}) over up to {args.n_docs:,} documents...")
    tok.train_from_iterator(_doc_iterator(args, args.n_docs), trainer=trainer)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok.save(str(out / "tokenizer.json"))
    (out / "meta.json").write_text(
        json.dumps(
            {
                "vocab_size": tok.get_vocab_size(),
                "special_tokens": SPECIAL_TOKENS,
                "split_digits": args.split_digits,
                "source": args.files or args.dataset,
                "langs": args.langs,
                "n_docs": args.n_docs,
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    sample = "Arbetsmarknadsutbildningen på Södermalm kostade 1 250 kronor år 2024."
    enc = tok.encode(sample)
    print(f"\nsaved -> {out}  (vocab {tok.get_vocab_size()})")
    print(f"sample: {sample}")
    print(f"tokens ({len(enc.ids)}): {enc.tokens}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
