"""Train a byte-level BPE tokenizer on Swedish text.

Multilingual tokenizers spend most of their vocabulary on other languages and
split Swedish compounds such as `arbetsmarknadsutbildning` into many pieces. A
Swedish-only vocabulary needs fewer tokens for the same text.

    gptsv-tokenizer --files data/raw/fineweb2-swe/train/000_00000.parquet \
        --vocab-size 65536 --n-docs 1000000 --out tokenizers/sv64k

Then measure it: `gptsv-fertility --tokenizers tokenizers/sv64k ...`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..data.sources import add_source_args, iter_texts
from . import SPECIAL_TOKENS


def main(argv: list[str] | None = None) -> int:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    p = argparse.ArgumentParser(prog="gptsv-tokenizer", description=__doc__)
    add_source_args(p)
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

    # Byte level, so any Unicode input is covered without an unknown token.
    # Digits are split one by one.
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
    tok.train_from_iterator(iter_texts(args, args.n_docs), trainer=trainer)

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
