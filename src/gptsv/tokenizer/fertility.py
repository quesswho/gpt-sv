"""Compare tokenizer fertility (tokens per word) on Swedish text.

A tokenizer with 25% lower fertility fits 25% more text into the same token
budget.

    gptsv-fertility --text-file data/eval/sv_sample.txt \
        --tokenizers tokenizers/sv64k-v2 \
        --hf-tokenizers AI-Sweden-Models/gpt-sw3-6.7b meta-llama/Llama-3.1-8B Qwen/Qwen3-8B

Compares against the first entry as baseline unless --baseline is given.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _load_texts(args) -> list[str]:
    if args.text_file:
        raw = Path(args.text_file).read_text(encoding="utf-8")
        docs = [d for d in raw.split("\n\n") if d.strip()]
        return docs[: args.n_docs]

    from datasets import load_dataset

    ds = load_dataset(args.dataset, args.config, split=args.split, streaming=True)
    langs = set(args.langs) if args.langs else None
    out = []
    for row in ds:
        if langs is not None and row.get(args.lang_field) not in langs:
            continue
        text = row.get(args.text_field)
        if text:
            out.append(text)
        if len(out) >= args.n_docs:
            break
    return out


def _encode_lengths(encode, texts: list[str]) -> int:
    return sum(len(encode(t)) for t in texts)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="gptsv-fertility", description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--text-file", help="UTF-8 text, documents separated by blank lines")
    src.add_argument("--dataset", help="HuggingFace dataset id")
    p.add_argument("--config", default=None)
    p.add_argument("--split", default="train")
    p.add_argument("--text-field", default="text")
    p.add_argument("--lang-field", default=None)
    p.add_argument("--langs", nargs="*", default=None)
    p.add_argument("--n-docs", type=int, default=2000)
    p.add_argument("--tokenizers", nargs="*", default=[], help="local dirs with tokenizer.json")
    p.add_argument("--hf-tokenizers", nargs="*", default=[], help="HuggingFace tokenizer ids")
    p.add_argument("--baseline", default=None, help="name to compare against (default: first)")
    args = p.parse_args(argv)

    texts = _load_texts(args)
    if not texts:
        print("no documents loaded", file=sys.stderr)
        return 1

    n_words = sum(len(t.split()) for t in texts)
    n_chars = sum(len(t) for t in texts)
    n_bytes = sum(len(t.encode("utf-8")) for t in texts)
    print(f"corpus: {len(texts):,} docs | {n_words:,} words | {n_bytes:,} bytes\n")

    encoders: list[tuple[str, callable, int]] = []

    for path in args.tokenizers:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(Path(path) / "tokenizer.json"))
        encoders.append(
            (Path(path).name, lambda t, _tok=tok: _tok.encode(t, add_special_tokens=False).ids, _tok_size(tok))
        )

    for name in args.hf_tokenizers:
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(name)
        except Exception as exc:  # gated repos, missing auth, network
            print(f"  skipping {name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        encoders.append(
            (name, lambda t, _tok=tok: _tok(t, add_special_tokens=False)["input_ids"], len(tok))
        )

    if not encoders:
        print("no tokenizers loaded", file=sys.stderr)
        return 1

    rows = []
    for name, encode, vocab in encoders:
        total = _encode_lengths(encode, texts)
        rows.append(
            {
                "name": name,
                "vocab": vocab,
                "tokens": total,
                "tokens_per_word": total / n_words,
                "bytes_per_token": n_bytes / total,
                "chars_per_token": n_chars / total,
            }
        )

    base = next((r for r in rows if r["name"] == args.baseline), rows[0])

    hdr = f"{'tokenizer':44s} {'vocab':>8s} {'tok/word':>9s} {'bytes/tok':>10s} {'vs base':>9s}"
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda r: r["tokens_per_word"]):
        ratio = r["tokens"] / base["tokens"]
        marker = "  <- baseline" if r is base else ""
        print(
            f"{r['name'][:44]:44s} {r['vocab']:>8,} {r['tokens_per_word']:>9.3f} "
            f"{r['bytes_per_token']:>10.3f} {ratio:>8.3f}x{marker}"
        )

    best = min(rows, key=lambda r: r["tokens_per_word"])
    if best is not base:
        saving = 1 - best["tokens"] / base["tokens"]
        print(
            f"\n{best['name']} needs {saving * 100:.1f}% fewer tokens than {base['name']} "
            f"for the same Swedish text - i.e. {saving * 100:.1f}% more data per token budget."
        )
    return 0


def _tok_size(tok) -> int:
    return tok.get_vocab_size()


if __name__ == "__main__":
    raise SystemExit(main())
