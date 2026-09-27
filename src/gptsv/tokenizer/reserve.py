"""Add special tokens to a trained BPE tokenizer without retraining it.

The BPE trainer assigns IDs in order: special tokens, the 256-byte alphabet,
then one ID per merge in the order the merges were learned. Dropping the last
k merges therefore gives exactly the tokenizer training would have produced
had it stopped k merges earlier, and frees the top k IDs. New special tokens go
there: every other token keeps its ID and the vocab size is unchanged, so token
shards stay uint16 and the model's embedding table needs no resize.

    python -m gptsv.tokenizer.reserve --base tokenizers/sv64k \\
        --out tokenizers/sv64k-v2 --total-specials 64

On sv64k, 64 special slots cost 59 merges and +0.005% tokens on FineWeb-2 val.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from . import CHAT_TOKENS


def reserve_special_tokens(tokenizer_json: dict, new_tokens: list[str]) -> dict:
    """Return a copy of `tokenizer_json` with `new_tokens` in the IDs of its last merges."""
    d = copy.deepcopy(tokenizer_json)
    model = d["model"]
    if model["type"] != "BPE":
        raise ValueError(f"expected a BPE model, got {model['type']}")
    vocab, merges = model["vocab"], model["merges"]
    n, k = len(vocab), len(new_tokens)
    cut = n - k
    if clash := set(new_tokens) & vocab.keys():
        raise ValueError(f"already in vocab: {sorted(clash)}")
    if any(a["id"] >= cut for a in d["added_tokens"]):
        raise ValueError(f"an added token already sits in the top {k} IDs")

    # Only valid if the last k merges produced exactly the top k IDs. Otherwise
    # ordinary tokens would be remapped without any error.
    pairs = [m.split(" ") if isinstance(m, str) else m for m in merges[-k:]]
    if {vocab.get("".join(p)) for p in pairs} != set(range(cut, n)):
        raise ValueError(f"the last {k} merges do not map onto the top {k} IDs")

    model["merges"] = merges[:-k]
    model["vocab"] = {t: i for t, i in vocab.items() if i < cut}
    for i, t in enumerate(new_tokens, start=cut):
        model["vocab"][t] = i
        d["added_tokens"].append(
            {
                "id": i,
                "content": t,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
        )
    return d


def main(argv: list[str] | None = None) -> int:
    from tokenizers import Tokenizer

    p = argparse.ArgumentParser(
        prog="gptsv-tokenizer-reserve",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--base", required=True, help="dir containing the tokenizer.json to extend")
    p.add_argument("--out", required=True)
    p.add_argument(
        "--total-specials",
        type=int,
        default=64,
        help="special tokens after extending, existing ones included",
    )
    args = p.parse_args(argv)

    base = Path(args.base)
    tok_json = json.loads((base / "tokenizer.json").read_text())
    existing = [a["content"] for a in tok_json["added_tokens"] if a["special"]]
    n_new = args.total_specials - len(existing)
    if n_new < len(CHAT_TOKENS):
        raise SystemExit(
            f"--total-specials {args.total_specials} leaves no room for {CHAT_TOKENS} "
            f"beyond the {len(existing)} existing special tokens"
        )
    new_tokens = CHAT_TOKENS + [f"<|reserved_{i}|>" for i in range(n_new - len(CHAT_TOKENS))]

    # Round-trip through the library so the saved file is known to load.
    tok = Tokenizer.from_str(json.dumps(reserve_special_tokens(tok_json, new_tokens)))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok.save(str(out / "tokenizer.json"))

    meta_path = base / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update(
        vocab_size=tok.get_vocab_size(),
        special_tokens=existing + new_tokens,
        derived_from=str(base),
        dropped_merges=len(new_tokens),
    )
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    ids = {t: tok.token_to_id(t) for t in CHAT_TOKENS}
    print(f"saved -> {out}  (vocab {tok.get_vocab_size()}, {len(new_tokens)} merges dropped)")
    print(f"chat tokens: {ids}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
