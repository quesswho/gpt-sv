"""Sample text from a gpt-sv checkpoint.

    gptsv-generate --ckpt out/phase0_130m/step_0004000.pt --prompt "Stockholm är"
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Iterator
from contextlib import nullcontext

import torch

from .config import ModelConfig
from .model import GPTSV, KVCache, precompute_rope
from .tokenizer import SPECIAL_TOKENS

EOT = SPECIAL_TOKENS[0]


def load_model(
    path: str, device: torch.device, dtype: torch.dtype | None = None
) -> tuple[GPTSV, int]:
    """Load a checkpoint for inference. Builds on `meta` to skip __init__'s
    trunc_normal_ over every parameter, ~42s at 448M and overwritten anyway."""
    ckpt = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    cfg = ModelConfig(**ckpt["config"]["model"])

    with torch.device("meta"):
        model = GPTSV(cfg)
    model.load_state_dict(ckpt["model"], assign=True, strict=True)

    # persistent=False, so absent from the state dict and still meta after load
    model.rope_cos, model.rope_sin = precompute_rope(
        cfg.head_dim, cfg.max_seq_len, cfg.rope_theta
    )

    model = model.to(device)
    if dtype is not None:
        model = model.to(dtype)
        # keep rope in fp32: bf16 cos/sin carries ~3 digits of a rotation angle
        model.rope_cos = model.rope_cos.float()
        model.rope_sin = model.rope_sin.float()
    if cfg.tie_embeddings:  # assign=True replaced the shared Parameter
        model.lm_head.weight = model.tok_emb.weight
    return model.eval(), int(ckpt["step"])


def sample(logits: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    return torch.multinomial(logits.softmax(dim=-1), num_samples=1)


@torch.no_grad()
def generate(
    model: GPTSV,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    eot_id: int,
) -> Iterator[int]:
    """Yield sampled token ids one at a time, stopping at <|endoftext|> or a full context."""
    device = model.tok_emb.weight.device
    on_cuda = device.type == "cuda"
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if on_cuda else nullcontext()
    max_len = min(model.cfg.max_seq_len, len(prompt_ids) + max_new_tokens)
    cache = KVCache(model.cfg, 1, max_len, device, torch.bfloat16 if on_cuda else torch.float32)

    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        with autocast:
            logits = model.lm_head(model.trunk(ids, cache)[:, -1]).float()
        ids = sample(logits, temperature, top_k)
        token = ids.item()
        yield token
        if token == eot_id or cache.pos == cache.max_len:
            return


def main(argv: list[str] | None = None) -> int:
    from tokenizers import Tokenizer

    p = argparse.ArgumentParser(
        prog="gptsv-generate", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--ckpt", required=True, help="checkpoint .pt written by gptsv-train")
    p.add_argument("--tokenizer", default="tokenizers/sv64k", help="dir containing tokenizer.json")
    p.add_argument("--prompt", default="", help="empty = start a fresh document")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8, help="0 = greedy")
    p.add_argument("--top-k", type=int, default=50, help="0 = no top-k filtering")
    p.add_argument("--num-samples", type=int, default=1)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--dtype",
        default=None,
        choices=["bfloat16", "float16", "float32"],
        help="weight dtype on the device; defaults to bfloat16 on cuda",
    )
    args = p.parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
    tok = Tokenizer.from_file(f"{args.tokenizer}/tokenizer.json")
    eot_id = tok.token_to_id(EOT)
    device = torch.device(args.device)
    if args.dtype is not None:
        dtype = getattr(torch, args.dtype)
    else:
        dtype = torch.bfloat16 if device.type == "cuda" else None
    t0 = time.perf_counter()
    model, step = load_model(args.ckpt, device, dtype)
    print(
        f"loaded step {step} ({model.num_params() / 1e6:.0f}M params) on {args.device}"
        f" as {next(model.parameters()).dtype} in {time.perf_counter() - t0:.1f}s\n"
    )

    prompt_ids = [eot_id] + tok.encode(args.prompt, add_special_tokens=False).ids
    if len(prompt_ids) >= model.cfg.max_seq_len:
        raise SystemExit(f"prompt is {len(prompt_ids)} tokens; the context is {model.cfg.max_seq_len}")

    for i in range(args.num_samples):
        print(f"--- sample {i + 1}\n{args.prompt}", end="", flush=True)
        out: list[int] = []
        shown = 0
        t0 = time.perf_counter()
        for token in generate(model, prompt_ids, args.max_new_tokens, args.temperature, args.top_k, eot_id):
            out.append(token)
            text = tok.decode(out, skip_special_tokens=True)
            if not text.endswith("�"):
                print(text[shown:], end="", flush=True)
                shown = len(text)
        dt = time.perf_counter() - t0
        print(f"\n[{len(out)} tokens, {len(out) / dt:.1f} tok/s]\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
