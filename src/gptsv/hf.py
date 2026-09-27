"""Export a gpt-sv checkpoint as a Hugging Face Qwen3ForCausalLM.

    gptsv-export-hf out/430m/step_0040000.pt hf/gpt-sv-430m --tokenizer tokenizers/sv64k-v2

GPTSV has the same architecture as Qwen3 (pre-norm, GQA, per-head QK RMSNorm
before rotate-half RoPE, SwiGLU, no biases), so exporting is a matter of
renaming weights. MTP heads are dropped. Every export is checked to produce the
same logits as GPTSV.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import ModelConfig
from .model import GPTSV
from .tokenizer import SPECIAL_TOKENS

_LAYER_RENAMES = {
    "attn_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "attn.wq.weight": "self_attn.q_proj.weight",
    "attn.wk.weight": "self_attn.k_proj.weight",
    "attn.wv.weight": "self_attn.v_proj.weight",
    "attn.wo.weight": "self_attn.o_proj.weight",
    "attn.q_norm.weight": "self_attn.q_norm.weight",
    "attn.k_norm.weight": "self_attn.k_norm.weight",
    "ffn.w1.weight": "mlp.gate_proj.weight",
    "ffn.w3.weight": "mlp.up_proj.weight",
    "ffn.w2.weight": "mlp.down_proj.weight",
}


def qwen3_config(cfg: ModelConfig, eot_id: int | None = None, pad_id: int | None = None):
    from transformers import Qwen3Config

    if not cfg.qk_norm:
        raise ValueError("Qwen3 always applies QK-norm; this checkpoint was trained without it")
    hf_cfg = Qwen3Config(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.dim,
        intermediate_size=cfg.ffn_hidden,
        num_hidden_layers=cfg.n_layers,
        num_attention_heads=cfg.n_heads,
        num_key_value_heads=cfg.n_kv_heads,
        head_dim=cfg.head_dim,
        hidden_act="silu",
        max_position_embeddings=cfg.max_seq_len,
        rms_norm_eps=cfg.norm_eps,
        rope_parameters={"rope_type": "default", "rope_theta": cfg.rope_theta},
        attention_bias=False,
        tie_word_embeddings=cfg.tie_embeddings,
        bos_token_id=eot_id,
        eos_token_id=eot_id,
        pad_token_id=pad_id,
    )
    # transformers 4.x reads rope_theta from the top level, not rope_parameters.
    hf_cfg.rope_theta = cfg.rope_theta
    return hf_cfg


def to_hf_state_dict(sd: dict[str, torch.Tensor], tie_embeddings: bool) -> dict[str, torch.Tensor]:
    out = {}
    for key, value in sd.items():
        if key.startswith("mtp_heads."):
            continue
        if key == "tok_emb.weight":
            out["model.embed_tokens.weight"] = value
        elif key == "norm_f.weight":
            out["model.norm.weight"] = value
        elif key == "lm_head.weight":
            if not tie_embeddings:
                out["lm_head.weight"] = value
        elif key.startswith("layers."):
            _, idx, rest = key.split(".", 2)
            out[f"model.layers.{idx}.{_LAYER_RENAMES[rest]}"] = value
        else:
            raise KeyError(f"no Qwen3 equivalent for {key}")
    return out


def build_hf_model(cfg: ModelConfig, sd: dict[str, torch.Tensor], **config_kw):
    from transformers import Qwen3ForCausalLM

    hf_cfg = qwen3_config(cfg, **config_kw)
    hf_cfg.dtype = "float32"
    model = Qwen3ForCausalLM(hf_cfg)
    missing, unexpected = model.load_state_dict(
        to_hf_state_dict(sd, cfg.tie_embeddings), strict=False
    )
    missing = [k for k in missing if not (cfg.tie_embeddings and k == "lm_head.weight")]
    if missing or unexpected:
        raise ValueError(f"export mismatch: missing {missing}, unexpected {unexpected}")
    if cfg.tie_embeddings:
        model.tie_weights()
    return model.eval()


@torch.no_grad()
def max_logit_diff(gptsv: GPTSV, hf_model, tokens: torch.Tensor) -> float:
    ref = gptsv.eval()(tokens).float()
    got = hf_model(input_ids=tokens).logits.float()
    return (ref - got).abs().max().item()


def main(argv: list[str] | None = None) -> int:
    from transformers import PreTrainedTokenizerFast

    p = argparse.ArgumentParser(
        prog="gptsv-export-hf",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("ckpt")
    p.add_argument("out_dir")
    p.add_argument("--tokenizer", required=True, help="dir containing tokenizer.json")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--tol", type=float, default=1e-3, help="max fp32 logit difference allowed")
    args = p.parse_args(argv)

    ckpt = torch.load(args.ckpt, map_location="cpu", mmap=True, weights_only=False)
    cfg = ModelConfig(**ckpt["config"]["model"])
    sd = {k: v.float() for k, v in ckpt["model"].items()}

    tok = PreTrainedTokenizerFast(
        tokenizer_file=str(Path(args.tokenizer) / "tokenizer.json"),
        bos_token=SPECIAL_TOKENS[0],
        eos_token=SPECIAL_TOKENS[0],
        pad_token="<|pad|>",
    )
    tok.model_max_length = cfg.max_seq_len

    hf_model = build_hf_model(
        cfg, sd, eot_id=tok.convert_tokens_to_ids(SPECIAL_TOKENS[0]), pad_id=tok.pad_token_id
    )

    reference = GPTSV(cfg)
    reference.load_state_dict(sd, strict=True)
    text = "Sveriges huvudstad är Stockholm, och Göteborg är den näst största staden."
    tokens = torch.tensor([tok.encode(text)])
    diff = max_logit_diff(reference, hf_model, tokens)
    print(f"parity: max |logit difference| = {diff:.2e} over {tokens.shape[1]} tokens")
    if diff > args.tol:
        raise SystemExit(f"parity check failed: {diff:.2e} > {args.tol:.0e}")
    del reference

    out = Path(args.out_dir)
    hf_model.to(getattr(torch, args.dtype)).save_pretrained(out)
    tok.save_pretrained(out)
    # transformers 5 saves a tokenizer class that 4.x does not have.
    tok_cfg_path = out / "tokenizer_config.json"
    tok_cfg = json.loads(tok_cfg_path.read_text())
    tok_cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
    tok_cfg_path.write_text(json.dumps(tok_cfg, indent=2, ensure_ascii=False) + "\n")
    print(f"saved {out} (step {ckpt['step']}, {args.dtype})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
