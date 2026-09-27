"""Decoder-only transformer for gpt-sv, depending on nothing but torch.

Architecture: RMSNorm (pre-norm), SwiGLU, RoPE, GQA, QK-norm, optional tied
embeddings, optional DeepSeek-V3-style multi-token-prediction heads.

Conventions follow Llama/HF (rotate-half RoPE, `wq/wk/wv/wo`, `w1/w2/w3`), so
the weights map directly onto Qwen3 in `transformers` (see `gptsv.hf`).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig

# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.size(-1),), self.weight, self.eps)

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)


def precompute_rope(head_dim: int, max_seq_len: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (cos, sin), each [max_seq_len, head_dim // 2], in fp32."""
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, H, T, head_dim]; cos/sin: [T, head_dim // 2]. Rotate-half convention."""
    x1, x2 = x.float().chunk(2, dim=-1)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    out = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
    return out.type_as(x)


class KVCache:
    """Preallocated keys/values for incremental decoding, one slot per layer.

    `pos` counts the positions already written. `GPTSV.trunk` advances it
    after every forward, so a caller feeds the prompt once and then one token
    at a time; RoPE offsets and causal masking both follow from `pos` alone.
    """

    def __init__(self, cfg: ModelConfig, batch_size: int, max_len: int, device, dtype):
        if max_len > cfg.max_seq_len:
            raise ValueError(f"max_len {max_len} exceeds max_seq_len {cfg.max_seq_len} (RoPE table)")
        shape = (cfg.n_layers, batch_size, cfg.n_kv_heads, max_len, cfg.head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.max_len = max_len
        self.pos = 0

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Write k/v [B, Hkv, T, D] at `pos`; return everything cached for `layer` so far."""
        end = self.pos + k.size(2)
        if end > self.max_len:
            raise ValueError(f"KV cache full: need {end} positions, have {self.max_len}")
        self.k[layer, :, :, self.pos : end] = k
        self.v[layer, :, :, self.pos : end] = v
        return self.k[layer, :, :, :end], self.v[layer, :, :, :end]


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.gqa = cfg.n_kv_heads != cfg.n_heads

        self.wq = nn.Linear(cfg.dim, cfg.n_heads * cfg.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.dim, bias=False)

        # QK-norm keeps the attention logits bounded, which avoids loss spikes.
        self.q_norm = RMSNorm(cfg.head_dim, cfg.norm_eps) if cfg.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(cfg.head_dim, cfg.norm_eps) if cfg.qk_norm else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: KVCache | None = None,
        layer: int = 0,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(self.q_norm(q), cos, sin)
        k = apply_rope(self.k_norm(k), cos, sin)

        if cache is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=self.gqa)
        else:
            pos = cache.pos
            k, v = cache.update(layer, k, v)
            mask = torch.ones(T, pos + T, dtype=torch.bool, device=x.device).tril(diagonal=pos)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=self.gqa)
        y = y.transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim)
        return self.wo(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.dim, cfg.ffn_hidden, bias=False)  # gate
        self.w3 = nn.Linear(cfg.dim, cfg.ffn_hidden, bias=False)  # up
        self.w2 = nn.Linear(cfg.ffn_hidden, cfg.dim, bias=False)  # down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: KVCache | None = None,
        layer: int = 0,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin, cache, layer)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class MTPHead(nn.Module):
    """One DeepSeek-V3 multi-token-prediction depth.

    Takes the previous depth's hidden state h at position i and the embedding of
    the token at i+k, and predicts the token at i+k+1. The embedding and output
    head are shared with the trunk, so each depth adds one block and one
    projection. `out_norm` is applied only to what goes into the output head;
    the next depth gets the unnormalised state.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.h_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.e_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.proj = nn.Linear(2 * cfg.dim, cfg.dim, bias=False)
        self.block = Block(cfg)
        self.out_norm = RMSNorm(cfg.dim, cfg.norm_eps) if cfg.mtp_out_norm else None

    def forward(self, h, emb, cos, sin):
        z = self.proj(torch.cat((self.h_norm(h), self.e_norm(emb)), dim=-1))
        return self.block(z, cos, sin)


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #


@dataclass
class LossOutput:
    loss: torch.Tensor  # the scalar that gets .backward()
    main_ce: torch.Tensor
    mtp_ce: torch.Tensor | None
    z_main: torch.Tensor | None = None  # mean logsumexp^2 of the main logits


class GPTSV(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm_f = RMSNorm(cfg.dim, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        self.mtp_heads = nn.ModuleList(MTPHead(cfg) for _ in range(cfg.n_mtp_heads))

        cos, sin = precompute_rope(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # Scale residual-output projections by 1/sqrt(2 * n_layers) so the
        # residual stream variance stays ~constant with depth at init.
        scale = (2 * cfg.n_layers) ** -0.5
        for name, p in self.named_parameters():
            if name.endswith(("attn.wo.weight", "ffn.w2.weight")):
                with torch.no_grad():
                    p.mul_(scale)

    # -- init ---------------------------------------------------------------

    def _init_weights(self, module: nn.Module) -> None:
        std = self.cfg.init_std
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
        elif isinstance(module, RMSNorm):
            module.reset_parameters()

    # -- forward ------------------------------------------------------------

    def trunk(self, idx: torch.Tensor, cache: KVCache | None = None) -> torch.Tensor:
        """Token ids [B, T] -> final hidden states [B, T, dim] (pre-lm_head).

        With a `cache`, `idx` continues the sequence already cached: its
        positions start at `cache.pos`, and the cache is advanced past them.
        """
        B, T = idx.shape
        pos = cache.pos if cache is not None else 0
        assert pos + T <= self.cfg.max_seq_len, f"position {pos + T} > max_seq_len {self.cfg.max_seq_len}"
        cos, sin = self.rope_cos[pos : pos + T], self.rope_sin[pos : pos + T]

        x = self.tok_emb(idx)
        for i, layer in enumerate(self.layers):
            if self.cfg.grad_checkpoint and self.training:
                x = checkpoint(layer, x, cos, sin, use_reentrant=False)
            else:
                x = layer(x, cos, sin, cache, i)
        if cache is not None:
            cache.pos += T
        return self.norm_f(x)

    def forward(self, idx: torch.Tensor, cache: KVCache | None = None) -> torch.Tensor:
        """Inference path: token ids [B, T] -> logits [B, T, vocab]."""
        return self.lm_head(self.trunk(idx, cache))

    def block_len(self) -> int:
        """Token block length the data loader must supply per sample.

        Main head needs T+1 tokens. MTP depth k needs up to index T+k, and
        predicts up to index T+k+1, so depth D needs T+D+1 tokens total.
        """
        return self.cfg.max_seq_len + self.cfg.n_mtp_heads + 1

    def _head_loss(
        self, h: torch.Tensor, targets: torch.Tensor, want_z: bool
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply lm_head + cross entropy, optionally in recomputed chunks.

        Returns (mean CE, mean squared logsumexp or None). Chunking keeps the
        peak logit allocation at `loss_chunk_size * vocab` instead of
        `B*T * vocab`, at the cost of recomputing the head in backward.
        """
        hf = h.reshape(-1, h.size(-1))
        tf = targets.reshape(-1)
        chunk = self.cfg.loss_chunk_size
        n = hf.size(0)

        # Sum over chunks and divide once at the end, since the last chunk can
        # be smaller than the others.
        def compute(h_chunk: torch.Tensor, t_chunk: torch.Tensor):
            logits = self.lm_head(h_chunk).float()
            ce = F.cross_entropy(logits, t_chunk, reduction="sum")
            if not want_z:
                return ce, ce.new_zeros(())
            return ce, torch.logsumexp(logits, dim=-1).pow(2).sum()

        if chunk <= 0 or chunk >= n:
            ce_sum, z_sum = compute(hf, tf)
        else:
            ce_sum = hf.new_zeros((), dtype=torch.float32)
            z_sum = hf.new_zeros((), dtype=torch.float32)
            for i in range(0, n, chunk):
                h_c, t_c = hf[i : i + chunk], tf[i : i + chunk]
                if self.training and torch.is_grad_enabled():
                    ce_c, z_c = checkpoint(compute, h_c, t_c, use_reentrant=False)
                else:
                    ce_c, z_c = compute(h_c, t_c)
                ce_sum = ce_sum + ce_c
                z_sum = z_sum + z_c

        return ce_sum / n, (z_sum / n) if want_z else None

    def loss(self, tokens: torch.Tensor) -> LossOutput:
        """tokens: [B, L] where L == block_len(). Returns the training loss.

        The loader always supplies the D+1 extra lookahead tokens, so slicing is
        static and needs no masking, which suits torch.compile.
        """
        cfg = self.cfg
        expected = self.block_len()
        if tokens.shape[1] != expected:
            raise ValueError(
                f"loss() expects blocks of exactly block_len()={expected} tokens "
                f"(seq_len {cfg.max_seq_len} + {cfg.n_mtp_heads} MTP lookahead + 1), "
                f"got {tokens.shape[1]}. Construct the loader with model.block_len()."
            )
        T = cfg.max_seq_len
        idx = tokens[:, :T]
        h = self.trunk(idx)

        want_z = cfg.z_loss_weight > 0
        main_ce, z_main = self._head_loss(h, tokens[:, 1 : T + 1], want_z)

        mtp_ce = None
        if cfg.n_mtp_heads > 0:
            cos, sin = self.rope_cos[:T], self.rope_sin[:T]
            losses = []
            h_k = h
            for k, head in enumerate(self.mtp_heads, start=1):
                emb = self.tok_emb(tokens[:, k : k + T])
                h_k = head(h_k, emb, cos, sin)
                h_out = h_k if head.out_norm is None else head.out_norm(h_k)
                ce_k, _ = self._head_loss(h_out, tokens[:, k + 1 : k + 1 + T], want_z=False)
                losses.append(ce_k)
            mtp_ce = torch.stack(losses).mean()

        total = main_ce
        if z_main is not None:
            total = total + cfg.z_loss_weight * z_main
        if mtp_ce is not None:
            total = total + cfg.mtp_loss_weight * mtp_ce

        return LossOutput(
            loss=total,
            main_ce=main_ce.detach(),
            mtp_ce=None if mtp_ce is None else mtp_ce.detach(),
            z_main=None if z_main is None else z_main.detach(),
        )

    @torch.no_grad()
    def calibrate_mtp_out_norm(self, tokens: torch.Tensor) -> None:
        """Fit each MTP out_norm gain so the normalised output matches the raw one.

        Used when resuming a checkpoint trained without out_norm. A per-channel
        least-squares gain keeps the head's output scale instead of resetting it
        to unit RMS. `tokens` is a [B, block_len()] batch.
        """
        T = self.cfg.max_seq_len
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        h_k = self.trunk(tokens[:, :T])
        for k, head in enumerate(self.mtp_heads, start=1):
            h_k = head(h_k, self.tok_emb(tokens[:, k : k + T]), cos, sin)
            if head.out_norm is None:
                continue
            x = h_k.float().reshape(-1, h_k.size(-1))
            n = F.rms_norm(x, (x.size(-1),), None, head.out_norm.eps)
            head.out_norm.weight.copy_((x * n).sum(0) / (n * n).sum(0))

    # -- introspection ------------------------------------------------------

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
            if not self.cfg.tie_embeddings:
                n -= self.lm_head.weight.numel()
        return n

    def flops_per_token(self) -> float:
        """Rough forward+backward FLOPs/token (6ND + attention), for MFU."""
        cfg = self.cfg
        n = self.num_params(non_embedding=True)
        dense = 6 * n
        attn = 12 * cfg.n_layers * cfg.n_heads * cfg.head_dim * cfg.max_seq_len
        return dense + attn

    def param_groups(self) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        """Split params into (muon_params, adam_params).

        Muon gets the 2D hidden matrices. Embeddings, the output head and all
        norms go to AdamW, since Muon works poorly on embedding tables and does
        not apply to 1D tensors.
        """
        muon, adam = [], []
        excluded = {id(self.tok_emb.weight), id(self.lm_head.weight)}
        for p in self.parameters():
            if p.ndim == 2 and id(p) not in excluded:
                muon.append(p)
            else:
                adam.append(p)
        return muon, adam


def build_model(cfg: ModelConfig) -> GPTSV:
    return GPTSV(cfg)
