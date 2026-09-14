"""Configuration dataclasses, loaded from TOML.

One config file fully describes a run. The same file is used on 1 GPU and on a
GH200 cluster; only `train.parallel`, batch sizes and the data paths change.
"""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    vocab_size: int = 65536
    dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int | None = None  # None -> MHA (n_kv_heads == n_heads)
    head_dim: int | None = None  # None -> dim // n_heads
    ffn_hidden: int | None = None  # None -> derived from ffn_mult, see below
    ffn_mult: float = 8 / 3  # SwiGLU keeps param count equal to a 4x GELU MLP
    ffn_multiple_of: int = 128
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    qk_norm: bool = True
    tie_embeddings: bool = True

    # Multi-token prediction (DeepSeek-V3 style sequential heads).
    # 0 disables it. Each head costs one extra transformer block + one extra
    # full-vocab logit matrix in the loss, so memory grows noticeably.
    n_mtp_heads: int = 0
    mtp_loss_weight: float = 0.3
    # RMSNorm on each MTP output before the shared lm_head, as in DeepSeek-V3.
    # Off by default so checkpoints saved without it rebuild unchanged.
    mtp_out_norm: bool = False

    z_loss_weight: float = 1e-4  # stabilises logit scale; 0 disables
    init_std: float = 0.02

    # Memory controls. Both matter enormously on a 12GB RTX 3060 and are
    # usually left off on a 96GB GH200.
    #
    # loss_chunk_size: rows of [B*T, vocab] logits materialised at a time.
    # The full logit tensor is the single largest allocation in a small model
    # with a 65k vocab (8*2048 rows * 65536 * 4B = 4.3GB in fp32, per head,
    # and MTP multiplies that). Chunks are recomputed in backward, so this
    # trades a little compute for a large memory saving. 0 disables.
    loss_chunk_size: int = 0
    # grad_checkpoint: recompute each transformer block in backward.
    grad_checkpoint: bool = False

    def __post_init__(self) -> None:
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        if self.head_dim is None:
            assert self.dim % self.n_heads == 0, "dim must be divisible by n_heads"
            self.head_dim = self.dim // self.n_heads
        if self.ffn_hidden is None:
            h = int(self.ffn_mult * self.dim)
            m = self.ffn_multiple_of
            self.ffn_hidden = m * ((h + m - 1) // m)
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"


@dataclass
class DataConfig:
    train_dir: str = "data/shards/train"
    val_dir: str = "data/shards/val"
    # Token dtype in the .bin shards. uint16 works up to vocab 65535.
    dtype: str = "uint16"


@dataclass
class OptimConfig:
    # Muon handles 2D hidden-layer matrices; AdamW handles embeddings, the
    # output head, norms and biases. This split is the standard recipe
    # (Moonlight / Kimi K2) - Muon on the embedding table hurts.
    use_muon: bool = True
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5
    muon_weight_decay: float = 0.01

    adam_lr: float = 3e-3
    adam_betas: tuple[float, float] = (0.9, 0.95)
    adam_eps: float = 1e-10
    adam_weight_decay: float = 0.01

    grad_clip: float = 1.0


@dataclass
class ScheduleConfig:
    warmup_steps: int = 200
    decay_frac: float = 0.2  # last 20% of steps decay to lr_min_frac
    lr_min_frac: float = 0.0


@dataclass
class TrainConfig:
    out_dir: str = "out/run"
    seed: int = 1337

    seq_len: int = 2048
    micro_batch_size: int = 8  # per-device
    grad_accum_steps: int = 1
    max_steps: int = 5000

    parallel: str = "ddp"  # "ddp" | "fsdp" | "single"
    dtype: str = "bfloat16"  # autocast dtype
    compile: bool = True

    eval_interval: int = 250
    eval_steps: int = 40
    log_interval: int = 10
    ckpt_interval: int = 1000
    keep_last_n_ckpts: int = 2
    keep_every_n_steps: int = 0  # never prune checkpoints at multiples of this; 0 disables

    wandb_project: str | None = None
    wandb_run_name: str | None = None

    def __post_init__(self) -> None:
        if self.keep_every_n_steps < 0:
            raise ValueError(f"keep_every_n_steps must be >= 0, got {self.keep_every_n_steps}")
        if self.keep_every_n_steps and self.keep_every_n_steps % self.ckpt_interval:
            raise ValueError(
                f"keep_every_n_steps={self.keep_every_n_steps} is not a multiple of "
                f"ckpt_interval={self.ckpt_interval}, so no checkpoint would be kept"
            )


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @property
    def tokens_per_step(self) -> int:
        """Tokens consumed per optimizer step, per device. Multiply by world size."""
        return self.train.micro_batch_size * self.train.grad_accum_steps * self.train.seq_len


def _build(cls: type, raw: dict[str, Any]) -> Any:
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**raw)


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    """Load a TOML config. `overrides` uses dotted keys, e.g. {"train.max_steps": 10}."""
    raw = tomllib.loads(Path(path).read_text())

    for dotted, value in (overrides or {}).items():
        section, _, key = dotted.partition(".")
        if not key:
            raise ValueError(f"override must be 'section.key', got {dotted!r}")
        raw.setdefault(section, {})[key] = value

    sections = {f.name: f.type for f in fields(Config)}
    unknown = set(raw) - set(sections)
    if unknown:
        raise ValueError(f"unknown config sections: {sorted(unknown)}")

    return Config(
        model=_build(ModelConfig, raw.get("model", {})),
        data=_build(DataConfig, raw.get("data", {})),
        optim=_build(OptimConfig, raw.get("optim", {})),
        schedule=_build(ScheduleConfig, raw.get("schedule", {})),
        train=_build(TrainConfig, raw.get("train", {})),
    )


def config_to_dict(cfg: Config) -> dict[str, Any]:
    return asdict(cfg)
