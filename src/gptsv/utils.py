"""Distributed setup, LR schedules, checkpointing, small helpers."""

from __future__ import annotations

import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from .config import ScheduleConfig

# --------------------------------------------------------------------------- #
# distributed
# --------------------------------------------------------------------------- #


@dataclass
class DistInfo:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    device: torch.device = torch.device("cpu")
    enabled: bool = False

    @property
    def is_master(self) -> bool:
        return self.rank == 0


def setup_distributed() -> DistInfo:
    """Initialise from torchrun env vars if present, else return a single-process info.

    Uses nccl on CUDA and gloo otherwise.
    """
    if "RANK" not in os.environ:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return DistInfo(device=device)

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ["WORLD_SIZE"])

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    dist.init_process_group(backend=backend)
    return DistInfo(rank, local_rank, world_size, device, enabled=True)


def cleanup_distributed(info: DistInfo) -> None:
    if info.enabled and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def all_reduce_mean(x: torch.Tensor, info: DistInfo) -> torch.Tensor:
    if info.enabled:
        x = x.clone()
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        x /= info.world_size
    return x


# --------------------------------------------------------------------------- #
# schedules
# --------------------------------------------------------------------------- #


def lr_multiplier(step: int, max_steps: int, cfg: ScheduleConfig) -> float:
    """Warmup-stable-decay multiplier in [0, 1], applied to every base LR.

    WSD holds a constant LR for most of training, so a run can be extended or
    branched without redoing the whole schedule.
    """
    warmup = cfg.warmup_steps
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup

    decay_start = int(max_steps * (1.0 - cfg.decay_frac))
    if step < decay_start:
        return 1.0
    progress = min(1.0, max(0.0, (step - decay_start) / max(1, max_steps - decay_start)))
    # 1 - sqrt decay tends to beat linear for WSD.
    return cfg.lr_min_frac + (1.0 - cfg.lr_min_frac) * (1.0 - math.sqrt(progress))


# --------------------------------------------------------------------------- #
# checkpoints
# --------------------------------------------------------------------------- #


_WRAPPER_PREFIXES = ("module.", "_orig_mod.", "model.")


def strip_wrapper_prefixes(sd: dict) -> dict:
    """Remove DDP / torch.compile / TrainWrapper key prefixes.

    Checkpoints then load into a bare `GPTSV` however they were trained.
    """
    out = {}
    for key, value in sd.items():
        changed = True
        while changed:
            changed = False
            for prefix in _WRAPPER_PREFIXES:
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    changed = True
        out[key] = value
    return out


def save_checkpoint(path: Path, model, optimizer, step: int, cfg_dict: dict, info: DistInfo) -> None:
    """Rank-0 full-state checkpoint.

    `model` is the DDP module when training in parallel. Wrapper key prefixes
    are stripped before writing.
    """
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
        get_optimizer_state_dict,
    )

    opts = StateDictOptions(full_state_dict=True, cpu_offload=True)

    # Every rank takes part in these calls; only rank 0 writes the file.
    model_sd = strip_wrapper_prefixes(get_model_state_dict(model, options=opts))
    optim_sd = [get_optimizer_state_dict(model, o, options=opts) for o in optimizer.optimizers]
    if not info.is_master:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model_sd,
            "optimizer": optim_sd,
            "optimizer_base_lrs": optimizer.state_dict()["base_lrs"],
            "step": step,
            "config": cfg_dict,
        },
        path,
    )


def find_latest_checkpoint(out_dir: Path) -> Path | None:
    ckpts = sorted(out_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    return ckpts[-1] if ckpts else None


def load_checkpoint(
    path: Path, model, optimizer, info: DistInfo, allow_missing: tuple[str, ...] = ()
) -> tuple[int, list[str]]:
    """Restore model + optimizer in place. Returns (step, missing keys).

    Every model weight must be in the checkpoint and vice versa, except missing
    keys that fully match one of the `allow_missing` regexes; those keep their
    initial values. set_model_state_dict does not enforce this itself.

    Checkpoints hold the full state, so one written on N GPUs can be resumed on M.
    """
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        set_model_state_dict,
        set_optimizer_state_dict,
    )

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    expected, saved = set(model.state_dict()), set(ckpt["model"])
    missing = sorted(expected - saved)
    unexpected = sorted(saved - expected)
    disallowed = [k for k in missing if not any(re.fullmatch(p, k) for p in allow_missing)]
    if disallowed or unexpected:
        raise ValueError(
            f"checkpoint {path} does not match the model: "
            f"missing {disallowed}, unexpected {unexpected}"
        )
    opts = StateDictOptions(
        full_state_dict=True, broadcast_from_rank0=info.enabled, strict=not missing
    )
    set_model_state_dict(model, ckpt["model"], options=opts)

    # Muon and AdamW each own only some of the parameters, so each optimizer
    # has no state for the rest. strict=False allows that.
    opt_opts = StateDictOptions(
        full_state_dict=True, broadcast_from_rank0=info.enabled, strict=False
    )
    for opt, sd in zip(optimizer.optimizers, ckpt["optimizer"], strict=True):
        set_optimizer_state_dict(model, opt, optim_state_dict=sd, options=opt_opts)
    return int(ckpt["step"]), missing


def prune_checkpoints(
    out_dir: Path, keep: int, protect: Path | None = None, keep_every: int = 0
) -> None:
    """Keep the `keep` highest-numbered checkpoints.

    `protect` is never deleted, so the checkpoint just written survives even
    if out_dir holds higher-numbered ones from an earlier run. Checkpoints at a
    multiple of `keep_every` are never deleted either.
    """
    if keep <= 0:
        return
    ckpts = sorted(out_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    protect = protect.resolve() if protect else None
    for old in ckpts[:-keep]:
        if protect is not None and old.resolve() == protect:
            continue
        if keep_every > 0 and int(old.stem.split("_")[1]) % keep_every == 0:
            continue
        old.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# misc
# --------------------------------------------------------------------------- #


def set_seed(seed: int, rank: int = 0) -> None:
    """Seed everything. Rank offset keeps data-order RNG distinct per process."""
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


def human_num(n: float) -> str:
    for unit in ("", "K", "M", "B", "T"):
        if abs(n) < 1000:
            return f"{n:.2f}{unit}"
        n /= 1000
    return f"{n:.2f}P"


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))

