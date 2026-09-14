"""Pretraining entrypoint. Same file for 1 GPU, 4x RTX 3060, and a GH200 node.

    # phase 0 - single GPU
    gptsv-train --config configs/phase0_150m.toml

    # phase 1 - 4x RTX 3060 (DDP; no NVLink, no P2P, so keep it data-parallel)
    torchrun --nproc_per_node=4 -m gptsv.train --config configs/phase1_500m.toml

    # phase 2 - single GH200 node, FSDP2
    torchrun --nproc_per_node=4 -m gptsv.train --config configs/phase2_1p5b.toml

Beyond a single node, hand the model over to torchtitan for TP/PP/CP - see
torchtitan. This loop is deliberately data-parallel only: no TP, PP or CP.
"""

from __future__ import annotations

import argparse
import math
import shutil
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn

from .config import Config, config_to_dict, load_config
from .data.loader import ShardDataset, TokenLoader
from .model import build_model
from .optim import build_optimizer
from .utils import (
    all_reduce_mean,
    cleanup_distributed,
    find_latest_checkpoint,
    human_num,
    load_checkpoint,
    lr_multiplier,
    prune_checkpoints,
    save_checkpoint,
    set_seed,
    setup_distributed,
    write_json,
)

DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


class TrainWrapper(nn.Module):
    """Puts the loss computation inside `forward`.

    This is not cosmetic: DDP prepares its gradient reducer inside
    `DistributedDataParallel.forward`, so calling `model.loss(...)` directly on
    the inner module would silently skip gradient synchronisation. Everything
    in the training loop goes through this wrapper.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tokens):
        return self.model.loss(tokens)


def parse_overrides(pairs: list[str]) -> dict:
    """--set train.max_steps=50 model.dim=256"""
    import ast

    out = {}
    for pair in pairs:
        key, _, raw = pair.partition("=")
        if not raw:
            raise SystemExit(f"--set expects key=value, got {pair!r}")
        try:
            out[key] = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            out[key] = raw
    return out


def wrap_parallel(model, cfg: Config, info):
    """Return (train_module, parallel_root).

    `parallel_root` is the module that owns gradient synchronisation, kept
    separate because `torch.compile` is applied afterwards.

    Both DDP and FSDP2 install their hooks on a module's `__call__`, so the
    root of the parallel wrapping must be the *same* module the training loop
    invokes - here, `TrainWrapper`. Sharding the raw model and then calling
    `model.loss(...)` through a wrapper silently skips the root pre-forward
    all-gather and fails with "mixed torch.Tensor and DTensor".
    """
    wrapped = TrainWrapper(model)

    if not info.enabled or cfg.train.parallel == "single":
        return wrapped, None

    if cfg.train.parallel == "ddp":
        from torch.nn.parallel import DistributedDataParallel

        ddp = DistributedDataParallel(
            wrapped,
            device_ids=[info.local_rank] if info.device.type == "cuda" else None,
            gradient_as_bucket_view=True,
        )
        return ddp, ddp

    if cfg.train.parallel == "fsdp":
        from torch.distributed.fsdp import fully_shard

        # Shard per block so each all-gather overlaps with the previous
        # block's compute. The root shard then covers what is left:
        # embeddings, the output head and the final norm.
        for block in model.layers:
            fully_shard(block)
        for head in model.mtp_heads:
            fully_shard(head)
        fully_shard(wrapped)
        return wrapped, wrapped

    raise SystemExit(f"unknown train.parallel: {cfg.train.parallel!r}")


def grad_norm_groups(model) -> dict[str, list[nn.Parameter]]:
    """Parameters bucketed for per-group gradient norms. Tied weights appear once."""
    groups: dict[str, list[nn.Parameter]] = {}
    for name, p in model.named_parameters():
        if name.startswith("layers."):
            key = "trunk"
        elif name.startswith("mtp_heads."):
            key = "mtp"
        elif name.startswith(("tok_emb.", "lm_head.")):
            key = "embed"
        else:
            key = "other"
        groups.setdefault(key, []).append(p)
    return groups


def set_grad_sync(parallel_root, cfg: Config, enabled: bool):
    """Suppress gradient sync during grad-accum micro-steps."""
    if parallel_root is None:
        return nullcontext()
    if cfg.train.parallel == "ddp":
        return nullcontext() if enabled else parallel_root.no_sync()
    if cfg.train.parallel == "fsdp" and hasattr(parallel_root, "set_requires_gradient_sync"):
        parallel_root.set_requires_gradient_sync(enabled)
    return nullcontext()


@torch.no_grad()
def evaluate(train_module, loader, steps: int, device, autocast_ctx, info) -> dict[str, float]:
    # The same val batches every time, so the eval curve tracks the model
    # rather than which slice of the val set happened to be drawn.
    loader.seek(0)
    train_module.eval()
    totals: dict[str, float] = {}
    for _ in range(steps):
        tokens = next(loader).to(device, non_blocking=True)
        with autocast_ctx:
            out = train_module(tokens)
        totals["val_loss"] = totals.get("val_loss", 0.0) + out.loss.item()
        totals["val_main_ce"] = totals.get("val_main_ce", 0.0) + out.main_ce.item()
        if out.mtp_ce is not None:
            totals["val_mtp_ce"] = totals.get("val_mtp_ce", 0.0) + out.mtp_ce.item()
        for part, z in (("main", out.z_main), ("mtp", out.z_mtp)):
            if z is not None:
                totals[f"val_z_{part}"] = totals.get(f"val_z_{part}", 0.0) + z.item()
    train_module.train()

    metrics = {k: v / steps for k, v in totals.items()}
    for k, v in metrics.items():
        metrics[k] = all_reduce_mean(torch.tensor(v, device=device), info).item()
    metrics["val_ppl"] = math.exp(min(20.0, metrics["val_main_ce"]))
    for part in ("main", "mtp"):
        z = metrics.pop(f"val_z_{part}", None)
        if z is not None:
            metrics[f"val_lse_rms_{part}"] = math.sqrt(z)
    return metrics


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="gptsv-train", description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. train.max_steps=50")
    ap.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="resume from a checkpoint path, or 'auto' for the latest in out_dir",
    )
    args = ap.parse_args(argv)

    cfg = load_config(args.config, parse_overrides(args.set))
    info = setup_distributed()
    set_seed(cfg.train.seed, info.rank)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    out_dir = Path(cfg.train.out_dir)
    if info.is_master:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(out_dir / "config.json", config_to_dict(cfg))

    # -- model ---------------------------------------------------------------
    cfg.model.max_seq_len = cfg.train.seq_len
    model = build_model(cfg.model).to(info.device)
    block_len = model.block_len()

    if info.is_master:
        print(f"params: {human_num(model.num_params())} total, "
              f"{human_num(model.num_params(non_embedding=True))} non-embedding")
        print(f"block_len: {block_len} (seq_len {cfg.train.seq_len} + {cfg.model.n_mtp_heads} MTP + 1)")

    train_module, parallel_root = wrap_parallel(model, cfg, info)
    if cfg.train.compile:
        train_module = torch.compile(train_module)

    optimizer = build_optimizer(model, cfg.optim)
    if info.is_master:
        n_muon = sum(p.numel() for p in model.param_groups()[0])
        n_adam = sum(p.numel() for p in model.param_groups()[1])
        print(f"optimizer: Muon on {human_num(n_muon)} params, AdamW on {human_num(n_adam)} params")

    # -- data ----------------------------------------------------------------
    train_ds = ShardDataset(cfg.data.train_dir, cfg.data.dtype)
    val_ds = ShardDataset(cfg.data.val_dir, cfg.data.dtype)
    if info.is_master:
        print(f"train: {train_ds}\nval:   {val_ds}")

    train_loader = TokenLoader(
        train_ds, cfg.train.micro_batch_size, block_len, cfg.train.seed, info.rank, info.world_size
    )
    val_loader = TokenLoader(
        val_ds, cfg.train.micro_batch_size, block_len, cfg.train.seed + 1, info.rank, info.world_size
    )

    tokens_per_step = cfg.tokens_per_step * info.world_size
    if info.is_master:
        print(f"tokens/step: {tokens_per_step:,} | total: {human_num(tokens_per_step * cfg.train.max_steps)}")

    # -- run -----------------------------------------------------------------
    amp_dtype = DTYPE_MAP[cfg.train.dtype]
    autocast_ctx = (
        torch.autocast(device_type=info.device.type, dtype=amp_dtype)
        if info.device.type == "cuda" and amp_dtype is not torch.float32
        else nullcontext()
    )

    run = None
    if cfg.train.wandb_project and info.is_master:
        import wandb

        # A resumed run reuses the id stored in out_dir, so its metrics keep
        # appending to the same wandb run instead of starting a new chart.
        id_file = out_dir / "wandb_run_id"
        resume_id = id_file.read_text().strip() if args.resume and id_file.exists() else None
        run = wandb.init(
            project=cfg.train.wandb_project,
            name=cfg.train.wandb_run_name,
            config=config_to_dict(cfg),
            id=resume_id,
            resume="allow" if resume_id else None,
        )
        id_file.write_text(run.id)

    # -- resume --------------------------------------------------------------
    start_step = 0
    if args.resume:
        path = (
            find_latest_checkpoint(out_dir) if args.resume == "auto" else Path(args.resume)
        )
        if path is None:
            if info.is_master:
                print(f"--resume auto: no checkpoint in {out_dir}, starting fresh")
        else:
            # Keys are stored unprefixed, so restore into the raw model. Its
            # parameters are the same objects DDP shares and FSDP2 sharded
            # in place, so this reaches the parallel wrappers too.
            start_step = load_checkpoint(path, model, optimizer, info)
            # The loader is a pure function of (seed, rank, batch index), so
            # seeking reproduces the exact data order the run would have had.
            train_loader.seek(start_step * cfg.train.grad_accum_steps)
            if info.is_master:
                print(f"resumed from {path} at step {start_step}")

    flops_per_token = model.flops_per_token()
    train_module.train()
    t0 = time.perf_counter()
    accum = cfg.train.grad_accum_steps
    norm_groups = grad_norm_groups(model)

    for step in range(start_step, cfg.train.max_steps):
        optimizer.set_lr_mult(lr_multiplier(step, cfg.train.max_steps, cfg.schedule))

        loss_sum = 0.0
        main_ce_sum = 0.0
        z_sums: dict[str, torch.Tensor] = {}
        for micro in range(accum):
            tokens = next(train_loader).to(info.device, non_blocking=True)
            ctx = set_grad_sync(parallel_root, cfg, enabled=(micro == accum - 1))
            with ctx, autocast_ctx:
                out = train_module(tokens)
                loss = out.loss / accum
            loss.backward()
            loss_sum += loss.item()
            main_ce_sum += out.main_ce.item() / accum
            for part, z in (("main", out.z_main), ("mtp", out.z_mtp)):
                if z is not None:
                    z_sums[part] = z_sums.get(part, 0.0) + z / accum

        log_step = (step + 1) % cfg.train.log_interval == 0
        group_norms: dict[str, float] = {}
        if log_step:
            # Pre-clip, and on every rank: under FSDP the norms are collectives.
            for key, params in norm_groups.items():
                grads = [p.grad for p in params if p.grad is not None]
                if grads:
                    group_norms[key] = float(torch.nn.utils.get_total_norm(grads))

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if log_step and info.is_master:
            if info.device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            steps_done = cfg.train.log_interval
            tok_per_s = tokens_per_step * steps_done / dt
            tflops = flops_per_token * tok_per_s / 1e12
            mem_gb = torch.cuda.max_memory_allocated() / 1e9 if info.device.type == "cuda" else 0.0
            lrs = optimizer.current_lrs()
            lse = {part: math.sqrt(float(z)) for part, z in z_sums.items()}
            groups_str = " ".join(f"{k} {v:.2f}" for k, v in group_norms.items())
            lse_str = " ".join(f"{k} {v:.2f}" for k, v in lse.items())
            print(
                f"step {step + 1:>7d} | loss {loss_sum:.4f} | ce {main_ce_sum:.4f} "
                f"| gnorm {float(grad_norm):.2f} ({groups_str}) | lse ({lse_str}) | lr {lrs[0]:.2e} "
                f"| {tok_per_s / 1e3:.1f}K tok/s | {tflops:.1f} TFLOP/s | mem {mem_gb:.1f}GB"
            )
            if run:
                run.log(
                    {
                        "train/loss": loss_sum,
                        "train/main_ce": main_ce_sum,
                        "train/grad_norm": float(grad_norm),
                        **{f"grad_norm/{k}": v for k, v in group_norms.items()},
                        **{f"train/lse_rms_{k}": v for k, v in lse.items()},
                        "train/lr": lrs[0],
                        "perf/tokens_per_s": tok_per_s,
                        "perf/tflops": tflops,
                        "perf/max_mem_gb": mem_gb,
                        "tokens": tokens_per_step * (step + 1),
                    },
                    step=step + 1,
                )
            t0 = time.perf_counter()

        if (step + 1) % cfg.train.eval_interval == 0 or step + 1 == cfg.train.max_steps:
            metrics = evaluate(
                train_module, val_loader, cfg.train.eval_steps, info.device, autocast_ctx, info
            )
            if info.is_master:
                pretty = " | ".join(f"{k} {v:.4f}" for k, v in metrics.items())
                print(f"step {step + 1:>7d} | EVAL | {pretty}")
                if run:
                    run.log({f"eval/{k}": v for k, v in metrics.items()}, step=step + 1)
            t0 = time.perf_counter()

        if (step + 1) % cfg.train.ckpt_interval == 0 or step + 1 == cfg.train.max_steps:
            ckpt_path = out_dir / f"step_{step + 1:07d}.pt"
            save_checkpoint(
                ckpt_path,
                parallel_root if parallel_root is not None else train_module,
                optimizer,
                step + 1,
                config_to_dict(cfg),
                info,
            )
            if info.is_master:
                prune_checkpoints(
                    out_dir,
                    cfg.train.keep_last_n_ckpts,
                    protect=ckpt_path,
                    keep_every=cfg.train.keep_every_n_steps,
                )
                print(f"step {step + 1:>7d} | checkpoint saved")
                # Milestones accumulate; warn before a save can hit a full disk.
                free, size = shutil.disk_usage(out_dir).free, ckpt_path.stat().st_size
                if free < 2 * size:
                    print(
                        f"step {step + 1:>7d} | WARNING: {free / 1e9:.1f} GB free in {out_dir}, "
                        f"checkpoints are {size / 1e9:.2f} GB"
                    )
            t0 = time.perf_counter()

    if run:
        run.finish()
    cleanup_distributed(info)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
