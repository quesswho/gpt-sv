"""Copy a training checkpoint, optionally smaller, for inference or transfer.

  --strip-optim  drops the optimizer state (Muon momentum, AdamW moments) and
                 `optimizer_base_lrs`.
  --dtype        casts the model weights

"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

# What gptsv-generate needs. Anything else in the checkpoint is training state.
INFERENCE_KEYS = ("model", "step", "config")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("ckpt", help="training checkpoint written by gptsv-train")
    p.add_argument("out_dir", help="destination directory (put this on an SSD)")
    p.add_argument(
        "--strip-optim",
        action="store_true",
        help="drop the optimizer state; the result can do inference but not resume",
    )
    p.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        help="cast the model weights (optimizer state is left alone)",
    )
    args = p.parse_args(argv)

    # A run with neither flag would just be a slower `cp`, and silently doing
    # nothing is a worse answer than saying so.
    if not args.strip_optim and args.dtype is None:
        p.error("nothing to do: pass --strip-optim and/or --dtype (a plain copy is `cp`)")

    src = Path(args.ckpt)
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    # Name the file after what was done to it, so a stripped checkpoint is not
    # mistaken for a resumable one months later.
    parts = [
        src.stem,
        *([args.dtype] if args.dtype else []),
        *(["weights"] if args.strip_optim else []),
    ]
    dst = out_dir / f"{'.'.join(parts)}.pt"

    t0 = time.perf_counter()
    ckpt = torch.load(src, map_location="cpu", mmap=True, weights_only=False)

    if args.dtype:
        dtype = getattr(torch, args.dtype)
        ckpt["model"] = {
            k: v.to(dtype) if torch.is_floating_point(v) else v for k, v in ckpt["model"].items()
        }
    read = time.perf_counter() - t0

    # Keep the whole checkpoint by default, rather than rebuilding it from the
    # keys known today: a later key added to save_checkpoint would otherwise be
    # dropped here without anything failing.
    out = {k: ckpt[k] for k in INFERENCE_KEYS} if args.strip_optim else ckpt
    torch.save(out, dst)

    src_gb = src.stat().st_size / 1e9
    dst_gb = dst.stat().st_size / 1e9
    print(f"read  {src} ({src_gb:.2f} GB) in {read:.1f}s")
    print(
        f"wrote {dst} ({dst_gb:.2f} GB, {src_gb / dst_gb:.1f}x smaller) "
        f"in {time.perf_counter() - t0 - read:.1f}s"
    )

    if args.strip_optim:
        print("\ninference only - no optimizer state, this cannot resume training")
        print(
            f"gptsv-generate --ckpt {dst} --tokenizer tokenizers/sv64k-v2 "
            "--prompt 'Sveriges huvudstad är'"
        )
    else:
        print(f"\nresumable: gptsv-train --config <cfg> --resume {dst}")
        if args.dtype:
            print(f"note: weights are {args.dtype}, so resuming loses the precision they had")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
