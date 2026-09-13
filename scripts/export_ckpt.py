"""Strip a training checkpoint down to weights, for inference.

    scripts/export_ckpt.py out/phase1_430m/step_0021200.pt ~/gptsv-ckpt/

Drops the optimizer state (Muon momentum, AdamW moments) and casts to bf16:
4.26GB -> 1.06GB. Output is a drop-in --ckpt for gptsv-generate.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("ckpt", help="training checkpoint written by gptsv-train")
    p.add_argument("out_dir", help="destination directory (put this on an SSD)")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = p.parse_args(argv)

    src = Path(args.ckpt)
    dtype = getattr(torch, args.dtype)
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{src.stem}.{args.dtype}.pt"

    t0 = time.perf_counter()
    ckpt = torch.load(src, map_location="cpu", mmap=True, weights_only=False)
    weights = {
        k: v.to(dtype) if torch.is_floating_point(v) else v for k, v in ckpt["model"].items()
    }
    read = time.perf_counter() - t0

    # Same shape as a training checkpoint minus the optimizer, so gptsv-generate
    # loads it with no special casing.
    torch.save({"model": weights, "step": ckpt["step"], "config": ckpt["config"]}, dst)

    src_gb = src.stat().st_size / 1e9
    dst_gb = dst.stat().st_size / 1e9
    print(f"read  {src} ({src_gb:.2f} GB) in {read:.1f}s")
    print(
        f"wrote {dst} ({dst_gb:.2f} GB, {src_gb / dst_gb:.1f}x smaller) "
        f"in {time.perf_counter() - t0 - read:.1f}s"
    )
    print(
        f"\ngptsv-generate --ckpt {dst} --tokenizer tokenizers/sv64k-v2 --prompt 'Sveriges huvudstad är'"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
