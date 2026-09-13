#!/usr/bin/env bash
# Phase 1: multi-GPU box, DDP.
#
#   scripts/run_phase1.sh [--resume out/phase1_430m/step_0000200.pt]
#
# GPU count and interconnect are detected, not hardcoded. The previous version
# assumed the 4x RTX 3060 box this config was written for and always set
# NCCL_P2P_DISABLE=1; on an NVLink machine that forces every all-reduce through
# host memory and throws away most of what the hardware can do.
#
# The global batch is pinned at 128 sequences (262k tokens/step) whatever the GPU
# count, so the loss curve stays comparable across phase 0 and across hosts -
# only the number of micro-steps per rank changes.
#
set -euo pipefail
cd "$(dirname "$0")/.."
TORCHRUN=${TORCHRUN:-$([ -x .venv/bin/torchrun ] && echo .venv/bin/torchrun || echo torchrun)}

GPUS=${GPUS:-$(nvidia-smi -L | wc -l)}
# Size the per-GPU micro-batch to the card actually present. The global batch is
# unchanged either way - only the number of micro-steps per rank changes - so a
# bigger micro-batch is free throughput: larger matmuls, fewer per-step overheads,
# and no effect on the optimization or on comparability with phase 0.
# Activations are ~2.4GB per sequence with grad_checkpoint off, plus ~7GB fixed.
if [ -z "${MICRO:-}" ]; then
    # Size from FREE memory on the WORST gpu, not from memory.total. On any
    # shared or multi-tenant box a neighbour's processes are invisible to us:
    # one 40GB A100 came with 34GB already occupied and "No running processes
    # found". Sizing off total there would OOM on step 1, and
    # supervise_phase1.sh would restart straight back into it.
    VRAM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
    if   [ "${VRAM:-0}" -ge 50000 ]; then MICRO=16   # ~45GB peak
    elif [ "${VRAM:-0}" -ge 31000 ]; then MICRO=8    # ~27GB peak
    elif [ "${VRAM:-0}" -ge 20000 ]; then MICRO=4    # ~17GB peak
    elif [ "${VRAM:-0}" -ge 14000 ]; then MICRO=2    # ~12GB peak
    else                                  MICRO=1    # ~9GB peak, last resort
    fi
    echo "least-free gpu has ${VRAM}MB -> micro_batch $MICRO"
fi
GLOBAL_SEQS=${GLOBAL_SEQS:-128} # 128 * 2048 = 262144 tokens/step
ACCUM=$(( GLOBAL_SEQS / (GPUS * MICRO) ))

if [ $(( ACCUM * GPUS * MICRO )) -ne "$GLOBAL_SEQS" ]; then
    echo "GPUS=$GPUS MICRO=$MICRO does not divide $GLOBAL_SEQS sequences/step;" >&2
    echo "pick a MICRO such that GPUS*MICRO divides $GLOBAL_SEQS" >&2
    exit 1
fi

# `topo -m` prints NV# between peers that share an NVLink. P2P is broken in the
# driver on GeForce cards, so disabling it there saves NCCL a long probe for a
# path that does not exist. Datacenter cards are the opposite: even without
# NVLink their PCIe P2P works, and disabling it would push every all-reduce
# through host memory for nothing. Hence the test is NVLink *or* card class, not
# NVLink alone - a 2x A100-SXM4 box whose `topo -m` reports NODE rather than NV#
# would otherwise have had P2P wrongly turned off.
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
if nvidia-smi topo -m 2>/dev/null | grep -qE '\bNV[0-9]+\b'; then
    echo "NVLink detected across $GPUS GPUs: P2P enabled"
elif echo "$GPU_NAME" | grep -qiE 'geforce|rtx [0-9]'; then
    echo "consumer GPUs ($GPU_NAME), no NVLink: disabling P2P (all-reduce via host)"
    export NCCL_P2P_DISABLE=1
else
    echo "datacenter GPUs ($GPU_NAME), no NVLink: leaving PCIe P2P enabled"
fi

export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1  # live log lines when piped to tee / a file

echo "launching: $GPUS GPUs x micro $MICRO x accum $ACCUM = $GLOBAL_SEQS seqs/step"
exec "$TORCHRUN" --standalone --nproc_per_node="$GPUS" -m gptsv.train \
    --config configs/phase1_430m.toml \
    --set "train.micro_batch_size=$MICRO" "train.grad_accum_steps=$ACCUM" "$@"
