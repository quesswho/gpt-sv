#!/usr/bin/env bash
# Train the 430M model with DDP on every GPU in the machine.
#
#   scripts/train.sh [--resume [out/430m/step_0000200.pt]]
#
# The global batch is fixed at 128 sequences (262k tokens/step) regardless of
# GPU count; only the micro-batch and the number of accumulation steps change.
#
set -euo pipefail
cd "$(dirname "$0")/.."
TORCHRUN=${TORCHRUN:-$([ -x .venv/bin/torchrun ] && echo .venv/bin/torchrun || echo torchrun)}

GPUS=${GPUS:-$(nvidia-smi -L | wc -l)}
# Pick the largest micro-batch that fits. Activations are ~2.4GB per sequence
# with grad_checkpoint off, plus ~7GB fixed. Sized from the free memory of the
# fullest GPU, since other processes on a shared machine may not be visible.
if [ -z "${MICRO:-}" ]; then
    VRAM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
    if   [ "${VRAM:-0}" -ge 50000 ]; then MICRO=16   # ~45GB peak
    elif [ "${VRAM:-0}" -ge 31000 ]; then MICRO=8    # ~27GB peak
    elif [ "${VRAM:-0}" -ge 20000 ]; then MICRO=4    # ~17GB peak
    elif [ "${VRAM:-0}" -ge 14000 ]; then MICRO=2    # ~12GB peak
    else                                  MICRO=1    # ~9GB peak
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

# GeForce cards have P2P disabled in the driver, so NCCL is told not to try.
# Datacenter cards keep PCIe P2P even without NVLink.
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
export PYTHONUNBUFFERED=1

echo "launching: $GPUS GPUs x micro $MICRO x accum $ACCUM = $GLOBAL_SEQS seqs/step"
exec "$TORCHRUN" --standalone --nproc_per_node="$GPUS" -m gptsv.train \
    --config configs/430m.toml \
    --set "train.micro_batch_size=$MICRO" "train.grad_accum_steps=$ACCUM" "$@"
