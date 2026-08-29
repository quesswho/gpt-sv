#!/usr/bin/env bash
#SBATCH --job-name=gptsv-1p2b
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#
# Single GH200 node. For multi-node, move to torchtitan - this loop is
# data-parallel only.
set -euo pipefail

CONTAINER=${CONTAINER:-$HOME/containers/pytorch-ngc.sif}
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=${HF_HOME:-$SCRATCH/hf}

srun apptainer exec --nv --bind "$PWD:$PWD" --bind "$SCRATCH:$SCRATCH" "$CONTAINER" \
    torchrun --nnodes=1 --nproc_per_node=4 \
        --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
        -m gptsv.train --config configs/phase2_1p2b.toml "$@"
