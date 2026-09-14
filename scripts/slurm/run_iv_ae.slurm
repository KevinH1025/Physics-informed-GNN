#!/bin/bash
#SBATCH --job-name=iv_ae
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH -o /dss/dsshome1/03/go49jit2/thesis/slurm_logs/%x_%j.out
#SBATCH -e /dss/dsshome1/03/go49jit2/thesis/slurm_logs/%x_%j.err

# Usage: sbatch scripts/slurm/run_iv_ae.slurm [extra args for train_iv_autoencoder.py]
# Default: bottleneck 64, batch 1024, all Vbs (32k surfaces).

cd /dss/dsshome1/03/go49jit2/thesis
export PATH=/dss/dsshome1/03/go49jit2/.local/bin:$PATH

echo "Job: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | GPU: $CUDA_VISIBLE_DEVICES"
echo "Extra args: $@"
echo "Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

srun python scripts/train_iv_autoencoder.py \
    --bottleneck 64 \
    --batch-size 1024 \
    --epochs 2000 \
    --warmup-epochs 10 \
    --plateau-patience 45 \
    --early-stop-patience 150 \
    --log-interval 5 \
    "$@"

echo "End: $(date)"
