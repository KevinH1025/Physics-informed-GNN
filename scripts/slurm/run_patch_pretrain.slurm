#!/bin/bash
#SBATCH --job-name=patch_pretrain
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH -o /dss/dsshome1/03/go49jit2/thesis/slurm_logs/%x_%j.out
#SBATCH -e /dss/dsshome1/03/go49jit2/thesis/slurm_logs/%x_%j.err

cd /dss/dsshome1/03/go49jit2/thesis
export PATH=/dss/dsshome1/03/go49jit2/.local/bin:$PATH
echo "Start: $(date)"

srun python -u scripts/patch_pretrain_combined.py

echo "End: $(date)"
