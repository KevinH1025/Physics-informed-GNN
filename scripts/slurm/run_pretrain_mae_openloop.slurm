#!/bin/bash
#SBATCH --job-name=gnn_mae
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH -o /dss/dsshome1/03/go49jit2/thesis/slurm_logs/%x_%j.out
#SBATCH -e /dss/dsshome1/03/go49jit2/thesis/slurm_logs/%x_%j.err

cd /dss/dsshome1/03/go49jit2/thesis
export PATH=/dss/dsshome1/03/go49jit2/.local/bin:$PATH
echo "Job: $SLURM_JOB_ID | Config: $1 | Name: $2"
echo "Start: $(date)"
srun python -u scripts/pretrain_mae_openloop.py --config "$1" --name "$2"
echo "End: $(date)"
