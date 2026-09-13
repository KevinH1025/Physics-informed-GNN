#!/bin/bash
#SBATCH --job-name=seedval3h
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=3:00:00
#SBATCH -o /dss/dsshome1/03/go49jit2/thesis/slurm_logs/%x_%j.out
#SBATCH -e /dss/dsshome1/03/go49jit2/thesis/slurm_logs/%x_%j.err

set -euo pipefail
cd /dss/dsshome1/03/go49jit2/thesis
export PATH=/dss/dsshome1/03/go49jit2/.local/bin:$PATH

CONFIG=$1; NAME=$2; SEED=$3; TOPO=$4; MAXN=$5; INIT_FROM=${6:-}
EXTRA=""; [ -n "$INIT_FROM" ] && EXTRA="--init-from $INIT_FROM"

echo "Job: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"
echo "name=$NAME seed=$SEED topo=$TOPO max-per-topo=$MAXN init-from=$INIT_FROM walltime=3h"
echo "Start: $(date)"

srun /usr/bin/python3 -u scripts/pretrain_supervised_v1.py \
  --config "$CONFIG" --name "$NAME" --seed "$SEED" \
  --lr 0.003 --warmup-epochs 0 --epochs 5000 --batch-size 64 \
  --topology "$TOPO" --max-per-topo "$MAXN" \
  --dataset-dir datasets/opamp_3stage_pretrain_combined_5topo $EXTRA

echo "End: $(date)"
