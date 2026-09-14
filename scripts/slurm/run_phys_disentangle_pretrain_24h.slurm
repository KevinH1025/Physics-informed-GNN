#!/bin/bash
#SBATCH --job-name=physdis24h
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH -o /dss/dsshome1/03/go49jit2/thesis/slurm_logs/%x_%j.out
#SBATCH -e /dss/dsshome1/03/go49jit2/thesis/slurm_logs/%x_%j.err

set -euo pipefail
cd /dss/dsshome1/03/go49jit2/thesis
export PATH=/dss/dsshome1/03/go49jit2/.local/bin:$PATH

CONFIG=$1; NAME=$2; EXCLUDE=$3; SEED=${4:-88}; BS=${5:-400}

echo "Job: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"
echo "name=$NAME exclude=$EXCLUDE seed=$SEED bs=$BS"
echo "Start: $(date)"

srun /usr/bin/python3 -u scripts/pretrain_supervised_v1.py \
  --config "$CONFIG" --name "$NAME" --seed "$SEED" \
  --lr 0.001 --warmup-epochs 50 --epochs 5000 --batch-size "$BS" \
  --exclude-topology "$EXCLUDE" \
  --dataset-dir datasets/opamp_3stage_pretrain_combined_5topo

echo "End: $(date)"
