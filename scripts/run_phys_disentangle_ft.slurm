#!/bin/bash
#SBATCH --job-name=physdis_ft
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

CONFIG=$1
NAME=$2
INIT=$3
TOPO=$4
MAXN=$5
SEED=${6:-88}
BS=${7:-200}

echo "Job: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"
echo "name=$NAME init=$INIT topo=$TOPO max-per-topo=$MAXN seed=$SEED bs=$BS"
echo "Start: $(date)"

# Wait for init checkpoint if needed
if [ ! -f "$INIT" ]; then
  echo "ERROR: init checkpoint not found: $INIT"
  exit 1
fi

srun /usr/bin/python3 -u scripts/pretrain_supervised_v1.py \
  --config "$CONFIG" \
  --name "$NAME" \
  --seed "$SEED" \
  --lr 0.001 \
  --warmup-epochs 0 \
  --epochs 5000 \
  --batch-size "$BS" \
  --topology "$TOPO" \
  --max-per-topo "$MAXN" \
  --dataset-dir datasets/opamp_3stage_pretrain_combined_5topo \
  --init-from "$INIT"

echo "End: $(date)"
