#!/bin/bash
# Canonical seed-valley launcher. The walltime/memory clones this file used to
# have (run_seed_valley_{1h,3h,4h,6h,bs256,bs256_3h}.slurm) are gone: pass
# --time / --mem on the sbatch command line instead, and set the batch size and
# learning rate through the optional BATCH_SIZE / LR arguments below.
#
#   sbatch --time=3:00:00 --mem=120G scripts/slurm/run_seed_valley.slurm \
#          <config> <name> <seed> <topo> <maxn> [init_from] [batch_size] [lr]
#
#SBATCH --job-name=seedval
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=2:00:00
#SBATCH -o /dss/dsshome1/03/go49jit2/thesis/slurm_logs/%x_%j.out
#SBATCH -e /dss/dsshome1/03/go49jit2/thesis/slurm_logs/%x_%j.err

set -euo pipefail
cd /dss/dsshome1/03/go49jit2/thesis
export PATH=/dss/dsshome1/03/go49jit2/.local/bin:$PATH

# Args: CONFIG NAME SEED TOPO MAXN [INIT_FROM_CKPT_OR_EMPTY] [BATCH_SIZE] [LR]
CONFIG=$1
NAME=$2
SEED=$3
TOPO=$4
MAXN=$5
INIT_FROM=${6:-}
BATCH_SIZE=${7:-64}
LR=${8:-0.003}

EXTRA=""
if [ -n "$INIT_FROM" ]; then
  EXTRA="--init-from $INIT_FROM"
fi

echo "Job: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"
echo "name=$NAME seed=$SEED topo=$TOPO max-per-topo=$MAXN init-from=$INIT_FROM bs=$BATCH_SIZE lr=$LR"
echo "Start: $(date)"

srun /usr/bin/python3 -u scripts/pretrain_supervised_v1.py \
  --config "$CONFIG" \
  --name "$NAME" \
  --seed "$SEED" \
  --lr "$LR" \
  --warmup-epochs 0 \
  --epochs 5000 \
  --batch-size "$BATCH_SIZE" \
  --topology "$TOPO" \
  --max-per-topo "$MAXN" \
  --dataset-dir datasets/opamp_3stage_pretrain_combined_5topo \
  $EXTRA

echo "End: $(date)"
