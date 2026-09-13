#!/bin/bash
#SBATCH --job-name=fan_kcl
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=4:00:00
#SBATCH -o /dss/dsshome1/03/go49jit2/thesis/slurm_logs/%x_%j.out
#SBATCH -e /dss/dsshome1/03/go49jit2/thesis/slurm_logs/%x_%j.err
set -euo pipefail
cd /dss/dsshome1/03/go49jit2/thesis
export PATH=/dss/dsshome1/03/go49jit2/.local/bin:$PATH

CONFIG=$1   # KCL ON or KCL OFF config
NAME=$2     # experiment name
SEED=$3     # 42 / 43 / 44
NMAX=$4     # max-train-samples

echo "Start: $(date)  config=$CONFIG  name=$NAME  seed=$SEED  N=$NMAX"
srun /usr/bin/python3 -u scripts/train_v3.py \
  --config "$CONFIG" \
  --name "$NAME" \
  --seed "$SEED" \
  --dataset datasets/opamp_3stage_fan_smc_openloop_5k \
  --max-train-samples "$NMAX"
echo "End: $(date)"
