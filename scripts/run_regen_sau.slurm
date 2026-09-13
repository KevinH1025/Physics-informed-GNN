#!/bin/bash
#SBATCH --job-name=regen_sau
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH -o /dss/dsshome1/03/go49jit2/thesis/slurm_logs/%x_%j.out
#SBATCH -e /dss/dsshome1/03/go49jit2/thesis/slurm_logs/%x_%j.err

cd /dss/dsshome1/03/go49jit2/thesis
export PATH=/dss/dsshome1/03/go49jit2/envs/thesis/bin:$PATH
# Use venv python directly (the env has no activate script)
PY=/dss/dsshome1/03/go49jit2/envs/thesis/bin/python

# Generate 5500 fresh sau_cfcc samples — enough buffer for the rail-margin
# filter rejection AND any V<0 cases that slip through.
mkdir -p datasets/opamp_3stage_sau_cfcc_5k_clean
$PY scripts/generate_dataset.py \
  --config configs/opamp_dataset/opamp_3stage_sau_cfcc_5k.yaml \
  --output-dir datasets/opamp_3stage_sau_cfcc_5k_clean \
  --num-samples 5500
