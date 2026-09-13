#!/bin/bash
#SBATCH --job-name=patch_open
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH -o /dss/dsshome1/03/go49jit2/thesis/slurm_logs/%x_%j.out
#SBATCH -e /dss/dsshome1/03/go49jit2/thesis/slurm_logs/%x_%j.err

cd /dss/dsshome1/03/go49jit2/thesis
export PATH=/dss/dsshome1/03/go49jit2/.local/bin:$PATH
echo "Start: $(date)"

srun python scripts/patch_dataset_mosfet_wl.py \
    --dataset-dir datasets/opamp_3stage_fan_smc_openloop_5k \
    --config configs/opamp_dataset/opamp_3stage_fan_smc_wide_5k_v7_nofil.yaml

echo "End: $(date)"
