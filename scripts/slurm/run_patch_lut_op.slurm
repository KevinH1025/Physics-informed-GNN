#!/bin/bash
#SBATCH --job-name=patch_lut_op
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH -o /dss/dsshome1/03/go49jit2/thesis/slurm_logs/%x_%j.out
#SBATCH -e /dss/dsshome1/03/go49jit2/thesis/slurm_logs/%x_%j.err

cd /dss/dsshome1/03/go49jit2/thesis
export PATH=/dss/dsshome1/03/go49jit2/.local/bin:$PATH

echo "Job: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | GPU: $CUDA_VISIBLE_DEVICES"
echo "Start: $(date)"

srun python scripts/migrations/patch_dataset_lut_op_features.py \
    --dataset-dir datasets/opamp_3stage_fan_smc_v9_5k_nofil \
    --pass1-ckpt datasets/opamp_3stage_fan_smc_v9_5k_nofil/experiments/ginbn_3layer_bb8_notower_vn_kcl_w10_v9_5k/best_model.pt \
    --pass1-config datasets/opamp_3stage_fan_smc_v9_5k_nofil/experiments/ginbn_3layer_bb8_notower_vn_kcl_w10_v9_5k/original_config.yaml \
    --lut datasets/lut/lut_v2/sky130_mosfet_lut_v2_id_gm_gds.h5

echo "End: $(date)"
