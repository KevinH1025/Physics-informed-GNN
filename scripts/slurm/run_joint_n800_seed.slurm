#!/bin/bash
# Joint 5-topology training with 800 samples per topology (= 4000 total),
# batch = 64 (same as the single-topo seed_valley), lr = 3e-3, 5000 epochs.
# Val loader: PretrainCombinedLoader on dataset_val.pkl with ALL 5 topos
# (per_topo_v_mae is computed every epoch across all 5 in the training log).
#SBATCH --job-name=joint_n800
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

CONFIG=$1   # v5_5topo_joint_n800/original_config.yaml
NAME=$2     # v5_5topo_joint_n800_s<seed>
SEED=$3

# bs=250 (50 per topo, divisible by 5) — closest analogue to single-topo
# seed_valley bs=64. lr sqrt-scaled 3e-3×√(250/64) ≈ 6e-3. 16 batches/epoch
# × 5000 = 80k gradient updates — should reach baseline territory.
echo "Start: $(date)  name=$NAME  seed=$SEED  N_per_topo=800  bs=250  lr=6e-3  epochs=5000"
srun /usr/bin/python3 -u scripts/pretrain_supervised_v1.py \
  --config "$CONFIG" \
  --name "$NAME" \
  --seed "$SEED" \
  --lr 0.006 \
  --warmup-epochs 50 \
  --epochs 5000 \
  --batch-size 250 \
  --max-per-topo 800 \
  --dataset-dir datasets/opamp_3stage_pretrain_combined_5topo
echo "End: $(date)"
