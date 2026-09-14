#!/bin/bash
#SBATCH --job-name=gnn_eval
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH -o /dss/dsshome1/03/go49jit2/thesis/slurm_logs/%x_%j.out
#SBATCH -e /dss/dsshome1/03/go49jit2/thesis/slurm_logs/%x_%j.err

cd /dss/dsshome1/03/go49jit2/thesis
export PATH=/dss/dsshome1/03/go49jit2/.local/bin:$PATH

# Eval all key checkpoints for the multi-topology comparison
EXPS=(
  v5_5topo_pertopo_sau_cfcc
  v5_5topo_pertopo_peng_tcfc
  v5_5topo_pertopo_leung_nmcf
  v5_5topo_pertopo_leung_nmcnr
  v5_5topo_pertopo_fan_smc
  v5_5topo_joint
  v5_5topo_joint_n100
  v5_5topo_joint_n250
  v5_5topo_joint_n500
  v5_5topo_joint_n800
  v5_5topo_joint_n1000
  v5_5topo_pertopo_fansmc_n100
  v5_5topo_pertopo_fansmc_n250
  v5_5topo_pertopo_fansmc_n500
  v5_5topo_pertopo_fansmc_n1000
  v5_5topo_pertopo_fansmc_n2500
  v5_5topo_zeroshot_fansmc
)

for exp in "${EXPS[@]}"; do
  echo ""
  echo "========================================================================"
  srun python scripts/analysis/eval_per_topo.py "$exp"
done
