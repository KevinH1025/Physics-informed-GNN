#!/bin/bash
#SBATCH --job-name=topo_stats
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --output=/dss/dsshome1/03/go49jit2/thesis/slurm_logs/topo_stats_%j.out
#SBATCH --error=/dss/dsshome1/03/go49jit2/thesis/slurm_logs/topo_stats_%j.err

source ~/envs/thesis/bin/activate
cd /dss/dsshome1/03/go49jit2/thesis
python -u scripts/topo_stats_and_dists.py
