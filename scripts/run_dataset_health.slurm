#!/bin/bash
#SBATCH --job-name=health_check
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=00:30:00
#SBATCH -o /dss/dsshome1/03/go49jit2/thesis/slurm_logs/%x_%j.out
#SBATCH -e /dss/dsshome1/03/go49jit2/thesis/slurm_logs/%x_%j.err

source ~/envs/thesis/bin/activate
cd /dss/dsshome1/03/go49jit2/thesis
python -u scripts/scan_dataset_health.py
