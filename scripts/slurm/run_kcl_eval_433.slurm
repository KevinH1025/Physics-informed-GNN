#!/bin/bash
#SBATCH --job-name=kcl_eval_433
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --output=/dss/dsshome1/03/go49jit2/thesis/slurm_logs/kcl_eval_433_%j.out
#SBATCH --error=/dss/dsshome1/03/go49jit2/thesis/slurm_logs/kcl_eval_433_%j.err

source ~/envs/thesis/bin/activate 2>/dev/null
cd /dss/dsshome1/03/go49jit2/thesis
/usr/bin/python3 -u scripts/analysis/eval_kcl_residual_section433.py
