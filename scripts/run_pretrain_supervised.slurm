#!/bin/bash
#SBATCH --job-name=gnn_suppre
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH -o /dss/dsshome1/03/go49jit2/thesis/slurm_logs/%x_%j.out
#SBATCH -e /dss/dsshome1/03/go49jit2/thesis/slurm_logs/%x_%j.err

CONFIG=$1
NAME=$2
shift 2
EXTRA="$@"

cd /dss/dsshome1/03/go49jit2/thesis
export PATH=/dss/dsshome1/03/go49jit2/.local/bin:$PATH
echo "Job: $SLURM_JOB_ID | Config: $CONFIG | Name: $NAME | Extra: $EXTRA"
echo "Start: $(date)"

srun python -u scripts/pretrain_supervised_v1.py --config "$CONFIG" --name "$NAME" $EXTRA

echo "End: $(date)"
