#!/bin/bash
# Canonical ablation-training launcher. The walltime clones this file used to
# have (run_abl_{45min,1h,1h15,1h30h,1h45,2h,3h,long}.slurm) are gone: override
# the defaults below from the sbatch command line instead, e.g.
#
#   sbatch --time=1:15:00 --mem=48G scripts/slurm/run_abl.slurm <config> <name>
#
#SBATCH --job-name=gnn_abl
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH -o /dss/dsshome1/03/go49jit2/thesis/slurm_logs/%x_%j.out
#SBATCH -e /dss/dsshome1/03/go49jit2/thesis/slurm_logs/%x_%j.err

CONFIG=$1
NAME=$2
shift 2
EXTRA_ARGS="$@"

if [ -z "$CONFIG" ] || [ -z "$NAME" ]; then
    echo "Usage: sbatch scripts/slurm/run_abl.slurm <config> <name> [extra args...]"
    exit 1
fi

cd /dss/dsshome1/03/go49jit2/thesis
export PATH=/dss/dsshome1/03/go49jit2/.local/bin:$PATH

echo "Job: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | GPU: $CUDA_VISIBLE_DEVICES"
echo "Config: $CONFIG | Name: $NAME | Extra: $EXTRA_ARGS"
echo "Start: $(date)"

srun python scripts/train_v3.py --config "$CONFIG" --name "$NAME" $EXTRA_ARGS

echo "End: $(date)"
