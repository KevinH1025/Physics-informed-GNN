#!/bin/bash
# Polling auto-submitter — pushes missing seed cells into slurm as queue space frees up.
# Walltime is chosen per N so we don't over-request:
#   N≤250  → 1h walltime (bs=64)
#   N=500  → 2h walltime (bs=64)
#   N=1000 → 3h walltime (bs=64)  — bumped after leungnr_ft hit 2h cap
#   N=2500 → 2h walltime (bs=256)
#   N=4000 → 3h walltime (bs=256)
set -uo pipefail
cd /dss/dsshome1/03/go49jit2/thesis
EXP=datasets/opamp_3stage_pretrain_combined_5topo/experiments
MAX_ITERS=200

submit() {
  local topo=$1 N=$2 method=$3 seed=$4
  local alias=$topo; [ "$topo" = "fan_smc" ] && alias=fansmc
  local name="v5_5topo_seedval_${method}_${alias}_n${N}_s${seed}"
  local config init_from
  if [ "$method" = "scratch" ]; then
    if [ "$N" = "4000" ]; then
      config="${EXP}/v5_5topo_pertopo_${topo}/original_config.yaml"
    else
      config="${EXP}/v5_5topo_pertopo_${topo}_n${N}/original_config.yaml"
    fi
    init_from=""
  else
    config="${EXP}/v5_5topo_ftzeroshot_${alias}_n${N}_v2/original_config.yaml"
    init_from="${EXP}/v5_5topo_zeroshot_${alias}/best.pt"
  fi
  [ -d "${EXP}/${name}" ] && return 2  # already exists, skip

  # Walltime-tiered script choice
  local script
  if [ "$N" -le 250 ]; then
    script=scripts/run_seed_valley_1h.slurm        # 1h, bs=64
  elif [ "$N" -eq 500 ]; then
    script=scripts/run_seed_valley.slurm           # 2h, bs=64
  elif [ "$N" -eq 1000 ]; then
    script=scripts/run_seed_valley_3h.slurm        # 3h, bs=64 (bumped from 2h after timeouts)
  elif [ "$N" -eq 2500 ]; then
    script=scripts/run_seed_valley_bs256.slurm     # 2h, bs=256 (verified sufficient)
  else                                              # N=4000
    script=scripts/run_seed_valley_bs256_3h.slurm  # 3h, bs=256 (N=4000 needs more)
  fi

  local out
  out=$(sbatch --parsable --job-name="all_${alias:0:5}_${method:0:3}_n${N}_s${seed}" \
    "$script" "$config" "$name" "$seed" "$topo" "$N" "$init_from" 2>&1)
  echo "$out" | grep -qE "^[0-9]+$" && return 0 || return 1
}

for iter in $(seq 1 $MAX_ITERS); do
  remaining=0
  submitted=0
  rejected=0
  for topo in fan_smc sau_cfcc peng_tcfc leung_nmcf leung_nmcnr; do
    for method in scratch ft; do
      for N in 100 250 500 1000 2500 4000; do
        for seed in 42 43 44; do
          submit "$topo" "$N" "$method" "$seed"
          rc=$?
          case $rc in
            0) submitted=$((submitted+1)) ;;
            1) rejected=$((rejected+1)); remaining=$((remaining+1)) ;;
            2) ;;
          esac
        done
      done
    done
  done
  echo "[iter $iter] submitted=$submitted rejected_full=$rejected remaining=$remaining $(date +'%H:%M:%S')"
  [ "$remaining" -eq 0 ] && { echo "ALL SUBMITTED"; exit 0; }
  sleep 300
done
echo "MAX_ITERS reached, $remaining cells still not submitted"
