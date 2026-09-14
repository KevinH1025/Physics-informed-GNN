#!/bin/bash
# Submit the seed-variance experiment for the §4.4.2 mid-N valley check.
# 32 jobs total (peng_tcfc 12 + fan_smc 8 + sau_cfcc 12).

set -euo pipefail
cd /dss/dsshome1/03/go49jit2/thesis

EXP_DIR=datasets/opamp_3stage_pretrain_combined_5topo/experiments

# Helper: alias for fan_smc (config dirs use 'fansmc', not 'fan_smc')
alias_topo() {
  case "$1" in
    fan_smc) echo fansmc ;;
    *) echo "$1" ;;
  esac
}

# Track submitted jobs for monitoring
submitted_log=/dss/dsshome1/03/go49jit2/thesis/slurm_logs/seed_valley_submitted_$(date +%s).txt
> "$submitted_log"

submit() {
  local topo=$1 N=$2 method=$3 seed=$4
  local alias=$(alias_topo "$topo")
  local name="v5_5topo_seedval_${method}_${alias}_n${N}_s${seed}"
  local config init_from

  if [ "$method" = "scratch" ]; then
    config="${EXP_DIR}/v5_5topo_pertopo_${alias}_n${N}/original_config.yaml"
    [ -f "$config" ] || config="${EXP_DIR}/v5_5topo_pertopo_${topo}_n${N}/original_config.yaml"
    init_from=""
  else
    config="${EXP_DIR}/v5_5topo_ftzeroshot_${alias}_n${N}_v2/original_config.yaml"
    [ -f "$config" ] || config="${EXP_DIR}/v5_5topo_ftzeroshot_${topo}_n${N}_v2/original_config.yaml"
    init_from="${EXP_DIR}/v5_5topo_zeroshot_${alias}/best.pt"
    [ -f "$init_from" ] || init_from="${EXP_DIR}/v5_5topo_zeroshot_${topo}/best.pt"
  fi

  if [ ! -f "$config" ]; then
    echo "SKIP (no config): $name"
    return
  fi
  if [ "$method" = "ft" ] && [ ! -f "$init_from" ]; then
    echo "SKIP (no init checkpoint): $name"
    return
  fi
  if [ -d "${EXP_DIR}/${name}" ]; then
    echo "SKIP (run already exists): $name"
    return
  fi

  local jobid
  jobid=$(sbatch --parsable --job-name="sv_${alias:0:4}_${method}_n${N}_s${seed}" \
                 --time=2:00:00 --mem=80G \
                 scripts/slurm/run_seed_valley.slurm "$config" "$name" "$seed" "$topo" "$N" "$init_from")
  echo "$jobid  $name  config=$(basename $(dirname $config))" | tee -a "$submitted_log"
}

# === PRIORITY: peng_tcfc (12 jobs) ===
for seed in 42 43 44; do
  for N in 500 1000; do
    submit peng_tcfc "$N" scratch "$seed"
    submit peng_tcfc "$N" ft      "$seed"
  done
done

# === SECONDARY 1: fan_smc (8 jobs — scratch ×6 + FT ×2 since s43/s44 FT already exist) ===
for seed in 42 43 44; do
  for N in 500 1000; do
    submit fan_smc "$N" scratch "$seed"
  done
done
# fan_smc FT seeds: s43/s44 exist, only need s42
for N in 500 1000; do
  submit fan_smc "$N" ft 42
done

# === SECONDARY 2: sau_cfcc (12 jobs) ===
for seed in 42 43 44; do
  for N in 500 1000; do
    submit sau_cfcc "$N" scratch "$seed"
    submit sau_cfcc "$N" ft      "$seed"
  done
done

echo
echo "Submitted jobs log: $submitted_log"
echo "Total submitted: $(wc -l < "$submitted_log")"
