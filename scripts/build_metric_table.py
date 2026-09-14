#!/usr/bin/env python
"""Run eval_per_topo on all relevant N=4000 checkpoints and assemble a master metric table.

Output: figures/thesis/metric_table.json + a printed markdown table covering:
  - methods: scratch (per-topo), ftzeroshot (per-topo), joint
  - topologies: 5
  - metrics: V MAE (mV), I MAE (uA), gm log-MAE, gds log-MAE, UGBW log-MAE (dec), PM MAE (deg), AM MAE (dB)

Usage:
    python scripts/build_metric_table.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
EXP = REPO / 'datasets/opamp_3stage_pretrain_combined_5topo/experiments'
OUT_DIR = REPO / 'figures/thesis'
OUT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR = OUT_DIR / 'eval_tmp'
TMP_DIR.mkdir(parents=True, exist_ok=True)

TOPOS = ['fan_smc', 'sau_cfcc', 'peng_tcfc', 'leung_nmcf', 'leung_nmcnr']

# Method registry: name → list of (topology, exp_name) at full data (N=4000)
METHODS = {
    'scratch': [(t, f'v5_5topo_pertopo_{t}') for t in TOPOS],
    'ftzeroshot': [(t, f'v5_5topo_ftzeroshot_{("fansmc" if t=="fan_smc" else t)}_n4000_v2')
                   for t in TOPOS],
    'joint': [('all', 'v5_5topo_joint')],
}


def run_eval(exp_name: str, json_out: Path) -> dict | None:
    if not (EXP / exp_name).exists():
        print(f'  [skip] {exp_name} not found')
        return None
    print(f'  running eval on {exp_name}...', flush=True)
    cmd = [sys.executable, str(REPO / 'scripts/analysis/eval_per_topo.py'),
           exp_name, '--json-out', str(json_out)]
    res = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
    if res.returncode != 0:
        print(f'  [error] {exp_name}:\n{res.stderr[-500:]}')
        return None
    if not json_out.exists():
        print(f'  [error] {exp_name}: no JSON output')
        return None
    with open(json_out) as f:
        return json.load(f)


def main():
    # Collect: method → topo → metrics dict
    table = {m: {} for m in METHODS}
    for method, runs in METHODS.items():
        print(f'\n=== {method} ===')
        for topo, exp in runs:
            jp = TMP_DIR / f'{method}_{topo}_{exp}.json'
            if jp.exists():
                with open(jp) as f:
                    d = json.load(f)
            else:
                d = run_eval(exp, jp)
                if d is None:
                    continue
            metrics_per_topo = d['metrics']
            if topo == 'all':
                # joint run produces metrics for all topos
                for t, m in metrics_per_topo.items():
                    if t in TOPOS:
                        table[method][t] = m
            else:
                # Single-topo run — pick the matching topo's metrics
                if topo in metrics_per_topo:
                    table[method][topo] = metrics_per_topo[topo]
                else:
                    # Some single-topo runs might list all topos (with most empty); take whichever has data
                    for t, m in metrics_per_topo.items():
                        if t == topo:
                            table[method][topo] = m

    # Print master table per metric
    metric_keys = [
        ('v_mae_mV', 'V MAE (mV)', '{:.2f}'),
        ('i_mae_uA', 'I MAE (µA)', '{:.2f}'),
        ('gm_log_mae', 'gm log-MAE', '{:.4f}'),
        ('gds_log_mae', 'gds log-MAE', '{:.4f}'),
        ('ugbw_log_mae_dec', 'UGBW log-MAE (dec)', '{:.3f}'),
        ('pm_mae_deg', 'PM MAE (deg)', '{:.2f}'),
        ('am_mae_db', 'AM MAE (dB)', '{:.2f}'),
    ]
    print('\n\n========== MASTER METRIC TABLE (full data, N=4000) ==========\n')
    method_order = ['scratch', 'ftzeroshot', 'joint']
    for key, label, fmt in metric_keys:
        print(f'### {label}\n')
        header = '| topology | ' + ' | '.join(method_order) + ' |'
        sep = '|' + '----------|' * (len(method_order) + 1)
        print(header)
        print(sep)
        for t in TOPOS:
            row = [t]
            for m in method_order:
                v = table.get(m, {}).get(t, {}).get(key)
                if v is None or (isinstance(v, float) and (v != v)):
                    row.append('-')
                else:
                    row.append(fmt.format(v))
            print('| ' + ' | '.join(row) + ' |')
        print()

    out_json = OUT_DIR / 'metric_table.json'
    with open(out_json, 'w') as f:
        json.dump(table, f, indent=2)
    print(f'\nSaved master table → {out_json}')


if __name__ == '__main__':
    main()
