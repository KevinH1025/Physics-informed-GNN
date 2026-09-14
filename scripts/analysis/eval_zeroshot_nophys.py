#!/usr/bin/env python
"""Test whether KCL-on pretrain is what makes I MAE the most-transferable quantity.

Eval same per-topo V/I/gm/gds on:
  (a) 5 nophys pretrains (no KCL, no loop attention — legacy nophys recipe)
  (b) 2 disentangle pretrains (fan_smc held-out, both with loop ON):
      A: KCL on
      B: KCL off
Compare against the phys-pretrain numbers we already have.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.analysis.eval_zeroshot_held_out import evaluate, EXP

import torch

OUT = Path(__file__).resolve().parents[2] / 'figures/thesis/section442_zeroshot_kcl_ablation.json'

# Baselines (same-topo scratch) for ratio computation
BASELINE = {
    'fan_smc':     {'V': 14.13, 'I': 9.43,  'gm': 0.0359, 'gds': 0.0555},
    'sau_cfcc':    {'V': 26.79, 'I': 26.57, 'gm': 0.0625, 'gds': 0.0812},
    'peng_tcfc':   {'V': 12.93, 'I': 4.91,  'gm': 0.0237, 'gds': 0.0435},
    'leung_nmcf':  {'V': 24.27, 'I': 10.27, 'gm': 0.0570, 'gds': 0.0885},
    'leung_nmcnr': {'V': 35.65, 'I': 16.96, 'gm': 0.0819, 'gds': 0.1161},
}

# (label, exp_dir, held_out_topology)
RUNS = [
    # Legacy nophys (kcl=0, loop=false) - the confounded recipe
    ('nophys_fansmc',      EXP / 'v5_5topo_zeroshot_fansmc_nophys',      'fan_smc'),
    ('nophys_sau',         EXP / 'v5_5topo_zeroshot_sau_cfcc_nophys',    'sau_cfcc'),
    ('nophys_peng',        EXP / 'v5_5topo_zeroshot_peng_tcfc_nophys',   'peng_tcfc'),
    ('nophys_leungf',      EXP / 'v5_5topo_zeroshot_leung_nmcf_nophys',  'leung_nmcf'),
    ('nophys_leungnr',     EXP / 'v5_5topo_zeroshot_leung_nmcnr_nophys', 'leung_nmcnr'),
    # Disentangle: loop ON for both, KCL is the only difference
    ('disent_A_phys_loopON',   EXP / 'v5_5topo_physdis_A_phys_fansmcHO',         'fan_smc'),
    ('disent_B_nophys_loopON', EXP / 'v5_5topo_physdis_B_nophys_looON_fansmcHO', 'fan_smc'),
]

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[setup] device={device}\n')
    results = {}
    for label, exp_dir, topo in RUNS:
        if not exp_dir.exists():
            print(f'[skip] {label}: {exp_dir.name} missing')
            continue
        print(f'[eval] {label}  topo={topo}  dir={exp_dir.name}')
        m = evaluate(exp_dir, topo, device)
        b = BASELINE[topo]
        ratios = {'V': m['V_mae_mV']/b['V'], 'I': m['I_mae_uA']/b['I'],
                  'gm': m['gm_logMAE']/b['gm'], 'gds': m['gds_logMAE']/b['gds']}
        results[label] = {**m, 'topo': topo, 'baseline': b, 'ratios': ratios}
        print(f'  V  ={m["V_mae_mV"]:7.2f} mV  (ratio {ratios["V"]:.2f}x)')
        print(f'  I  ={m["I_mae_uA"]:7.2f} µA  (ratio {ratios["I"]:.2f}x)')
        print(f'  gm =       {m["gm_logMAE"]:.4f}  (ratio {ratios["gm"]:.2f}x)')
        print(f'  gds=       {m["gds_logMAE"]:.4f}  (ratio {ratios["gds"]:.2f}x)\n')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved → {OUT}')

    print('\n' + '=' * 100)
    print('Per-quantity degradation RATIO comparison (zs / baseline)')
    print('=' * 100)
    print(f'{"label":<28}{"topo":<14}{"V":>7}{"I":>7}{"gm":>7}{"gds":>7}')
    print('-' * 100)
    # Show the canonical (already-evaluated) phys runs from earlier JSON for direct comparison
    phys_path = Path(__file__).resolve().parents[2] / 'figures/thesis/section442_zeroshot_per_topo.json'
    if phys_path.exists():
        phys = json.load(open(phys_path))
        for topo in ('fan_smc','sau_cfcc','peng_tcfc','leung_nmcf','leung_nmcnr'):
            if topo in phys:
                m = phys[topo]; b = m['baseline']
                print(f"{'PHYS  zs_'+topo[:8]:<28}{topo:<14}"
                      f"{m['V_mae_mV']/b['V']:>7.2f}{m['I_mae_uA']/b['I']:>7.2f}"
                      f"{m['gm_logMAE']/b['gm']:>7.2f}{m['gds_logMAE']/b['gds']:>7.2f}")
        print('-' * 100)
    for label, _, topo in RUNS:
        if label not in results: continue
        r = results[label]['ratios']
        print(f"{label:<28}{topo:<14}{r['V']:>7.2f}{r['I']:>7.2f}{r['gm']:>7.2f}{r['gds']:>7.2f}")


if __name__ == '__main__':
    main()
