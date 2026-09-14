#!/usr/bin/env python
"""Evaluate I MAE (µA) and V MAE (mV) for the 4 KCL placements (col1/2/3/4)
on fan_smc and sau_cfcc at N=500. Uses eval_one from eval_all_checkpoints."""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from circuitgnn.evaluation import eval_one, EXP

CELLS = {
    'col1_KCLpre+KCLft':  {'fan_smc': 'v5_5topo_ftzeroshot_fansmc_n500_v2',
                           'sau_cfcc': 'v5_5topo_ftzeroshot_sau_cfcc_n500_v2'},
    'col2_KCLpre+noFT':   {'fan_smc': 'v5_5topo_ftzs_fansmc_phys_nophysFT_n500',
                           'sau_cfcc': 'v5_5topo_ftzs_sau_phys_nophysFT_n500'},
    'col3_no_KCL_either': {'fan_smc': 'v5_5topo_ftzs_fansmc_nophys_nophysFT_n500',
                           'sau_cfcc': 'v5_5topo_ftzs_sau_nophys_nophysFT_n500'},
    'col4_nopre+KCLft':   {'fan_smc': 'v5_5topo_ftzs_fansmc_nophysPRE_kclFT_n500',
                           'sau_cfcc': 'v5_5topo_ftzs_sau_nophysPRE_kclFT_n500'},
}

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[setup] device={device}')
    out = {}
    for cell, topo_dir in CELLS.items():
        out[cell] = {}
        for topo, run in topo_dir.items():
            path = EXP / run
            if not (path / 'best.pt').exists():
                print(f'  MISSING {cell}/{topo}: {path}'); out[cell][topo] = None; continue
            print(f'  eval {cell}/{topo}: {run}', flush=True)
            try:
                m = eval_one(path, device, topo_filter=topo)
                out[cell][topo] = m.get(topo)
            except Exception as e:
                print(f'    ERROR: {e}'); out[cell][topo] = None
    print('\n=== I MAE (µA) at N=500, 4 KCL placements ===')
    print(f"{'cell':<22} | {'fan_smc':>12} | {'sau_cfcc':>12}")
    for cell in CELLS:
        f = out[cell].get('fan_smc'); s = out[cell].get('sau_cfcc')
        def fmt(m, k): return f'{m[k]:.2f}' if (m and m.get(k) is not None) else 'NA'
        print(f"{cell:<22} | {fmt(f,'i_mae_uA'):>12} | {fmt(s,'i_mae_uA'):>12}")
    print('\n=== V MAE (mV) at N=500 — for cross-check ===')
    print(f"{'cell':<22} | {'fan_smc':>12} | {'sau_cfcc':>12}")
    for cell in CELLS:
        f = out[cell].get('fan_smc'); s = out[cell].get('sau_cfcc')
        def fmt(m, k): return f'{m[k]:.2f}' if (m and m.get(k) is not None) else 'NA'
        print(f"{cell:<22} | {fmt(f,'v_mae_mV'):>12} | {fmt(s,'v_mae_mV'):>12}")
    json.dump(out, open(Path(__file__).resolve().parents[2] / 'figures/thesis/cell_4_current.json', 'w'), indent=2)

if __name__ == '__main__':
    main()
