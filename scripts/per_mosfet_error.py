#!/usr/bin/env python
"""Per-MOSFET-position error breakdown for usingnow on fan_smc val.

Reuses extraction code from tsne_usingnow_embeddings (no t-SNE — just dumps tables).
For each MOSFET position M0..M23, compute:
  - NMOS/PMOS, region distribution, V_ov range
  - V error at drain net (mV)
  - I error  (|log10 I_pred − log10 I_gt|)
  - g_m error
  - g_ds error
Then rank by each error.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tsne_usingnow_embeddings import load_model_and_run

# fan_smc has 24 MOSFETs. From the standard fan_smc netlist (verify below).
ROLE_NAME = {0:'bias_mirror', 1:'diff_pair', 2:'stage1_load', 3:'stage2', 4:'output'}

def main():
    d = load_model_and_run()
    midx = d['midx']
    print(f"\nTotal nodes: {len(midx)}  (24 MOSFETs × ~1000 samples)\n")
    print(f"{'M':<4}{'role':<14}{'NMOS?':<7}{'n':>6}  | {'V_ov_med (mV)':>14} {'V_DS_med (mV)':>14}  | {'V err (mV)':>11} {'I err (log)':>12} {'gm err':>9} {'gds err':>9}  | regions cut/tri/sat")
    print('-' * 150)
    rows = []
    for m in range(24):
        mask = midx == m
        if not mask.any():
            continue
        is_nmos = d['is_nmos'][mask].mean() > 0.5
        role = int(np.bincount(d['role'][mask]).argmax())
        vov_med = np.median(d['vov'][mask]) * 1000
        vds_med = np.median(d['vds'][mask]) * 1000
        v_err = np.mean(d['v_err_mV'][mask])
        i_err = np.mean(d['i_err_log'][mask])
        gm_err = np.mean(d['gm_err_log'][mask])
        gds_err = np.mean(d['gds_err_log'][mask])
        regs = np.bincount(d['region'][mask], minlength=3)
        regs_str = f"{regs[0]:4d}/{regs[1]:4d}/{regs[2]:4d}"
        rows.append((m, role, is_nmos, mask.sum(), vov_med, vds_med, v_err, i_err, gm_err, gds_err, regs_str))
        print(f"M{m:<3}{ROLE_NAME[role]:<14}{'NMOS' if is_nmos else 'PMOS':<7}{int(mask.sum()):>6}  | "
              f"{vov_med:>14.0f} {vds_med:>14.0f}  | "
              f"{v_err:>11.2f} {i_err:>12.4f} {gm_err:>9.4f} {gds_err:>9.4f}  | {regs_str}")

    print(f"\n=== Top-5 by each error metric ===")
    for col, name in [(7,'V err mV'), (8,'I err log10'), (9,'gm err log10'), (10,'gds err log10')]:
        print(f"\n  {name}:")
        top = sorted(rows, key=lambda r: -r[col-1])[:5]
        for r in top:
            print(f"    M{r[0]:<3} {ROLE_NAME[r[1]]:<14} {'NMOS' if r[2] else 'PMOS':<7} V_ov={r[4]:.0f}mV  → {r[col-1]:.4f}")

    print(f"\n=== Bottom-5 by each error metric (cleanest) ===")
    for col, name in [(7,'V err mV'), (8,'I err log10'), (9,'gm err log10'), (10,'gds err log10')]:
        print(f"\n  {name}:")
        bot = sorted(rows, key=lambda r: r[col-1])[:5]
        for r in bot:
            print(f"    M{r[0]:<3} {ROLE_NAME[r[1]]:<14} {'NMOS' if r[2] else 'PMOS':<7} V_ov={r[4]:.0f}mV  → {r[col-1]:.4f}")


if __name__ == '__main__':
    main()
