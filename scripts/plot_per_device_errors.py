#!/usr/bin/env python
"""Per-device prediction error boxplots for the usingnow reference model
on fan_smc val (1000 samples). Reads the parity arrays already dumped to
section41_parity_fan_smc.npz, reshapes to per-device, and plots |pred - tgt|
distribution per net (V) / per MOSFET (I, gm, gds).

Output: figures/thesis/fig_per_device_errors_usingnow.png  (2x2 grid)
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Patch

REPO = Path('/dss/dsshome1/03/go49jit2/thesis')
NPZ = REPO / 'figures/thesis/section41_parity_fan_smc.npz'
OUT = REPO / 'figures/thesis/fig_per_device_errors_usingnow.png'

# fan_smc stage assignment from netlist
STAGE_OF_MOSFET = {
    0:'bias', 1:'bias', 2:'bias', 3:'bias', 4:'bias', 5:'bias', 6:'bias', 7:'bias',
    8:'S1', 9:'S1', 15:'S1', 16:'S1', 19:'S1', 20:'S1',
    10:'S2', 21:'S2', 22:'S2',
    11:'S3', 23:'S3',
    12:'bias', 13:'bias', 14:'bias', 17:'bias', 18:'bias',
}
STAGE_COLOR = {'bias':'#888888', 'S1':'#1f77b4', 'S2':'#ff7f0e', 'S3':'#d62728'}
STAGE_LABEL = {'bias':'Bias', 'S1':'Stage 1', 'S2':'Stage 2', 'S3':'Stage 3'}

N_SAMPLES = 1000
N_NETS = 14
N_MOSFETS = 24


def box(ax, data, xlabels, ylabel, title, tick_size=10, mosfet_colored=False):
    bp = ax.boxplot(data, showfliers=False, patch_artist=True,
                    medianprops=dict(color='black', linewidth=1.5),
                    whiskerprops=dict(color='black', linewidth=0.6),
                    capprops=dict(color='black', linewidth=0.6))
    if mosfet_colored:
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(STAGE_COLOR[STAGE_OF_MOSFET[i]])
            patch.set_edgecolor('black'); patch.set_linewidth(0.6)
        handles = [Patch(facecolor=STAGE_COLOR[k], edgecolor='black', label=STAGE_LABEL[k])
                   for k in ['bias','S1','S2','S3']]
        ax.legend(handles=handles, loc='upper left', fontsize=10, frameon=True, framealpha=0.95)
    else:
        for patch in bp['boxes']:
            patch.set_facecolor('#9ecae1'); patch.set_edgecolor('black'); patch.set_linewidth(0.6)
    ax.set_xticks(range(1, len(xlabels)+1))
    ax.set_xticklabels(xlabels, fontsize=tick_size, rotation=45, ha='right')
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(True, axis='y', alpha=0.3, linestyle=':')


def main():
    d = np.load(NPZ)

    # --- V error per net (mV) ---
    V_pred = d['V_pred'].reshape(N_SAMPLES, N_NETS)
    V_tgt  = d['V_tgt'].reshape(N_SAMPLES, N_NETS)
    V_err_mv = np.abs(V_pred - V_tgt) * 1000   # (N_samples, N_nets) in mV
    V_per_net = [V_err_mv[:, k] for k in range(N_NETS)]

    # --- I error per MOSFET (log10|I| space).
    # I_pred has 48 entries per sample = 24 (drain) + 24 (source) interleaved (d,s,d,s,...)
    # device_pooling broadcasts so drain == source value, take drain only.
    I_pred = d['I_pred_log10'].reshape(N_SAMPLES, N_MOSFETS, 2)   # [drain, source]
    I_tgt  = d['I_tgt_log10'].reshape(N_SAMPLES, N_MOSFETS, 2)
    I_err = np.abs(I_pred[:, :, 0] - I_tgt[:, :, 0])              # use drain
    I_per_mosfet = [I_err[:, k] for k in range(N_MOSFETS)]

    # --- gm error per MOSFET ---
    gm_pred = d['gm_pred_log10'].reshape(N_SAMPLES, N_MOSFETS)
    gm_tgt  = d['gm_tgt_log10'].reshape(N_SAMPLES, N_MOSFETS)
    gm_err = np.abs(gm_pred - gm_tgt)
    gm_per_mosfet = [gm_err[:, k] for k in range(N_MOSFETS)]

    # --- gds error per MOSFET ---
    gds_pred = d['gds_pred_log10'].reshape(N_SAMPLES, N_MOSFETS)
    gds_tgt  = d['gds_tgt_log10'].reshape(N_SAMPLES, N_MOSFETS)
    gds_err = np.abs(gds_pred - gds_tgt)
    gds_per_mosfet = [gds_err[:, k] for k in range(N_MOSFETS)]

    # ---- 2x2 grid ----
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    box(axes[0, 0], V_per_net, [f'net{i}' for i in range(N_NETS)],
        '|V_pred − V_tgt| (mV)',
        '(a) Per-net voltage prediction error', tick_size=10)
    box(axes[0, 1], I_per_mosfet, [f'M{i}' for i in range(N_MOSFETS)],
        '|log₁₀|I_pred| − log₁₀|I_tgt||',
        '(b) Per-MOSFET current prediction error', tick_size=10, mosfet_colored=True)
    box(axes[1, 0], gm_per_mosfet, [f'M{i}' for i in range(N_MOSFETS)],
        '|log₁₀(g_m,pred) − log₁₀(g_m,tgt)|',
        '(c) Per-MOSFET gm prediction error', tick_size=10, mosfet_colored=True)
    box(axes[1, 1], gds_per_mosfet, [f'M{i}' for i in range(N_MOSFETS)],
        '|log₁₀(g_ds,pred) − log₁₀(g_ds,tgt)|',
        '(d) Per-MOSFET gds prediction error', tick_size=10, mosfet_colored=True)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {OUT}')

    # Print summary: median per-device MAE
    print('\n=== median per-device errors ===')
    print(f"V (mV):  {' '.join(f'{np.median(V_err_mv[:, k]):.2f}' for k in range(N_NETS))}")
    print(f"I (log): {' '.join(f'{np.median(I_err[:, k]):.3f}' for k in range(N_MOSFETS))}")
    print(f"gm log:  {' '.join(f'{np.median(gm_err[:, k]):.3f}' for k in range(N_MOSFETS))}")
    print(f"gds log: {' '.join(f'{np.median(gds_err[:, k]):.3f}' for k in range(N_MOSFETS))}")


if __name__ == '__main__':
    main()
