#!/usr/bin/env python
"""Parity / scatter plots for §4.1 reference model (fan_smc usingnow).

Reads figures/thesis/section41_parity_fan_smc.npz, produces:
  figures/thesis/fig_parity_reference.png  — 4 panels: V, I, gm, gds
DC gain is skipped (the npz dump is from a suspect denorm path; use the
training-log MAE 4.36 dB in the headline table instead).
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

REPO = Path('/dss/dsshome1/03/go49jit2/thesis')
NPZ = REPO / 'figures/thesis/section41_parity_fan_smc.npz'
OUT = REPO / 'figures/thesis/fig_parity_reference.png'


def parity_panel(ax, pred, tgt, *, title, units, mae_str, log_axes=False, alpha=0.06):
    """Scatter pred vs tgt with y=x reference line and MAE annotation."""
    # Subsample if too many points for clean rendering
    n = pred.size
    if n > 40000:
        idx = np.random.default_rng(0).choice(n, 40000, replace=False)
        pred, tgt = pred[idx], tgt[idx]
    ax.scatter(tgt, pred, s=2, alpha=alpha, color='#1f77b4', rasterized=True)
    lo = float(min(tgt.min(), pred.min()))
    hi = float(max(tgt.max(), pred.max()))
    rng = hi - lo
    lo, hi = lo - 0.03 * rng, hi + 0.03 * rng
    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1, alpha=0.7, label='y = x')
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    if log_axes:
        ax.set_xscale('symlog'); ax.set_yscale('symlog')
    ax.set_xlabel(f'SPICE ground truth ({units})', fontsize=10)
    ax.set_ylabel(f'GNN prediction ({units})', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.text(0.04, 0.96, mae_str, transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_aspect('equal')
    # show how many points
    ax.text(0.96, 0.04, f'n = {n:,}', transform=ax.transAxes, fontsize=8,
            ha='right', va='bottom', color='gray')


def main():
    d = np.load(NPZ)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # --- V (Volts) ---
    v_p, v_t = d['V_pred'], d['V_tgt']
    v_mae = np.abs(v_p - v_t).mean() * 1000  # mV
    v_med = np.median(np.abs(v_p - v_t)) * 1000
    parity_panel(axes[0, 0], v_p, v_t,
                 title='(a) Node voltages V (internal nets)',
                 units='V', mae_str=f'MAE = {v_mae:.2f} mV\nmedian = {v_med:.2f} mV')

    # --- I (log10|I|) ---
    i_p, i_t = d['I_pred_log10'], d['I_tgt_log10']
    i_mae = np.abs(i_p - i_t).mean()
    i_med = np.median(np.abs(i_p - i_t))
    i_mae_uA = np.abs(d['I_pred_uA'] - d['I_tgt_uA']).mean()
    parity_panel(axes[0, 1], i_p, i_t,
                 title='(b) Drain currents I (log scale)',
                 units='log₁₀|I| (log A)',
                 mae_str=f'logMAE = {i_mae:.4f}\nmedian = {i_med:.4f}\n→ {i_mae_uA:.2f} µA')

    # --- gm (log10) ---
    g_p, g_t = d['gm_pred_log10'], d['gm_tgt_log10']
    g_mae = np.abs(g_p - g_t).mean()
    g_med = np.median(np.abs(g_p - g_t))
    parity_panel(axes[1, 0], g_p, g_t,
                 title='(c) Small-signal transconductance gm',
                 units='log₁₀(g_m) (log S)',
                 mae_str=f'logMAE = {g_mae:.4f}\nmedian = {g_med:.4f}\n→ {(10**g_mae - 1)*100:.1f}% rel')

    # --- gds (log10) ---
    ds_p, ds_t = d['gds_pred_log10'], d['gds_tgt_log10']
    ds_mae = np.abs(ds_p - ds_t).mean()
    ds_med = np.median(np.abs(ds_p - ds_t))
    parity_panel(axes[1, 1], ds_p, ds_t,
                 title='(d) Small-signal output conductance gds',
                 units='log₁₀(g_ds) (log S)',
                 mae_str=f'logMAE = {ds_mae:.4f}\nmedian = {ds_med:.4f}\n→ {(10**ds_mae - 1)*100:.1f}% rel')

    # --- (e) DC gain (physics formula only) ---
    phys, tgt_dc = d['DC_physics_dB'], d['DC_tgt_dB']
    phys_mae = np.abs(phys - tgt_dc).mean()
    phys_med = np.median(np.abs(phys - tgt_dc))
    parity_panel(axes[0, 2], phys, tgt_dc,
                 title='(e) DC gain — physics formula only',
                 units='dB',
                 mae_str=f'MAE = {phys_mae:.2f} dB\nmedian = {phys_med:.2f} dB\n(no MLP refinement)',
                 alpha=0.4)

    # --- (f) DC gain (MLP-refined head output) ---
    mlp, tgt_dc = d['DC_pred_dB'], d['DC_tgt_dB']
    mlp_mae = np.abs(mlp - tgt_dc).mean()
    mlp_med = np.median(np.abs(mlp - tgt_dc))
    parity_panel(axes[1, 2], mlp, tgt_dc,
                 title='(f) DC gain — MLP-refined head output',
                 units='dB',
                 mae_str=f'MAE = {mlp_mae:.2f} dB\nmedian = {mlp_med:.2f} dB\n(model collapsed near 0)',
                 alpha=0.4)

    fig.suptitle('Reference model — fan_smc, full data (usingnow @ epoch 4022)\n'
                 'GNN prediction vs SPICE ground truth, validation set',
                 fontsize=13, fontweight='bold', y=1.00)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {OUT}')


if __name__ == '__main__':
    main()
