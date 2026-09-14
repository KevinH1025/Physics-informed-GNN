#!/usr/bin/env python
"""5-panel parity grid (3 + 2 layout) for §4.1 reference model.
Row 1: V, I, gm
Row 2: gds, DC gain
DC panel uses all 1000 samples (valid + invalid color-coded).
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NPZ_PARITY = REPO / 'figures/thesis/section41_parity_fan_smc.npz'
NPZ_DC = REPO / 'figures/thesis/section41_dc_all_samples.npz'
OUT = REPO / 'figures/thesis/fig_parity_grid.png'


def parity_panel(ax, pred, tgt, *, title, units, mae_str, color='#1f77b4', valid=None,
                 overlay=None):
    """overlay: optional dict {pred, tgt, label, color, s, alpha} for a second series."""
    n = pred.size
    if n > 40000:
        idx = np.random.default_rng(0).choice(n, 40000, replace=False)
        pred, tgt = pred[idx], tgt[idx]
        if valid is not None: valid = valid[idx]
    if n <= 2000:        s, a = 22, 0.7
    elif n <= 15000:     s, a = 12, 0.4
    elif n <= 30000:     s, a = 7,  0.25
    else:                s, a = 5,  0.18
    main_lbl = 'MOSFETs' if overlay is not None else None
    ax.scatter(tgt, pred, s=s, alpha=a, color=color, rasterized=True,
               edgecolors='none', label=main_lbl)
    if overlay is not None:
        op, ot = overlay['pred'], overlay['tgt']
        ax.scatter(ot, op, s=overlay.get('s', 30), alpha=overlay.get('alpha', 0.7),
                   color=overlay.get('color', '#d62728'), rasterized=True,
                   edgecolors='none', label=overlay.get('label'))
    lo = float(min(tgt.min(), pred.min())); hi = float(max(tgt.max(), pred.max()))
    rng = hi - lo
    lo, hi = lo - 0.03 * rng, hi + 0.03 * rng
    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1, alpha=0.7)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(f'SPICE ground truth ({units})', fontsize=12)
    ax.set_ylabel(f'GNN prediction ({units})', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.text(0.04, 0.96, mae_str, transform=ax.transAxes, fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_aspect('equal')
    # Sample count in lower-right corner of every panel
    ax.text(0.96, 0.04, f'N = {n:,}', transform=ax.transAxes, fontsize=10,
            ha='right', va='bottom', color='dimgray',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='lightgray'))
    if overlay is not None:
        ax.legend(loc='lower right', fontsize=10, frameon=True, framealpha=0.9)


def main():
    d = np.load(NPZ_PARITY)
    dc = np.load(NPZ_DC)

    # 3+2 layout sized for A4 (figure rendered at the actual printed dimensions
    # so the embedded fonts don't get scaled down further by the document).
    # Target: full A4 page text width ≈ 16 cm ≈ 6.3", scaled up to give panels
    # breathing room. Use figsize that will scale gracefully to A4.
    fig = plt.figure(figsize=(14, 9.5), constrained_layout=False)
    gs = fig.add_gridspec(2, 6, hspace=0.55, wspace=0.95,
                          left=0.06, right=0.98, top=0.96, bottom=0.07)
    ax_V   = fig.add_subplot(gs[0, 0:2])
    ax_I   = fig.add_subplot(gs[0, 2:4])
    ax_gm  = fig.add_subplot(gs[0, 4:6])
    ax_gds = fig.add_subplot(gs[1, 1:3])
    ax_DC  = fig.add_subplot(gs[1, 3:5])

    # (a) V
    v_p, v_t = d['V_pred'], d['V_tgt']
    v_mae = np.abs(v_p - v_t).mean() * 1000
    v_med = np.median(np.abs(v_p - v_t)) * 1000
    parity_panel(ax_V, v_p, v_t,
                 title='(a) Node voltages V (internal nets)',
                 units='V',
                 mae_str=f'MAE = {v_mae:.2f} mV\nmedian = {v_med:.2f} mV')

    # (b) I in log10 space — usingnow uses device_pooling so each MOSFET has ONE
    # prediction broadcast to BOTH drain and source terminals. Dedupe to per-MOSFET
    # by taking every other entry (drain only).
    i_p = d['I_pred_log10'][::2]   # 48000 → 24000 (drain values only)
    i_t = d['I_tgt_log10'][::2]
    i_mae = np.abs(i_p - i_t).mean()
    i_med = np.median(np.abs(i_p - i_t))
    i_mae_uA = np.abs(d['I_pred_uA'][::2] - d['I_tgt_uA'][::2]).mean()
    parity_panel(ax_I, i_p, i_t,
                 title='(b) Current I (log scale)',
                 units='log₁₀|I| (log A)',
                 mae_str=f'logMAE = {i_mae:.4f}\nmedian = {i_med:.4f}\n→ {i_mae_uA:.2f} µA')

    # (c) gm
    g_p, g_t = d['gm_pred_log10'], d['gm_tgt_log10']
    g_mae = np.abs(g_p - g_t).mean()
    g_med = np.median(np.abs(g_p - g_t))
    parity_panel(ax_gm, g_p, g_t,
                 title='(c) Small-signal transconductance gm',
                 units='log₁₀(g_m) (log S)',
                 mae_str=f'logMAE = {g_mae:.4f}\nmedian = {g_med:.4f}\n→ {(10**g_mae - 1)*100:.1f}% rel')

    # (d) gds
    ds_p, ds_t = d['gds_pred_log10'], d['gds_tgt_log10']
    ds_mae = np.abs(ds_p - ds_t).mean()
    ds_med = np.median(np.abs(ds_p - ds_t))
    parity_panel(ax_gds, ds_p, ds_t,
                 title='(d) Small-signal output conductance gds',
                 units='log₁₀(g_ds) (log S)',
                 mae_str=f'logMAE = {ds_mae:.4f}\nmedian = {ds_med:.4f}\n→ {(10**ds_mae - 1)*100:.1f}% rel')

    # (e) DC gain — all 1000 samples, uniform color
    dc_pred = dc['model_MLP_dB']
    dc_tgt = dc['SPICE_target_dB']
    err = np.abs(dc_pred - dc_tgt)
    parity_panel(ax_DC, dc_pred, dc_tgt,
                 title='(e) DC gain',
                 units='dB',
                 mae_str=f'MAE = {err.mean():.2f} dB\nmedian = {np.median(err):.2f} dB')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {OUT}')


if __name__ == '__main__':
    main()
