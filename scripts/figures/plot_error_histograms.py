#!/usr/bin/env python
"""Error histograms for the usingnow model on fan_smc val.
2x2 grid: V (mV), I, gm, gds.
For log-quantities, converts to RELATIVE error (10^logErr − 1) so it reads as %.
Each panel: histogram of all per-sample-per-device errors,
with vertical lines at median / 95th / 99th percentile.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NPZ = REPO / 'figures/thesis/section41_parity_fan_smc.npz'
OUT = REPO / 'figures/thesis/fig_error_histograms.png'


def hist_panel(ax, errors, *, title, unit, color, log_x=False, x_clip=None):
    """Histogram of an error array, with median/p95/p99 verticals + summary box."""
    n = errors.size
    arr = errors
    if x_clip is not None:
        arr = errors[errors < x_clip]
    if log_x:
        bins = np.logspace(np.log10(max(arr[arr > 0].min(), 1e-6)),
                           np.log10(arr.max()), 80)
        ax.set_xscale('log')
    else:
        bins = 80
    ax.hist(arr, bins=bins, color=color, alpha=0.7, edgecolor='black', linewidth=0.3)
    mae = float(errors.mean()); med = float(np.median(errors))
    p95 = float(np.percentile(errors, 95)); p99 = float(np.percentile(errors, 99))
    for v, lbl, ls, c in [(med, 'median', '-', 'black'),
                          (p95, '95%', '--', '#d62728'),
                          (p99, '99%', ':', '#9467bd')]:
        ax.axvline(v, color=c, linewidth=1.6, linestyle=ls, label=f'{lbl} = {v:.3f}')
    ax.set_xlabel(f'absolute error ({unit})', fontsize=12)
    ax.set_ylabel('count', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    summary = (f'N = {n:,}\n'
               f'MAE = {mae:.4f}\n'
               f'median = {med:.4f}\n'
               f'p95 = {p95:.3f}\n'
               f'p99 = {p99:.3f}')
    ax.text(0.03, 0.97, summary, transform=ax.transAxes, fontsize=10,
            ha='left', va='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.92, edgecolor='gray'))
    ax.legend(loc='upper right', fontsize=9, frameon=True, framealpha=0.92)
    ax.grid(True, alpha=0.3, linestyle=':')


def main():
    d = np.load(NPZ)

    # V error in mV
    V_err_mV = np.abs(d['V_pred'] - d['V_tgt']) * 1000

    # log10 errors → relative error in %
    # log_err = |log10(pred) - log10(tgt)|
    # → rel_err = pred/tgt (or tgt/pred, take the larger ratio) − 1 = 10^log_err − 1
    I_log_err   = np.abs(d['I_pred_log10']   - d['I_tgt_log10'])
    gm_log_err  = np.abs(d['gm_pred_log10']  - d['gm_tgt_log10'])
    gds_log_err = np.abs(d['gds_pred_log10'] - d['gds_tgt_log10'])

    I_rel_pct   = (10**I_log_err   - 1) * 100
    gm_rel_pct  = (10**gm_log_err  - 1) * 100
    gds_rel_pct = (10**gds_log_err - 1) * 100

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    hist_panel(axes[0, 0], V_err_mV,
               title='(a) Voltage prediction error',
               unit='mV', color='#9ecae1',
               x_clip=np.percentile(V_err_mV, 99.5))

    hist_panel(axes[0, 1], I_rel_pct,
               title='(b) Current prediction relative error',
               unit='%', color='#fdae6b',
               log_x=True, x_clip=None)

    hist_panel(axes[1, 0], gm_rel_pct,
               title='(c) gm prediction relative error',
               unit='%', color='#a1d99b',
               log_x=True, x_clip=None)

    hist_panel(axes[1, 1], gds_rel_pct,
               title='(d) gds prediction relative error',
               unit='%', color='#dadaeb',
               log_x=True, x_clip=None)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {OUT}')

    # Summary printout
    print('\n=== summary (median rel error) ===')
    print(f'V    : median {np.median(V_err_mV):.2f} mV,  p95 {np.percentile(V_err_mV,95):.1f} mV')
    print(f'I    : median {np.median(I_rel_pct):.2f}%,  p95 {np.percentile(I_rel_pct,95):.1f}%')
    print(f'gm   : median {np.median(gm_rel_pct):.2f}%,  p95 {np.percentile(gm_rel_pct,95):.1f}%')
    print(f'gds  : median {np.median(gds_rel_pct):.2f}%, p95 {np.percentile(gds_rel_pct,95):.1f}%')


if __name__ == '__main__':
    main()
