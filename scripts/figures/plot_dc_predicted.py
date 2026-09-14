#!/usr/bin/env python
"""Two scatter panels of DC gain prediction on the valid subset:
  (a) formula applied to PREDICTED gm/gds (the model's dc_gain_physics_est_dB)
  (b) MLP-refined head output (the model's dc_gain_pred)
both compared against SPICE GT. Reads existing npz dump.

Output: figures/thesis/fig_dc_predicted.png
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

REPO = Path('/dss/dsshome1/03/go49jit2/thesis')
NPZ = REPO / 'figures/thesis/section41_parity_fan_smc.npz'
OUT = REPO / 'figures/thesis/fig_dc_predicted.png'


def panel(ax, pred, tgt, title, color):
    ax.scatter(tgt, pred, s=12, alpha=0.55, color=color)
    lo = min(tgt.min(), pred.min()); hi = max(tgt.max(), pred.max())
    rng = hi - lo
    lo, hi = lo - 0.05 * rng, hi + 0.05 * rng
    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1, alpha=0.7, label='y = x')
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    err = np.abs(pred - tgt)
    ax.set_xlabel('SPICE DC gain (dB)', fontsize=11)
    ax.set_ylabel('Predicted DC gain (dB)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.text(0.04, 0.96,
            f'N = {pred.size}\nMAE = {err.mean():.2f} dB\nmedian = {np.median(err):.2f} dB\ncorr = {np.corrcoef(pred, tgt)[0,1]:.3f}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_aspect('equal')


def main():
    d = np.load(NPZ)
    tgt = d['DC_tgt_dB']
    phys = d['DC_physics_dB']    # formula(predicted gm/gds)
    mlp = d['DC_pred_dB']        # MLP refined head output

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    panel(axes[0], phys, tgt,
          '(a) Formula with predicted gm/gds (no MLP)',
          color='#2ca02c')
    panel(axes[1], mlp, tgt,
          '(b) Directly predicted DC gain (MLP-refined head)',
          color='#d62728')
    fig.suptitle('DC gain prediction on fan_smc validation subset (177 valid samples, usingnow)\n'
                 'Compares the analytical formula on predicted gm/gds vs the learned MLP head',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {OUT}')


if __name__ == '__main__':
    main()
