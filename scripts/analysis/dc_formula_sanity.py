#!/usr/bin/env python
"""Sanity check: apply the usingnow DC-gain closed-form formula to GROUND-TRUTH
gm/gds (from SPICE) and compare against SPICE-reported DC gain. This isolates
the FORMULA's irreducible error from any model prediction error.

Two scatter plots produced:
  (a) all fan_smc val samples
  (b) only the "valid" subset (ac_valid & ac_dc_gain > 0)

Output: figures/thesis/fig_dc_formula_sanity.png
"""
import pickle, math
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
PKL = REPO / 'datasets/opamp_3stage_pretrain_combined_5topo/dataset_val.pkl'
OUT = REPO / 'figures/thesis/fig_dc_formula_sanity.png'

# 14 signal-path MOSFETs in usingnow's order (indices into the 24 mosfets)
DC_INDICES = [8, 9, 5, 6, 12, 13, 10, 11, 7, 19, 20, 21, 22, 23]


def log10_add(a, b):
    """log10(10^a + 10^b), numerically stable."""
    mx = np.maximum(a, b)
    return mx + np.log10(np.exp((a - mx) * np.log(10)) + np.exp((b - mx) * np.log(10)))


def dc_gain_formula(log10_gm_b, log10_gds_b, openloop=True):
    """Compute usingnow's open-loop DC gain formula in dB from per-MOSFET log10(gm/gds).
    Inputs: shape [B, 14] (14 signal-path MOSFETs, log10 of gm or gds, in S).
    Returns: [B] DC gain in dB.
    """
    gm = log10_gm_b
    gds = log10_gds_b
    # Stage 1 R_out (cascode variant — usingnow uses it):
    #   log R_out1 = -log10(gds_M6 + gds_M16*gds_M20/gm_M16)
    log_cascode_term = gds[:, 5] + gds[:, 7] - gm[:, 5]
    log_rout1 = -log10_add(gds[:, 3], log_cascode_term)
    log_A1 = gm[:, 0] + log_rout1
    # Stage 2
    log_denom1 = log10_add(log10_add(gm[:, 10], gds[:, 10]), gds[:, 9])
    log_A2_p1 = gm[:, 9] - log_denom1
    log_denom2 = log10_add(gds[:, 8], gds[:, 11])
    log_A2 = log_A2_p1 + gm[:, 11] - log_denom2
    # Stage 3 (open-loop: drop feedback factor)
    log_rout3 = -log10_add(gds[:, 12], gds[:, 13])
    log_sum_gm = log10_add(gm[:, 12], log_A2 + gm[:, 13])
    log_T = log_A1 + log_rout3 + log_sum_gm
    return 20.0 * log_T   # dB


def main():
    print(f'Loading {PKL}...')
    with open(PKL, 'rb') as f:
        data = pickle.load(f)
    samples = [s for s in data if s.get('topology') == 'fan_smc']
    print(f'fan_smc val samples: {len(samples)}')

    log_gm_all, log_gds_all, dc_gt_all, ac_valid_all = [], [], [], []
    for s in samples:
        g = s['graph']
        # MOSFET drain indices: mosfet_info[:, 1] are local drain indices
        mi = g.mosfet_info
        drain_idx = mi[:, 1].long()
        # Per-MOSFET log10(gm), log10(gds) at drain positions
        log_gm = g.node_log_gm[drain_idx].cpu().numpy()    # shape [24]
        log_gds = g.node_log_gds[drain_idx].cpu().numpy()
        log_gm_all.append(log_gm)
        log_gds_all.append(log_gds)
        # GT DC gain
        v = g.ac_dc_gain.item() if hasattr(g.ac_dc_gain, 'item') else float(g.ac_dc_gain[0])
        dc_gt_all.append(v)
        ac_valid_all.append(bool(g.ac_valid))

    log_gm_arr = np.stack(log_gm_all)      # [N, 24]
    log_gds_arr = np.stack(log_gds_all)
    dc_gt = np.array(dc_gt_all)
    ac_valid = np.array(ac_valid_all)

    # Select 14 signal-path
    gm_key = log_gm_arr[:, DC_INDICES]
    gds_key = log_gds_arr[:, DC_INDICES]
    # Apply formula
    dc_formula = dc_gain_formula(gm_key, gds_key, openloop=True)
    print(f'\nformula DC gain (from GT gm/gds): range [{dc_formula.min():.1f}, {dc_formula.max():.1f}] dB')
    print(f'SPICE DC gain:                     range [{dc_gt.min():.1f}, {dc_gt.max():.1f}] dB')

    # Filters
    valid_mask = ac_valid & (dc_gt > 0)
    print(f'  valid subset (ac_valid & gain>0): {valid_mask.sum()}/{len(samples)}')

    def stats_block(pred, tgt, label):
        err = np.abs(pred - tgt)
        return (f'{label}: N={pred.size}, MAE={err.mean():.2f} dB, '
                f'median={np.median(err):.2f} dB, corr={np.corrcoef(pred, tgt)[0,1]:.3f}')

    print('\n' + stats_block(dc_formula, dc_gt, 'ALL samples'))
    print(stats_block(dc_formula[valid_mask], dc_gt[valid_mask], 'VALID subset'))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    def scatter_panel(ax, pred, tgt, title):
        ax.scatter(tgt, pred, s=8, alpha=0.4, color='#1f77b4')
        lo = min(tgt.min(), pred.min()); hi = max(tgt.max(), pred.max())
        rng = hi - lo
        lo, hi = lo - 0.05 * rng, hi + 0.05 * rng
        ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1, alpha=0.7, label='y = x')
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        err = np.abs(pred - tgt)
        ax.set_xlabel('SPICE DC gain (dB)', fontsize=11)
        ax.set_ylabel('Formula DC gain from GT gm/gds (dB)', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.text(0.04, 0.96,
                f'N = {pred.size}\nMAE = {err.mean():.2f} dB\nmedian = {np.median(err):.2f} dB\ncorr = {np.corrcoef(pred, tgt)[0,1]:.3f}',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.set_aspect('equal')

    scatter_panel(axes[0], dc_formula, dc_gt,
                  '(a) All fan_smc val samples')
    scatter_panel(axes[1], dc_formula[valid_mask], dc_gt[valid_mask],
                  '(b) Only valid samples (ac_valid & gain > 0)')

    fig.suptitle('DC gain formula sanity — applied to GT gm/gds (no model)\n'
                 'usingnow open-loop cascade formula; fan_smc validation set',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nsaved {OUT}')


if __name__ == '__main__':
    main()
