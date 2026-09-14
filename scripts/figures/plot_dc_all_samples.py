#!/usr/bin/env python
"""Two-panel scatter on ALL fan_smc val samples (1000), valid + invalid:
  (a) formula using GT gm/gds (no model)         vs SPICE GT
  (b) MLP-refined DC gain (usingnow model)       vs SPICE GT

Color: valid samples (positive gain, ac_valid=True) vs invalid (SPICE-failed sentinels).
Both panels cover the whole deployment range — that's what inference sees.
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PKL = REPO / 'datasets/opamp_3stage_pretrain_combined_5topo/dataset_val.pkl'
NPZ = REPO / 'figures/thesis/section41_dc_proper.npz'   # has DC_pred_dB_all, ac_valid_all
OUT = REPO / 'figures/thesis/fig_dc_all_samples.png'

DC_INDICES = [8, 9, 5, 6, 12, 13, 10, 11, 7, 19, 20, 21, 22, 23]


def log10_add(a, b):
    mx = np.maximum(a, b)
    return mx + np.log10(np.exp((a - mx) * np.log(10)) + np.exp((b - mx) * np.log(10)))


def dc_gain_formula(log10_gm_b, log10_gds_b):
    gm, gds = log10_gm_b, log10_gds_b
    log_cascode_term = gds[:, 5] + gds[:, 7] - gm[:, 5]
    log_rout1 = -log10_add(gds[:, 3], log_cascode_term)
    log_A1 = gm[:, 0] + log_rout1
    log_denom1 = log10_add(log10_add(gm[:, 10], gds[:, 10]), gds[:, 9])
    log_A2_p1 = gm[:, 9] - log_denom1
    log_denom2 = log10_add(gds[:, 8], gds[:, 11])
    log_A2 = log_A2_p1 + gm[:, 11] - log_denom2
    log_rout3 = -log10_add(gds[:, 12], gds[:, 13])
    log_sum_gm = log10_add(gm[:, 12], log_A2 + gm[:, 13])
    return 20.0 * (log_A1 + log_rout3 + log_sum_gm)


def panel(ax, pred, tgt, valid, title):
    """Scatter with valid/invalid color coding + MAE annotations for both subsets."""
    ax.scatter(tgt[~valid], pred[~valid], s=10, alpha=0.4, color='#888888',
               label=f'invalid ({(~valid).sum()})', rasterized=True)
    ax.scatter(tgt[valid], pred[valid], s=14, alpha=0.65, color='#d62728',
               label=f'valid ({valid.sum()})', rasterized=True)
    lo = float(min(tgt.min(), pred.min()))
    hi = float(max(tgt.max(), pred.max()))
    rng = hi - lo
    lo, hi = lo - 0.04 * rng, hi + 0.04 * rng
    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1, alpha=0.7)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    err = np.abs(pred - tgt)
    err_v = np.abs(pred[valid] - tgt[valid])
    mae_all = err.mean(); med_all = np.median(err)
    mae_v   = err_v.mean(); med_v   = np.median(err_v)
    corr_all = np.corrcoef(pred, tgt)[0, 1]
    corr_v   = np.corrcoef(pred[valid], tgt[valid])[0, 1]
    txt = (
        f'ALL ({pred.size}):  MAE={mae_all:.2f} dB, med={med_all:.2f}, corr={corr_all:.3f}\n'
        f'VALID ({valid.sum()}): MAE={mae_v:.2f} dB, med={med_v:.2f}, corr={corr_v:.3f}'
    )
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
    ax.set_xlabel('SPICE DC gain (dB)', fontsize=11)
    ax.set_ylabel('Predicted DC gain (dB)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10, frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_aspect('equal')


def main():
    # --- (a) formula with GT gm/gds, all samples ---
    print('Loading raw fan_smc val samples...')
    with open(PKL, 'rb') as f:
        data = pickle.load(f)
    samples = [s for s in data if s.get('topology') == 'fan_smc']
    print(f'  {len(samples)} samples')
    log_gm_arr, log_gds_arr, dc_gt, ac_valid = [], [], [], []
    for s in samples:
        g = s['graph']
        di = g.mosfet_info[:, 1].long()
        log_gm_arr.append(g.node_log_gm[di].cpu().numpy())
        log_gds_arr.append(g.node_log_gds[di].cpu().numpy())
        v = g.ac_dc_gain.item() if hasattr(g.ac_dc_gain, 'item') else float(g.ac_dc_gain[0])
        dc_gt.append(v)
        ac_valid.append(bool(g.ac_valid))
    log_gm_arr = np.stack(log_gm_arr)
    log_gds_arr = np.stack(log_gds_arr)
    dc_gt = np.array(dc_gt)
    ac_valid = np.array(ac_valid)

    # Formula with GT gm/gds
    gm_key = log_gm_arr[:, DC_INDICES]
    gds_key = log_gds_arr[:, DC_INDICES]
    formula_gt = dc_gain_formula(gm_key, gds_key)

    # --- (b) MLP-refined predictions (loaded from prior dump) ---
    d = np.load(NPZ)
    mlp_pred = d['DC_pred_dB_all']
    mlp_tgt = d['DC_tgt_dB_all']
    mlp_valid = d['ac_valid_all'].astype(bool)
    # Sanity check the ordering matches our raw extraction
    assert mlp_tgt.size == dc_gt.size, f'size mismatch: {mlp_tgt.size} vs {dc_gt.size}'

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    panel(axes[0], formula_gt, dc_gt, ac_valid,
          '(a) Analytical formula with GT gm/gds (no model)')
    panel(axes[1], mlp_pred, mlp_tgt, mlp_valid,
          '(b) MLP-refined DC gain (usingnow model)')
    fig.suptitle('DC gain prediction on all fan_smc validation samples (N=1000)\n'
                 'Includes both valid (red, gain > 0) and invalid (gray, SPICE-failed) samples',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {OUT}')


if __name__ == '__main__':
    main()
