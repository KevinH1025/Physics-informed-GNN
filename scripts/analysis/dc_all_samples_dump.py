#!/usr/bin/env python
"""Dump DC gain arrays for ALL 1000 fan_smc val samples (no valid filter):
  - SPICE_target_dB           (ground truth)
  - formula_GT_dB             (analytical cascade applied to GT gm/gds)
  - formula_predicted_dB      (analytical cascade applied to model-predicted gm/gds)
  - model_MLP_dB              (the MLP-refined head output)
  - ac_valid                  (bool mask, 177 True)

Also produces fig_dc_all_4ways.png — four panels, all on the same 1000 samples.
"""
import pickle, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

REPO = Path('/dss/dsshome1/03/go49jit2/thesis')
PROPER_NPZ = REPO / 'figures/thesis/section41_dc_proper.npz'   # has MLP all-samples
PARITY_NPZ = REPO / 'figures/thesis/section41_parity_fan_smc.npz'  # has DC_physics for valid only
OUT_NPZ = REPO / 'figures/thesis/section41_dc_all_samples.npz'
OUT_PNG = REPO / 'figures/thesis/fig_dc_all_4ways.png'

DC_INDICES = [8, 9, 5, 6, 12, 13, 10, 11, 7, 19, 20, 21, 22, 23]


def log10_add(a, b):
    mx = np.maximum(a, b)
    return mx + np.log10(np.exp((a - mx) * np.log(10)) + np.exp((b - mx) * np.log(10)))


def formula(log10_gm_b, log10_gds_b):
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


def panel(ax, pred, tgt, valid, title, color_invalid='#888888', color_valid='#d62728'):
    ax.scatter(tgt[~valid], pred[~valid], s=8, alpha=0.4, color=color_invalid,
               label=f'invalid ({(~valid).sum()})', rasterized=True)
    ax.scatter(tgt[valid], pred[valid], s=14, alpha=0.65, color=color_valid,
               label=f'valid ({valid.sum()})', rasterized=True)
    lo = float(min(tgt.min(), pred.min())); hi = float(max(tgt.max(), pred.max()))
    rng = hi - lo
    lo, hi = lo - 0.04 * rng, hi + 0.04 * rng
    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1, alpha=0.7)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    err_all = np.abs(pred - tgt); err_v = np.abs(pred[valid] - tgt[valid])
    txt = (
        f'ALL ({pred.size}):  MAE={err_all.mean():.2f} dB, med={np.median(err_all):.2f}, '
        f'corr={np.corrcoef(pred,tgt)[0,1]:.3f}\n'
        f'VALID ({valid.sum()}): MAE={err_v.mean():.2f} dB, med={np.median(err_v):.2f}, '
        f'corr={np.corrcoef(pred[valid],tgt[valid])[0,1]:.3f}'
    )
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, fontsize=9.5,
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
    ax.set_xlabel('SPICE DC gain (dB)', fontsize=11)
    ax.set_ylabel('Predicted DC gain (dB)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9, frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_aspect('equal')


def main():
    # --- (1) Load raw GT gm/gds + ac_dc_gain + ac_valid for all 1000 samples ---
    print('Loading raw fan_smc val samples...')
    with open(REPO / 'datasets/opamp_3stage_pretrain_combined_5topo/dataset_val.pkl','rb') as f:
        data = pickle.load(f)
    samples = [s for s in data if s.get('topology') == 'fan_smc']
    log_gm_arr, log_gds_arr, dc_gt, ac_valid = [], [], [], []
    for s in samples:
        g = s['graph']
        di = g.mosfet_info[:, 1].long()
        log_gm_arr.append(g.node_log_gm[di].cpu().numpy())
        log_gds_arr.append(g.node_log_gds[di].cpu().numpy())
        v = g.ac_dc_gain.item() if hasattr(g.ac_dc_gain,'item') else float(g.ac_dc_gain[0])
        dc_gt.append(v)
        ac_valid.append(bool(g.ac_valid))
    log_gm_arr = np.stack(log_gm_arr); log_gds_arr = np.stack(log_gds_arr)
    dc_gt = np.array(dc_gt); ac_valid = np.array(ac_valid)
    formula_GT = formula(log_gm_arr[:, DC_INDICES], log_gds_arr[:, DC_INDICES])

    # --- (2) Pull existing MLP-on-all-samples from prior dump ---
    proper = np.load(PROPER_NPZ)
    model_MLP = proper['DC_pred_dB_all']
    assert proper['DC_tgt_dB_all'].shape == dc_gt.shape
    # Sanity check ordering matches
    diff = np.abs(proper['DC_tgt_dB_all'] - dc_gt).max()
    assert diff < 1e-3, f'target ordering mismatch: max diff {diff}'

    # --- (3) Compute formula with PREDICTED gm/gds on all 1000 samples ---
    # The model's predicted gm/gds are in section41_parity_fan_smc.npz but only for
    # MOSFETs of fan_smc (24 per graph × 1000 = 24000). Reshape and apply formula.
    parity = np.load(PARITY_NPZ)
    gm_pred = parity['gm_pred_log10'].reshape(1000, 24)   # log10 gm per MOSFET
    gds_pred = parity['gds_pred_log10'].reshape(1000, 24)
    formula_pred = formula(gm_pred[:, DC_INDICES], gds_pred[:, DC_INDICES])

    # --- Save self-contained dump ---
    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_NPZ,
             SPICE_target_dB=dc_gt,
             formula_GT_dB=formula_GT,           # formula(GT gm/gds)
             formula_predicted_dB=formula_pred,  # formula(model-predicted gm/gds)
             model_MLP_dB=model_MLP,             # MLP-refined model output
             ac_valid=ac_valid)
    print(f'saved {OUT_NPZ}')

    # --- Plot 4-panel comparison (all on all 1000 samples) ---
    fig, axes = plt.subplots(2, 2, figsize=(15, 14))
    panel(axes[0, 0], formula_GT, dc_gt, ac_valid,
          '(a) Formula on GROUND-TRUTH gm/gds (no model)')
    panel(axes[0, 1], formula_pred, dc_gt, ac_valid,
          '(b) Formula on PREDICTED gm/gds')
    panel(axes[1, 0], model_MLP, dc_gt, ac_valid,
          '(c) Full deployed model output (MLP-refined head)')
    # (d) Diff: model vs formula on predicted gm/gds — does the MLP help vs hurt?
    ax = axes[1, 1]
    diff = model_MLP - formula_pred
    ax.scatter(dc_gt[~ac_valid], diff[~ac_valid], s=8, alpha=0.4, color='#888888',
               label='invalid')
    ax.scatter(dc_gt[ac_valid], diff[ac_valid], s=14, alpha=0.65, color='#d62728',
               label='valid')
    ax.axhline(0, color='k', linewidth=1, alpha=0.5)
    ax.set_xlabel('SPICE DC gain (dB)', fontsize=11)
    ax.set_ylabel('model_MLP − formula_predicted (dB)', fontsize=11)
    ax.set_title('(d) MLP refinement = (c) − (b): does the MLP help the formula?',
                 fontsize=12, fontweight='bold')
    ax.text(0.04, 0.96,
            f'mean(diff)={diff.mean():.2f} dB, std={diff.std():.2f} dB\n'
            f'mean(diff) valid={diff[ac_valid].mean():.2f}, std={diff[ac_valid].std():.2f}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle=':')

    fig.suptitle('DC gain on ALL fan_smc validation samples (N=1000)\n'
                 'analytical formula vs deployed model, ground-truth vs predicted gm/gds',
                 fontsize=13, fontweight='bold', y=1.00)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {OUT_PNG}')

    # Print summary
    print('\n=== summary: MAE / median / corr on ALL 1000 samples ===')
    def m(p, t, lbl):
        e = np.abs(p - t)
        ev = np.abs(p[ac_valid] - t[ac_valid])
        return f'  {lbl}: ALL MAE={e.mean():.2f}/med={np.median(e):.2f} | VALID MAE={ev.mean():.2f}/med={np.median(ev):.2f}'
    print(m(formula_GT, dc_gt, 'formula(GT gm/gds)   '))
    print(m(formula_pred, dc_gt, 'formula(pred gm/gds) '))
    print(m(model_MLP, dc_gt, 'MLP-refined (deployed)'))


if __name__ == '__main__':
    main()
