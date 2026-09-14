#!/usr/bin/env python
"""Per-region scatter plots (cutoff / triode / saturation) for the usingnow model
on fan_smc val. Splits per-MOSFET predictions by the SPICE region label.

Output: figures/thesis/fig_parity_per_region.png — 2 rows × 3 cols
  rows: gm, gds
  cols: cutoff, triode, saturation
"""
import sys, yaml, argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
from circuitgnn.data.pretrain_loader import PretrainCombinedLoader
from circuitgnn.training.checkpoint import create_model_from_args
from circuitgnn.training.config import parse_training_config
from eval_all_checkpoints import attach_norm

REPO = Path('/dss/dsshome1/03/go49jit2/thesis')
EXP = REPO / 'datasets/opamp_3stage_fan_smc_openloop_5k/experiments'
CKPT = EXP / 'ginebn_3layer_bb8_notower_vn_kcl_w10_edge6_netrole_clip03_openloop_5k_usingnow' / 'best_model.pt'
DATA = REPO / 'datasets/opamp_3stage_pretrain_combined_5topo/dataset_val.pkl'
OUT = REPO / 'figures/thesis/fig_parity_per_region.png'

# label -> region name (from graph_builder.py:821)
REGION_NAMES = {0: 'cutoff', 1: 'triode', 2: 'saturation'}
REGION_COLORS = {0: '#9467bd', 1: '#ff7f0e', 2: '#1f77b4'}


def main():
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    cfg = yaml.safe_load(open(CKPT.parent/'original_config.yaml'))
    s = ck['stats']
    stats = {'v_mean': s['vdc']['mean'], 'v_std': s['vdc']['std'],
             'i_mean': s['current']['mean'], 'i_std': s['current']['std'],
             'gm_mean': s['ss_gm']['mean'], 'gm_std': s['ss_gm']['std'],
             'gds_mean': s['ss_gds']['mean'], 'gds_std': s['ss_gds']['std']}
    gm_mean, gm_std = stats['gm_mean'], stats['gm_std']
    gds_mean, gds_std = stats['gds_mean'], stats['gds_std']

    dl = PretrainCombinedLoader(str(DATA), batch_size=200, device='cpu', shuffle=False,
                                drop_last=False, topology_filter='fan_smc')
    b0 = next(iter(dl))
    # Region labels are not propagated by PretrainCombinedLoader → load from raw samples.
    import pickle
    with open(DATA, 'rb') as f:
        raw = pickle.load(f)
    raw = [r for r in raw if r.get('topology') == 'fan_smc']
    raw_regions = np.stack([r['graph'].mosfet_region_labels.cpu().numpy() for r in raw])  # [N_samples, 24]
    cli = argparse.Namespace()
    for k, v in parse_training_config(cfg).items():
        if v is not None: setattr(cli, k, v)
    cli.device='cpu'; cli.dataset=str(DATA.parent); cli.predict_currents=True
    model, _ = create_model_from_args(cli, b0.x.shape[-1]+b0.type_tens.shape[-1], 'cpu')
    model.load_state_dict(ck['model_state_dict'])
    model.set_normalization_stats(vdc_mean=stats['v_mean'], vdc_std=stats['v_std'],
        ss_gm_mean=stats['gm_mean'], ss_gm_std=stats['gm_std'],
        ss_gds_mean=stats['gds_mean'], ss_gds_std=stats['gds_std'],
        current_mean=stats['i_mean'], current_std=stats['i_std'])
    model.current_epoch = ck.get('epoch', 4022); model.eval()

    # Collect per-MOSFET: gm_pred, gm_tgt, gds_pred, gds_tgt, region
    gm_p, gm_t, gds_p, gds_t = [], [], [], []
    raw_idx = 0
    regions_all = []
    with torch.no_grad():
        for b in dl:
            attach_norm(b, stats)
            out = model(b)
            dm = b.mosfet_drain_mask
            log_gm_p = (out['mosfet_gm_pred'] * gm_std + gm_mean).cpu().numpy()
            log_gds_p = (out['mosfet_gds_pred'] * gds_std + gds_mean).cpu().numpy()
            log_gm_t = b.node_log_gm[dm].cpu().numpy()
            log_gds_t = b.node_log_gds[dm].cpu().numpy()
            ng = int(b.num_graphs)
            # Region labels from raw samples (24 per graph), flattened in batch order
            regions_all.append(raw_regions[raw_idx:raw_idx+ng].reshape(-1))
            raw_idx += ng
            gm_p.append(log_gm_p); gm_t.append(log_gm_t)
            gds_p.append(log_gds_p); gds_t.append(log_gds_t)
    regions = np.concatenate(regions_all)
    gm_p = np.concatenate(gm_p); gm_t = np.concatenate(gm_t)
    gds_p = np.concatenate(gds_p); gds_t = np.concatenate(gds_t)
    print(f'Total MOSFETs: {len(regions)}')
    print(f'  cutoff:     {(regions==0).sum()} ({100*(regions==0).mean():.1f}%)')
    print(f'  triode:     {(regions==1).sum()} ({100*(regions==1).mean():.1f}%)')
    print(f'  saturation: {(regions==2).sum()} ({100*(regions==2).mean():.1f}%)')

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    region_order = [0, 1, 2]   # cutoff, triode, saturation
    for col, r in enumerate(region_order):
        mask = (regions == r)
        if not mask.any():
            continue
        for row, (pred, tgt, name, units) in enumerate([
            (gm_p, gm_t, 'gm',  'log₁₀(g_m) (log S)'),
            (gds_p, gds_t, 'gds', 'log₁₀(g_ds) (log S)'),
        ]):
            ax = axes[row, col]
            p, t = pred[mask], tgt[mask]
            ax.scatter(t, p, s=8, alpha=0.35, color=REGION_COLORS[r],
                       edgecolors='none', rasterized=True)
            lo, hi = min(t.min(), p.min()), max(t.max(), p.max())
            rng = hi - lo
            lo, hi = lo - 0.04*rng, hi + 0.04*rng
            ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1, alpha=0.7)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            err = np.abs(p - t)
            mae = err.mean(); med = np.median(err)
            rel = (10**err - 1) * 100
            relmed = np.median(rel)
            ax.text(0.04, 0.96,
                    f'N = {p.size:,}\nlogMAE = {mae:.4f}\nmedian = {med:.4f}\nmed rel = {relmed:.1f}%',
                    transform=ax.transAxes, fontsize=10, va='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.92, edgecolor='gray'))
            ax.set_xlabel(f'SPICE ground truth — {units}', fontsize=11)
            ax.set_ylabel(f'GNN prediction — {units}', fontsize=11)
            ax.set_title(f'({chr(97 + col + 3*row)}) {name} — {REGION_NAMES[r]} region', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle=':')
            ax.set_aspect('equal')
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {OUT}')


if __name__ == '__main__':
    main()
