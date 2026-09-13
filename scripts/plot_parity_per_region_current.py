#!/usr/bin/env python
"""Per-region scatter plots of drain current (log10|I_d|) for the usingnow model on fan_smc val.
Split by SPICE-labeled operating region: cutoff / triode / saturation."""
import sys, yaml, argparse, pickle
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
from src.data.pretrain_loader import PretrainCombinedLoader
from src.training.checkpoint import create_model_from_args
from src.training.config import parse_training_config
from eval_all_checkpoints import attach_norm

REPO = Path('/dss/dsshome1/03/go49jit2/thesis')
CKPT = REPO / 'datasets/opamp_3stage_fan_smc_openloop_5k/experiments/ginebn_3layer_bb8_notower_vn_kcl_w10_edge6_netrole_clip03_openloop_5k_usingnow/best_model.pt'
DATA = REPO / 'datasets/opamp_3stage_pretrain_combined_5topo/dataset_val.pkl'
OUT = REPO / 'figures/thesis/fig_parity_current_per_region.png'

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
    i_mean, i_std = stats['i_mean'], stats['i_std']

    dl = PretrainCombinedLoader(str(DATA), batch_size=200, device='cpu', shuffle=False,
                                drop_last=False, topology_filter='fan_smc')
    b0 = next(iter(dl))

    raw = pickle.load(open(DATA, 'rb'))
    raw = [r for r in raw if r.get('topology') == 'fan_smc']
    # 24 MOSFETs per sample, drain idx per mosfet from mosfet_info[:,1]
    raw_regions = np.stack([r['graph'].mosfet_region_labels.cpu().numpy() for r in raw])

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

    # Collect per-MOSFET (drain) predicted+target log10|I| and region
    i_p, i_t, regions = [], [], []
    raw_idx = 0
    with torch.no_grad():
        for b in dl:
            attach_norm(b, stats)
            out = model(b)
            ng = int(b.num_graphs)
            # Drain index per MOSFET in the batched node space:
            mi = b.mosfet_info.long()
            # offsets per MOSFET
            mptr = getattr(b, 'mosfet_ptr', None)
            if mptr is not None:
                m_graph_idx = torch.bucketize(torch.arange(mi.shape[0]), mptr[1:], right=True)
            else:
                m_graph_idx = torch.arange(mi.shape[0]) // (mi.shape[0] // ng)
            node_offsets = b.ptr[m_graph_idx]
            drain_nodes = mi[:, 1] + node_offsets
            log_p_drain = (out['node_currents'][drain_nodes] * i_std + i_mean).cpu().numpy()
            log_t_drain = b.node_current_targets[drain_nodes].abs().clamp_min(1e-15).log10().cpu().numpy()
            i_p.append(log_p_drain); i_t.append(log_t_drain)
            regions.append(raw_regions[raw_idx:raw_idx+ng].reshape(-1))
            raw_idx += ng
    i_p = np.concatenate(i_p); i_t = np.concatenate(i_t); regions = np.concatenate(regions)
    print(f'Total MOSFETs: {len(regions)}')
    for r in (0,1,2):
        print(f'  {REGION_NAMES[r]}: {(regions==r).sum()} ({100*(regions==r).mean():.1f}%)')

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for col, r in enumerate([0, 1, 2]):
        ax = axes[col]
        mask = regions == r
        p, t = i_p[mask], i_t[mask]
        ax.scatter(t, p, s=10, alpha=0.4, color=REGION_COLORS[r], edgecolors='none', rasterized=True)
        lo, hi = min(t.min(), p.min()), max(t.max(), p.max())
        rng = hi - lo
        lo, hi = lo - 0.04*rng, hi + 0.04*rng
        ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1, alpha=0.7)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        err = np.abs(p - t); rel = (10**err - 1) * 100
        Ip = 10 ** p; It = 10 ** t
        rel_direct = np.abs(Ip - It) / np.maximum(It, 1e-20) * 100
        mean_rel = rel_direct.mean()
        ax.text(0.04, 0.96,
                f'N = {p.size:,}\nlogMAE = {err.mean():.4f}\nmedian = {np.median(err):.4f}\nmed rel = {np.median(rel):.1f}%\nmean rel = {mean_rel:.1f}%',
                transform=ax.transAxes, fontsize=11, va='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.92, edgecolor='gray'))
        ax.set_xlabel('SPICE ground truth — log₁₀|I_d| (log A)', fontsize=11)
        ax.set_ylabel('GNN prediction — log₁₀|I_d| (log A)', fontsize=11)
        ax.set_title(f'({chr(97+col)}) {REGION_NAMES[r]} region', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.set_aspect('equal')
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {OUT}')


if __name__ == '__main__':
    main()
