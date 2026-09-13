#!/usr/bin/env python
"""§exploration: t-SNE of usingnow model's per-MOSFET embeddings on fan_smc val.

EMBEDDING: concat of gate+drain+source terminal embeddings from
out['node_embeddings'] (= state_repr in tower_genconv.py:2644 — JK-aggregated
backbone output + input skip connection). This is the per-MOSFET vector the
SS head (g_m, g_ds) actually uses. Dim = 3 * 149 = 447.

12 panels (3 rows × 4 cols), same t-SNE layout:
  Row 1 (IDENTITY):
    (a) NMOS vs PMOS
    (b) BSIM4 operating region (cutoff/triode/saturation)
    (c) Coarse device role (bias / diff pair / stage1 / stage2 / output)
    (d) MOSFET position M0..M23 (fine-grained)
  Row 2 (OPERATING POINT):
    (e) V_ov  = V_GS - V_th
    (f) log10(|I_D|)  in µA
    (g) log10(g_m)
    (h) log10(g_ds)
  Row 3 (PREDICTION ERRORS):
    (i) V prediction error at drain net (mV)
    (j) I prediction error  |log10(I_pred) - log10(I_gt)|
    (k) g_m prediction error
    (l) g_ds prediction error

Output: figures/thesis/fig_tsne_usingnow.png
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from circuitgnn.data.pretrain_loader import PretrainCombinedLoader
from circuitgnn.training.checkpoint import create_model_from_args
from circuitgnn.training.config import parse_training_config
from eval_all_checkpoints import attach_norm

REPO = Path('/dss/dsshome1/03/go49jit2/thesis')
EXP = REPO / 'datasets/opamp_3stage_fan_smc_openloop_5k/experiments'
USINGNOW = EXP / 'ginebn_3layer_bb8_notower_vn_kcl_w10_edge6_netrole_clip03_openloop_5k_usingnow'
VAL_PKL = REPO / 'datasets/opamp_3stage_pretrain_combined_5topo/dataset_val.pkl'
OUT_PNG = REPO / 'figures/thesis/fig_tsne_usingnow.png'


def mosfet_role(midx):
    if midx in (0, 1, 2, 3, 4, 5, 6, 7):   return 0  # bias mirrors
    if midx in (8, 9):                      return 1  # diff pair
    if midx in (10, 11):                    return 2  # stage1 load
    if midx in (12, 13, 14, 15):            return 3  # stage2
    return 4                                          # output stage (16-23)


def load_per_sample_extras():
    """Load mosfet_region_labels + mosfet_vth from raw pickle in fan_smc order."""
    import pickle as _pkl
    with open(str(VAL_PKL), 'rb') as f:
        raw = _pkl.load(f)
    fan = [s for s in raw if s.get('topology') == 'fan_smc']
    regs = np.stack([s['graph'].mosfet_region_labels.numpy() for s in fan], axis=0)
    vth = np.stack([np.abs(s['graph'].mosfet_vth.numpy()) for s in fan], axis=0)  # |Vth|
    print(f'  loaded extras: region shape={regs.shape}, vth shape={vth.shape}')
    return regs, vth


def load_model_and_run():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[setup] device={device}')

    region_all, vth_all = load_per_sample_extras()

    val = PretrainCombinedLoader(str(VAL_PKL), batch_size=50, device=device,
                                 shuffle=False, drop_last=False, topology_filter='fan_smc')
    sample = next(iter(val))
    ck_path = USINGNOW / 'best.pt'
    if not ck_path.exists():
        ck_path = USINGNOW / 'best_model.pt'
    ckpt = torch.load(ck_path, map_location=device, weights_only=False)
    with open(USINGNOW / 'original_config.yaml') as f:
        cfg = yaml.safe_load(f)
    if 'norm_stats' in ckpt:
        stats = ckpt['norm_stats']
    else:
        s = ckpt['stats']
        stats = {'v_mean': s['vdc']['mean'], 'v_std': s['vdc']['std'],
                 'i_mean': s['current']['mean'], 'i_std': s['current']['std'],
                 'gm_mean': s['ss_gm']['mean'], 'gm_std': s['ss_gm']['std'],
                 'gds_mean': s['ss_gds']['mean'], 'gds_std': s['ss_gds']['std']}

    import argparse as _ap
    cli = _ap.Namespace()
    for k, v in parse_training_config(cfg).items():
        if v is not None:
            setattr(cli, k, v)
    cli.device = str(device); cli.dataset = str(VAL_PKL.parent); cli.predict_currents = True
    node_dim = sample.x.shape[-1]; type_dim = sample.type_tens.shape[-1]
    model, _ = create_model_from_args(cli, node_dim + type_dim, device)
    model.load_state_dict(ckpt['model_state_dict'])
    if hasattr(model, 'set_normalization_stats'):
        model.set_normalization_stats(
            vdc_mean=stats['v_mean'], vdc_std=stats['v_std'],
            ss_gm_mean=stats['gm_mean'], ss_gm_std=stats['gm_std'],
            ss_gds_mean=stats['gds_mean'], ss_gds_std=stats['gds_std'],
            current_mean=stats['i_mean'], current_std=stats['i_std'])
    if hasattr(model, 'current_epoch'):
        model.current_epoch = ckpt.get('epoch', 99999)
    model.eval()

    val = PretrainCombinedLoader(str(VAL_PKL), batch_size=50, device=device,
                                 shuffle=False, drop_last=False, topology_filter='fan_smc')

    cols = {k: [] for k in [
        'emb', 'is_nmos', 'region', 'role', 'midx',
        'vov', 'vgs', 'vds', 'vth',
        'log_id_uA', 'log_gm', 'log_gds',
        'v_err_mV', 'i_err_log', 'gm_err_log', 'gds_err_log',
    ]}
    sample_offset = 0
    with torch.no_grad():
        for batch in val:
            attach_norm(batch, stats)
            out = model(batch)
            emb_full = out['node_embeddings']                     # [num_nodes, 149]
            mi = batch.mosfet_info.long()                         # [num_mosfets, 7]
            ptr = batch.ptr
            num_mosfets = mi.shape[0]
            num_graphs = len(ptr) - 1
            mosfets_per_graph = num_mosfets // num_graphs

            # Per-MOSFET global indices for gate/drain/source terminals
            graph_idx = torch.arange(num_mosfets, device=mi.device) // mosfets_per_graph
            node_offsets = ptr[graph_idx]
            gate_idx   = mi[:, 0] + node_offsets
            drain_idx  = mi[:, 1] + node_offsets
            source_idx = mi[:, 2] + node_offsets
            # Net indices (V predictions live here)
            gate_net_idx   = mi[:, 3] + node_offsets
            drain_net_idx  = mi[:, 4] + node_offsets
            source_net_idx = mi[:, 5] + node_offsets

            # CONCAT embedding: gate ⊕ drain ⊕ source (3 * 149 = 447)
            emb_concat = torch.cat([emb_full[gate_idx], emb_full[drain_idx],
                                     emb_full[source_idx]], dim=-1)

            # Voltage at gate/drain/source NETS (GT)
            v_gate   = batch.node_voltage_targets[gate_net_idx]
            v_drain  = batch.node_voltage_targets[drain_net_idx]
            v_source = batch.node_voltage_targets[source_net_idx]
            is_nmos  = mi[:, 6].bool()
            vgs_raw = v_gate - v_source
            vds_raw = v_drain - v_source
            vgs = torch.where(is_nmos,  vgs_raw, -vgs_raw)
            vds = torch.where(is_nmos,  vds_raw, -vds_raw)
            # V_th: looked up per MOSFET from pre-loaded array
            vth_per = vth_all[sample_offset:sample_offset + num_graphs].flatten()
            vth_t = torch.tensor(vth_per, dtype=torch.float, device=mi.device)
            vov = vgs - vth_t

            # Region
            reg_per = region_all[sample_offset:sample_offset + num_graphs].flatten()

            # Coarse role + position
            mid_pg = (torch.arange(num_mosfets, device=mi.device) - graph_idx * mosfets_per_graph).cpu().numpy()
            role = np.array([mosfet_role(int(m)) for m in mid_pg])

            # Targets and predictions
            id_log_gt = batch.node_current_targets[drain_idx].abs().clamp_min(1e-15).log10().cpu().numpy()
            id_log_uA = id_log_gt + 6.0
            id_log_pred = (out['node_currents'].squeeze()[drain_idx] * stats['i_std'] + stats['i_mean']).cpu().numpy()
            i_err = np.abs(id_log_pred - id_log_gt)
            # gm/gds (per-MOSFET predictions from SS head)
            gm_log_gt  = batch.node_log_gm[drain_idx].cpu().numpy()
            gds_log_gt = batch.node_log_gds[drain_idx].cpu().numpy()
            if 'mosfet_gm_pred' in out and 'mosfet_gds_pred' in out:
                gm_log_pred  = (out['mosfet_gm_pred'].squeeze()  * stats['gm_std']  + stats['gm_mean']).cpu().numpy()
                gds_log_pred = (out['mosfet_gds_pred'].squeeze() * stats['gds_std'] + stats['gds_mean']).cpu().numpy()
                gm_err  = np.abs(gm_log_pred  - gm_log_gt)
                gds_err = np.abs(gds_log_pred - gds_log_gt)
            else:
                gm_err  = np.zeros(num_mosfets)
                gds_err = np.zeros(num_mosfets)
            # V prediction error at drain net (mV)
            v_pred = (out['node_voltages'][drain_net_idx] * stats['v_std'] + stats['v_mean']).cpu().numpy()
            v_tgt  = batch.node_voltage_targets[drain_net_idx].cpu().numpy()
            v_err_mV = np.abs(v_pred - v_tgt) * 1000.0

            cols['emb'].append(emb_concat.cpu().numpy())
            cols['is_nmos'].append(is_nmos.cpu().numpy().astype(int))
            cols['region'].append(reg_per.astype(int))
            cols['role'].append(role.astype(int))
            cols['midx'].append(mid_pg.astype(int))
            cols['vov'].append(vov.cpu().numpy())
            cols['vgs'].append(vgs.cpu().numpy())
            cols['vds'].append(vds.cpu().numpy())
            cols['vth'].append(vth_per)
            cols['log_id_uA'].append(id_log_uA)
            cols['log_gm'].append(gm_log_gt)
            cols['log_gds'].append(gds_log_gt)
            cols['v_err_mV'].append(v_err_mV)
            cols['i_err_log'].append(i_err)
            cols['gm_err_log'].append(gm_err)
            cols['gds_err_log'].append(gds_err)

            sample_offset += num_graphs

    return {k: np.concatenate(v) for k, v in cols.items()}


def main():
    print('[1] loading model + extracting embeddings...')
    d = load_model_and_run()
    print(f'  embeddings: shape={d["emb"].shape}  (gate⊕drain⊕source concat)')
    print(f'  region dist: {np.bincount(d["region"], minlength=3)} (cut/tri/sat)')
    print(f'  NMOS/PMOS dist: NMOS={(d["is_nmos"]==1).sum()}  PMOS={(d["is_nmos"]==0).sum()}')

    # Subsample
    N = d['emb'].shape[0]
    if N > 12000:
        rng = np.random.default_rng(42)
        idx = rng.choice(N, size=12000, replace=False)
        d = {k: v[idx] for k, v in d.items()}
        print(f'  subsampled {N} → 12000')

    print('[2] running t-SNE (perplexity=30)...')
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42,
                learning_rate='auto', max_iter=1000, verbose=1)
    X2 = tsne.fit_transform(d['emb'])
    print(f'  done, X2 shape={X2.shape}')

    print('[3] plotting 3x4 panel figure...')
    fig, axes = plt.subplots(3, 4, figsize=(22, 16.5))
    PS = 4; ALPHA = 0.6

    def cont(ax, color, title, cmap, clip_pct=(2, 98), units=''):
        lo, hi = np.percentile(color, clip_pct)
        c_clip = np.clip(color, lo, hi)
        sc = ax.scatter(X2[:, 0], X2[:, 1], s=PS, c=c_clip, cmap=cmap,
                        alpha=ALPHA, edgecolors='none')
        cb = plt.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
        cb.set_label(units, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight='bold', loc='left')
        ax.set_xticks([]); ax.set_yticks([])

    def cat(ax, labels, names, colors, title):
        for i, (lbl, col) in enumerate(zip(names, colors)):
            m = labels == i
            if m.any():
                ax.scatter(X2[m, 0], X2[m, 1], s=PS, c=col, alpha=ALPHA,
                           label=f'{lbl} (N={m.sum()})', edgecolors='none')
        ax.legend(loc='upper right', markerscale=3, fontsize=7,
                  frameon=True, framealpha=0.9, handletextpad=0.3)
        ax.set_title(title, fontsize=11, fontweight='bold', loc='left')
        ax.set_xticks([]); ax.set_yticks([])

    # === ROW 1: IDENTITY ===
    cat(axes[0, 0], d['is_nmos'],
        ['PMOS', 'NMOS'], ['#d62728', '#1f77b4'],
        '(a) device type (NMOS vs PMOS)')
    cat(axes[0, 1], d['region'],
        ['cutoff', 'triode', 'saturation'], ['#9467bd', '#ff7f0e', '#1f77b4'],
        '(b) BSIM4 operating region')
    cat(axes[0, 2], d['role'],
        ['bias (M0-7)', 'diff pair (M8-9)', 'stage1 load (M10-11)',
         'stage2 (M12-15)', 'output (M16-23)'],
        ['#2ca02c', '#d62728', '#1f77b4', '#ff7f0e', '#8c564b'],
        '(c) coarse role (5 stages)')
    # Position: 24 colors via tab20 + extras
    import matplotlib.colors as mcolors
    pos_colors = list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors[:4])
    ax = axes[0, 3]
    for i in range(24):
        m = d['midx'] == i
        if m.any():
            ax.scatter(X2[m, 0], X2[m, 1], s=PS, c=[pos_colors[i]], alpha=ALPHA, edgecolors='none')
    ax.set_title('(d) MOSFET position M0..M23', fontsize=11, fontweight='bold', loc='left')
    ax.set_xticks([]); ax.set_yticks([])
    # Mini legend with position numbers
    handles = [plt.Line2D([0], [0], marker='o', linestyle='', color=pos_colors[i],
                          label=f'M{i}', markersize=4) for i in range(24)]
    ax.legend(handles=handles, loc='upper right', ncol=4, fontsize=6,
              frameon=True, framealpha=0.9, handletextpad=0.1, columnspacing=0.5)

    # === ROW 2: OPERATING POINT ===
    cont(axes[1, 0], d['vov']*1000, '(e) V_ov = V_GS − V_th',  'coolwarm', units='mV')
    cont(axes[1, 1], d['log_id_uA'], '(f) log10(|I_D|)',        'viridis', units='log µA')
    cont(axes[1, 2], d['log_gm'],    '(g) log10(g_m)',           'viridis', units='log S')
    cont(axes[1, 3], d['log_gds'],   '(h) log10(g_ds)',          'viridis', units='log S')

    # === ROW 3: PREDICTION ERRORS ===
    cont(axes[2, 0], d['v_err_mV'],    '(i) V prediction error at drain net', 'RdYlGn_r', units='mV')
    cont(axes[2, 1], d['i_err_log'],   '(j) I prediction error',              'RdYlGn_r', units='|log10|')
    cont(axes[2, 2], d['gm_err_log'],  '(k) g_m prediction error',            'RdYlGn_r', units='|log10|')
    cont(axes[2, 3], d['gds_err_log'], '(l) g_ds prediction error',           'RdYlGn_r', units='|log10|')

    fig.suptitle('t-SNE of usingnow per-MOSFET embeddings (gate⊕drain⊕source concat) — fan_smc val',
                 fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved → {OUT_PNG}')

    # ===== Diagnostics =====
    print('\n=== Structural read ===')
    REGIONS = ['cutoff', 'triode', 'saturation']
    for r in range(3):
        m = d['region'] == r
        if m.sum() > 50:
            print(f'  region {REGIONS[r]:<11}: centroid=({X2[m,0].mean():6.1f}, {X2[m,1].mean():6.1f})  '
                  f'σ=({X2[m,0].std():5.1f}, {X2[m,1].std():5.1f})  N={m.sum()}')
    print()
    for r in range(5):
        m = d['role'] == r
        names = ['bias    ', 'diff_pair', 'stage1_l', 'stage2  ', 'output  ']
        if m.sum() > 50:
            print(f'  role {names[r]}: centroid=({X2[m,0].mean():6.1f}, {X2[m,1].mean():6.1f})  '
                  f'σ=({X2[m,0].std():5.1f}, {X2[m,1].std():5.1f})  N={m.sum()}')
    print()
    for name, key in [('I err', 'i_err_log'), ('V err', 'v_err_mV'),
                      ('gm err', 'gm_err_log'), ('gds err', 'gds_err_log')]:
        hi = d[key] > np.percentile(d[key], 90)
        lo = d[key] < np.percentile(d[key], 10)
        print(f'  {name:<8}: HI-10% centroid=({X2[hi,0].mean():6.1f}, {X2[hi,1].mean():6.1f})  '
              f'LO-10% centroid=({X2[lo,0].mean():6.1f}, {X2[lo,1].mean():6.1f})  '
              f'Δ=({X2[hi,0].mean()-X2[lo,0].mean():+5.1f}, {X2[hi,1].mean()-X2[lo,1].mean():+5.1f})')


if __name__ == '__main__':
    main()
