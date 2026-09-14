#!/usr/bin/env python
"""Three deliverables for §4.x:
  1. Per-topology stats table (V, I, gm, gds, DC gain, amplifying fraction)
     for all 5 topologies over their full 5000 samples.
  2. Aggregate distribution figure: histograms of log10|I| and DC gain in dB,
     one curve per topology overlaid.
  3. Per-device distribution figure (fan_smc only): box plots of V (per net),
     log10|I| (per terminal/device), log10(gm) (per MOSFET), log10(gds) (per MOSFET).

Loads pickles in two passes (4-topo combined then fan_smc) so 5k-per-topo fits in RAM.
"""
import pickle, gc, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch

REPO = Path(__file__).resolve().parents[2]
OUT_TABLE = REPO / 'figures/thesis/topo_stats.json'
OUT_AGG = REPO / 'figures/thesis/fig_topo_distributions.png'
OUT_DEV_VI = REPO / 'figures/thesis/fig_per_device_fansmc_VI.png'
OUT_DEV_SS = REPO / 'figures/thesis/fig_per_device_fansmc_SS.png'
OUT_DEV_ALL = REPO / 'figures/thesis/fig_per_device_fansmc_ALL.png'        # 4x1 tall
OUT_DEV_2x2 = REPO / 'figures/thesis/fig_per_device_fansmc_ALL_2x2.png'    # 2x2 grid

TOPOS = ['fan_smc', 'sau_cfcc', 'peng_tcfc', 'leung_nmcf', 'leung_nmcnr']
TOPO_COLORS = {
    'fan_smc': '#1f77b4', 'sau_cfcc': '#ff7f0e', 'peng_tcfc': '#2ca02c',
    'leung_nmcf': '#d62728', 'leung_nmcnr': '#9467bd',
}


def gather_topology(samples, topo_filter=None, drop_unphysical_V=True,
                    v_min=-0.001, v_max=1.801):
    """Return dict of per-quantity arrays for a list of samples.
    If drop_unphysical_V, exclude samples where any internal-net V is outside
    [v_min, v_max] (catches SPICE convergence artifacts like sau_cfcc's net2).
    """
    Vs, Is, gms, gdss, dcs, ac_valids = [], [], [], [], [], []
    n_used = 0; n_dropped = 0
    for s in samples:
        if topo_filter is not None and s.get('topology') != topo_filter:
            continue
        g = s['graph']
        # V on internal nets only (~known_voltage_mask, since terminal V isn't supervised)
        if hasattr(g, 'train_mask'):
            mask = g.train_mask
        else:
            mask = ~g.known_voltage_mask & g.output_node_mask if hasattr(g, 'output_node_mask') else ~g.known_voltage_mask
        v_arr = g.node_voltage_targets[mask].cpu().numpy().astype(np.float32)
        if drop_unphysical_V and (v_arr.min() < v_min or v_arr.max() > v_max):
            n_dropped += 1
            continue
        Vs.append(v_arr)
        # I supervised positions
        if hasattr(g, 'has_current_mask') and g.has_current_mask is not None and g.has_current_mask.any():
            i_vals = g.node_current_targets[g.has_current_mask].abs().clamp_min(1e-15).log10().cpu().numpy()
            Is.append(i_vals.astype(np.float32))
        # gm/gds at drain positions
        if hasattr(g, 'node_log_gm') and hasattr(g, 'mosfet_drain_mask'):
            gms.append(g.node_log_gm[g.mosfet_drain_mask].cpu().numpy().astype(np.float32))
            gdss.append(g.node_log_gds[g.mosfet_drain_mask].cpu().numpy().astype(np.float32))
        # DC gain
        if hasattr(g, 'ac_dc_gain'):
            v = g.ac_dc_gain.item() if hasattr(g.ac_dc_gain, 'item') else float(g.ac_dc_gain[0])
            dcs.append(v)
            ac_valids.append(bool(g.ac_valid) if hasattr(g, 'ac_valid') else (v > 0))
        n_used += 1
    return {
        'V': np.concatenate(Vs) if Vs else np.zeros(0),
        'I_log10': np.concatenate(Is) if Is else np.zeros(0),
        'gm_log10': np.concatenate(gms) if gms else np.zeros(0),
        'gds_log10': np.concatenate(gdss) if gdss else np.zeros(0),
        'DC_dB': np.array(dcs, dtype=np.float32),
        'ac_valid': np.array(ac_valids, dtype=bool),
        'n_samples': n_used,
        'n_dropped_unphysical_V': n_dropped,
    }


def stats_of(arr):
    if arr.size == 0:
        return {'n': 0, 'min': None, 'max': None, 'mean': None, 'std': None}
    return {'n': int(arr.size), 'min': float(arr.min()), 'max': float(arr.max()),
            'mean': float(arr.mean()), 'std': float(arr.std())}


def build_table(topo_data):
    table = {}
    for t, d in topo_data.items():
        row = {
            'n_samples': d['n_samples'],
            'n_dropped_unphysical_V': d.get('n_dropped_unphysical_V', 0),
            'V':         stats_of(d['V']),
            'I_log10':   stats_of(d['I_log10']),
            'gm_log10':  stats_of(d['gm_log10']),
            'gds_log10': stats_of(d['gds_log10']),
            'DC_gain_dB': stats_of(d['DC_dB']),
            'amplifying_fraction': float(((d['DC_dB'] > 0) & d['ac_valid']).mean()) if d['DC_dB'].size else None,
        }
        table[t] = row
    return table


def make_aggregate_fig(topo_data):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    # (a) log10|I| histogram per topology
    ax = axes[0]
    for t in TOPOS:
        v = topo_data[t]['I_log10']
        if v.size == 0: continue
        ax.hist(v, bins=80, range=(-15, 0), density=True, histtype='step',
                linewidth=2, color=TOPO_COLORS[t], label=t)
    ax.set_xlabel('log₁₀|I| (log A)', fontsize=12)
    ax.set_ylabel('density', fontsize=12)
    ax.set_title('(a) Per-device current distribution by topology', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10, frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=':')
    # (b) DC gain histogram per topology
    ax = axes[1]
    for t in TOPOS:
        v = topo_data[t]['DC_dB']
        if v.size == 0: continue
        ax.hist(v, bins=80, range=(-250, 130), density=True, histtype='step',
                linewidth=2, color=TOPO_COLORS[t], label=t)
    ax.axvline(0, color='black', linewidth=1, alpha=0.5, linestyle='--')
    ax.text(2, ax.get_ylim()[1]*0.92, 'amplifying →', fontsize=9, color='gray')
    ax.set_xlabel('DC gain (dB)', fontsize=12)
    ax.set_ylabel('density', fontsize=12)
    ax.set_title('(b) DC gain distribution by topology', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10, frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=':')
    fig.tight_layout()
    OUT_AGG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_AGG, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {OUT_AGG}')


def make_per_device_fig(fansmc_samples):
    """Box plots per device on fan_smc. Each panel one quantity, x-axis is device/net index."""
    # Walk one sample to determine the supervised positions per quantity
    s0 = fansmc_samples[0]['graph']
    train_mask = s0.train_mask if hasattr(s0, 'train_mask') else ~s0.known_voltage_mask
    mosfet_info = s0.mosfet_info
    # Position indices
    v_positions = train_mask.nonzero(as_tuple=True)[0].cpu().numpy()    # per-net V indices
    # I is one PER MOSFET (drain == source magnitude due to device_pooling),
    # so we sample only at the drain position → 24 not 48.
    mos_drain_positions = mosfet_info[:, 1].cpu().numpy()
    print(f'  fan_smc: {len(v_positions)} V positions (nets), {len(mos_drain_positions)} MOSFETs')

    # Accumulate per-position values across all samples
    V_pp = [[] for _ in v_positions]
    I_pp = [[] for _ in mos_drain_positions]
    gm_pp = [[] for _ in mos_drain_positions]
    gds_pp = [[] for _ in mos_drain_positions]
    for s in fansmc_samples:
        g = s['graph']
        for k, p in enumerate(v_positions):
            V_pp[k].append(float(g.node_voltage_targets[p]))
        for k, p in enumerate(mos_drain_positions):
            iv = float(g.node_current_targets[p])
            I_pp[k].append(np.log10(max(abs(iv), 1e-15)))
            gm_pp[k].append(float(g.node_log_gm[p]))
            gds_pp[k].append(float(g.node_log_gds[p]))

    # fan_smc stage assignment from netlist
    STAGE_OF_MOSFET = {
        0:'bias', 1:'bias', 2:'bias', 3:'bias', 4:'bias', 5:'bias', 6:'bias', 7:'bias',
        8:'S1', 9:'S1', 15:'S1', 16:'S1', 19:'S1', 20:'S1',
        10:'S2', 21:'S2', 22:'S2',
        11:'S3', 23:'S3',
        12:'bias', 13:'bias', 14:'bias', 17:'bias', 18:'bias',
    }
    STAGE_COLOR = {'bias':'#888888', 'S1':'#1f77b4', 'S2':'#ff7f0e', 'S3':'#d62728'}
    STAGE_LABEL = {'bias':'Bias', 'S1':'Stage 1', 'S2':'Stage 2', 'S3':'Stage 3'}

    def box(ax, data, xlabels, ylabel, title, ylim=None, tick_size=10, mosfet_colored=False):
        bp = ax.boxplot(data, showfliers=False, patch_artist=True,
                        medianprops=dict(color='black', linewidth=1.5),
                        whiskerprops=dict(color='black', linewidth=0.6),
                        capprops=dict(color='black', linewidth=0.6))
        if mosfet_colored:
            # color each box by the stage of its MOSFET (index from M0..M23)
            for i, patch in enumerate(bp['boxes']):
                patch.set_facecolor(STAGE_COLOR[STAGE_OF_MOSFET[i]])
                patch.set_edgecolor('black'); patch.set_linewidth(0.6)
            # legend
            from matplotlib.patches import Patch
            handles = [Patch(facecolor=STAGE_COLOR[k], edgecolor='black', label=STAGE_LABEL[k])
                       for k in ['bias','S1','S2','S3']]
            ax.legend(handles=handles, loc='lower left', fontsize=10, frameon=True, framealpha=0.95)
        else:
            for patch in bp['boxes']:
                patch.set_facecolor('#9ecae1'); patch.set_edgecolor('black'); patch.set_linewidth(0.6)
        ax.set_xticks(range(1, len(xlabels)+1))
        ax.set_xticklabels(xlabels, fontsize=tick_size, rotation=45, ha='right')
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.tick_params(axis='y', labelsize=10)
        ax.grid(True, axis='y', alpha=0.3, linestyle=':')
        if ylim: ax.set_ylim(*ylim)

    # === Figure 1: V (top) + I (bottom) ===
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    box(axes[0], V_pp, [f'net{i}' for i in range(len(v_positions))],
        'V (Volts)', '(a) Per-net voltage distribution', tick_size=10)
    box(axes[1], I_pp, [f'M{i}' for i in range(len(mos_drain_positions))],
        'log₁₀|I| (log A)', '(b) Per-MOSFET current distribution (log scale)', tick_size=10,
        mosfet_colored=True)
    fig.tight_layout()
    OUT_DEV_VI.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DEV_VI, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {OUT_DEV_VI}')

    # === Figure 2: gm (top) + gds (bottom) ===
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    box(axes[0], gm_pp, [f'M{i}' for i in range(len(mos_drain_positions))],
        'log₁₀(g_m) (log S)', '(a) Per-MOSFET transconductance gm distribution', tick_size=10,
        mosfet_colored=True)
    box(axes[1], gds_pp, [f'M{i}' for i in range(len(mos_drain_positions))],
        'log₁₀(g_ds) (log S)', '(b) Per-MOSFET output conductance gds distribution', tick_size=10,
        mosfet_colored=True)
    fig.tight_layout()
    fig.savefig(OUT_DEV_SS, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {OUT_DEV_SS}')

    # === Figure 3: ALL FOUR in one tall figure ===
    fig, axes = plt.subplots(4, 1, figsize=(16, 20))
    box(axes[0], V_pp, [f'net{i}' for i in range(len(v_positions))],
        'V (Volts)', '(a) Per-net voltage distribution', tick_size=11)
    box(axes[1], I_pp, [f'M{i}' for i in range(len(mos_drain_positions))],
        'log₁₀|I| (log A)', '(b) Per-MOSFET current distribution (log scale)', tick_size=11,
        mosfet_colored=True)
    box(axes[2], gm_pp, [f'M{i}' for i in range(len(mos_drain_positions))],
        'log₁₀(g_m) (log S)', '(c) Per-MOSFET transconductance gm distribution', tick_size=11,
        mosfet_colored=True)
    box(axes[3], gds_pp, [f'M{i}' for i in range(len(mos_drain_positions))],
        'log₁₀(g_ds) (log S)', '(d) Per-MOSFET output conductance gds distribution', tick_size=11,
        mosfet_colored=True)
    fig.tight_layout()
    fig.savefig(OUT_DEV_ALL, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {OUT_DEV_ALL}')

    # === Figure 4: 2x2 grid (V, I, gm, gds) — wide for A4 landscape ===
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    box(axes[0, 0], V_pp, [f'net{i}' for i in range(len(v_positions))],
        'V (Volts)', '(a) Per-net voltage distribution', tick_size=10)
    box(axes[0, 1], I_pp, [f'M{i}' for i in range(len(mos_drain_positions))],
        'log₁₀|I| (log A)', '(b) Per-MOSFET current distribution (log scale)', tick_size=10,
        mosfet_colored=True)
    box(axes[1, 0], gm_pp, [f'M{i}' for i in range(len(mos_drain_positions))],
        'log₁₀(g_m) (log S)', '(c) Per-MOSFET transconductance gm distribution', tick_size=10,
        mosfet_colored=True)
    box(axes[1, 1], gds_pp, [f'M{i}' for i in range(len(mos_drain_positions))],
        'log₁₀(g_ds) (log S)', '(d) Per-MOSFET output conductance gds distribution', tick_size=10,
        mosfet_colored=True)
    fig.tight_layout()
    fig.savefig(OUT_DEV_2x2, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {OUT_DEV_2x2}')


def main():
    topo_data = {}
    print('=== Pass 1: load fan_smc (standalone) ===')
    fan_train = pickle.load(open(REPO / 'datasets/opamp_3stage_fan_smc_openloop_5k/dataset_train.pkl', 'rb'))
    fan_val = pickle.load(open(REPO / 'datasets/opamp_3stage_fan_smc_openloop_5k/dataset_val.pkl', 'rb'))
    fan_all = fan_train + fan_val
    print(f'  fan_smc total samples: {len(fan_all)}')
    topo_data['fan_smc'] = gather_topology(fan_all, topo_filter=None)

    # Hold onto fan_all for the per-device fig later
    del fan_train, fan_val; gc.collect()

    print('\n=== Pass 2: load 4-topo combined train + val (full 5k/topo) ===')
    c4_train = pickle.load(open(REPO / 'datasets/opamp_3stage_pretrain_combined/dataset_train.pkl', 'rb'))
    c4_val = pickle.load(open(REPO / 'datasets/opamp_3stage_pretrain_combined/dataset_val.pkl', 'rb'))
    c4_all = c4_train + c4_val
    del c4_train, c4_val; gc.collect()
    print(f'  combined total samples: {len(c4_all)}')
    for t in ('sau_cfcc', 'peng_tcfc', 'leung_nmcf', 'leung_nmcnr'):
        topo_data[t] = gather_topology(c4_all, topo_filter=t)
        print(f"  {t}: n={topo_data[t]['n_samples']}, V_n={topo_data[t]['V'].size}")
    del c4_all; gc.collect()

    # Build table + save JSON
    table = build_table(topo_data)
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_TABLE, 'w') as f:
        json.dump(table, f, indent=2)
    print(f'\nsaved table → {OUT_TABLE}')
    # Pretty print
    print('\n=== Per-topology stats (after dropping samples with V outside [0, 1.8] V) ===')
    for t, row in table.items():
        nd = row.get('n_dropped_unphysical_V', 0)
        drop_msg = f', {nd} dropped (unphysical V)' if nd else ''
        print(f'\n  {t} ({row["n_samples"]} samples used{drop_msg}):')
        for q in ('V', 'I_log10', 'gm_log10', 'gds_log10', 'DC_gain_dB'):
            r = row[q]
            print(f'    {q:<12}: n={r["n"]:>7,}, min={r["min"]:>9.3f}, max={r["max"]:>9.3f}, μ={r["mean"]:>9.3f}, σ={r["std"]:>8.3f}')
        print(f'    amplifying_fraction (gain>0 & ac_valid): {row["amplifying_fraction"]:.3f}')

    # Aggregate figure
    make_aggregate_fig(topo_data)

    # Per-device fig (fan_smc only)
    print('\n=== Per-device boxplots (fan_smc) ===')
    make_per_device_fig(fan_all)


if __name__ == '__main__':
    main()
