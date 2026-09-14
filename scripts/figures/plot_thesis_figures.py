#!/usr/bin/env python
"""Generate thesis figures from experiment logs.

Outputs to figures/thesis/:
  fig1_data_efficiency.png       — main 5-panel grid: V MAE vs N for all methods
  fig2_joint_vs_pertopo.png       — bar chart, joint beats per-topo on every topology
  fig3_zeroshot_heatmap.png       — LOO heatmap, diagonal = zero-shot V MAE
  fig4_speedup.png                — speedup of ftzeroshot over scratch, by topology
  fig5_gnn_vs_mlp.png              — architecture comparison, fan_smc + sau_cfcc
  fig6_full_data_summary.png      — final V MAE at N=4000 across methods × topos
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


EXP_DIR = Path('datasets/opamp_3stage_pretrain_combined_5topo/experiments')
OUT_DIR = Path('figures/thesis')
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOPOS = ['fan_smc', 'sau_cfcc', 'peng_tcfc', 'leung_nmcf', 'leung_nmcnr']
TOPO_COLORS = {
    'fan_smc': '#1f77b4',
    'sau_cfcc': '#ff7f0e',
    'peng_tcfc': '#2ca02c',
    'leung_nmcf': '#d62728',
    'leung_nmcnr': '#9467bd',
}

# Method colors and styles — picked to be colorblind-friendly and printer-safe
METHODS = {
    'scratch':    {'color': '#d62728', 'marker': 'o', 'ls': '-',  'label': 'Train from scratch'},
    'ftzeroshot': {'color': '#1f77b4', 'marker': 's', 'ls': '-',  'label': 'Few-shot from cross-topology pretrain'},
    'joint':      {'color': '#2ca02c', 'marker': '^', 'ls': '-',  'label': 'Joint (N per topology = 5N total)'},
    'joint_total':{'color': '#17becf', 'marker': 'v', 'ls': '--', 'label': 'Joint (N total budget, N/5 per topology)'},
    'mlp':        {'color': '#7f7f7f', 'marker': 'd', 'ls': '--', 'label': 'MLP baseline'},
}

# Bigger fonts everywhere for readability
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})


def extract_topo_v_mae(log_path: Path, topo: str) -> float | None:
    # Best *observed* validation V MAE across all logged epochs — NOT the [BEST]-marked
    # line. Runs use different checkpoint-selection criteria by vintage (old: val_sched
    # v+i+gm+gds; new: avg per-topo V MAE), so the [BEST] line understates V MAE for the
    # old runs. Taking the min over all epochs scores every method identically.
    if not log_path.exists():
        return None
    pat = re.compile(rf'{topo}=([0-9.]+)')
    vmin = None
    with open(log_path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                v = float(m.group(1))
                vmin = v if vmin is None else min(vmin, v)
    return vmin


def extract_mlp_v_mae(log_path: Path) -> float | None:
    # Best observed V MAE across all epochs (see extract_topo_v_mae): the MLP saves on
    # total val loss, so its [BEST] line also understates V MAE.
    if not log_path.exists():
        return None
    pat = re.compile(r'V MAE=([0-9.]+)mV')
    vmin = None
    with open(log_path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                v = float(m.group(1))
                vmin = v if vmin is None else min(vmin, v)
    return vmin


def collect_data():
    d = {
        'scratch': {t: {} for t in TOPOS},
        'ftzeroshot': {t: {} for t in TOPOS},
        'joint': {t: {} for t in TOPOS},
        'joint_total': {t: {} for t in TOPOS},
        'mlp': {t: {} for t in TOPOS},
        'pertopo_full': {},
        'joint_full': {},
        'zeroshot_loo': {},
    }
    for t in TOPOS:
        t2 = 'fansmc' if t == 'fan_smc' else t
        for N in (100, 250, 500, 1000, 2500):
            log = EXP_DIR / f'v5_5topo_pertopo_{t2}_n{N}/training.log'
            v = extract_topo_v_mae(log, t)
            if v is not None: d['scratch'][t][N] = v
        # Average across seeds where reseed runs exist (multi-seed for variance check)
        seed_logs = [EXP_DIR / f'v5_5topo_pertopo_{t}/training.log']
        for s in (43, 44):
            sl = EXP_DIR / f'v5_5topo_pertopo_{t}_s{s}/training.log'
            if sl.exists(): seed_logs.append(sl)
        seed_vs = [extract_topo_v_mae(p, t) for p in seed_logs]
        seed_vs = [v for v in seed_vs if v is not None]
        if seed_vs:
            full_v = sum(seed_vs) / len(seed_vs)
            d['scratch'][t][4000] = full_v
            d['pertopo_full'][t] = full_v
            if len(seed_vs) > 1:
                d.setdefault('pertopo_full_std', {})[t] = (
                    sum((v - full_v) ** 2 for v in seed_vs) / len(seed_vs)) ** 0.5
                d.setdefault('pertopo_full_n_seeds', {})[t] = len(seed_vs)
    for t in TOPOS:
        t2 = 'fansmc' if t == 'fan_smc' else t
        for N in (10, 50, 100, 250, 500, 1000, 2500, 4000):
            log_v2 = EXP_DIR / f'v5_5topo_ftzeroshot_{t2}_n{N}_v2/training.log'
            log_v1 = EXP_DIR / f'v5_5topo_ftzeroshot_{t2}_n{N}/training.log'
            log = log_v2 if log_v2.exists() else log_v1
            v = extract_topo_v_mae(log, t)
            if v is not None: d['ftzeroshot'][t][N] = v
    for N in (100, 250, 500, 800, 1000):
        log = EXP_DIR / f'v5_5topo_joint_n{N}/training.log'
        for t in TOPOS:
            v = extract_topo_v_mae(log, t)
            if v is not None: d['joint'][t][N] = v
    full_log = EXP_DIR / 'v5_5topo_joint/training.log'
    for t in TOPOS:
        v = extract_topo_v_mae(full_log, t)
        if v is not None:
            d['joint'][t][4000] = v
            d['joint_full'][t] = v
    # Apples-to-apples joint: x-axis = TOTAL samples (per-topo cap × 5 topologies).
    # Original 'joint' above is plotted at the per-topo N; this curve places the SAME
    # joint runs (plus the lower-N reseeds) at their true total-sample budget.
    joint_total_runs = {
        100:  'v5_5topo_joint_n20',      # 20/topo  × 5
        250:  'v5_5topo_joint_n50',      # 50/topo  × 5
        500:  'v5_5topo_joint_n100_v2',  # 100/topo × 5 (re-run; monotone fix)
        1000: 'v5_5topo_joint_n200',     # 200/topo × 5
        2500: 'v5_5topo_joint_n500',     # 500/topo × 5
        4000: 'v5_5topo_joint_n800',     # 800/topo × 5
        5000: 'v5_5topo_joint_n1000',    # 1000/topo × 5
    }
    for total, run in joint_total_runs.items():
        log = EXP_DIR / f'{run}/training.log'
        for t in TOPOS:
            v = extract_topo_v_mae(log, t)
            if v is not None: d['joint_total'][t][total] = v
    for t in TOPOS:
        t2 = 'fansmc' if t == 'fan_smc' else t
        for N in (100, 250, 500, 1000, 2500):
            log = EXP_DIR / f'mlp_{t2}_n{N}/training.log'
            v = extract_mlp_v_mae(log)
            if v is not None: d['mlp'][t][N] = v
        full_log = EXP_DIR / f'mlp_{t2}_full/training.log'
        v = extract_mlp_v_mae(full_log)
        if v is not None: d['mlp'][t][4000] = v
    for held in TOPOS:
        t2 = 'fansmc' if held == 'fan_smc' else held
        log = EXP_DIR / f'v5_5topo_zeroshot_{t2}/training.log'
        if not log.exists(): continue
        d['zeroshot_loo'][held] = {}
        for ev in TOPOS:
            v = extract_topo_v_mae(log, ev)
            if v is not None: d['zeroshot_loo'][held][ev] = v
    return d


# =====================================================================
# Figure 1 — Main result: data-efficiency curves per topology
# =====================================================================

def fig1_data_efficiency(d):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    legend_handles = None
    for i, topo in enumerate(TOPOS):
        ax = axes[i]
        for method_key in ('mlp', 'scratch', 'joint', 'joint_total', 'ftzeroshot'):
            cfg = METHODS[method_key]
            data = d[method_key][topo]
            Ns = sorted(data.keys())
            if not Ns: continue
            ys = [data[n] for n in Ns]
            ax.plot(Ns, ys, marker=cfg['marker'], linestyle=cfg['ls'],
                    color=cfg['color'], label=cfg['label'],
                    linewidth=2, markersize=8, alpha=0.9 if method_key != 'mlp' else 0.6)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(f'{topo}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Training samples N (joint: total budget, split N/5 per topology)', fontsize=11)
        ax.set_ylabel('Voltage MAE (mV) — lower is better', fontsize=11)
        ax.grid(True, which='both', alpha=0.3, linestyle=':')
        ax.set_xticks([10, 50, 100, 500, 1000, 4000])
        ax.set_xticklabels(['10', '50', '100', '500', '1k', '4k'])
        if i == 0:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    axes[5].axis('off')
    if legend_handles:
        axes[5].legend(legend_handles, legend_labels, loc='center', fontsize=13,
                       frameon=True, framealpha=0.95)

    fig.suptitle('Data Efficiency: V MAE vs target-topology training samples',
                 fontsize=15, fontweight='bold', y=1.00)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig1_data_efficiency.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  saved fig1_data_efficiency.png')


# =====================================================================
# Figure 2 — Joint vs per-topology bars
# =====================================================================

def fig2_joint_vs_pertopo(d):
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(TOPOS))
    width = 0.35
    pertopo = [d['pertopo_full'].get(t, 0) for t in TOPOS]
    joint = [d['joint_full'].get(t, 0) for t in TOPOS]
    pertopo_err = [d.get('pertopo_full_std', {}).get(t, 0) for t in TOPOS]
    b1 = ax.bar(x - width/2, pertopo, width, yerr=pertopo_err, capsize=4,
                label='Per-topology training (4000 samples on target only)',
                color='#d62728', edgecolor='black', linewidth=0.5,
                error_kw={'ecolor': 'black', 'lw': 1.2})
    b2 = ax.bar(x + width/2, joint, width,
                label='Joint multi-topology training (4000/topo × 5 topos)',
                color='#2ca02c', edgecolor='black', linewidth=0.5)
    for i, (p, j) in enumerate(zip(pertopo, joint)):
        if p > 0 and j > 0:
            delta = (p - j) / p * 100
            ax.text(i, max(p, j) * 1.04, f'−{delta:.0f}%',
                    ha='center', fontsize=11, color='darkgreen' if delta > 0 else 'red',
                    fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(TOPOS, fontsize=11)
    ax.set_ylabel('Voltage MAE (mV) on this topology — lower is better', fontsize=12)
    ax.set_title('Joint multi-topology training beats per-topology training\non 4 of 5 opamp designs (loses on sau_cfcc)',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, axis='y', alpha=0.3, linestyle=':')
    # Add subtitle below
    fig.text(0.5, -0.02,
             'Each pair: a model trained on 1 topology (4000 samples) vs the SAME architecture trained jointly on all 5.\n'
             'Joint sees 5× more total data; it wins on 4/5 but the gap is small — cross-topology benefit is real but modest.',
             ha='center', fontsize=10, style='italic')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig2_joint_vs_pertopo.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  saved fig2_joint_vs_pertopo.png')


# =====================================================================
# Figure 3 — Zero-shot LOO heatmap
# =====================================================================

def fig3_zeroshot_heatmap(d):
    fig, ax = plt.subplots(figsize=(9, 7))
    matrix = np.full((len(TOPOS), len(TOPOS)), np.nan)
    for i, held in enumerate(TOPOS):
        for j, ev in enumerate(TOPOS):
            v = d['zeroshot_loo'].get(held, {}).get(ev)
            if v is not None: matrix[i, j] = v
    im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=180)
    ax.set_xticks(range(len(TOPOS)))
    ax.set_yticks(range(len(TOPOS)))
    ax.set_xticklabels(TOPOS, fontsize=11, rotation=15)
    ax.set_yticklabels(TOPOS, fontsize=11)
    ax.set_xlabel('Evaluated on (which topology)', fontsize=12)
    ax.set_ylabel('Held out from training (NEVER seen)', fontsize=12)
    for i in range(len(TOPOS)):
        for j in range(len(TOPOS)):
            v = matrix[i, j]
            if not np.isnan(v):
                color = 'white' if v > 90 else 'black'
                weight = 'bold' if i == j else 'normal'
                marker = ' ★' if i == j else ''
                ax.text(j, i, f'{v:.0f}{marker}', ha='center', va='center',
                        color=color, fontsize=11, fontweight=weight)
    ax.set_title(
        'Zero-shot LOO matrix — V MAE (mV)\n'
        '★ Diagonal = the held-out topology (model NEVER trained on it).\n'
        'Off-diagonal = the model\'s normal in-distribution performance.',
        fontsize=12)
    plt.colorbar(im, ax=ax, label='V MAE (mV)')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig3_zeroshot_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  saved fig3_zeroshot_heatmap.png')


# =====================================================================
# Figure 4 — Speedup of ftzeroshot over scratch (clearer than overlay)
# =====================================================================

def fig4_error_ratio(d):
    """Error ratio: scratch V MAE / ftzeroshot V MAE at the same N."""
    Ns = [100, 250, 500, 1000, 2500, 4000]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    x = np.arange(len(Ns))

    for i, topo in enumerate(TOPOS):
        ax = axes[i]
        ratios = []
        for N in Ns:
            scr = d['scratch'][topo].get(N)
            ft = d['ftzeroshot'][topo].get(N)
            ratios.append(scr / ft if (scr and ft) else np.nan)
        bar_colors = ['#2ca02c' if (not np.isnan(s) and s >= 1.0) else '#d62728'
                      for s in ratios]
        ax.bar(x, ratios, 0.7, color=bar_colors, edgecolor='black', linewidth=0.5)
        for xi, s in enumerate(ratios):
            if not np.isnan(s):
                ax.text(xi, s + 0.05, f'{s:.2f}×', ha='center', fontsize=10,
                        fontweight='bold')
        ax.axhline(1.0, color='black', linewidth=1.2, linestyle='--', alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in Ns], fontsize=10)
        ax.set_xlabel('Target N (same for both methods)', fontsize=11)
        ax.set_ylabel('Scratch V MAE / ftzeroshot V MAE', fontsize=10)
        ax.set_title(topo, fontsize=13, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3, linestyle=':')
        ax.set_ylim(0, max(4.6, np.nanmax(ratios) * 1.15 if not np.all(np.isnan(ratios)) else 4.6))

    axes[5].axis('off')

    fig.suptitle('Error ratio at the same N — how much lower is ftzeroshot V MAE than scratch V MAE',
                 fontsize=14, fontweight='bold', y=1.00)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig4_error_ratio.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  saved fig4_error_ratio.png')


def fig4b_sample_efficiency(d):
    """Sample-efficiency speedup: how many MORE samples scratch needs to match ftzeroshot at a given N."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, topo in enumerate(TOPOS):
        ax = axes[i]
        # Build scratch curve
        scratch_pts = sorted(d['scratch'][topo].items())
        if len(scratch_pts) < 2:
            ax.axis('off'); continue
        N_scratch_arr = np.array([p[0] for p in scratch_pts], dtype=float)
        V_scratch_arr = np.array([p[1] for p in scratch_pts], dtype=float)
        # Interp in log-log space: given target V MAE, find N
        # Make sure V_scratch is monotonically decreasing in N — interp wants increasing xp
        # We want N(V), so use V as x (must be increasing → reverse)
        order = np.argsort(V_scratch_arr)
        Vsx = V_scratch_arr[order]
        Nsx = N_scratch_arr[order]

        N_ft = []; speedup = []; labels = []
        for N, V_ft in sorted(d['ftzeroshot'][topo].items()):
            if V_ft <= 0 or V_ft >= max(V_scratch_arr): continue
            if V_ft < min(V_scratch_arr):
                # ftzs better than scratch's best — scratch never reaches this V MAE.
                # Bar height = 1.0 (just enough to be visible at the tied line)
                sp = 1.0
                cap = True
            else:
                logV = np.log(Vsx); logN = np.log(Nsx)
                eq_N = float(np.exp(np.interp(np.log(V_ft), logV, logN)))
                sp = eq_N / N
                cap = False
            N_ft.append(N); speedup.append(sp); labels.append(cap)

        if not N_ft:
            ax.axis('off'); continue

        x = np.arange(len(N_ft))
        colors = []
        for s, cap in zip(speedup, labels):
            if cap:                colors.append('#9467bd')   # purple — ftzeroshot beats scratch's best
            elif s >= 1.0:         colors.append('#2ca02c')   # green — ftzeroshot wins
            else:                  colors.append('#d62728')   # red — scratch wins
        bars = ax.bar(x, speedup, 0.7, color=colors, edgecolor='black', linewidth=0.5,
                      hatch=['' if not c else '///' for c in labels])
        # Add headroom so the value labels stay inside the plot area
        ymax = max([s for s, c in zip(speedup, labels) if not c] + [1.0]) * 1.30
        for xi, (s, cap) in enumerate(zip(speedup, labels)):
            if not cap:
                ax.text(xi, s + ymax * 0.02, f'{s:.1f}×', ha='center',
                        fontsize=10, fontweight='bold')
        ax.axhline(1.0, color='black', linewidth=1.2, linestyle='--', alpha=0.6)
        ax.set_ylim(0, ymax)
        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in N_ft], fontsize=10)
        ax.set_xlabel('ftzeroshot N (target topology samples used)', fontsize=11)
        ax.set_ylabel('Scratch samples needed / ftzeroshot N', fontsize=10)
        ax.set_title(topo, fontsize=13, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3, linestyle=':')

    axes[5].axis('off')
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ca02c', edgecolor='black', label='ftzeroshot wins (≥1×)'),
        Patch(facecolor='#d62728', edgecolor='black', label='scratch wins (<1×)'),
        Patch(facecolor='#9467bd', edgecolor='black', hatch='///',
              label='scratch never reaches ft'),
    ]
    axes[5].legend(handles=legend_elements, loc='center', fontsize=12,
                   frameon=True, framealpha=0.95)
    fig.suptitle('Sample-efficiency ratio — scratch samples needed to match ftzeroshot at N',
                 fontsize=14, fontweight='bold', y=1.00)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig4b_sample_efficiency.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  saved fig4b_sample_efficiency.png')


# =====================================================================
# Figure 5 — GNN vs MLP architecture comparison
# =====================================================================

def fig5_gnn_vs_mlp(d):
    plot_topos = [t for t in TOPOS if t in ['fan_smc', 'sau_cfcc']]
    fig, axes = plt.subplots(1, len(plot_topos), figsize=(13, 5.5))
    if len(plot_topos) == 1: axes = [axes]
    for i, topo in enumerate(plot_topos):
        ax = axes[i]
        Ns_g = sorted(d['scratch'][topo].keys())
        Ns_m = sorted(d['mlp'][topo].keys())
        if Ns_g:
            ax.plot(Ns_g, [d['scratch'][topo][n] for n in Ns_g],
                    'o-', color='#1f77b4', label='GNN (per-topology training)',
                    linewidth=2.5, markersize=9)
        if Ns_m:
            ax.plot(Ns_m, [d['mlp'][topo][n] for n in Ns_m],
                    'd--', color='#7f7f7f', label='MLP baseline',
                    linewidth=2, markersize=8)
        # Annotate the gap at full data
        if 4000 in d['scratch'][topo] and 4000 in d['mlp'][topo]:
            g = d['scratch'][topo][4000]
            m = d['mlp'][topo][4000]
            ratio = m / g
            mid_y = (g + m) / 2
            ax.annotate(f'{ratio:.1f}× gap\nat full data',
                        xy=(4000, mid_y), xytext=(1500, mid_y * 0.95),
                        fontsize=11, fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Target topology training samples (N)', fontsize=11)
        ax.set_ylabel('Voltage MAE (mV) — lower is better', fontsize=11)
        ax.set_title(f'{topo}', fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, which='both', alpha=0.3, linestyle=':')
    fig.suptitle('GNN architecture beats MLP by 2-4× on V MAE\n(why graph structure matters for circuit prediction)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig5_gnn_vs_mlp.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  saved fig5_gnn_vs_mlp.png')


# =====================================================================
# Figure 6 — Final V MAE comparison at full data (N=4000) across all methods
# =====================================================================

def fig6_full_data_summary(d):
    fig, ax = plt.subplots(figsize=(14, 6.5))
    width = 0.15
    x = np.arange(len(TOPOS))

    # Match fig1's method set: 5 bars, including BOTH joint variants
    # (per-topo budget vs total-N budget) so the comparison is honest.
    methods_order = [
        ('scratch',     'Scratch per-topo (4000 on target)'),
        ('ftzeroshot',  'ftzeroshot (pretrain on 4 others + 4000 FT on target)'),
        ('joint',       'Joint (N per topology = 4000/topo, 20 000 total)'),
        ('joint_total', 'Joint (N total budget = 4000 total, 800/topo)'),
        ('mlp',         'MLP baseline (4000 on target)'),
    ]
    n = len(methods_order)
    all_max = 0
    for i, (key, label) in enumerate(methods_order):
        offset = (i - (n - 1) / 2) * width
        vals = [d[key][t].get(4000, np.nan) for t in TOPOS]
        cfg = METHODS[key]
        ls = cfg.get('ls', '-')
        hatch = '///' if ls == '--' else None  # distinguish dashed/total-budget method visually
        ax.bar(x + offset, vals, width,
               label=label, color=cfg['color'], edgecolor='black', linewidth=0.4, hatch=hatch)
        valid = [v for v in vals if v == v]
        if valid: all_max = max(all_max, max(valid))
        for xi, v in enumerate(vals):
            if v == v:
                ax.text(x[xi] + offset, v + 0.5, f'{v:.1f}',
                        ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(TOPOS, fontsize=11)
    ax.set_ylabel('Voltage MAE (mV) at full data (N=4000 total)', fontsize=12)
    ax.set_title(
        'Final V MAE at full data across all methods × all topologies\n'
        'Joint shown twice: per-topology budget (20k total) vs honest total-budget (4k total, N/5 per topo)',
        fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, axis='y', alpha=0.3, linestyle=':')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig6_full_data_summary.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  saved fig6_full_data_summary.png')


def fig7_multi_metric_curves():
    """Data-efficiency curves V/I/gm/gds vs N for all 3 methods, averaged across 5 topos."""
    import json
    pn = OUT_DIR / 'metric_table_per_n.json'
    if not pn.exists():
        print('  fig7 skipped — metric_table_per_n.json not found (run eval_all_checkpoints first)')
        return
    with open(pn) as f:
        table = json.load(f)
    methods = [
        ('mlp', METHODS['mlp']),
        ('scratch', METHODS['scratch']),
        ('ftzeroshot', METHODS['ftzeroshot']),
        ('joint', METHODS['joint']),
        ('joint_total', METHODS['joint_total']),
    ]
    metric_specs = [
        ('v_mae_mV', 'V MAE (mV)'),
        ('i_mae_uA', 'I MAE (µA)'),
        ('gm_log_mae', 'gm log10-MAE'),
        ('gds_log_mae', 'gds log10-MAE'),
    ]

    # 4 metrics × 5 topos = 20 panels (one figure per metric, each with 5 topo curves)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    for ai, (key, label) in enumerate(metric_specs):
        ax = axes[ai]
        for m_key, cfg in methods:
            # Aggregate across topos: for each N, take mean over topos
            n_to_vals = {}  # N → list of values (one per topo where present)
            for t in TOPOS:
                topo_data = table.get(m_key, {}).get(t, {})
                for N_str, mv in topo_data.items():
                    v = mv.get(key)
                    if v is None or (isinstance(v, float) and (v != v)):
                        continue
                    n_to_vals.setdefault(int(N_str), []).append(v)
            Ns = sorted(n_to_vals.keys())
            ys = [np.mean(n_to_vals[n]) for n in Ns]
            if not Ns: continue
            ax.plot(Ns, ys, marker=cfg['marker'], linestyle=cfg['ls'],
                    color=cfg['color'], label=cfg['label'],
                    linewidth=2, markersize=8)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('Training samples N (joint: total budget, split N/5 per topology)', fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label + ' (mean over 5 topologies)', fontsize=13, fontweight='bold')
        ax.grid(True, which='both', alpha=0.3, linestyle=':')
        ax.set_xticks([10, 50, 100, 500, 1000, 4000])
        ax.set_xticklabels(['10', '50', '100', '500', '1k', '4k'])
    # Single shared legend below the grid — never overlaps the curves.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels),
               fontsize=11, frameon=True, framealpha=0.95,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('Multi-metric data efficiency: V / I / gm / gds vs N',
                 fontsize=14, fontweight='bold', y=1.00)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(OUT_DIR / 'fig7_multi_metric_curves.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  saved fig7_multi_metric_curves.png')


def fig8_joint_advantage_heatmap():
    """Heatmap: joint's % improvement over scratch per (topology × metric)."""
    import json
    table_path = OUT_DIR / 'metric_table.json'
    if not table_path.exists():
        print('  fig8 skipped — metric_table.json not found')
        return
    with open(table_path) as f:
        table = json.load(f)
    metric_keys = [
        ('v_mae_mV', 'V MAE'),
        ('i_mae_uA', 'I MAE'),
        ('gm_log_mae', 'gm MAE'),
        ('gds_log_mae', 'gds MAE'),
    ]
    H = np.full((len(TOPOS), len(metric_keys)), np.nan)
    for i, t in enumerate(TOPOS):
        for j, (k, _) in enumerate(metric_keys):
            scr = table.get('scratch', {}).get(t, {}).get(k)
            jnt = table.get('joint', {}).get(t, {}).get(k)
            if scr and jnt and scr > 0:
                H[i, j] = (scr - jnt) / scr * 100  # +ve means joint wins

    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(H, cmap='RdYlGn', aspect='auto', vmin=-50, vmax=80)
    ax.set_xticks(range(len(metric_keys)))
    ax.set_xticklabels([m[1] for m in metric_keys], fontsize=11)
    ax.set_yticks(range(len(TOPOS)))
    ax.set_yticklabels(TOPOS, fontsize=11)
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Topology', fontsize=12)
    for i in range(len(TOPOS)):
        for j in range(len(metric_keys)):
            v = H[i, j]
            if not np.isnan(v):
                color = 'white' if abs(v) > 50 else 'black'
                ax.text(j, i, f'{v:+.0f}%', ha='center', va='center',
                        color=color, fontsize=12, fontweight='bold')
    ax.set_title('Joint advantage over per-topology scratch (N=4000, full data)\n'
                 'Positive % = joint wins (lower MAE) — green = bigger win',
                 fontsize=13, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('% improvement of joint over scratch', fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'fig8_joint_advantage_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  saved fig8_joint_advantage_heatmap.png')


def main():
    print('Collecting data from training logs...')
    d = collect_data()
    print(f'Found data for {len(d["pertopo_full"])} per-topo, '
          f'{len(d["joint_full"])} joint, '
          f'{len(d["zeroshot_loo"])} LOO zero-shot.\n')
    fig1_data_efficiency(d)
    fig2_joint_vs_pertopo(d)
    fig3_zeroshot_heatmap(d)
    fig4_error_ratio(d)
    fig4b_sample_efficiency(d)
    fig5_gnn_vs_mlp(d)
    fig6_full_data_summary(d)
    fig7_multi_metric_curves()
    fig8_joint_advantage_heatmap()
    print(f'\nAll figures saved to {OUT_DIR}')


if __name__ == '__main__':
    main()
