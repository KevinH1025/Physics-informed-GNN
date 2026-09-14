#!/usr/bin/env python
"""Consolidated physics-ablation figure for the thesis. 4 panels:
  (a) DC gain head benefit — V MAE reduction, scratch full data, fan/sau
  (b) KCL residual — KCL-FT vs noKCL-FT relative current-conservation violation, all 5 topos
  (c) smaxt scorecard — Δ%(smaxt vs baseline) across 4 placements × 2 topos
  (d) Physics-pretrain 2×2 — col2 (phys) vs col3 (nophys), all 5 topos at N=500

Output: figures/thesis/fig_physics_ablation.png
"""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / 'datasets/opamp_3stage_pretrain_combined_5topo/experiments'
OUT = REPO / 'figures/thesis/fig_physics_ablation.png'

TOPOS = ['fan_smc', 'sau_cfcc', 'peng_tcfc', 'leung_nmcf', 'leung_nmcnr']
TOPO_LABEL = {t: t for t in TOPOS}
SH = {'fan_smc': 'fansmc', 'sau_cfcc': 'sau', 'peng_tcfc': 'peng',
      'leung_nmcf': 'leungf', 'leung_nmcnr': 'leungnr'}
T2 = {t: ('fansmc' if t == 'fan_smc' else t) for t in TOPOS}


def true_min(log: Path, topo: str):
    if not log.exists():
        return None
    pat = re.compile(rf'{topo}=([0-9.]+)')
    vmin = None
    for line in open(log):
        m = pat.search(line)
        if m:
            v = float(m.group(1))
            vmin = v if vmin is None else min(vmin, v)
    return vmin


def panel_a_dc(ax):
    """DC gain head benefit, scratch full data."""
    topos = ['fan_smc', 'sau_cfcc']
    no_dc, with_dc = [], []
    for t in topos:
        nd = true_min(EXP / f'v5_5topo_pertopo_{t}/training.log', t)
        no_dc.append(nd)
        if t == 'fan_smc':
            dc = true_min(EXP / 'v5_5topo_pertopo_fansmc_dcgain_n4000/training.log', t)
        else:
            vals = [true_min(EXP / f'v5_5topo_pertopo_sau_cfcc_dcgain_n4000_s{s}/training.log', t)
                    for s in (42, 43, 44)]
            vals = [v for v in vals if v is not None]
            dc = np.mean(vals) if vals else None
        with_dc.append(dc)
    x = np.arange(len(topos))
    w = 0.35
    ax.bar(x - w/2, no_dc, w, label='no DC head', color='#d62728', edgecolor='black', linewidth=0.5)
    ax.bar(x + w/2, with_dc, w, label='+ DC gain head (physics)', color='#2ca02c', edgecolor='black', linewidth=0.5)
    for i, (a, b) in enumerate(zip(no_dc, with_dc)):
        delta = (a - b) / a * 100
        ax.text(i, max(a, b) * 1.03, f'−{delta:.0f}%', ha='center', fontsize=12,
                color='darkgreen', fontweight='bold')
        ax.text(i - w/2, a + 0.3, f'{a:.1f}', ha='center', fontsize=10)
        ax.text(i + w/2, b + 0.3, f'{b:.1f}', ha='center', fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(topos, fontsize=11)
    ax.set_ylabel('V MAE (mV) — lower is better', fontsize=11)
    ax.set_title('(a) DC gain head: physics-equation architecture\n'
                 'nearly halves V MAE (scratch, full data)',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, axis='y', alpha=0.3, linestyle=':')
    ax.set_ylim(0, max(no_dc) * 1.2)


def panel_b_kcl(ax):
    """KCL residual: KCL-FT vs noKCL-FT, all 5 topos."""
    p = REPO / 'figures/thesis/kcl_residual_indist.json'
    if not p.exists():
        ax.text(0.5, 0.5, 'kcl_residual_indist.json missing', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title('(b) KCL residual (data missing)', fontsize=12, fontweight='bold')
        return
    data = json.load(open(p))
    topos = TOPOS
    kcl_ft = [data[t]['kcl_ft'][0] * 100 for t in topos]
    no_kcl = [data[t]['nokcl_ft'][0] * 100 for t in topos]
    x = np.arange(len(topos))
    w = 0.35
    ax.bar(x - w/2, no_kcl, w, label='no-KCL FT', color='#d62728', edgecolor='black', linewidth=0.5)
    ax.bar(x + w/2, kcl_ft, w, label='+ KCL loss FT (physics)', color='#2ca02c', edgecolor='black', linewidth=0.5)
    for i, (n, k) in enumerate(zip(no_kcl, kcl_ft)):
        delta = (n - k) / n * 100
        ax.text(i, max(n, k) * 1.03, f'−{delta:.0f}%', ha='center', fontsize=11,
                color='darkgreen', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels([t.replace('_', '\n') for t in topos], fontsize=9)
    ax.set_ylabel('Mean rel. KCL violation (%) — lower is better', fontsize=11)
    ax.set_title('(b) KCL loss → physically consistent predictions\n'
                 'currents conserve at every node, all 5 topologies',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, axis='y', alpha=0.3, linestyle=':')


def panel_c_smaxt(ax):
    """smaxt Δ% across 4 placements × 2 topos."""
    # Placements: joint, scratch, pretrain→FT, FT-only
    # For each, compute Δ% (smaxt vs baseline). Negative = smaxt helps.
    # joint: avg V MAE across topos — smaxt 13.42/16.44/17.24 mean=15.7 vs baseline 17.2 → -8.7% (but represents avg, not per-topo)
    # We show the joint result under both fan and sau columns (it's an avg, mark with hatch)
    # scratch: fan (smaxt 6.97 vs base 7.57 = -8%), sau (smaxt 14.27 vs base 13.13 = +9%)
    # pretrain→FT avg of N=100/500/1000: fan avg, sau avg
    # FT-only avg of N=100/500/1000: fan avg, sau avg

    def avg_delta(sm_dirs, base_dirs, topo):
        rs = []
        for sd, bd in zip(sm_dirs, base_dirs):
            sv = true_min(EXP / f'{sd}/training.log', topo)
            bv = true_min(EXP / f'{bd}/training.log', topo)
            if sv is not None and bv is not None:
                rs.append((sv - bv) / bv * 100)
        return np.mean(rs) if rs else None

    # joint smaxt avg vs baseline avg
    j_smaxt_means = []
    for s in ('', '_s43', '_s44'):
        log = EXP / f'v5_5topo_joint_pl_smaxt{s}/training.log'
        line = [l for l in open(log) if '[BEST]' in l][-1] if log.exists() else ''
        vals = [float(m) for m in re.findall(r'[a-z_]+=([0-9.]+)', line.split('per-topo V MAE (mV):')[-1])] if 'per-topo' in line else []
        if vals: j_smaxt_means.append(np.mean(vals))
    j_smaxt = np.mean(j_smaxt_means) if j_smaxt_means else 15.7
    log_b = EXP / 'v5_5topo_joint_pl_baseline/training.log'
    line_b = [l for l in open(log_b) if '[BEST]' in l][-1] if log_b.exists() else ''
    bv = [float(m) for m in re.findall(r'[a-z_]+=([0-9.]+)', line_b.split('per-topo V MAE (mV):')[-1])] if 'per-topo' in line_b else []
    j_base = np.mean(bv) if bv else 17.2
    j_delta = (j_smaxt - j_base) / j_base * 100

    # scratch (single-topo full): 3 seeds each
    def avg_seeds(dirs, topo):
        v = [true_min(EXP / d, topo) for d in dirs]; v = [x for x in v if x is not None]
        return np.mean(v) if v else None
    fan_sc_sm = avg_seeds([f'v5_5topo_pertopo_fansmc_dcgain_smaxt_n4000_s{s}/training.log' for s in (42,43,44)], 'fan_smc')
    fan_sc_b  = avg_seeds([f'v5_5topo_pertopo_fansmc_dcgain_n4000{("_s"+str(s)) if s!=42 else ""}/training.log' for s in (42,43,44)], 'fan_smc')
    sau_sc_sm = avg_seeds([f'v5_5topo_pertopo_sau_cfcc_dcgain_smaxt_n4000_s{s}/training.log' for s in (42,43,44)], 'sau_cfcc')
    sau_sc_b  = avg_seeds([f'v5_5topo_pertopo_sau_cfcc_dcgain_n4000_s{s}/training.log' for s in (42,43,44)], 'sau_cfcc')
    fan_sc_d = (fan_sc_sm - fan_sc_b)/fan_sc_b*100
    sau_sc_d = (sau_sc_sm - sau_sc_b)/sau_sc_b*100

    # pretrain→FT avg of N=100/500/1000
    fan_pre_d = avg_delta([f'v5_5topo_ftzs_fansmc_smaxtPRE_n{n}' for n in (100,500,1000)],
                          [f'v5_5topo_ftzs_fansmc_phys_nophysFT_n{n}' for n in (100,500,1000)], 'fan_smc')
    sau_pre_d = avg_delta([f'v5_5topo_ftzs_sau_smaxtPRE_n{n}' for n in (100,500,1000)],
                          [f'v5_5topo_ftzs_sau_phys_nophysFT_n{n}' for n in (100,500,1000)], 'sau_cfcc')
    # FT-only avg of N=100/500/1000
    fan_ft_d = avg_delta([f'v5_5topo_ftzs_fansmc_smaxtFT_n{n}' for n in (100,500,1000)],
                          [f'v5_5topo_ftzs_fansmc_phys_nophysFT_n{n}' for n in (100,500,1000)], 'fan_smc')
    sau_ft_d = avg_delta([f'v5_5topo_ftzs_sau_smaxtFT_n{n}' for n in (100,500,1000)],
                          [f'v5_5topo_ftzs_sau_phys_nophysFT_n{n}' for n in (100,500,1000)], 'sau_cfcc')

    placements = ['joint\n(avg)', 'scratch', 'pretrain→FT', 'FT only']
    fan_deltas = [j_delta, fan_sc_d, fan_pre_d, fan_ft_d]
    sau_deltas = [j_delta, sau_sc_d, sau_pre_d, sau_ft_d]
    x = np.arange(len(placements))
    w = 0.35
    bf = ax.bar(x - w/2, fan_deltas, w, label='fan_smc (easy)', color='#1f77b4', edgecolor='black', linewidth=0.5)
    bs = ax.bar(x + w/2, sau_deltas, w, label='sau_cfcc (hard)', color='#ff7f0e', edgecolor='black', linewidth=0.5)
    for i, d in enumerate(fan_deltas):
        ax.text(i - w/2, d + (1 if d >= 0 else -2), f'{d:+.0f}%', ha='center', fontsize=10)
    for i, d in enumerate(sau_deltas):
        ax.text(i + w/2, d + (1 if d >= 0 else -2), f'{d:+.0f}%', ha='center', fontsize=10)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(placements, fontsize=10)
    ax.set_ylabel('Δ V MAE (%) — negative = smaxt helps', fontsize=11)
    ax.set_title('(c) smaxt gm-loss tested in 4 placements × 2 topos\n'
                 'no reliable benefit anywhere; topology-dep noise',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, axis='y', alpha=0.3, linestyle=':')


def panel_c_pretrain(ax):
    """Physics-pretrain 2x2: col2 (phys) vs col3 (nophys), all 5 topos, N=500.
    Renamed from panel (d) → (c) after removing smaxt panel."""
    N = 500
    col2, col3 = [], []
    for t in TOPOS:
        s = SH[t]; t2 = T2[t]
        c2 = true_min(EXP / f'v5_5topo_ftzs_{s}_phys_nophysFT_n{N}/training.log', t)
        c3 = true_min(EXP / f'v5_5topo_ftzs_{s}_nophys_nophysFT_n{N}/training.log', t)
        col2.append(c2); col3.append(c3)
    x = np.arange(len(TOPOS))
    w = 0.35
    ax.bar(x - w/2, col2, w, label='phys pretrain (KCL+loop ON)', color='#2ca02c', edgecolor='black', linewidth=0.5)
    ax.bar(x + w/2, col3, w, label='nophys pretrain (KCL+loop OFF)', color='#d62728', edgecolor='black', linewidth=0.5)
    for i, (a, b) in enumerate(zip(col2, col3)):
        winner = 'phys' if a < b else 'nophys'
        col = 'darkgreen' if a < b else 'firebrick'
        delta = abs(a - b) / max(a, b) * 100
        ax.text(i, max(a, b) * 1.03, f'{winner} −{delta:.0f}%', ha='center', fontsize=10,
                color=col, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels([t.replace('_', '\n') for t in TOPOS], fontsize=9)
    ax.set_ylabel('V MAE (mV) — lower is better', fontsize=11)
    ax.set_title('(c) Physics-pretrain (KCL+loop) effect, N=500\n'
                 'helps sau strongly, hurts leung_nmcnr; topology-dependent',
                 fontsize=12, fontweight='bold')
    # Headroom so winner-label text + bars stay below the legend
    ymax = max(max(col2), max(col3)) * 1.35
    ax.set_ylim(0, ymax)
    # Legend anchored outside the top-right corner so it never overlaps bars/labels
    ax.legend(loc='upper right', fontsize=10, framealpha=0.95,
              bbox_to_anchor=(1.0, 1.0))
    ax.grid(True, axis='y', alpha=0.3, linestyle=':')


def main():
    # 3-panel layout: (a) DC head, (b) KCL residual, (c) physics-pretrain 2x2.
    # smaxt scorecard removed per request.
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    panel_a_dc(axes[0])
    panel_b_kcl(axes[1])
    panel_c_pretrain(axes[2])
    fig.suptitle('Physics-Informed GNN — Ablation of Physics Components',
                 fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {OUT}')


if __name__ == '__main__':
    main()
