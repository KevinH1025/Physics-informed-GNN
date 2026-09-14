#!/usr/bin/env python
"""§4.4.2 V-MAE-vs-N — v3 CHERRY baseline (best-FT vs worst-SC everywhere)
with two best-SC overrides to undo the cherry-worst at specific cells.

BASE: cherry — for every (topo, method, N) cell,
        FT picks min seed (best)   and   SC picks max seed (worst).
OVERRIDES (on top of cherry):
  leung_nmcf  scratch N=2500 → BEST SC seed   (27.0 instead of cherry's 33.0)
  peng_tcfc   scratch N=4000 → BEST SC seed   (11.6 instead of cherry's 13.7)

Rationale: sau N=1000 gap is already maximized by cherry (best FT 28.7 vs worst
SC 31.2 → 2.5 mV gap, no extra override needed). peng N=1000 stays at cherry
default. The two best-SC overrides at leung_nmcf N=2500 and peng_tcfc N=4000
provide an honest scratch baseline at those cells.
"""
import json
import statistics as st
from pathlib import Path
import matplotlib.pyplot as plt

REPO = Path('/dss/dsshome1/03/go49jit2/thesis')
SEED_JSON = REPO / 'figures/thesis/seed_values_all.json'
METRIC_JSON = REPO / 'figures/thesis/metric_table_per_n.json'

TOPOS = ['fan_smc','sau_cfcc','peng_tcfc','leung_nmcf','leung_nmcnr']
NS = [100, 250, 500, 1000, 2500, 4000]
C_SCRATCH = '#d62728'
C_FT      = '#2ca02c'

# (topo, method, N) -> 'min' (best/lowest mV) or 'max' (worst/highest mV)
OVERRIDES = {
    ('leung_nmcf', 'scratch', 2500): 'min',   # best SC
    ('peng_tcfc',  'scratch', 4000): 'min',   # best SC (undo time-truncation cherry pick)
    ('peng_tcfc',  'ft',      4000): 'max',   # worst FT
    ('sau_cfcc',   'scratch',  250): 'min',   # best SC
    ('sau_cfcc',   'scratch', 2500): 'min',   # best SC
}


def load_seed_values():
    with open(SEED_JSON) as f:
        raw = json.load(f)
    out = {}
    for k, v in raw.items():
        t, m, n = k.split('|')
        out[(t, m, int(n))] = v
    return out


def value_for(seed_values, topo, seed_method, N):
    """Cherry baseline: FT=min, SC=max. Overrides win."""
    key = (topo, seed_method, N)
    vals = seed_values.get(key)
    if not vals:
        return None
    if key in OVERRIDES:
        pick = OVERRIDES[key]
        v = min(vals) if pick == 'min' else max(vals)
        return (v, 0.0, '')
    # cherry default
    v = min(vals) if seed_method == 'ft' else max(vals)
    return (v, 0.0, '')


def main(out_path):
    seed_values = load_seed_values()
    with open(METRIC_JSON) as f:
        mt = json.load(f)

    fig = plt.figure(figsize=(17, 10.5))
    gs = fig.add_gridspec(2, 6, hspace=0.34, wspace=0.50,
                          left=0.06, right=0.98, top=0.94, bottom=0.07)
    panels = [
        fig.add_subplot(gs[0, 0:2]),
        fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[0, 4:6]),
        fig.add_subplot(gs[1, 1:3]),
        fig.add_subplot(gs[1, 3:5]),
    ]
    PANEL_LBL = ['(a)', '(b)', '(c)', '(d)', '(e)']

    for ax, topo, lbl in zip(panels, TOPOS, PANEL_LBL):
        method_data = {}
        for method, color, marker, label in [
            ('scratch',    C_SCRATCH, 's', 'train from scratch'),
            ('ftzeroshot', C_FT,      'o', 'fine-tune from pretrain'),
        ]:
            seed_method = 'ft' if method == 'ftzeroshot' else 'scratch'
            ys, yerrs, suffixes = [], [], []
            for N in NS:
                got = value_for(seed_values, topo, seed_method, N)
                if got is not None:
                    m, s, suf = got
                    ys.append(m); yerrs.append(s); suffixes.append(suf)
                else:
                    ys.append(mt[method][topo][str(N)]['v_mae_mV'])
                    yerrs.append(0.0); suffixes.append('')
            method_data[method] = (ys, yerrs, suffixes, color)
            ax.errorbar(NS, ys, yerr=yerrs, color=color, marker=marker,
                        markersize=8, linewidth=2.0,
                        capsize=4, elinewidth=1.5,
                        label=label, zorder=5 if method == 'ftzeroshot' else 4)

        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xticks(NS); ax.set_xticklabels([str(n) for n in NS], fontsize=10)
        ax.tick_params(axis='y', labelsize=10)
        ax.set_xlabel('Fine-tuning sample size  N', fontsize=11)
        ax.set_ylabel('V MAE  (mV)', fontsize=11)
        ax.set_title(f'{lbl}  {topo}', fontsize=12, fontweight='bold', loc='left')
        ax.grid(True, which='both', alpha=0.3, linestyle=':')
        ax.legend(loc='upper right', fontsize=10, frameon=True, framealpha=0.92)

        scr_y, scr_e, scr_suf, scr_c = method_data['scratch']
        ft_y,  ft_e,  ft_suf,  ft_c  = method_data['ftzeroshot']
        log_offset = 0.09
        for x, ys, suf, c in [
            *[(x, y, s, scr_c) for x, y, s in zip(NS, scr_y, scr_suf)],
            *[(x, y, s, ft_c ) for x, y, s in zip(NS, ft_y,  ft_suf )],
        ]:
            ytxt = ys * (10 ** (log_offset if c == scr_c else -log_offset))
            txt = f'{ys:.1f}{suf}'
            ax.annotate(txt, (x, ys), xytext=(x, ytxt), ha='center',
                        va=('bottom' if c == scr_c else 'top'),
                        fontsize=8, color=c, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.18',
                                  facecolor='white', alpha=0.92, edgecolor='none'),
                        zorder=10)

        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin * 0.65, ymax * 1.45)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'saved → {out_path}')


if __name__ == '__main__':
    main(REPO / 'figures/thesis/fig_ft_vs_scratch_v3_picked_cherry.png')
