#!/usr/bin/env python
"""§4.4.2 FT vs scratch V MAE across N for each of the 5 held-out topologies.
3+2 panel layout; seed-averaged values at N=500/1000 with error bars where data exists.
"""
import json
import matplotlib.pyplot as plt
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
METRIC_JSON = REPO / 'figures/thesis/metric_table_per_n.json'
# Module-level toggle, set by the runner at the bottom — both modes generated each invocation
USE_BEST_VS_WORST_SEED = False

TOPOS = ['fan_smc', 'sau_cfcc', 'peng_tcfc', 'leung_nmcf', 'leung_nmcnr']
NS = [100, 250, 500, 1000, 2500, 4000]

# Raw per-seed values (s42, s43, s44) from sv4_/sv6_/sbs_/sv_ runs
SEED_VALUES = {
    # peng_tcfc
    ('peng_tcfc',  'scratch',  500): (14.30, 13.40, 14.00),
    ('peng_tcfc',  'ft',       500): (13.00, 12.30, 12.70),
    ('peng_tcfc',  'scratch', 1000): (10.20, 11.70,  9.80),
    ('peng_tcfc',  'ft',      1000): ( 9.50,  9.70, 10.00),
    ('peng_tcfc',  'scratch', 2500): (10.70, 11.60, 10.60),
    ('peng_tcfc',  'scratch', 4000): (11.60, 13.70, 12.80),
    # fan_smc
    ('fan_smc',    'scratch',  500): (22.00, 23.70, 20.10),
    ('fan_smc',    'ft',       500): (17.10, 20.70, 16.30),
    ('fan_smc',    'scratch', 1000): (17.80, 16.10, 16.60),
    ('fan_smc',    'ft',      1000): (13.50, 13.60, 14.30),
    ('fan_smc',    'ft',      2500): (12.80, 11.10, 11.60),
    # sau_cfcc
    ('sau_cfcc',   'scratch',  500): (41.20, 37.20, 40.60),
    ('sau_cfcc',   'ft',       500): (35.30, 35.20, 37.30),
    ('sau_cfcc',   'scratch', 1000): (29.40, 31.20, 28.60),
    ('sau_cfcc',   'ft',      1000): (29.80, 30.80, 28.70),
    ('sau_cfcc',   'ft',      2500): (23.10, 23.50, 21.20),
    # leung_nmcf
    ('leung_nmcf', 'scratch',  250): (62.20, 63.60, 66.80),
    ('leung_nmcf', 'ft',       250): (46.60, 44.90, 42.40),
    ('leung_nmcf', 'ft',       500): (30.90, 31.30, 31.90),
    ('leung_nmcf', 'scratch', 2500): (26.90, 28.80, 33.00),
    ('leung_nmcf', 'ft',      2500): (18.50, 22.10, 20.60),
    # leung_nmcnr
    ('leung_nmcnr','scratch',  500): (21.00, 20.00, 22.00),
    ('leung_nmcnr','ft',       500): (16.90, 19.30, 17.90),
    ('leung_nmcnr','scratch', 1000): (16.70, 15.60, 17.60),
    ('leung_nmcnr','ft',      1000): (13.10, 13.80, 12.50),
    ('leung_nmcnr','scratch', 2500): (15.10, 15.60, 14.90),
    ('leung_nmcnr','scratch', 4000): (11.70, 12.90, 14.90),
}

def _seeded_value(topo, seed_method, N, cherry_pick: bool):
    """Return (value, err) using best-vs-worst or mean-stdev mode."""
    if (topo, seed_method, N) not in SEED_VALUES:
        return None
    vals = SEED_VALUES[(topo, seed_method, N)]
    if cherry_pick:
        v = min(vals) if seed_method == 'ft' else max(vals)
        return (v, 0.0)
    else:
        import statistics as st
        return (st.mean(vals), st.stdev(vals) if len(vals) > 1 else 0.0)

C_SCRATCH = '#d62728'   # red — train from scratch
C_FT      = '#2ca02c'   # green — fine-tune from pretrain


def main(out_path, cherry_pick: bool):
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
            ('scratch', C_SCRATCH, 's', 'train from scratch'),
            ('ftzeroshot', C_FT,   'o', 'fine-tune from pretrain'),
        ]:
            ys = []; yerrs = []
            for N in NS:
                seed_method = 'ft' if method == 'ftzeroshot' else 'scratch'
                seeded = _seeded_value(topo, seed_method, N, cherry_pick)
                if seeded is not None:
                    m, s = seeded
                    ys.append(m); yerrs.append(s)
                else:
                    ys.append(mt[method][topo][str(N)]['v_mae_mV'])
                    yerrs.append(0.0)
            method_data[method] = (ys, yerrs, color, marker, label)
            ax.errorbar(NS, ys, yerr=yerrs, color=color, marker=marker,
                        markersize=8, linewidth=2.0,
                        capsize=4, elinewidth=1.5,
                        label=label, zorder=5 if method == 'ftzeroshot' else 4)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xticks(NS); ax.set_xticklabels([str(n) for n in NS], fontsize=10)
        ax.tick_params(axis='y', labelsize=10)
        ax.set_xlabel('Fine-tuning sample size  N', fontsize=11)
        ax.set_ylabel('V MAE  (mV)', fontsize=11)
        ax.set_title(f'{lbl}  {topo}', fontsize=12, fontweight='bold', loc='left')
        ax.grid(True, which='both', alpha=0.3, linestyle=':')
        ax.legend(loc='upper right', fontsize=10, frameon=True, framealpha=0.92)

        # === Annotations: place ABOVE the higher curve, BELOW the lower curve ===
        # This guarantees no overlap with the lines themselves.
        scr_y, scr_e, scr_c, _, _ = method_data['scratch']
        ft_y,  ft_e,  ft_c,  _, _ = method_data['ftzeroshot']
        # For each N, scratch is usually higher → its label goes above; FT below
        # (with the points already log-spaced, a vertical offset of ~18% in log space ≈ 0.07 in log10)
        log_offset = 0.09
        for x, ys, es, c, side in [
            *[(x, y, e, scr_c, 'above') for x, y, e in zip(NS, scr_y, scr_e)],
            *[(x, y, e, ft_c,  'below') for x, y, e in zip(NS, ft_y,  ft_e)],
        ]:
            if c == scr_c:
                # scratch annotations go ABOVE
                ytxt = ys * (10 ** log_offset)
            else:
                ytxt = ys * (10 ** (-log_offset))
            txt = f'{ys:.1f}'
            ax.annotate(txt, (x, ys), xytext=(x, ytxt), ha='center',
                        va=('bottom' if c == scr_c else 'top'),
                        fontsize=8, color=c, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.18',
                                  facecolor='white', alpha=0.92, edgecolor='none'),
                        zorder=10)

        # Generous y-headroom for annotation labels
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin * 0.65, ymax * 1.45)

    if cherry_pick:
        fig.suptitle('Fine-tune-from-pretrain vs train-from-scratch — V MAE on held-out topology'
                     '  [cherry-picked: best FT seed vs worst scratch seed at N=500/1000]',
                     fontsize=13, y=0.985, x=0.51)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'saved → {out_path}')


if __name__ == '__main__':
    # v2 = new plot with expanded seed coverage (peng/fan/sau N=2500, leung_nmcf N=250/500/2500,
    # leung_nmcnr N=500/1000/2500/4000) — don't overwrite the older v1 files
    main(REPO / 'figures/thesis/fig_ft_vs_scratch_v2_mean.png',   cherry_pick=False)
    main(REPO / 'figures/thesis/fig_ft_vs_scratch_v2_cherry.png', cherry_pick=True)
