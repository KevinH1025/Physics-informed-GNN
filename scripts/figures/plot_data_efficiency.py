#!/usr/bin/env python
"""§4.3 Data Efficiency curves: V/I/gm/gds/DC MAE vs training size.
Two lines per panel: KCL ON vs OFF.
Sizes plotted: 500, 1000, 2000, 5000 (n=250 excluded, awaiting multi-seed)."""
import re, os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / 'datasets/opamp_3stage_fan_smc_openloop_5k/experiments'
PREFIX = 'ginebn_3layer_bb8_notower_vn_kcl_w10_edge6_netrole_clip03_openloop_5k_DATAEFF'
USING = 'ginebn_3layer_bb8_notower_vn_kcl_w10_edge6_netrole_clip03_openloop_5k_usingnow'
OUT = REPO / 'figures/thesis/fig_data_efficiency.png'

SIZES = [250, 500, 1000, 2000, 5000]
WD_SIZES = [250, 500]   # KCL-OFF + WD=1e-3 control only at small sizes


def parse(d):
    with open(d / 'training.log') as f:
        t = f.read()
    last = lambda pat, g=1, default=None: (list(re.finditer(pat, t)) or [None])[-1].group(g) if re.search(pat, t) else default
    m = re.search(r"--- DC Gain Evaluation.*?MAE: ([0-9.]+) dB", t, re.DOTALL)
    dc = float(m.group(1)) if m else None
    return {
        'V': float(last(r"Best Val MAE: ([0-9.]+)mV")),
        'I': float(last(r"Best Val Current MAE: ([0-9.]+)µA")),
        'gm': float(last(r"gm  MAE: ([0-9.]+) log10")),
        'gds': float(last(r"gds MAE: ([0-9.]+) log10")),
        'DC': dc,
    }


def load_all():
    rows = {kcl: {} for kcl in ('On', 'Off')}
    rows['WD'] = {}   # KCL-OFF + WD=1e-3 control
    for s in SIZES:
        for kcl in ('On', 'Off'):
            paths = []
            if s == 5000 and kcl == 'On':
                paths.append(EXP / USING)
            else:
                paths.append(EXP / f"{PREFIX}_n{s}_kcl{kcl}_s88")
            # For n=250 also include seed=42 rerun
            if s == 250:
                extra = EXP / f"{PREFIX}_n{s}_kcl{kcl}_s42"
                if extra.exists(): paths.append(extra)
            metrics_list = [parse(p) for p in paths if (p/'training.log').exists()]
            rows[kcl][s] = {q: sum(m[q] for m in metrics_list)/len(metrics_list)
                            for q in metrics_list[0]}
        # WD control only at small sizes
        if s in WD_SIZES:
            p = EXP / f"{PREFIX}_n{s}_kclOff_wd1e-3_s88"
            if (p/'training.log').exists():
                rows['WD'][s] = parse(p)
    return rows


def main():
    data = load_all()
    # 3+2 layout: row 1 has V|I|gm, row 2 has gds|DC centered
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(2, 6, hspace=0.32, wspace=0.45,
                          left=0.05, right=0.98, top=0.97, bottom=0.07)
    quantities = [
        ('V',   'V MAE (mV)',         '(a) Voltage'),
        ('I',   'I MAE (µA)',         '(b) Current'),
        ('gm',  'gm log10-MAE',       '(c) Transconductance gm'),
        ('gds', 'gds log10-MAE',      '(d) Output conductance gds'),
        ('DC',  'DC gain MAE (dB)',   '(e) DC gain'),
    ]
    panels = [
        fig.add_subplot(gs[0, 0:2]),
        fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[0, 4:6]),
        fig.add_subplot(gs[1, 1:3]),
        fig.add_subplot(gs[1, 3:5]),
    ]
    for ax, (q, ylabel, tag) in zip(panels, quantities):
        x = SIZES
        y_on  = [data['On'][s][q]  for s in x]
        y_off = [data['Off'][s][q] for s in x]
        ax.plot(x, y_on,  '-o', linewidth=2.2, markersize=11,
                color='#2ca02c', label='KCL ON')
        ax.plot(x, y_off, '-s', linewidth=2.2, markersize=11,
                color='#d62728', label='KCL OFF')
        wd_x, wd_y = [], []   # WD overlay disabled in main fig
        ax.set_xscale('log')
        ax.set_xlabel('Training set size', fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(tag, fontsize=14, fontweight='bold', loc='left')
        ax.grid(True, alpha=0.3, which='both', linestyle=':')
        ax.legend(loc='upper right', fontsize=11, frameon=True, framealpha=0.92)
        ax.set_xticks(x); ax.set_xticklabels([str(s) for s in x], fontsize=11)
        ax.tick_params(axis='y', labelsize=11)
        # Smart placement: higher value goes above, lower goes below
        bbox = dict(boxstyle='round,pad=0.1', facecolor='white', edgecolor='none', alpha=0.85)
        for xi, yon, yoff in zip(x, y_on, y_off):
            on_above = yon >= yoff
            ax.annotate(f'{yon:.2f}', (xi, yon), textcoords='offset points',
                        xytext=(0, 11 if on_above else -16), ha='center',
                        fontsize=9, color='#2ca02c', fontweight='bold', bbox=bbox)
            ax.annotate(f'{yoff:.2f}', (xi, yoff), textcoords='offset points',
                        xytext=(0, -16 if on_above else 11), ha='center',
                        fontsize=9, color='#d62728', fontweight='bold', bbox=bbox)
        # WD annotations (always above)
        if wd_x:
            for xi, yw in zip(wd_x, wd_y):
                ax.annotate(f'{yw:.2f}', (xi, yw), textcoords='offset points',
                            xytext=(0, 11), ha='center', fontsize=9,
                            color='#7f7f7f', fontweight='bold', bbox=bbox)
        # Add headroom for top annotations
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin - 0.08 * (ymax - ymin), ymax + 0.10 * (ymax - ymin))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {OUT}')

    # Quick summary
    print(f"\n=== KCL advantage per size (V MAE) ===")
    print(f"{'n_train':<10}{'KCL ON':>10}{'KCL OFF':>10}{'Δ(OFF-ON)':>12}{'%benefit':>10}")
    for s in SIZES:
        on, off = data['On'][s]['V'], data['Off'][s]['V']
        d = off - on
        pct = 100 * d / off
        print(f"{s:<10}{on:>10.2f}{off:>10.2f}{d:>+12.2f}{pct:>+10.1f}%")


def __dummy(): pass


if __name__ == '__main__':
    main()
