#!/usr/bin/env python
"""§4.3 Data Efficiency curves — v2 with proper labels + 3-seed mean.

Differences from v1:
  - Uses the new fan_kcl seed sweep: fan_kcl_{on,off}_n{N}_s{seed} for
    seed ∈ {42, 43, 44}. Mean across seeds is plotted; error bars = std.
  - Correct N labels: 100, 250, 500, 1000, 2500, 4000 (v1 called
    the full-train-set point "5000" but fan_smc actually has 4000 train +
    1000 val = 5000 TOTAL; the training-set size is 4000, not 5000).
  - Never overwrites v1: writes fig_data_efficiency_v2.png.
"""
import re, statistics as st
from pathlib import Path
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / 'datasets/opamp_3stage_fan_smc_openloop_5k/experiments'
OUT = REPO / 'figures/thesis/fig_data_efficiency_v2.png'
SIZES = [100, 250, 500, 1000, 2500, 4000]
SEEDS = [42, 43, 44, 45, 46, 47]


def parse(d):
    p = d / 'training.log'
    if not p.exists():
        return None
    with open(p) as f:
        t = f.read()
    last = lambda pat, g=1: (list(re.finditer(pat, t)) or [None])[-1].group(g) if re.search(pat, t) else None
    m = re.search(r"--- DC Gain Evaluation.*?MAE:\s+([0-9.]+)\s+dB", t, re.DOTALL)
    dc = float(m.group(1)) if m else None
    try:
        return {
            'V':   float(last(r"Best Val MAE:\s*([0-9.]+)\s*mV")),
            'I':   float(last(r"Best Val Current MAE:\s*([0-9.]+)\s*µA")),
            'gm':  float(last(r"gm  MAE:\s*([0-9.]+)\s*log10")),
            'gds': float(last(r"gds MAE:\s*([0-9.]+)\s*log10")),
            'DC':  dc,
        }
    except (TypeError, ValueError):
        return None


def load_all():
    """rows[kcl][N][metric] = list of per-seed values; also _stats derived."""
    rows = {kcl: {} for kcl in ('On', 'Off')}
    for N in SIZES:
        for kcl in ('On', 'Off'):
            per_seed = []
            for s in SEEDS:
                d = EXP / f'fan_kcl_{kcl.lower()}_n{N}_s{s}'
                m = parse(d)
                if m is not None:
                    per_seed.append(m)
            if not per_seed:
                rows[kcl][N] = None
                continue
            aggr = {}
            for q in ('V','I','gm','gds','DC'):
                vals = [m[q] for m in per_seed if m.get(q) is not None]
                aggr[q] = {
                    'mean': st.mean(vals) if vals else None,
                    'std':  st.stdev(vals) if len(vals) > 1 else 0.0,
                    'n':    len(vals),
                }
            rows[kcl][N] = aggr
    return rows


def main():
    data = load_all()
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(2, 6, hspace=0.32, wspace=0.45,
                          left=0.05, right=0.98, top=0.95, bottom=0.07)
    quantities = [
        ('V',   'V MAE (mV)',        '(a) Voltage'),
        ('I',   'I MAE (µA)',        '(b) Current'),
        ('gm',  'gm log10-MAE',      '(c) Transconductance gm'),
        ('gds', 'gds log10-MAE',     '(d) Output conductance gds'),
        ('DC',  'DC gain MAE (dB)',  '(e) DC gain'),
    ]
    panels = [
        fig.add_subplot(gs[0, 0:2]),
        fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[0, 4:6]),
        fig.add_subplot(gs[1, 1:3]),
        fig.add_subplot(gs[1, 3:5]),
    ]
    for ax, (q, ylabel, tag) in zip(panels, quantities):
        for kcl, color, marker, label in [
            ('On',  '#2ca02c', 'o', 'KCL ON'),
            ('Off', '#d62728', 's', 'KCL OFF'),
        ]:
            xs, ys, es, ns = [], [], [], []
            for N in SIZES:
                row = data[kcl].get(N)
                if row is None or row[q]['mean'] is None:
                    continue
                xs.append(N); ys.append(row[q]['mean']); es.append(row[q]['std']); ns.append(row[q]['n'])
            ax.errorbar(xs, ys, yerr=es, fmt='-'+marker, linewidth=2.2, markersize=10,
                        color=color, label=label, capsize=4, elinewidth=1.3)
        ax.set_xscale('log')
        ax.set_xlabel('Training set size', fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(tag, fontsize=14, fontweight='bold', loc='left')
        ax.grid(True, alpha=0.3, which='both', linestyle=':')
        ax.legend(loc='upper right', fontsize=11, frameon=True, framealpha=0.92)
        ax.set_xticks(SIZES); ax.set_xticklabels([str(s) for s in SIZES], fontsize=11)
        ax.tick_params(axis='y', labelsize=11)
        # Annotate points with value + n_seeds indicator
        bbox = dict(boxstyle='round,pad=0.1', facecolor='white', edgecolor='none', alpha=0.85)
        y_on  = [(data['On'][N][q]['mean'] if data['On'].get(N) and data['On'][N][q]['mean'] is not None else None) for N in SIZES]
        y_off = [(data['Off'][N][q]['mean'] if data['Off'].get(N) and data['Off'][N][q]['mean'] is not None else None) for N in SIZES]
        for N, yon, yoff in zip(SIZES, y_on, y_off):
            if yon is None or yoff is None: continue
            on_above = yon >= yoff
            ax.annotate(f'{yon:.2f}', (N, yon), textcoords='offset points',
                        xytext=(0, 11 if on_above else -16), ha='center',
                        fontsize=9, color='#2ca02c', fontweight='bold', bbox=bbox)
            ax.annotate(f'{yoff:.2f}', (N, yoff), textcoords='offset points',
                        xytext=(0, -16 if on_above else 11), ha='center',
                        fontsize=9, color='#d62728', fontweight='bold', bbox=bbox)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin - 0.08 * (ymax - ymin), ymax + 0.10 * (ymax - ymin))
    fig.suptitle('Data-efficiency: KCL-on vs KCL-off across training-set size (mean over seeds 42/43/44)',
                 fontsize=13, y=0.99)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {OUT}')

    print("\n=== Summary — mean V MAE per training-size ===")
    print(f"{'N':<6}{'KCL ON':>12}{'KCL OFF':>12}{'Δ(off-on)':>12}{'% benefit':>10}{'n_seeds':>10}")
    for N in SIZES:
        on = data['On'].get(N); off = data['Off'].get(N)
        if on is None or off is None:
            print(f'{N:<6}{"—":>12}{"—":>12}{"—":>12}')
            continue
        von, voff = on['V']['mean'], off['V']['mean']
        if von is None or voff is None:
            print(f'{N:<6}{"—":>12}{"—":>12}{"—":>12}')
            continue
        d = voff - von; pct = 100 * d / voff
        print(f'{N:<6}{von:>12.2f}{voff:>12.2f}{d:>+12.2f}{pct:>+10.1f}%   ({on["V"]["n"]}/{off["V"]["n"]} seeds)')


if __name__ == '__main__':
    main()
