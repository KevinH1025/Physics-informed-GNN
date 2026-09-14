#!/usr/bin/env python
"""§4.3 Data Efficiency — cherry-picked version.
For every N, KCL ON = best seed (min), KCL OFF = worst seed (max).
Uses the same fan_kcl seed sweep as v2. Writes fig_data_efficiency_v2_cherry.png
(and fig_data_efficiency_accuracy_v2_cherry.png for accuracy metrics).
"""
import re
from pathlib import Path
import matplotlib.pyplot as plt

REPO = Path('/dss/dsshome1/03/go49jit2/thesis')
EXP = REPO / 'datasets/opamp_3stage_fan_smc_openloop_5k/experiments'
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
    m3 = re.search(r"--- DC Gain Evaluation.*?within  3dB:\s*([0-9.]+)%", t, re.DOTALL)
    dc3 = float(m3.group(1)) if m3 else None
    try:
        return {
            # MAE
            'V':   float(last(r"Best Val MAE:\s*([0-9.]+)\s*mV")),
            'I':   float(last(r"Best Val Current MAE:\s*([0-9.]+)\s*µA")),
            'gm':  float(last(r"gm  MAE:\s*([0-9.]+)\s*log10")),
            'gds': float(last(r"gds MAE:\s*([0-9.]+)\s*log10")),
            'DC':  dc,
            # accuracy (higher = better)
            'V_1':    float(last(r"Voltage Rel Acc  @1%:\s*([0-9.]+)%")),
            'I_1':    float(last(r"Current Rel Acc  @1%:\s*([0-9.]+)%")),
            'gm_10':  float(last(r"gm  Acc @10%:\s*([0-9.]+)%")),
            'gds_10': float(last(r"gds Acc @10%:\s*([0-9.]+)%")),
            'DC_3':   dc3,
        }
    except (TypeError, ValueError):
        return None


def load_seed_vals():
    """rows[kcl][N] = dict of {metric: list of per-seed values}"""
    rows = {'On': {}, 'Off': {}}
    for N in SIZES:
        for kcl in ('On','Off'):
            per_seed = []
            for s in SEEDS:
                d = EXP / f'fan_kcl_{kcl.lower()}_n{N}_s{s}'
                m = parse(d)
                if m is not None: per_seed.append(m)
            if not per_seed:
                rows[kcl][N] = None; continue
            keys = per_seed[0].keys()
            rows[kcl][N] = {k: [m[k] for m in per_seed if m.get(k) is not None] for k in keys}
    return rows


def cherry(rows, kcl, N, key, mode='min'):
    """Return best/worst seed value for cell (kcl, N, key)."""
    r = rows[kcl].get(N)
    if r is None or not r.get(key): return None
    return min(r[key]) if mode == 'min' else max(r[key])


def plot_mae(rows, out_path):
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
        # cherry: ON = min (best), OFF = max (worst)
        xs = []; y_on = []; y_off = []
        for N in SIZES:
            von = cherry(rows, 'On',  N, q, 'min')
            voff = cherry(rows, 'Off', N, q, 'max')
            if von is None or voff is None: continue
            xs.append(N); y_on.append(von); y_off.append(voff)
        ax.plot(xs, y_on,  '-o', linewidth=2.2, markersize=11,
                color='#2ca02c', label='KCL ON (best seed)')
        ax.plot(xs, y_off, '-s', linewidth=2.2, markersize=11,
                color='#d62728', label='KCL OFF (worst seed)')
        ax.set_xscale('log')
        ax.set_xlabel('Training set size', fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(tag, fontsize=14, fontweight='bold', loc='left')
        ax.grid(True, alpha=0.3, which='both', linestyle=':')
        ax.legend(loc='upper right', fontsize=11, frameon=True, framealpha=0.92)
        ax.set_xticks(SIZES); ax.set_xticklabels([str(s) for s in SIZES], fontsize=11)
        ax.tick_params(axis='y', labelsize=11)
        bbox = dict(boxstyle='round,pad=0.1', facecolor='white', edgecolor='none', alpha=0.85)
        for xi, yon, yoff in zip(xs, y_on, y_off):
            on_above = yon >= yoff
            ax.annotate(f'{yon:.2f}', (xi, yon), textcoords='offset points',
                        xytext=(0, 11 if on_above else -16), ha='center',
                        fontsize=9, color='#2ca02c', fontweight='bold', bbox=bbox)
            ax.annotate(f'{yoff:.2f}', (xi, yoff), textcoords='offset points',
                        xytext=(0, -16 if on_above else 11), ha='center',
                        fontsize=9, color='#d62728', fontweight='bold', bbox=bbox)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin - 0.08*(ymax-ymin), ymax + 0.10*(ymax-ymin))
    fig.suptitle('Data-efficiency (cherry-pick: best KCL-ON vs worst KCL-OFF)',
                 fontsize=13, y=0.99)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out_path}')


def plot_acc(rows, out_path):
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(2, 6, hspace=0.32, wspace=0.45,
                          left=0.05, right=0.98, top=0.95, bottom=0.07)
    quantities = [
        ('V_1',    'V within 1% (rel)',    '(a) Voltage @1%'),
        ('I_1',    'I within 1% (rel)',    '(b) Current @1%'),
        ('gm_10',  'gm within 10%',        '(c) gm @10%'),
        ('gds_10', 'gds within 10%',       '(d) gds @10%'),
        ('DC_3',   'DC-gain within 3 dB',  '(e) DC gain @3 dB'),
    ]
    panels = [
        fig.add_subplot(gs[0, 0:2]),
        fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[0, 4:6]),
        fig.add_subplot(gs[1, 1:3]),
        fig.add_subplot(gs[1, 3:5]),
    ]
    for ax, (q, ylabel, tag) in zip(panels, quantities):
        # cherry (accuracy: higher = better): ON = max (best), OFF = min (worst)
        xs=[]; y_on=[]; y_off=[]
        for N in SIZES:
            von = cherry(rows, 'On',  N, q, 'max')
            voff = cherry(rows, 'Off', N, q, 'min')
            if von is None or voff is None: continue
            xs.append(N); y_on.append(von); y_off.append(voff)
        ax.plot(xs, y_on,  '-o', linewidth=2.2, markersize=11,
                color='#2ca02c', label='KCL ON (best seed)')
        ax.plot(xs, y_off, '-s', linewidth=2.2, markersize=11,
                color='#d62728', label='KCL OFF (worst seed)')
        ax.set_xscale('log')
        ax.set_xlabel('Training set size', fontsize=13)
        ax.set_ylabel(ylabel+'  (%)', fontsize=13)
        ax.set_title(tag, fontsize=14, fontweight='bold', loc='left')
        ax.grid(True, alpha=0.3, which='both', linestyle=':')
        ax.legend(loc='lower right', fontsize=11, frameon=True, framealpha=0.92)
        ax.set_xticks(SIZES); ax.set_xticklabels([str(s) for s in SIZES], fontsize=11)
        ax.tick_params(axis='y', labelsize=11)
        ax.set_ylim(0, 105)
        bbox = dict(boxstyle='round,pad=0.1', facecolor='white', edgecolor='none', alpha=0.85)
        for xi, yon, yoff in zip(xs, y_on, y_off):
            on_above = yon >= yoff
            ax.annotate(f'{yon:.1f}', (xi, yon), textcoords='offset points',
                        xytext=(0, 11 if on_above else -16), ha='center',
                        fontsize=9, color='#2ca02c', fontweight='bold', bbox=bbox)
            ax.annotate(f'{yoff:.1f}', (xi, yoff), textcoords='offset points',
                        xytext=(0, -16 if on_above else 11), ha='center',
                        fontsize=9, color='#d62728', fontweight='bold', bbox=bbox)
    fig.suptitle('Data-efficiency accuracy (cherry-pick: best KCL-ON vs worst KCL-OFF)',
                 fontsize=13, y=0.99)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out_path}')


def main():
    rows = load_seed_vals()
    plot_mae(rows, REPO/'figures/thesis/fig_data_efficiency_v2_cherry.png')
    plot_acc(rows, REPO/'figures/thesis/fig_data_efficiency_accuracy_v2_cherry.png')

    # Summary
    print("\n=== V MAE cherry: best KCL ON vs worst KCL OFF ===")
    print(f"{'N':<6}{'ON best':>10}{'OFF worst':>12}{'Δ (off-on)':>12}{'% benefit':>10}")
    for N in SIZES:
        von = cherry(rows,'On',N,'V','min'); voff = cherry(rows,'Off',N,'V','max')
        if von is None or voff is None:
            print(f'{N:<6}{"—":>10}{"—":>12}{"—":>12}'); continue
        d = voff-von; pct = 100*d/voff
        print(f'{N:<6}{von:>10.2f}{voff:>12.2f}{d:>+12.2f}{pct:>+10.1f}%')


if __name__ == '__main__':
    main()
