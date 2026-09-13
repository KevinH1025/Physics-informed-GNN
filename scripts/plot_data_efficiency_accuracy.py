#!/usr/bin/env python
"""§4.3 Data Efficiency curves — WITHIN-THRESHOLD ACCURACY version.
Same layout as MAE plot but Y axis is "% of samples within X threshold":
  V @1 % rel,  I @1 % rel,  gm @10 %,  gds @10 %,  DC @3 dB."""
import re
from pathlib import Path
import matplotlib.pyplot as plt

REPO = Path('/dss/dsshome1/03/go49jit2/thesis')
EXP = REPO / 'datasets/opamp_3stage_fan_smc_openloop_5k/experiments'
PREFIX = 'ginebn_3layer_bb8_notower_vn_kcl_w10_edge6_netrole_clip03_openloop_5k_DATAEFF'
USING = 'ginebn_3layer_bb8_notower_vn_kcl_w10_edge6_netrole_clip03_openloop_5k_usingnow'
OUT = REPO / 'figures/thesis/fig_data_efficiency_accuracy.png'

SIZES = [250, 500, 1000, 2000, 5000]
WD_SIZES = [250, 500]


def parse(d):
    with open(d / 'training.log') as f:
        t = f.read()
    last = lambda pat, g=1: (list(re.finditer(pat, t)) or [None])[-1].group(g) if re.search(pat, t) else None
    # DC @3dB from DC Gain Evaluation block
    m = re.search(r"--- DC Gain Evaluation.*?within  3dB:\s*([0-9.]+)%", t, re.DOTALL)
    dc3 = float(m.group(1)) if m else None
    return {
        'V_1':   float(last(r"Voltage Rel Acc  @1%:\s*([0-9.]+)%")),
        'I_1':   float(last(r"Current Rel Acc  @1%:\s*([0-9.]+)%")),
        'gm_10': float(last(r"gm  Acc @10%:\s*([0-9.]+)%")),
        'gds_10': float(last(r"gds Acc @10%:\s*([0-9.]+)%")),
        'DC_3':  dc3,
    }


def load_all():
    rows = {kcl: {} for kcl in ('On', 'Off')}
    rows['WD'] = {}
    for s in SIZES:
        for kcl in ('On', 'Off'):
            paths = []
            if s == 5000 and kcl == 'On':
                paths.append(EXP / USING)
            else:
                paths.append(EXP / f"{PREFIX}_n{s}_kcl{kcl}_s88")
            if s == 250:
                extra = EXP / f"{PREFIX}_n{s}_kcl{kcl}_s42"
                if extra.exists(): paths.append(extra)
            metrics_list = [parse(p) for p in paths if (p/'training.log').exists()]
            rows[kcl][s] = {q: sum(m[q] for m in metrics_list)/len(metrics_list)
                            for q in metrics_list[0]}
        if s in WD_SIZES:
            p = EXP / f"{PREFIX}_n{s}_kclOff_wd1e-3_s88"
            if (p/'training.log').exists():
                rows['WD'][s] = parse(p)
    return rows


def main():
    data = load_all()
    # 3+2 layout: row 1 V|I|gm, row 2 gds|DC centered
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(2, 6, hspace=0.32, wspace=0.45,
                          left=0.05, right=0.98, top=0.97, bottom=0.07)
    quantities = [
        ('V_1',   'V within 1 % rel (%)',     '(a) Voltage @1 % rel'),
        ('I_1',   'I within 1 % rel (%)',     '(b) Current @1 % rel'),
        ('gm_10', 'gm within 10 % rel (%)',   '(c) gm @10 % rel'),
        ('gds_10','gds within 10 % rel (%)',  '(d) gds @10 % rel'),
        ('DC_3',  'DC gain within 3 dB (%)',  '(e) DC gain @3 dB'),
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
        ax.legend(loc='lower right', fontsize=11, frameon=True, framealpha=0.92)
        ax.set_xticks(x); ax.set_xticklabels([str(s) for s in x], fontsize=11)
        ax.tick_params(axis='y', labelsize=11)
        bbox = dict(boxstyle='round,pad=0.1', facecolor='white', edgecolor='none', alpha=0.85)
        for xi, yon, yoff in zip(x, y_on, y_off):
            on_above = yon >= yoff
            ax.annotate(f'{yon:.1f}', (xi, yon), textcoords='offset points',
                        xytext=(0, 11 if on_above else -16), ha='center',
                        fontsize=9, color='#2ca02c', fontweight='bold', bbox=bbox)
            ax.annotate(f'{yoff:.1f}', (xi, yoff), textcoords='offset points',
                        xytext=(0, -16 if on_above else 11), ha='center',
                        fontsize=9, color='#d62728', fontweight='bold', bbox=bbox)
        if wd_x:
            for xi, yw in zip(wd_x, wd_y):
                ax.annotate(f'{yw:.1f}', (xi, yw), textcoords='offset points',
                            xytext=(0, -16), ha='center', fontsize=9,
                            color='#7f7f7f', fontweight='bold', bbox=bbox)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin - 0.08 * (ymax - ymin), ymax + 0.10 * (ymax - ymin))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {OUT}')

    # Summary
    print(f"\n=== KCL advantage per size (within-threshold %) ===")
    for q, lbl, _ in quantities:
        print(f"\n-- {lbl} --")
        print(f"{'n_train':<10}{'KCL ON':>10}{'KCL OFF':>10}{'Δ(ON-OFF)':>12}")
        for s in SIZES:
            on, off = data['On'][s][q], data['Off'][s][q]
            print(f"{s:<10}{on:>10.1f}{off:>10.1f}{on-off:>+12.1f}")


if __name__ == '__main__':
    main()
