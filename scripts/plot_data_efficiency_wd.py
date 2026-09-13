#!/usr/bin/env python
"""Compact 3-bar comparison: at n=250 and n=500, V MAE for
   KCL ON  vs  KCL OFF  vs  KCL OFF + WD=1e-3 (generic regularizer control)."""
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

REPO = Path('/dss/dsshome1/03/go49jit2/thesis')
EXP = REPO / 'datasets/opamp_3stage_fan_smc_openloop_5k/experiments'
PREFIX = 'ginebn_3layer_bb8_notower_vn_kcl_w10_edge6_netrole_clip03_openloop_5k_DATAEFF'
OUT = REPO / 'figures/thesis/fig_data_efficiency_wd_control.png'


def parse(d):
    if not (d/'training.log').exists(): return None
    with open(d/'training.log') as f: t = f.read()
    last = lambda pat, g=1: (list(re.finditer(pat, t)) or [None])[-1].group(g) if re.search(pat, t) else None
    m = re.search(r"--- DC Gain Evaluation.*?MAE: ([0-9.]+) dB", t, re.DOTALL)
    return {
        'V':   float(last(r"Best Val MAE: ([0-9.]+)mV")),
        'I':   float(last(r"Best Val Current MAE: ([0-9.]+)µA")),
        'gm':  float(last(r"gm  MAE: ([0-9.]+) log10")),
        'gds': float(last(r"gds MAE: ([0-9.]+) log10")),
        'DC':  float(m.group(1)) if m else None,
    }


def avg(paths):
    ms = [parse(p) for p in paths if (p/'training.log').exists()]
    return {q: sum(m[q] for m in ms)/len(ms) for q in ms[0]}


def main():
    data = {}
    for s in (250, 500):
        on_paths  = [EXP/f"{PREFIX}_n{s}_kclOn_s88"]
        off_paths = [EXP/f"{PREFIX}_n{s}_kclOff_s88"]
        if s == 250:
            on_paths.append(EXP/f"{PREFIX}_n{s}_kclOn_s42")
            off_paths.append(EXP/f"{PREFIX}_n{s}_kclOff_s42")
        data[s] = {
            'on': avg(on_paths),
            'off': avg(off_paths),
            'wd': parse(EXP/f"{PREFIX}_n{s}_kclOff_wd1e-3_s88"),
        }

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    quantities = [
        ('V',   'V MAE (mV)',         '(a) Voltage'),
        ('I',   'I MAE (µA)',         '(b) Current'),
        ('gm',  'gm log10-MAE',       '(c) Transconductance gm'),
        ('gds', 'gds log10-MAE',      '(d) Output conductance gds'),
        ('DC',  'DC gain MAE (dB)',   '(e) DC gain'),
    ]
    sizes = [250, 500]
    bar_labels = ['KCL ON', 'KCL OFF', 'KCL OFF\n+ WD=1e-3']
    colors = ['#2ca02c', '#d62728', '#7f7f7f']
    for ax, (q, ylabel, title) in zip(axes, quantities):
        x_positions = np.arange(len(sizes))
        width = 0.27
        for i, (key, lbl, color) in enumerate(zip(['on', 'off', 'wd'], bar_labels, colors)):
            vals = [data[s][key][q] for s in sizes]
            bars = ax.bar(x_positions + (i - 1) * width, vals, width,
                          color=color, edgecolor='black', linewidth=0.7, label=lbl)
            for bar, v in zip(bars, vals):
                ax.annotate(f'{v:.2f}', xy=(bar.get_x() + bar.get_width()/2, v),
                            xytext=(0, 3), textcoords='offset points',
                            ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_xticks(x_positions)
        ax.set_xticklabels([f'n={s}' for s in sizes], fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold', loc='left')
        ax.grid(True, axis='y', alpha=0.3, linestyle=':')
        ax.legend(fontsize=9, loc='upper left', frameon=True, framealpha=0.92)
        # headroom for top annotations
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax * 1.12)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {OUT}')

    print('\n=== summary table ===')
    print(f"{'n_train':<8}{'KCL ON':>12}{'KCL OFF':>12}{'KCL OFF+WD':>14}")
    for s in sizes:
        for q, _, _ in quantities:
            on, off, wd = data[s]['on'][q], data[s]['off'][q], data[s]['wd'][q]
            print(f"n={s:<5} {q:<5} ON={on:8.3f}   OFF={off:8.3f}   WD={wd:8.3f}")
        print()


if __name__ == '__main__':
    main()
