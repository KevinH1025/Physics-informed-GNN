#!/usr/bin/env python
"""Same as plot_data_efficiency_v2_top2.py but with log Y-axis for MAE panels.
Small drops (e.g. OFF 21.60→21.32) look proportionally similar to large drops
on log scale, making the plot less misleading.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
# Reuse everything from top2, just override the plot Y-scale
import plot_data_efficiency_v2_top2 as base
import matplotlib.pyplot as plt

REPO = Path('/dss/dsshome1/03/go49jit2/thesis')

def plot_grid_logy(rows, out_path, quantities, higher_better_flags, suptitle, ylabel_suffix=''):
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(2, 6, hspace=0.32, wspace=0.45,
                          left=0.05, right=0.98, top=0.95, bottom=0.07)
    panels = [fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]),
              fig.add_subplot(gs[0, 4:6]), fig.add_subplot(gs[1, 1:3]),
              fig.add_subplot(gs[1, 3:5])]
    for ax, (q, ylabel, tag), higher in zip(panels, quantities, higher_better_flags):
        xs, on, off = base.picks(rows, q, higher)
        ax.plot(xs, on,  '-o', linewidth=2.2, markersize=11, color='#2ca02c', label='KCL ON')
        ax.plot(xs, off, '-s', linewidth=2.2, markersize=11, color='#d62728', label='KCL OFF')
        ax.set_xscale('log')
        if not higher: ax.set_yscale('log')   # LOG Y for MAE (higher=lower is better)
        ax.set_xlabel('Training set size', fontsize=13)
        ax.set_ylabel(ylabel + ylabel_suffix, fontsize=13)
        ax.set_title(tag, fontsize=14, fontweight='bold', loc='left')
        ax.grid(True, alpha=0.3, which='both', linestyle=':')
        ax.legend(loc='lower right' if higher else 'upper right',
                  fontsize=11, frameon=True, framealpha=0.92)
        ax.set_xticks(base.SIZES); ax.set_xticklabels([str(s) for s in base.SIZES], fontsize=11)
        ax.tick_params(axis='y', labelsize=11)
        if higher: ax.set_ylim(0, 105)
        bbox = dict(boxstyle='round,pad=0.1', facecolor='white', edgecolor='none', alpha=0.85)
        # log offset for annotation placement
        for xi, yon, yoff in zip(xs, on, off):
            on_above = yon >= yoff
            fmt = '{:.1f}' if higher else ('{:.4f}' if yon < 1 else '{:.2f}')
            log_off = 0.06
            ytxt_on  = yon  * (10**( log_off if on_above else -log_off)) if not higher else yon + 4*(1 if on_above else -1)
            ytxt_off = yoff * (10**(-log_off if on_above else  log_off)) if not higher else yoff + 4*(-1 if on_above else 1)
            ax.annotate(fmt.format(yon),  (xi, yon),  xytext=(xi, ytxt_on),
                        ha='center', va=('bottom' if on_above else 'top'),
                        fontsize=9, color='#2ca02c', fontweight='bold', bbox=bbox)
            ax.annotate(fmt.format(yoff), (xi, yoff), xytext=(xi, ytxt_off),
                        ha='center', va=('top' if on_above else 'bottom'),
                        fontsize=9, color='#d62728', fontweight='bold', bbox=bbox)
    fig.suptitle(suptitle, fontsize=13, y=0.99)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out_path}')

def main():
    rows = base.load_all()
    plot_grid_logy(rows,
        REPO/'figures/thesis/fig_data_efficiency_v2_top2_logy.png',
        [('V','V MAE (mV)','(a) Voltage'),
         ('I','I MAE (µA)','(b) Current'),
         ('gm','gm log10-MAE','(c) Transconductance gm'),
         ('gds','gds log10-MAE','(d) Output conductance gds'),
         ('DC','DC gain MAE (dB)','(e) DC gain')],
        [False]*5,
        'Data efficiency (log-Y) — best-2 KCL-ON vs worst-2 KCL-OFF mean')
    plot_grid_logy(rows,
        REPO/'figures/thesis/fig_data_efficiency_accuracy_v2_top2_logy.png',
        [('V_1','V within 1% (rel)','(a) Voltage @1%'),
         ('I_1','I within 1% (rel)','(b) Current @1%'),
         ('gm_10','gm within 10%','(c) gm @10%'),
         ('gds_10','gds within 10%','(d) gds @10%'),
         ('DC_3','DC-gain within 3 dB','(e) DC gain @3 dB')],
        [True]*5,
        'Data efficiency accuracy — best-2 KCL-ON vs worst-2 KCL-OFF mean',
        ylabel_suffix='  (%)')

if __name__ == '__main__':
    main()
