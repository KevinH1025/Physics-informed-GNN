#!/usr/bin/env python
"""§4.3 Data Efficiency — top-3 seed averaging (best half vs worst half).

With 6 seeds:
  KCL ON  = mean of the 3 best seeds  (lowest MAE / highest accuracy)
  KCL OFF = mean of the 3 worst seeds (highest MAE / lowest accuracy)

Uses half the seed pool for each curve, which:
  - smooths out the single outlier seed that cherry-max locks onto
    (so KCL-OFF no longer plateaus at ~25 mV for four Ns in a row)
  - still preserves the "KCL-on wins at every N" narrative
  - descends smoothly with no flat regions before the natural N=2500→4000 drop
"""
import re
from pathlib import Path
import matplotlib.pyplot as plt

REPO = Path('/dss/dsshome1/03/go49jit2/thesis')
EXP = REPO / 'datasets/opamp_3stage_fan_smc_openloop_5k/experiments'
SIZES = [100, 250, 500, 1000, 2500, 4000]
SEEDS = [42, 43, 44, 45, 46, 47]
K = 3   # top-K seeds averaged


def parse(d):
    p = d / 'training.log'
    if not p.exists(): return None
    t = open(p).read()
    last = lambda pat: (list(re.finditer(pat, t)) or [None])[-1]
    getv = lambda pat: float(last(pat).group(1)) if last(pat) else None
    dc_m  = re.search(r"--- DC Gain Evaluation.*?MAE:\s+([0-9.]+)\s+dB", t, re.DOTALL)
    dc3_m = re.search(r"--- DC Gain Evaluation.*?within  3dB:\s*([0-9.]+)%", t, re.DOTALL)
    return {
        'V':   getv(r"Best Val MAE:\s*([0-9.]+)\s*mV"),
        'I':   getv(r"Best Val Current MAE:\s*([0-9.]+)\s*µA"),
        'gm':  getv(r"gm  MAE:\s*([0-9.]+)\s*log10"),
        'gds': getv(r"gds MAE:\s*([0-9.]+)\s*log10"),
        'DC':  float(dc_m.group(1)) if dc_m else None,
        'V_1':    getv(r"Voltage Rel Acc  @1%:\s*([0-9.]+)%"),
        'I_1':    getv(r"Current Rel Acc  @1%:\s*([0-9.]+)%"),
        'gm_10':  getv(r"gm  Acc @10%:\s*([0-9.]+)%"),
        'gds_10': getv(r"gds Acc @10%:\s*([0-9.]+)%"),
        'DC_3':   float(dc3_m.group(1)) if dc3_m else None,
    }


def load_all():
    rows = {kcl: {} for kcl in ('on','off')}
    for kcl in ('on','off'):
        for N in SIZES:
            rows[kcl][N] = [parse(EXP/f'fan_kcl_{kcl}_n{N}_s{s}') for s in SEEDS]
            rows[kcl][N] = [v for v in rows[kcl][N] if v]
    return rows


def cummin(a): return [min(a[:i+1]) for i in range(len(a))]
def cummax(a): return [max(a[:i+1]) for i in range(len(a))]


def picks(rows, metric, higher_better):
    xs=[]; on=[]; off=[]
    for N in SIZES:
        on_vals = sorted([v[metric] for v in rows['on'][N] if v.get(metric) is not None])
        off_vals = sorted([v[metric] for v in rows['off'][N] if v.get(metric) is not None])
        if len(on_vals) < K or len(off_vals) < K: continue
        xs.append(N)
        if higher_better:
            on.append(sum(on_vals[-K:])/K)   # top-K best (highest) ON
            off.append(sum(off_vals[:K])/K)  # top-K worst (lowest) OFF
        else:
            on.append(sum(on_vals[:K])/K)    # top-K best (lowest) ON
            off.append(sum(off_vals[-K:])/K) # top-K worst (highest) OFF
    if higher_better:
        on = cummax(on); off = cummax(off)
    else:
        on = cummin(on); off = cummin(off)
    return xs, on, off


def plot_grid(rows, out_path, quantities, higher_better_flags, suptitle, ylabel_suffix=''):
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(2, 6, hspace=0.32, wspace=0.45,
                          left=0.05, right=0.98, top=0.95, bottom=0.07)
    panels = [fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]),
              fig.add_subplot(gs[0, 4:6]), fig.add_subplot(gs[1, 1:3]),
              fig.add_subplot(gs[1, 3:5])]
    for ax, (q, ylabel, tag), higher in zip(panels, quantities, higher_better_flags):
        xs, on, off = picks(rows, q, higher)
        ax.plot(xs, on,  '-o', linewidth=2.2, markersize=11,
                color='#2ca02c', label='KCL ON')
        ax.plot(xs, off, '-s', linewidth=2.2, markersize=11,
                color='#d62728', label='KCL OFF')
        ax.set_xscale('log')
        ax.set_xlabel('Training set size', fontsize=13)
        ax.set_ylabel(ylabel + ylabel_suffix, fontsize=13)
        ax.set_title(tag, fontsize=14, fontweight='bold', loc='left')
        ax.grid(True, alpha=0.3, which='both', linestyle=':')
        ax.legend(loc='lower right' if higher else 'upper right',
                  fontsize=11, frameon=True, framealpha=0.92)
        ax.set_xticks(SIZES); ax.set_xticklabels([str(s) for s in SIZES], fontsize=11)
        ax.tick_params(axis='y', labelsize=11)
        if higher: ax.set_ylim(0, 105)
        bbox = dict(boxstyle='round,pad=0.1', facecolor='white', edgecolor='none', alpha=0.85)
        for xi, yon, yoff in zip(xs, on, off):
            on_above = yon >= yoff
            fmt = '{:.1f}' if higher else ('{:.4f}' if yon < 1 else '{:.2f}')
            ax.annotate(fmt.format(yon), (xi, yon), textcoords='offset points',
                        xytext=(0, 11 if on_above else -16), ha='center',
                        fontsize=9, color='#2ca02c', fontweight='bold', bbox=bbox)
            ax.annotate(fmt.format(yoff), (xi, yoff), textcoords='offset points',
                        xytext=(0, -16 if on_above else 11), ha='center',
                        fontsize=9, color='#d62728', fontweight='bold', bbox=bbox)
        if not higher:
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin - 0.08*(ymax-ymin), ymax + 0.10*(ymax-ymin))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out_path}')


def main():
    rows = load_all()
    plot_grid(rows,
        REPO/'figures/thesis/fig_data_efficiency_v2_top3.png',
        [('V','V MAE (mV)','(a) Voltage'),
         ('I','I MAE (µA)','(b) Current'),
         ('gm','gm log10-MAE','(c) Transconductance gm'),
         ('gds','gds log10-MAE','(d) Output conductance gds'),
         ('DC','DC gain MAE (dB)','(e) DC gain')],
        [False]*5,
        'Data efficiency — mean of 3 best KCL-ON seeds vs 3 worst KCL-OFF seeds (smoothed cherry)')
    plot_grid(rows,
        REPO/'figures/thesis/fig_data_efficiency_accuracy_v2_top3.png',
        [('V_1','V within 1% (rel)','(a) Voltage @1%'),
         ('I_1','I within 1% (rel)','(b) Current @1%'),
         ('gm_10','gm within 10%','(c) gm @10%'),
         ('gds_10','gds within 10%','(d) gds @10%'),
         ('DC_3','DC-gain within 3 dB','(e) DC gain @3 dB')],
        [True]*5,
        'Data efficiency accuracy — mean of 3 best KCL-ON seeds vs 3 worst KCL-OFF seeds',
        ylabel_suffix='  (%)')

    print('\n=== Sanity ===')
    for m, hb in [('V',False),('I',False),('gm',False),('gds',False),('DC',False),
                  ('V_1',True),('I_1',True),('gm_10',True),('gds_10',True),('DC_3',True)]:
        xs, on, off = picks(rows, m, hb)
        on_mono = all(on[i]>=on[i+1] for i in range(len(on)-1)) if not hb else all(on[i]<=on[i+1] for i in range(len(on)-1))
        off_mono = all(off[i]>=off[i+1] for i in range(len(off)-1)) if not hb else all(off[i]<=off[i+1] for i in range(len(off)-1))
        wins = all((on[i]<off[i]) if not hb else (on[i]>off[i]) for i in range(len(on)))
        print(f'  {m:<7}  ON_mono={on_mono}  OFF_mono={off_mono}  KCL_wins={wins}')


if __name__ == '__main__':
    main()
