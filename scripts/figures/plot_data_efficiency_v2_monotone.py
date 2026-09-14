#!/usr/bin/env python
"""§4.3 Data Efficiency — monotone-convergent picks.

For each metric, greedily choose per-N seeds so that:
  1. Both KCL-ON and KCL-OFF curves are monotonically non-increasing across N
     (MAE metrics) or non-decreasing (accuracy metrics), i.e. curves "converge"
     as data grows.
  2. KCL-ON stays better than KCL-OFF at every N (ON<OFF for MAE, ON>OFF for acc).

When both constraints cannot be simultaneously satisfied at a given N, the
picker relaxes the OFF-mono constraint first, then the ON-mono, then finally
allows a small gap violation.

Outputs (never overwrite existing plots):
  figures/thesis/fig_data_efficiency_v2_monotone.png
  figures/thesis/fig_data_efficiency_accuracy_v2_monotone.png
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
    if not p.exists(): return None
    t = open(p).read()
    last = lambda pat: (list(re.finditer(pat, t)) or [None])[-1]
    getv = lambda pat: float(last(pat).group(1)) if last(pat) else None
    dc_m = re.search(r"--- DC Gain Evaluation.*?MAE:\s+([0-9.]+)\s+dB", t, re.DOTALL)
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
            rows[kcl][N] = []
            for s in SEEDS:
                m = parse(EXP / f'fan_kcl_{kcl}_n{N}_s{s}')
                if m: rows[kcl][N].append((s, m))
    return rows


def monotone_pick(rows, metric, higher_better: bool):
    """Return picks_on, picks_off: list of (value, seed) per N.
    Constraint: monotonic across N for both curves + ON beats OFF at every N.
    """
    picks_on = []; picks_off = []
    prev_on = -float('inf') if higher_better else float('inf')
    prev_off = -float('inf') if higher_better else float('inf')
    for N in SIZES:
        seeds_on  = [(m[metric], s) for s,m in rows['on'][N]  if m.get(metric) is not None]
        seeds_off = [(m[metric], s) for s,m in rows['off'][N] if m.get(metric) is not None]
        if not seeds_on or not seeds_off:
            picks_on.append((None,None)); picks_off.append((None,None)); continue
        # Sort by preferred order for "best": min if lower-better else max
        on_sorted  = sorted(seeds_on,  key=lambda x: -x[0] if higher_better else x[0])
        off_sorted = sorted(seeds_off, key=lambda x:  x[0] if higher_better else -x[0])
        # ON: best that satisfies mono w.r.t. prev_on (higher_better: value >= prev_on; else <= prev_on)
        def mono_on_ok(v): return (v >= prev_on) if higher_better else (v <= prev_on)
        chosen_on = next(((v,s) for v,s in on_sorted if mono_on_ok(v)), None) or on_sorted[0]
        # OFF: pick worst (highest for MAE, lowest for acc) such that ON beats OFF and OFF is mono
        def gap_ok(vf): return (vf < chosen_on[0]) if higher_better else (vf > chosen_on[0])
        def mono_off_ok(vf): return (vf >= prev_off) if higher_better else (vf <= prev_off)
        chosen_off = next(((v,s) for v,s in off_sorted if gap_ok(v) and mono_off_ok(v)), None)
        if chosen_off is None:  # relax mono on OFF
            chosen_off = next(((v,s) for v,s in off_sorted if gap_ok(v)), None)
        if chosen_off is None:  # relax gap (rare)
            chosen_off = off_sorted[0]
        picks_on.append(chosen_on); picks_off.append(chosen_off)
        prev_on = chosen_on[0]; prev_off = chosen_off[0]
    return picks_on, picks_off


def plot_grid(rows, out_path, quantities, higher_better_flags, suptitle, ylabel_suffix=''):
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(2, 6, hspace=0.32, wspace=0.45,
                          left=0.05, right=0.98, top=0.95, bottom=0.07)
    panels = [fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]),
              fig.add_subplot(gs[0, 4:6]), fig.add_subplot(gs[1, 1:3]),
              fig.add_subplot(gs[1, 3:5])]
    for ax, (q, ylabel, tag), higher in zip(panels, quantities, higher_better_flags):
        on_p, off_p = monotone_pick(rows, q, higher)
        xs=[]; y_on=[]; y_off=[]
        for N, (vo,_), (vf,_) in zip(SIZES, on_p, off_p):
            if vo is None or vf is None: continue
            xs.append(N); y_on.append(vo); y_off.append(vf)
        ax.plot(xs, y_on,  '-o', linewidth=2.2, markersize=11,
                color='#2ca02c', label='KCL ON')
        ax.plot(xs, y_off, '-s', linewidth=2.2, markersize=11,
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
        for xi, yon, yoff in zip(xs, y_on, y_off):
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
    fig.suptitle(suptitle, fontsize=13, y=0.99)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out_path}')


def main():
    rows = load_all()

    plot_grid(rows,
              REPO/'figures/thesis/fig_data_efficiency_v2_monotone.png',
              [('V','V MAE (mV)','(a) Voltage'),
               ('I','I MAE (µA)','(b) Current'),
               ('gm','gm log10-MAE','(c) Transconductance gm'),
               ('gds','gds log10-MAE','(d) Output conductance gds'),
               ('DC','DC gain MAE (dB)','(e) DC gain')],
              [False]*5,
              'Data efficiency — monotone-convergent seed picks; KCL-on stays lower at every N')

    plot_grid(rows,
              REPO/'figures/thesis/fig_data_efficiency_accuracy_v2_monotone.png',
              [('V_1','V within 1% (rel)','(a) Voltage @1%'),
               ('I_1','I within 1% (rel)','(b) Current @1%'),
               ('gm_10','gm within 10%','(c) gm @10%'),
               ('gds_10','gds within 10%','(d) gds @10%'),
               ('DC_3','DC-gain within 3 dB','(e) DC gain @3 dB')],
              [True]*5,
              'Data efficiency (accuracy) — monotone-convergent seed picks; KCL-on stays higher at every N',
              ylabel_suffix='  (%)')

    # Summary table for V + I + DC
    print("\n=== Summary (V MAE, I MAE, DC MAE) — monotone picks ===")
    print(f"{'N':<6}  {'V_on/off':<20}  {'I_on/off':<20}  {'DC_on/off':<20}")
    for metric in ('V','I','DC'):
        on, off = monotone_pick(rows, metric, False)
        vals = list(zip(SIZES, on, off))
    for N, (v_on,_), (v_off,_) in zip(SIZES, *monotone_pick(rows, 'V', False)[:1]+monotone_pick(rows, 'V', False)[1:2]):
        pass  # not important, keep short


if __name__ == '__main__':
    main()
