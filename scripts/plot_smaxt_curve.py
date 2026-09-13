#!/usr/bin/env python
"""Visualize the SMAXT g_m formula and its 3 asymptotic limits.
Single-panel V_ov sweep showing cutoff -> saturation -> triode transition.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path('/dss/dsshome1/03/go49jit2/thesis/figures/thesis/fig_smaxt_curves.png')

V_T = 0.02585     # thermal voltage at 300 K
n = 1.5            # NMOS slope factor
n_vt = n * V_T     # ~38.8 mV


def softplus(x):
    return np.log1p(np.exp(np.clip(x, -50, 50)))


def smaxt_gm(Id, Vov, Vds):
    arg1 = (Vds - Vov) / n_vt
    Vds_eff = Vds - n_vt * softplus(arg1)
    arg2 = (Vov - Vds_eff / 2 - n_vt) / n_vt
    denom = n_vt + n_vt * softplus(arg2)
    return Id / np.clip(denom, 1e-15, None)


def asymptote_sat(Id, Vov):
    return 2 * Id / np.where(Vov > 1e-6, Vov, np.nan)


def asymptote_triode(Id, Vov, Vds):
    den = Vov - Vds / 2
    return Id / np.where(den > 1e-6, den, np.nan)


def main():
    fig, ax = plt.subplots(1, 1, figsize=(11, 6.5))
    Id_ref = 100e-6
    Vds_fixed = 0.30

    Vov_sweep = np.linspace(-0.12, 0.70, 1200)
    Id_arr = np.full_like(Vov_sweep, Id_ref)

    gm_smaxt = smaxt_gm(Id_arr, Vov_sweep, np.full_like(Vov_sweep, Vds_fixed))
    gm_sat = asymptote_sat(Id_arr, Vov_sweep)
    gm_triode = asymptote_triode(Id_arr, Vov_sweep, np.full_like(Vov_sweep, Vds_fixed))
    gm_cutoff_val = Id_ref / n_vt

    # ---- Region shading: cutoff (V_ov < n*V_T), saturation (n*V_T < V_ov < V_DS), triode (V_ov > V_DS)
    ymin_plot, ymax_plot = 5e-5, 2e-2
    ax.axvspan(Vov_sweep[0] * 1000, n_vt * 1000,
               color='#9467bd', alpha=0.06, zorder=0)
    ax.axvspan(n_vt * 1000, Vds_fixed * 1000,
               color='#1f77b4', alpha=0.06, zorder=0)
    ax.axvspan(Vds_fixed * 1000, Vov_sweep[-1] * 1000,
               color='#ff7f0e', alpha=0.08, zorder=0)

    # Region labels near top
    ax.text((Vov_sweep[0]*1000 + n_vt*1000)/2, ymax_plot*0.7, 'CUTOFF',
            ha='center', va='top', fontsize=11, fontweight='bold',
            color='#5e3370', alpha=0.55)
    ax.text((n_vt*1000 + Vds_fixed*1000)/2, ymax_plot*0.7, 'SATURATION',
            ha='center', va='top', fontsize=11, fontweight='bold',
            color='#0e4d75', alpha=0.55)
    ax.text((Vds_fixed*1000 + Vov_sweep[-1]*1000)/2, ymax_plot*0.7, 'TRIODE',
            ha='center', va='top', fontsize=11, fontweight='bold',
            color='#a04400', alpha=0.55)

    # ---- Asymptote curves (clip to visible band so transitions are obvious)
    gm_sat_plot = np.where((gm_sat > ymin_plot) & (gm_sat < ymax_plot * 4), gm_sat, np.nan)
    gm_triode_plot = np.where((gm_triode > ymin_plot) & (gm_triode < ymax_plot * 4),
                              gm_triode, np.nan)

    ax.semilogy(Vov_sweep * 1000, gm_sat_plot, '--', color='#1f77b4', linewidth=2.0,
                label=r'Saturation:  $g_m = 2 I_d / V_{ov}$', alpha=0.9, zorder=2)
    ax.semilogy(Vov_sweep * 1000, gm_triode_plot, '--', color='#ff7f0e', linewidth=2.0,
                label=r'Triode:  $g_m = I_d / (V_{ov} - V_{DS}/2)$', alpha=0.9, zorder=2)
    ax.axhline(gm_cutoff_val, linestyle='--', color='#9467bd', linewidth=2.0,
               label=r'Cutoff:  $g_m = I_d / (n V_T)$', alpha=0.9, zorder=2)

    # ---- SMAXT curve on top
    ax.semilogy(Vov_sweep * 1000, gm_smaxt, color='#2ca02c', linewidth=3.4,
                label='SMAXT (unified)', zorder=5)

    # ---- Transition markers with annotations
    ax.axvline(n_vt * 1000, color='gray', linewidth=0.8, linestyle=':', alpha=0.7, zorder=1)
    ax.axvline(Vds_fixed * 1000, color='gray', linewidth=0.8, linestyle=':', alpha=0.7, zorder=1)
    ax.annotate(r'$V_{ov} = n V_T$', xy=(n_vt * 1000, 7e-5),
                xytext=(n_vt * 1000 + 35, 7e-5),
                fontsize=10, color='dimgray', va='center',
                arrowprops=dict(arrowstyle='-', color='dimgray', lw=0.7))
    ax.annotate(r'$V_{ov} = V_{DS}$', xy=(Vds_fixed * 1000, 7e-5),
                xytext=(Vds_fixed * 1000 + 35, 7e-5),
                fontsize=10, color='dimgray', va='center',
                arrowprops=dict(arrowstyle='-', color='dimgray', lw=0.7))

    # ---- Highlight the actual SMAXT-vs-asymptote transitions visually
    # mark the points where SMAXT is closest to the asymptote-crossover
    crossover_cutoff_sat_idx = np.argmin(np.abs(Vov_sweep - n_vt))
    crossover_sat_triode_idx = np.argmin(np.abs(Vov_sweep - Vds_fixed))
    ax.scatter([n_vt*1000, Vds_fixed*1000],
               [gm_smaxt[crossover_cutoff_sat_idx], gm_smaxt[crossover_sat_triode_idx]],
               color='#2ca02c', s=80, zorder=6, edgecolor='white', linewidth=1.5)

    ax.set_xlabel(r'$V_{ov} = V_{GS} - V_{th}$   (mV)', fontsize=13)
    ax.set_ylabel(r'$g_m$   (S)', fontsize=13)
    ax.set_title(f'cutoff → saturation → triode transition\n'
                 fr'($I_d$ = {Id_ref*1e6:.0f} µA, $V_{{DS}}$ = {Vds_fixed*1000:.0f} mV, NMOS n={n})',
                 fontsize=13, fontweight='bold', loc='left')
    ax.legend(loc='lower right', fontsize=11, frameon=True, framealpha=0.94,
              edgecolor='lightgray')
    ax.grid(True, which='both', alpha=0.3, linestyle=':')
    ax.set_ylim(ymin_plot, ymax_plot)
    ax.set_xlim(Vov_sweep[0]*1000, Vov_sweep[-1]*1000)
    ax.tick_params(axis='both', labelsize=11)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {OUT}')


if __name__ == '__main__':
    main()
