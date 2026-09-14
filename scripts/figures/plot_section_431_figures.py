#!/usr/bin/env python
"""§4.3.1 — Generates 4 (+1) figures for the constraint-choice subsection.
All computations on SPICE ground truth (no model in the loop)."""
import pickle, math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / 'datasets/opamp_3stage_fan_smc_openloop_5k/dataset_val.pkl'
OUT_DIR = REPO / 'figures/thesis'
OUT_DIR.mkdir(parents=True, exist_ok=True)

V_T = 0.02585
N_NMOS, N_PMOS = 1.5, 2.0

# Color palette matching existing thesis figures
C_PARITY = '#1f77b4'
C_CUTOFF = '#9467bd'
C_TRIODE = '#ff7f0e'
C_SAT    = '#1f77b4'
C_SMAXT  = '#2ca02c'
C_NMOS   = '#1f77b4'
C_PMOS   = '#d62728'


def parity_panel(ax, x_log, y_log, title, color, units='log₁₀(g_m) (log S)', extra_metric_lines=None):
    """Scatter (x = SPICE truth, y = formula prediction) in log space + diagonal + metric box."""
    ax.scatter(x_log, y_log, s=8, alpha=0.4, color=color, edgecolors='none', rasterized=True)
    lo = float(min(x_log.min(), y_log.min())); hi = float(max(x_log.max(), y_log.max()))
    rng = hi - lo
    lo, hi = lo - 0.04 * rng, hi + 0.04 * rng
    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1, alpha=0.7)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    err = np.abs(x_log - y_log)
    rho = np.corrcoef(x_log, y_log)[0, 1] if x_log.size > 1 else float('nan')
    rel = (10 ** err - 1) * 100
    lines = [
        f'N = {x_log.size:,}',
        f'logMAE = {err.mean():.4f}',
        f'median = {np.median(err):.4f}',
        f'med rel = {np.median(rel):.2f}%',
        f'corr r = {rho:.4f}',
    ]
    if extra_metric_lines: lines.extend(extra_metric_lines)
    print(f'[parity] {title}')
    for ln in lines:
        print(f'    {ln}')
    ax.text(0.04, 0.96, '\n'.join(lines), transform=ax.transAxes,
            fontsize=10, va='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.92, edgecolor='gray'))
    ax.set_xlabel(f'SPICE ground truth, {units}', fontsize=11)
    ax.set_ylabel(f'Formula prediction, {units}', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold', loc='left')
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_aspect('equal')


def load_fan_data():
    data = pickle.load(open(DATA, 'rb'))
    fan = [s for s in data if s.get('topology', 'fan_smc') == 'fan_smc']
    N = len(fan)
    print(f'fan_smc val: {N} samples × 24 MOSFETs')
    # Per-MOSFET arrays
    Vov = np.zeros((N, 24)); Vds = np.zeros((N, 24)); Vgs = np.zeros((N, 24))
    Vth = np.zeros((N, 24)); Id = np.zeros((N, 24))
    gm  = np.zeros((N, 24)); gds = np.zeros((N, 24))
    region = np.zeros((N, 24), dtype=int); is_n = np.zeros((N, 24), dtype=bool)
    # Per-sample
    UGBW = np.zeros(N); ac_valid = np.zeros(N, dtype=bool)
    C_C = np.zeros(N); gm_M8 = np.zeros(N)
    for i, s in enumerate(fan):
        g = s['graph']
        mi = g.mosfet_info.long().numpy()
        Vt_arr = g.node_voltage_targets.numpy()
        Vg = Vt_arr[mi[:, 3]]; Vd = Vt_arr[mi[:, 4]]; Vs_ = Vt_arr[mi[:, 5]]
        is_nmos = mi[:, 6].astype(bool)
        Vgs_i = np.where(is_nmos, Vg - Vs_, Vs_ - Vg)
        Vds_i = np.where(is_nmos, Vd - Vs_, Vs_ - Vd)
        Vth_i = np.abs(g.node_mosfet_vth.numpy()[mi[:, 1]])
        Vov[i] = Vgs_i - Vth_i; Vds[i] = Vds_i; Vgs[i] = Vgs_i; Vth[i] = Vth_i
        Id[i] = np.abs(g.node_current_targets.numpy()[mi[:, 1]])
        gm[i] = 10 ** g.node_log_gm.numpy()[mi[:, 1]]
        gds[i] = 10 ** g.node_log_gds.numpy()[mi[:, 1]]
        region[i] = g.mosfet_region_labels.numpy()
        is_n[i] = is_nmos
        if hasattr(g, 'ac_ugbw'):
            UGBW[i] = float(g.ac_ugbw[0])
        if hasattr(g, 'ac_valid'):
            ac_valid[i] = bool(g.ac_valid)
        C_C[i] = s['params'].get('C_COMP', 1e-12)
        gm_M8[i] = gm[i, 8]   # diff-pair MOSFET (gm1)
    return dict(Vov=Vov, Vds=Vds, Vgs=Vgs, Vth=Vth, Id=Id, gm=gm, gds=gds,
                region=region, is_n=is_n, UGBW=UGBW, ac_valid=ac_valid,
                C_C=C_C, gm_M8=gm_M8, N=N)


def figure_1_region_parity(d):
    log_gm = np.log10(np.clip(d['gm'], 1e-15, None))
    log_gds = np.log10(np.clip(d['gds'], 1e-15, None))

    # (a) Triode: gm = I_D/(V_ov - V_DS/2)
    # Apply training-time dynamic filter: V_ov > 1mV, V_ds > 1µV, V_ov > V_ds (true triode)
    m_tri_region = (d['region'] == 1)
    Vov_all = d['Vov']; Vds_all = d['Vds']; Id_all = d['Id']
    m_tri_dyn = m_tri_region & (Vov_all > 1e-3) & (Vds_all > 1e-6) & (Vov_all > Vds_all)
    Vov_t = Vov_all[m_tri_dyn]; Vds_t = Vds_all[m_tri_dyn]; Id_t = Id_all[m_tri_dyn]
    denom_t = np.clip(Vov_t - Vds_t/2, 1e-6, None)
    gm_tri = Id_t / denom_t
    log_gm_tri_t = log_gm[m_tri_dyn]
    # (b) Cutoff
    m_co = (d['region'] == 0)
    n_vt = np.where(d['is_n'][m_co], N_NMOS, N_PMOS) * V_T
    gm_co = d['Id'][m_co] / n_vt
    # (c) Saturation — V_ov > 100 mV (strong-inversion regime, where 2·I_D/V_ov is defensible)
    m_sat = (d['region'] == 2)
    Vov_s = d['Vov'][m_sat]; Id_s = d['Id'][m_sat]
    filt = Vov_s > 0.100
    gm_sat = 2 * Id_s[filt] / Vov_s[filt]
    log_gm_sat_t = log_gm[m_sat][filt]
    # (d) SMAXT (all)
    n_vt_all = np.where(d['is_n'], N_NMOS, N_PMOS) * V_T
    sp = lambda x: np.log1p(np.exp(np.clip(x, -50, 50)))
    Vds_eff = d['Vds'] - n_vt_all * sp((d['Vds'] - d['Vov']) / n_vt_all)
    denom = n_vt_all + n_vt_all * sp((d['Vov'] - Vds_eff/2 - n_vt_all) / n_vt_all)
    gm_smaxt = d['Id'] / np.clip(denom, 1e-12, None)

    fig, axes = plt.subplots(2, 2, figsize=(13, 13))
    parity_panel(axes[0, 0],
                 log_gm_tri_t,
                 np.log10(np.clip(gm_tri, 1e-15, None)),
                 '(a) Triode region: gm = I_D / (V_ov − V_DS/2)  (V_ov > V_DS)',
                 C_TRIODE)
    parity_panel(axes[0, 1],
                 log_gm[m_co],
                 np.log10(np.clip(gm_co, 1e-15, None)),
                 '(b) Cutoff region: gm = I_D / (n·V_T)',
                 C_CUTOFF)
    parity_panel(axes[1, 0],
                 log_gm_sat_t,
                 np.log10(np.clip(gm_sat, 1e-15, None)),
                 '(c) Saturation region: gm = 2·I_D / V_ov  (V_ov > 100 mV)',
                 C_SAT)
    parity_panel(axes[1, 1],
                 log_gm.flatten(),
                 np.log10(np.clip(gm_smaxt, 1e-15, None)).flatten(),
                 '(d) SMAXT (all-region unified)',
                 C_SMAXT)
    fig.tight_layout()
    out = OUT_DIR / 'fig_region_formula_parity.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out}')


def figure_2_ugbw_parity(d):
    """UGBW = g_m1 / (2π·C_C) vs SPICE UGBW. Use per-sample C_C."""
    valid = d['ac_valid'] & (d['UGBW'] > 0)
    gm1 = d['gm_M8'][valid]
    CC  = d['C_C'][valid]
    UGBW_pred = gm1 / (2 * np.pi * CC)
    UGBW_gt   = d['UGBW'][valid]
    log_p = np.log10(np.clip(UGBW_pred, 1, None))
    log_t = np.log10(np.clip(UGBW_gt, 1, None))
    err = np.abs(log_p - log_t)
    rel = (10 ** err - 1) * 100
    within5  = (rel < 5).mean() * 100
    within10 = (rel < 10).mean() * 100
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 7))
    parity_panel(ax, log_t, log_p,
                 '(a) Analytical UGBW = g_m1 / (2π · C_C)  vs SPICE UGBW',
                 '#d62728',
                 units='log₁₀(UGBW) (log Hz)',
                 extra_metric_lines=[
                     f'within 5%  = {within5:.1f}%',
                     f'within 10% = {within10:.1f}%',
                 ])
    fig.tight_layout()
    out = OUT_DIR / 'fig_ugbw_formula_parity.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out}')


def figure_3_vth(d):
    Vth_all = d['Vth'].flatten()
    is_n_flat = d['is_n'].flatten()
    Vth_n = Vth_all[is_n_flat]; Vth_p = Vth_all[~is_n_flat]
    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
    bins = np.linspace(0.45, 1.10, 80)
    ax.hist(Vth_n, bins=bins, alpha=0.7, color=C_NMOS, edgecolor='black', linewidth=0.4,
            label=f'NMOS (N={Vth_n.size:,}) — range {Vth_n.min():.3f}–{Vth_n.max():.3f} V, '
                  f'mean {Vth_n.mean():.3f} V, σ={Vth_n.std()*1000:.0f} mV')
    ax.hist(Vth_p, bins=bins, alpha=0.7, color=C_PMOS, edgecolor='black', linewidth=0.4,
            label=f'PMOS (N={Vth_p.size:,}) — range {Vth_p.min():.3f}–{Vth_p.max():.3f} V, '
                  f'mean {Vth_p.mean():.3f} V, σ={Vth_p.std()*1000:.0f} mV')
    ax.set_xlabel('SPICE per-device |V_th| (V)', fontsize=12)
    ax.set_ylabel('count (device-instances)', fontsize=12)
    ax.set_title('(a) V_th distribution across 24 000 device-instances',
                 fontsize=13, fontweight='bold', loc='left')
    ax.legend(loc='upper center', fontsize=10, frameon=True, framealpha=0.92)
    ax.grid(True, axis='y', alpha=0.3, linestyle=':')
    ax.text(0.03, 0.96,
            f'V_th spans 0.503 → 1.063 V\n(560 mV range; not constant)',
            transform=ax.transAxes, fontsize=10, va='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.92, edgecolor='gray'))
    fig.tight_layout()
    out = OUT_DIR / 'fig_vth_distribution.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out}')


def figure_4_mucox(d):
    m_sat = (d['region'] == 2)
    Vov_s = d['Vov'][m_sat]; Id_s = d['Id'][m_sat]
    filt = Vov_s > 0.050
    mu = 2 * Id_s[filt] / (Vov_s[filt] ** 2)
    log_mu = np.log10(mu)
    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
    bins = np.linspace(log_mu.min() - 0.1, log_mu.max() + 0.1, 70)
    ax.hist(log_mu, bins=bins, alpha=0.85, color='#2ca02c', edgecolor='black', linewidth=0.4)
    ax.set_xlabel('log₁₀ ( µCox · W/L )  (log A/V²)', fontsize=12)
    ax.set_ylabel('count (device-instances)', fontsize=12)
    ax.set_title('(b) µCox·(W/L) distribution from saturation devices  (V_ov > 50 mV)',
                 fontsize=13, fontweight='bold', loc='left')
    ax.grid(True, axis='y', alpha=0.3, linestyle=':')
    ax.text(0.03, 0.96,
            f'N = {mu.size:,}\n'
            f'range  : {mu.min():.2e} – {mu.max():.2e} A/V²\n'
            f'spans  : {np.log10(mu.max()/mu.min()):.2f} decades\n'
            f'median : {np.median(mu):.2e} A/V²\n'
            f'log₁₀ σ: {log_mu.std():.3f} ({10**log_mu.std():.1f}× spread)',
            transform=ax.transAxes, fontsize=10, va='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.92, edgecolor='gray'))
    fig.tight_layout()
    out = OUT_DIR / 'fig_mucox_distribution.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out}')


def figure_5_combined_nonconst(d):
    """Combined 2-panel: V_th hist + µCox·W/L hist."""
    Vth_all = d['Vth'].flatten()
    is_n_flat = d['is_n'].flatten()
    Vth_n = Vth_all[is_n_flat]; Vth_p = Vth_all[~is_n_flat]
    m_sat = (d['region'] == 2)
    Vov_s = d['Vov'][m_sat]; Id_s = d['Id'][m_sat]
    filt = Vov_s > 0.050
    mu = 2 * Id_s[filt] / (Vov_s[filt] ** 2)
    log_mu = np.log10(mu)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    # Panel a — V_th
    ax = axes[0]
    bins = np.linspace(0.45, 1.10, 80)
    ax.hist(Vth_n, bins=bins, alpha=0.7, color=C_NMOS, edgecolor='black', linewidth=0.4,
            label=f'NMOS  (range {Vth_n.min():.3f}–{Vth_n.max():.3f} V)')
    ax.hist(Vth_p, bins=bins, alpha=0.7, color=C_PMOS, edgecolor='black', linewidth=0.4,
            label=f'PMOS  (range {Vth_p.min():.3f}–{Vth_p.max():.3f} V)')
    ax.set_xlabel('SPICE per-device |V_th| (V)', fontsize=12)
    ax.set_ylabel('count (device-instances)', fontsize=12)
    ax.set_title('(a) V_th distribution (24 000 device-instances)',
                 fontsize=13, fontweight='bold', loc='left')
    ax.legend(loc='upper center', fontsize=10, frameon=True, framealpha=0.92)
    ax.grid(True, axis='y', alpha=0.3, linestyle=':')
    ax.text(0.03, 0.96,
            f'NMOS σ = {Vth_n.std()*1000:.0f} mV\n'
            f'PMOS σ = {Vth_p.std()*1000:.0f} mV\n'
            f'overall range: 560 mV',
            transform=ax.transAxes, fontsize=10, va='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.92, edgecolor='gray'))
    # Panel b — µCox·W/L
    ax = axes[1]
    bins = np.linspace(log_mu.min() - 0.1, log_mu.max() + 0.1, 70)
    ax.hist(log_mu, bins=bins, alpha=0.85, color='#2ca02c', edgecolor='black', linewidth=0.4)
    ax.set_xlabel('log₁₀ ( µCox · W/L )  (log A/V²)', fontsize=12)
    ax.set_ylabel('count (device-instances)', fontsize=12)
    ax.set_title('(b) µCox·(W/L) (saturation devices, V_ov > 50 mV)',
                 fontsize=13, fontweight='bold', loc='left')
    ax.grid(True, axis='y', alpha=0.3, linestyle=':')
    ax.text(0.03, 0.96,
            f'N = {mu.size:,}\n'
            f'range: {mu.min():.2e} – {mu.max():.2e} A/V²\n'
            f'spans {np.log10(mu.max()/mu.min()):.1f} decades\n'
            f'median {np.median(mu):.2e}\nlog₁₀ σ = {log_mu.std():.3f} ({10**log_mu.std():.1f}× spread)',
            transform=ax.transAxes, fontsize=10, va='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.92, edgecolor='gray'))
    fig.tight_layout()
    out = OUT_DIR / 'fig_nonconstant_params.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out}')


def main():
    d = load_fan_data()
    figure_1_region_parity(d)
    figure_2_ugbw_parity(d)
    figure_3_vth(d)
    figure_4_mucox(d)
    figure_5_combined_nonconst(d)


if __name__ == '__main__':
    main()
