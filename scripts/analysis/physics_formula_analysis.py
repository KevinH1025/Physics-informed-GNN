#!/usr/bin/env python
"""§4.3.1 — Comprehensive analytical-formula vs BSIM4 analysis.
Inputs: SPICE GROUND TRUTH V, I, gm, gds, Vth, region labels (no model in loop).

Reports for each formula:
  - logMAE, median log err, median rel %, correlation r
And for V_th and µCox·W/L (the "effective transconductance parameter"):
  - per-device-instance distribution, range, std
And V_ov sensitivity:
  - how formula errors change if V_th is perturbed by ±50 mV
"""
import pickle
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / 'datasets/opamp_3stage_pretrain_combined_5topo/dataset_val.pkl'

V_T = 0.02585
N_NMOS, N_PMOS = 1.5, 2.0


def stats(log_p, log_t, name):
    """Return logMAE, median log err, median rel%, mean rel%, correlation r."""
    err = np.abs(log_p - log_t)
    rel = (10 ** err - 1) * 100
    rho = np.corrcoef(log_p, log_t)[0, 1] if log_p.size > 1 else float('nan')
    return f"{name:<28} N={log_p.size:>6,}  logMAE={err.mean():.4f}  med_log={np.median(err):.4f}  med_rel={np.median(rel):.2f}%  mean_rel={rel.mean():.2f}%  r={rho:.4f}"


def main():
    data = pickle.load(open(DATA, 'rb'))
    fan = [s for s in data if s.get('topology') == 'fan_smc']
    N = len(fan)
    print(f'fan_smc val: {N} samples × 24 MOSFETs = {N*24} device-instances\n')

    # Per-MOSFET GT arrays
    Vov   = np.zeros((N, 24)); Vds = np.zeros((N, 24)); Vgs = np.zeros((N, 24))
    Vth   = np.zeros((N, 24))
    Id    = np.zeros((N, 24)); gm  = np.zeros((N, 24)); gds = np.zeros((N, 24))
    region= np.zeros((N, 24), dtype=int); is_n = np.zeros((N, 24), dtype=bool)
    W_norm = np.zeros((N, 24)); L_norm = np.zeros((N, 24))
    for i, s in enumerate(fan):
        g = s['graph']; mi = g.mosfet_info.long().numpy()
        Vt_arr = g.node_voltage_targets.numpy()
        Vg = Vt_arr[mi[:, 3]]; Vd = Vt_arr[mi[:, 4]]; Vs_ = Vt_arr[mi[:, 5]]
        is_nmos = mi[:, 6].astype(bool)
        Vgs_i = np.where(is_nmos, Vg - Vs_, Vs_ - Vg)
        Vds_i = np.where(is_nmos, Vd - Vs_, Vs_ - Vd)
        Vth_i = np.abs(g.node_mosfet_vth.numpy()[mi[:, 1]])
        Vov[i]    = Vgs_i - Vth_i
        Vds[i]    = Vds_i; Vgs[i] = Vgs_i; Vth[i] = Vth_i
        Id[i]     = np.abs(g.node_current_targets.numpy()[mi[:, 1]])
        gm[i]     = 10 ** g.node_log_gm.numpy()[mi[:, 1]]
        gds[i]    = 10 ** g.node_log_gds.numpy()[mi[:, 1]]
        region[i] = g.mosfet_region_labels.numpy()
        is_n[i]   = is_nmos
        W_norm[i] = g.x.numpy()[mi[:, 1], 0]
        L_norm[i] = g.x.numpy()[mi[:, 1], 1]

    log_gm  = np.log10(np.clip(gm,  1e-15, None))
    log_gds = np.log10(np.clip(gds, 1e-15, None))

    # ============== 1. Region equation accuracy ==============
    print("=" * 100)
    print("PART 1 — Standalone region-equation accuracy (SPICE inputs, comparing to SPICE outputs)")
    print("=" * 100)

    # --- Saturation: gm = 2·I_D/V_ov ---
    # The formula requires V_ov > 0; "saturation" SPICE region can still have
    # small or near-zero V_ov (right at threshold). Report with no filter,
    # then with progressively stricter V_ov filters.
    m_sat = (region == 2)
    Vov_s = Vov[m_sat]; Id_s = Id[m_sat]
    print('\n[Saturation region]')
    print(f'  V_ov distribution in saturation devices:')
    print(f'    min={Vov_s.min():.4f} V, max={Vov_s.max():.4f} V, frac with V_ov ≤ 0: {(Vov_s<=0).mean()*100:.1f}%')
    for thr in [0, 0.01, 0.05, 0.1]:
        m = Vov_s > thr
        gm_eq = 2 * Id_s[m] / Vov_s[m]
        print(stats(np.log10(np.clip(gm_eq, 1e-15, None)), log_gm[m_sat][m],
                    f"  gm = 2·I_D/V_ov, filter V_ov>{thr*1000:.0f}mV"))

    # --- Triode eq1: gm = I_D/(V_ov - V_DS/2) ---
    m_tri = (region == 1)
    Vov_t = Vov[m_tri]; Vds_t = Vds[m_tri]; Id_t = Id[m_tri]
    gm_tri_eq = Id_t / np.clip(Vov_t - Vds_t/2, 1e-6, None)
    gds_tri_eq = (Id_t / np.clip(Vds_t, 1e-6, None)) * (Vov_t - Vds_t) / np.clip(Vov_t - Vds_t/2, 1e-6, None)
    print('\n[Triode region]')
    print(stats(np.log10(np.clip(gm_tri_eq, 1e-15, None)), log_gm[m_tri],   "  gm = I_D/(V_ov - V_DS/2)"))
    print(stats(np.log10(np.clip(gds_tri_eq, 1e-15, None)), log_gds[m_tri], "  gds (square-law)"))

    # --- Cutoff: gm = I_D/(n·V_T) ---
    m_co = (region == 0)
    n_vt_co = np.where(is_n[m_co], N_NMOS, N_PMOS) * V_T
    gm_co_eq = Id[m_co] / n_vt_co
    print('\n[Cutoff (sub-threshold) region]')
    print(stats(np.log10(np.clip(gm_co_eq, 1e-15, None)), log_gm[m_co], "  gm = I_D / (n·V_T)"))

    # --- SMAXT (all-region unified) ---
    n_vt = np.where(is_n, N_NMOS, N_PMOS) * V_T
    sp = lambda x: np.log1p(np.exp(np.clip(x, -50, 50)))
    Vds_eff = Vds - n_vt * sp((Vds - Vov) / n_vt)
    denom = n_vt + n_vt * sp((Vov - Vds_eff/2 - n_vt) / n_vt)
    gm_smaxt = Id / np.clip(denom, 1e-12, None)
    print('\n[SMAXT (all regions, smooth-max unified)]')
    print(stats(np.log10(np.clip(gm_smaxt, 1e-15, None)).flatten(), log_gm.flatten(), "  gm_SMAXT (all 24k)"))
    for r, name in [(0,'cutoff'), (1,'triode'), (2,'saturation')]:
        mr = (region == r)
        print(stats(np.log10(np.clip(gm_smaxt[mr], 1e-15, None)), log_gm[mr], f"    SMAXT — {name:<10}"))

    # ============== 2. V_ov sensitivity ==============
    print("\n" + "=" * 100)
    print("PART 2 — V_ov sensitivity: how does formula error change if V_th is perturbed?")
    print("=" * 100)
    print("  V_ov = V_GS − V_th. In our setup V_th is the SPICE BSIM4 per-device |V_th| ")
    print("  (node_mosfet_vth, exact ground truth). Below we PERTURB V_th by ±25, ±50 mV ")
    print("  to see how sensitive each formula is to V_ov mis-specification.\n")

    def perturbed_smaxt(Vth_shift):
        Vov_p = Vgs - (Vth + Vth_shift)
        Vds_eff_p = Vds - n_vt * sp((Vds - Vov_p) / n_vt)
        denom_p = n_vt + n_vt * sp((Vov_p - Vds_eff_p/2 - n_vt) / n_vt)
        gm_p = Id / np.clip(denom_p, 1e-12, None)
        err = np.abs(np.log10(np.clip(gm_p, 1e-15, None)) - log_gm)
        return err.mean(), np.median(err), (10**np.median(err)-1)*100

    def perturbed_sat(Vth_shift):
        Vov_p = Vgs[m_sat] - (Vth[m_sat] + Vth_shift)
        gm_eq_p = 2 * Id[m_sat] / np.clip(Vov_p, 1e-6, None)
        err = np.abs(np.log10(np.clip(gm_eq_p, 1e-15, None)) - log_gm[m_sat])
        return err.mean(), np.median(err), (10**np.median(err)-1)*100

    print('  V_th shift  →  SMAXT (all)               saturation gm = 2I/V_ov')
    print('              logMAE / med_rel%             logMAE / med_rel%')
    for shift in [-0.050, -0.025, 0.000, +0.025, +0.050]:
        s_mae, s_med, s_relmed = perturbed_smaxt(shift)
        sat_mae, sat_med, sat_relmed = perturbed_sat(shift)
        print(f'  {shift*1000:+5.0f} mV   →  {s_mae:.4f} / {s_relmed:>6.2f}%       {sat_mae:.4f} / {sat_relmed:>6.2f}%')

    # ============== 3. V_th and µCox·W/L distributions ==============
    print("\n" + "=" * 100)
    print("PART 3 — V_th and µCox·(W/L) ranges across devices (non-constancy)")
    print("=" * 100)

    # V_th
    print('\n[V_th (SPICE BSIM4 |V_th| at each drain terminal)]')
    Vth_all = Vth.flatten()
    print(f'  All devices:                  N={Vth_all.size:,}')
    print(f'    range:  {Vth_all.min():.4f} V to {Vth_all.max():.4f} V')
    print(f'    mean ± std:  {Vth_all.mean():.4f} ± {Vth_all.std():.4f} V')
    print(f'    median:  {np.median(Vth_all):.4f} V')
    # NMOS only
    for label, mask in [('NMOS only', is_n.flatten()), ('PMOS only', ~is_n.flatten())]:
        v = Vth_all[mask]
        print(f'  {label}: N={v.size:,}, range {v.min():.3f}–{v.max():.3f} V, mean {v.mean():.3f} V, std {v.std():.3f} V')

    # µCox·W/L extraction from saturation: I_D = ½ µCox (W/L) V_ov² → µCox·W/L = 2 I_D / V_ov²
    print('\n[µCox·W/L (effective conductance parameter, A/V²)]')
    print('   from I_D = ½ µCox (W/L) V_ov² in saturation → µCox·W/L = 2 I_D / V_ov²')
    Vov_sat = Vov[m_sat]; Id_sat = Id[m_sat]
    # Filter out near-threshold devices where V_ov² blows the formula up
    valid = Vov_sat > 0.05
    mu_cox_wl = 2 * Id_sat[valid] / (Vov_sat[valid] ** 2)
    print(f'  (filter: V_ov > 50 mV to avoid near-threshold blow-up)')
    print(f'  N = {mu_cox_wl.size:,}')
    print(f'    range:  {mu_cox_wl.min():.3e} to {mu_cox_wl.max():.3e} A/V²')
    print(f'    range in orders of magnitude: {np.log10(mu_cox_wl.max()/mu_cox_wl.min()):.2f}')
    print(f'    median:  {np.median(mu_cox_wl):.3e} A/V²')
    print(f'    mean ± std (linear):  {mu_cox_wl.mean():.3e} ± {mu_cox_wl.std():.3e}')
    print(f'    log10(µCox·W/L) range: [{np.log10(mu_cox_wl.min()):.2f}, {np.log10(mu_cox_wl.max()):.2f}]')
    print(f'    log10 std (a "decade spread"): {np.log10(mu_cox_wl).std():.3f} (so ~{10**np.log10(mu_cox_wl).std():.1f}× spread around median)')

    # NMOS vs PMOS µCox·W/L
    is_n_sat = is_n[m_sat][valid]
    for label, m in [('NMOS sat', is_n_sat), ('PMOS sat', ~is_n_sat)]:
        v = mu_cox_wl[m]
        if v.size:
            print(f'    {label}: N={v.size}, range {v.min():.2e}–{v.max():.2e} A/V², median {np.median(v):.2e}')
        else:
            print(f'    {label}: N=0 (no devices satisfied V_ov>50mV filter)')


if __name__ == '__main__':
    main()
