#!/usr/bin/env python
"""For each §4.3.1 physics constraint, evaluate the formula's accuracy against
SPICE ground truth (V, I, gm, gds, V_th, UGBW) on fan_smc val (1000 samples).
This isolates the formula's intrinsic mismatch with BSIM4 from any model-prediction error.

Constraints checked:
  - SMAXT: gm_smaxt = I_D / softplus_denom(V_ov, V_DS, n_VT) vs gm_GT
  - Triode eq1: gm = I_D / (V_ov - V_DS/2) vs gm_GT (triode-region only)
  - Triode eq2: gds = (I_D/V_DS)·(V_ov-V_DS)/(V_ov-V_DS/2) vs gds_GT
  - Cutoff: gm = I_D / (n·V_T) vs gm_GT (cutoff-region only)
  - Mirror: I_mirror/I_ref = (W/L)_mirror/(W/L)_ref for 10 fan_smc hardcoded pairs
  - Diff-pair: I_M8 + I_M9 = 4·I_M0 (M4 is 4× M0 tail current)
  - UGBW: gm_M8 / (2π·C_C) vs SPICE UGBW
"""
import pickle, math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

REPO = Path('/dss/dsshome1/03/go49jit2/thesis')
DATA = REPO / 'datasets/opamp_3stage_pretrain_combined_5topo/dataset_val.pkl'

V_T = 0.02585  # thermal voltage at 27 C
N_NMOS, N_PMOS = 1.5, 2.0  # subthreshold slope factors

# fan_smc hardcoded mirror pairs from the run's training log:
#   M0=M1, M0=M2, M0=M3, M0=M7, M4=4*M0, M5=M6, M17=M18, M19=M20, M19=2*M17, M21=M22
MIRROR_PAIRS = [
    # (ref_idx, mirror_idx, expected_ratio_mirror/ref)
    (0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0), (0, 7, 1.0),
    (0, 4, 4.0),
    (5, 6, 1.0),
    (17, 18, 1.0),
    (19, 20, 1.0),
    (17, 19, 2.0),
    (21, 22, 1.0),
]


def main():
    print('Loading raw fan_smc val samples...')
    data = pickle.load(open(DATA, 'rb'))
    fan = [s for s in data if s.get('topology') == 'fan_smc']
    N = len(fan)
    print(f'  {N} samples\n')

    # Pre-extract per-sample tensors
    # mosfet_info columns: [gate_t, drain_t, source_t, gate_n, drain_n, source_n, is_nmos]
    # We need per-MOSFET: V_GS, V_DS, V_th, I_d, gm, gds, region, W, L, mult
    Vov_per   = np.zeros((N, 24))
    Vds_per   = np.zeros((N, 24))
    Id_per    = np.zeros((N, 24))   # |I_d| in amps
    gm_per    = np.zeros((N, 24))   # linear gm in S
    gds_per   = np.zeros((N, 24))
    region_per= np.zeros((N, 24), dtype=np.int32)
    is_nmos   = np.zeros((N, 24), dtype=bool)
    W_per     = np.zeros((N, 24))   # W in meters
    L_per     = np.zeros((N, 24))
    UGBW_GT   = np.zeros(N)
    ac_valid  = np.zeros(N, dtype=bool)

    # MOSFET W, L per sample (x feature 0=W, 1=L for terminals — but values are at terminal nodes)
    # Use mosfet_info[:, 1] = drain terminal node index, read x[drain, 0]/x[drain, 1]
    for i, s in enumerate(fan):
        g = s['graph']
        mi = g.mosfet_info.long().numpy()
        # Terminal voltages from voltage targets at gate/drain/source NET indices (cols 3,4,5)
        Vt_arr = g.node_voltage_targets.numpy()
        Vg = Vt_arr[mi[:, 3]]; Vd = Vt_arr[mi[:, 4]]; Vs = Vt_arr[mi[:, 5]]
        is_n = mi[:, 6].astype(bool)
        Vgs = np.where(is_n, Vg - Vs, Vs - Vg)
        Vds = np.where(is_n, Vd - Vs, Vs - Vd)
        Vth = np.abs(g.node_mosfet_vth.numpy()[mi[:, 1]])   # at drain terminal
        Vov = Vgs - Vth
        Id  = np.abs(g.node_current_targets.numpy()[mi[:, 1]])
        gm  = 10 ** g.node_log_gm.numpy()[mi[:, 1]]
        gds = 10 ** g.node_log_gds.numpy()[mi[:, 1]]
        rl  = g.mosfet_region_labels.numpy()
        # W, L from x feature (first 2 columns are W, L normalized): un-normalize?
        # The dataset feature is W/L raw values (in meters, log-scaled in some configs)
        # x[drain, 0] = W, x[drain, 1] = L — but they may be normalized. For mirror,
        # only the ratio matters, so the normalization cancels out.
        x = g.x.numpy()
        W = x[mi[:, 1], 0]
        L = x[mi[:, 1], 1]
        Vov_per[i]    = Vov;   Vds_per[i] = Vds
        Id_per[i]     = Id;    gm_per[i]  = gm; gds_per[i] = gds
        region_per[i] = rl;    is_nmos[i] = is_n
        W_per[i]      = W;     L_per[i]   = L
        if hasattr(g, 'ac_ugbw'):
            UGBW_GT[i] = float(g.ac_ugbw[0])
        if hasattr(g, 'ac_valid'):
            ac_valid[i] = bool(g.ac_valid)

    print(f'GT data extracted: 24 MOSFETs × {N} samples = {24*N} device-instances\n')

    # ============== 1. SMAXT (all-region) ==============
    n_vt = np.where(is_nmos, N_NMOS, N_PMOS) * V_T
    # softplus-smooth min/max
    sp = lambda x: np.log1p(np.exp(np.clip(x, -50, 50)))
    Vds_eff = Vds_per - n_vt * sp((Vds_per - Vov_per) / n_vt)
    denom   = n_vt + n_vt * sp((Vov_per - Vds_eff/2 - n_vt) / n_vt)
    gm_smaxt = Id_per / np.clip(denom, 1e-12, None)
    smaxt_log_err = np.abs(np.log10(np.clip(gm_smaxt, 1e-15, None)) - np.log10(np.clip(gm_per, 1e-15, None)))

    print('=== SMAXT (gm formula on GT inputs) ===')
    print(f'  All 24 MOSFETs:  logMAE = {smaxt_log_err.mean():.4f}, median = {np.median(smaxt_log_err):.4f}')
    # Per-region
    for rname, rid in [('cutoff', 0), ('triode', 1), ('saturation', 2)]:
        m = region_per == rid
        if m.any():
            e = smaxt_log_err[m]
            print(f'    {rname:<12}: N={m.sum():>6,}  logMAE={e.mean():.4f}  median={np.median(e):.4f}  med_rel={(10**np.median(e)-1)*100:.2f}%')
    # Per V_ov bucket (saturation only)
    print('  Per |V_ov| bucket (saturation):')
    mask_sat = region_per == 2
    Vov_sat = Vov_per[mask_sat]; err_sat = smaxt_log_err[mask_sat]
    for lo, hi in [(0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 1.0)]:
        m = (Vov_sat >= lo) & (Vov_sat < hi)
        if m.any():
            print(f'    {lo:.2f} ≤ Vov < {hi:.2f}: N={m.sum():>5}  logMAE={err_sat[m].mean():.4f}')

    # ============== 2. Triode eq1: gm = Id / (Vov - Vds/2) ==============
    mask_tri = region_per == 1
    if mask_tri.any():
        Vov_t = Vov_per[mask_tri]; Vds_t = Vds_per[mask_tri]
        Id_t = Id_per[mask_tri]; gm_t = gm_per[mask_tri]; gds_t = gds_per[mask_tri]
        denom_tri = np.clip(Vov_t - Vds_t/2, 1e-6, None)
        gm_eq1 = Id_t / denom_tri
        err_eq1 = np.abs(np.log10(np.clip(gm_eq1,1e-15,None)) - np.log10(np.clip(gm_t,1e-15,None)))
        print(f'\n=== TRIODE eq1: gm = Id/(Vov - Vds/2) — triode region only ===')
        print(f'  N={mask_tri.sum()}  logMAE={err_eq1.mean():.4f}  median={np.median(err_eq1):.4f}  med_rel={(10**np.median(err_eq1)-1)*100:.2f}%')
        # ============== 3. Triode eq2: gds ==============
        gds_eq2 = (Id_t / np.clip(Vds_t, 1e-6, None)) * (Vov_t - Vds_t) / np.clip(Vov_t - Vds_t/2, 1e-6, None)
        gds_eq2 = np.clip(gds_eq2, 1e-15, None)
        err_eq2 = np.abs(np.log10(gds_eq2) - np.log10(np.clip(gds_t, 1e-15, None)))
        print(f'=== TRIODE eq2: gds = (Id/Vds)(Vov-Vds)/(Vov-Vds/2) — triode region ===')
        print(f'  N={mask_tri.sum()}  logMAE={err_eq2.mean():.4f}  median={np.median(err_eq2):.4f}  med_rel={(10**np.median(err_eq2)-1)*100:.2f}%')

    # ============== 4. Cutoff: gm = Id / (n * Vt) ==============
    mask_co = region_per == 0
    if mask_co.any():
        n_vt_co = np.where(is_nmos[mask_co], N_NMOS, N_PMOS) * V_T
        gm_cutoff = Id_per[mask_co] / n_vt_co
        err_co = np.abs(np.log10(np.clip(gm_cutoff, 1e-15, None)) - np.log10(np.clip(gm_per[mask_co], 1e-15, None)))
        print(f'\n=== CUTOFF: gm = Id / (n·Vt)  with n=1.5(NMOS)/2.0(PMOS), Vt=25.85 mV ===')
        print(f'  N={mask_co.sum()}  logMAE={err_co.mean():.4f}  median={np.median(err_co):.4f}  med_rel={(10**np.median(err_co)-1)*100:.2f}%')

    # ============== 5. Mirror pairs (10 hardcoded fan_smc pairs) ==============
    print(f'\n=== MIRROR (10 hardcoded fan_smc pairs) — does I_mirror/I_ref = expected_ratio? ===')
    print(f'  pair             N      median |Δratio/expected|   mean |Δ|     max |Δ|')
    for (ri, mi_, exp) in MIRROR_PAIRS:
        I_ref = Id_per[:, ri]; I_mir = Id_per[:, mi_]
        valid = (I_ref > 1e-12)
        actual = I_mir[valid] / I_ref[valid]
        rel_err = np.abs(actual - exp) / exp
        nameA = f'M{ri}=M{mi_}' if exp == 1.0 else f'M{ri}=({exp:.0f}x)·M{mi_}' if exp > 1 else f'M{ri}={exp:.2f}·M{mi_}'
        print(f'  {nameA:<18} {valid.sum():>5}  med={np.median(rel_err)*100:>8.2f}%  mean={rel_err.mean()*100:>8.2f}%  max={rel_err.max()*100:>8.2f}%')

    # ============== 6. Diff-pair: I_M8 + I_M9 = I_M4 (tail = 4x bias) ==============
    print(f'\n=== DIFF-PAIR: I_M8 + I_M9 == I_M4 (tail) — should hold by KCL ===')
    Itail = Id_per[:, 4]
    Isum = Id_per[:, 8] + Id_per[:, 9]
    rel = np.abs(Isum - Itail) / np.clip(Itail, 1e-15, None)
    print(f'  N={N}  median rel err = {np.median(rel)*100:.3f}%  mean = {rel.mean()*100:.2f}%  max = {rel.max()*100:.2f}%')

    # ============== 7. UGBW formula ==============
    # UGBW ≈ gm_M8 / (2π · C_C) where C_C is the Miller compensation cap
    # fan_smc has a single C_C compensation cap; we extract it from the dataset config
    # Approximate: use a typical C_C ≈ 1 pF (the dataset varies it via sampling)
    print(f'\n=== UGBW: gm_M8 / (2π · C_C) — single-pole approximation ===')
    print(f'  (cannot evaluate without per-sample C_C — using typical 1 pF for order-of-magnitude check)')
    C_C = 1e-12   # placeholder; real value varies per sample
    gm_M8 = gm_per[:, 8]
    UGBW_est = gm_M8 / (2 * np.pi * C_C)
    valid = ac_valid & (UGBW_GT > 0)
    if valid.any():
        rel = np.abs(UGBW_est[valid] - UGBW_GT[valid]) / UGBW_GT[valid]
        print(f'  N_valid={valid.sum()}  median rel err = {np.median(rel)*100:.1f}%  mean = {rel.mean()*100:.1f}%')
        print(f'  (gm_M8 range: {gm_M8[valid].min():.2e} to {gm_M8[valid].max():.2e} S)')
        print(f'  (UGBW_GT range: {UGBW_GT[valid].min():.2e} to {UGBW_GT[valid].max():.2e} Hz)')
        print(f'  (formula estimate range: {UGBW_est[valid].min():.2e} to {UGBW_est[valid].max():.2e} Hz)')


if __name__ == '__main__':
    main()
