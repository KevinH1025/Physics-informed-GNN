#!/usr/bin/env python
"""Three analyses on the usingnow per-MOSFET embedding space:
  (1) Per-cluster fingerprints — operating-point profile of each role / position cluster
  (2) Symmetry check — distance within matched pairs vs unrelated pairs
  (3) Principal-direction analysis — what each PCA axis encodes

Output: figures/thesis/tsne_meaning.json + console tables.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.analysis.tsne_usingnow_embeddings import load_model_and_run

OUT_JSON = Path(__file__).resolve().parents[2] / 'figures/thesis/tsne_meaning.json'

# fan_smc matched pairs — devices that should be electrically/structurally identical
# (verify against netlist if needed; common 3-stage opamp conventions)
MATCHED_PAIRS = [
    ('M0',  'M1',  'bias_mirror_pair_1'),
    ('M2',  'M3',  'bias_mirror_pair_2'),
    ('M4',  'M5',  'bias_mirror_pair_3'),
    ('M6',  'M7',  'bias_mirror_pair_4'),
    ('M8',  'M9',  'diff_pair_input'),
    ('M10', 'M11', 'stage1_active_load'),
    ('M12', 'M13', 'stage2_pair'),
    ('M14', 'M15', 'stage2_aux_pair'),
    ('M16', 'M17', 'output_pair_1'),
    ('M18', 'M19', 'output_pair_2'),
    ('M20', 'M21', 'output_pair_3'),
    ('M22', 'M23', 'output_pair_4'),
]


def main():
    print('[1] Loading model + extracting per-MOSFET embeddings (concat g+d+s)...')
    d = load_model_and_run()
    emb = d['emb']               # [N, 447]
    midx = d['midx']             # [N], values 0..23
    is_nmos = d['is_nmos']
    region = d['region']
    role = d['role']
    vov = d['vov']; vgs = d['vgs']; vds = d['vds']; vth = d['vth']
    log_id_uA = d['log_id_uA']
    log_gm = d['log_gm']; log_gds = d['log_gds']
    n_nodes = emb.shape[0]
    print(f'  embeddings: {emb.shape}, samples × MOSFETs = {n_nodes}')

    results = {}

    # =====================================================================
    # (1) Per-cluster fingerprints
    # =====================================================================
    print('\n' + '=' * 80)
    print('(1) Per-CLUSTER FINGERPRINTS')
    print('=' * 80)
    ROLE_NAMES = ['bias_mirror', 'diff_pair', 'stage1_load', 'stage2', 'output_stage']
    REGION_NAMES = ['cutoff', 'triode', 'saturation']

    role_fp = {}
    print(f"\n  By ROLE:")
    print(f"  {'role':<14}{'N':>6}{'NMOS%':>8}{'V_GS(mV)':>10}{'V_DS(mV)':>10}{'V_ov(mV)':>10}{'V_th(mV)':>10}{'log|I_D|':>10}{'log gm':>9}{'log gds':>9}  region split")
    for r in range(5):
        m = role == r
        if not m.any(): continue
        fp = {
            'N': int(m.sum()),
            'pct_nmos': float((is_nmos[m] == 1).mean() * 100),
            'V_GS_mV_med': float(np.median(vgs[m]) * 1000),
            'V_DS_mV_med': float(np.median(vds[m]) * 1000),
            'V_ov_mV_med': float(np.median(vov[m]) * 1000),
            'V_th_mV_med': float(np.median(vth[m]) * 1000),
            'log_ID_uA_med': float(np.median(log_id_uA[m])),
            'log_gm_med': float(np.median(log_gm[m])),
            'log_gds_med': float(np.median(log_gds[m])),
            'region_dist_pct': {REGION_NAMES[i]: float((region[m] == i).mean() * 100) for i in range(3)},
        }
        role_fp[ROLE_NAMES[r]] = fp
        rd = fp['region_dist_pct']
        print(f"  {ROLE_NAMES[r]:<14}{fp['N']:>6}{fp['pct_nmos']:>7.0f}%"
              f"{fp['V_GS_mV_med']:>10.0f}{fp['V_DS_mV_med']:>10.0f}{fp['V_ov_mV_med']:>10.0f}{fp['V_th_mV_med']:>10.0f}"
              f"{fp['log_ID_uA_med']:>10.2f}{fp['log_gm_med']:>9.2f}{fp['log_gds_med']:>9.2f}"
              f"  cut={rd['cutoff']:.0f}% tri={rd['triode']:.0f}% sat={rd['saturation']:.0f}%")
    results['role_fingerprints'] = role_fp

    # Per-position fingerprint (compact)
    pos_fp = {}
    print(f"\n  By POSITION (M0..M23):")
    print(f"  {'M':<4}{'NMOS?':<7}{'role':<14}{'V_GS':>7}{'V_DS':>7}{'V_ov':>7}{'log|ID|':>9}{'log gm':>8}  region split")
    for mid in range(24):
        m = midx == mid
        if not m.any(): continue
        is_n = (is_nmos[m] == 1).mean() > 0.5
        r = int(np.bincount(role[m]).argmax())
        fp = {
            'is_nmos': bool(is_n),
            'role': ROLE_NAMES[r],
            'V_GS_mV_med': float(np.median(vgs[m]) * 1000),
            'V_DS_mV_med': float(np.median(vds[m]) * 1000),
            'V_ov_mV_med': float(np.median(vov[m]) * 1000),
            'log_ID_uA_med': float(np.median(log_id_uA[m])),
            'log_gm_med': float(np.median(log_gm[m])),
            'region_dist_pct': {REGION_NAMES[i]: float((region[m] == i).mean() * 100) for i in range(3)},
        }
        pos_fp[f'M{mid}'] = fp
        rd = fp['region_dist_pct']
        print(f"  M{mid:<3}{'NMOS' if is_n else 'PMOS':<7}{ROLE_NAMES[r]:<14}"
              f"{fp['V_GS_mV_med']:>7.0f}{fp['V_DS_mV_med']:>7.0f}{fp['V_ov_mV_med']:>7.0f}"
              f"{fp['log_ID_uA_med']:>9.2f}{fp['log_gm_med']:>8.2f}  "
              f"cut={rd['cutoff']:.0f}/tri={rd['triode']:.0f}/sat={rd['saturation']:.0f}")
    results['position_fingerprints'] = pos_fp

    # =====================================================================
    # (2) Symmetry check — matched pairs vs unrelated baseline
    # =====================================================================
    print('\n' + '=' * 80)
    print('(2) SYMMETRY CHECK — matched-pair vs unrelated-pair embedding distance')
    print('=' * 80)
    # For each MOSFET position, compute mean embedding (across all samples)
    pos_centroids = np.zeros((24, emb.shape[1]))
    pos_counts = np.zeros(24, dtype=int)
    for mid in range(24):
        m = midx == mid
        if m.any():
            pos_centroids[mid] = emb[m].mean(axis=0)
            pos_counts[mid] = m.sum()

    def euclid(a, b):
        return float(np.linalg.norm(a - b))

    matched_dists = []
    print(f"\n  Matched-pair embedding distance (centroid-to-centroid):")
    print(f"  {'Pair':<25}{'distance':>10}")
    for a, b, label in MATCHED_PAIRS:
        ai = int(a[1:]); bi = int(b[1:])
        dist = euclid(pos_centroids[ai], pos_centroids[bi])
        matched_dists.append((label, dist))
        print(f"  {a:>3}↔{b:<3}  {label:<19}{dist:>10.3f}")

    # Unrelated baseline: random pairs of MOSFETs from DIFFERENT roles
    def role_of(mid):
        return int(np.bincount(role[midx == mid]).argmax()) if (midx == mid).any() else -1
    rng = np.random.default_rng(42)
    unrelated_dists = []
    n_baseline = 100
    while len(unrelated_dists) < n_baseline:
        i, j = rng.integers(0, 24, size=2)
        if i == j: continue
        if role_of(i) == role_of(j): continue
        # Skip if either is part of a matched pair pair (so we don't double-count)
        unrelated_dists.append(euclid(pos_centroids[i], pos_centroids[j]))

    matched_arr = np.array([d for _, d in matched_dists])
    unrelated_arr = np.array(unrelated_dists)
    print(f"\n  Summary:")
    print(f"    matched-pair mean dist:    {matched_arr.mean():.3f}  (median {np.median(matched_arr):.3f})")
    print(f"    unrelated-pair mean dist:  {unrelated_arr.mean():.3f}  (median {np.median(unrelated_arr):.3f})")
    print(f"    ratio (unrelated / matched): {unrelated_arr.mean() / matched_arr.mean():.2f}×")
    print(f"    → matched pairs are {unrelated_arr.mean() / matched_arr.mean():.1f}× closer than unrelated devices")
    results['symmetry'] = {
        'matched_pairs': [{'pair': f'{a}↔{b}', 'label': l, 'dist': d} for (a, b, l), (_, d) in zip(MATCHED_PAIRS, matched_dists)],
        'matched_mean': float(matched_arr.mean()),
        'matched_median': float(np.median(matched_arr)),
        'unrelated_mean': float(unrelated_arr.mean()),
        'unrelated_median': float(np.median(unrelated_arr)),
        'ratio': float(unrelated_arr.mean() / matched_arr.mean()),
    }

    # =====================================================================
    # (3) PCA — what is each principal axis?
    # =====================================================================
    print('\n' + '=' * 80)
    print('(3) PCA — what does each principal axis encode?')
    print('=' * 80)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=5)
    Z = pca.fit_transform(emb)   # [N, 5]
    print(f"\n  Top-5 variance explained: {pca.explained_variance_ratio_}")
    print(f"  Cumulative: {pca.explained_variance_ratio_.cumsum()}")

    # Correlate each PC with categorical + continuous features
    print(f"\n  {'PC':<5}{'var %':>8}  {'top features (Pearson r)':<60}")
    pca_table = []
    feat_names = {
        'is_nmos':   is_nmos.astype(float),
        'role':      role.astype(float),
        'position':  midx.astype(float),
        'region':    region.astype(float),
        'V_GS':      vgs,
        'V_DS':      vds,
        'V_ov':      vov,
        'V_th':      vth,
        'log|I_D|':  log_id_uA,
        'log gm':    log_gm,
        'log gds':   log_gds,
    }
    for pc in range(5):
        z = Z[:, pc]
        corrs = []
        for name, vals in feat_names.items():
            r = float(np.corrcoef(z, vals)[0, 1])
            corrs.append((name, r))
        corrs.sort(key=lambda x: -abs(x[1]))
        top3 = corrs[:3]
        s = '  '.join([f'{n}={r:+.2f}' for n, r in top3])
        print(f"  PC{pc+1:<3} {pca.explained_variance_ratio_[pc] * 100:>7.1f}%  {s}")
        pca_table.append({
            'pc': pc + 1,
            'variance_explained': float(pca.explained_variance_ratio_[pc]),
            'top_correlations': [{'feature': n, 'r': r} for n, r in corrs[:5]],
        })
    results['pca'] = pca_table

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved → {OUT_JSON}')


if __name__ == '__main__':
    main()
