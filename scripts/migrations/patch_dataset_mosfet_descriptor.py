#!/usr/bin/env python
"""Add a 5-dim per-MOSFET physical descriptor to a prebatched dataset.

For each MOSFET (W, L, polarity), query the SKY130 LUT at canonical bias points:
  feat[0] = log10(Id)  at (Vgs=0.7, Vds=0.9, Vbs=0)   strong-inversion drive current
  feat[1] = log10(gm)  at (Vgs=0.7, Vds=0.9, Vbs=0)   saturation transconductance
  feat[2] = log10(gds) at (Vgs=0.7, Vds=0.9, Vbs=0)   output conductance
  feat[3] = Vth        at (Vbs=0)                      threshold (estimated from Id-Vgs sweep)
  feat[4] = log10(gm)  at (Vgs=0.5, Vds=0.5, Vbs=0)   weak-inversion / subthreshold signature

Writes `mosfet_descriptor` [M, 5] per graph, in the prebatched DataBatch format.

Usage:
    python scripts/migrations/patch_dataset_mosfet_descriptor.py --dataset-dir datasets/opamp_3stage_fan_smc_openloop_5k
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator


LUT_PATH = 'datasets/lut/lut_v2/sky130_mosfet_lut_v2_id_gm_gds.h5'
LOG_FLOOR = 1e-15


def build_lut_interpolators(lut_path: str):
    """Returns dicts of {'n':..., 'p':...} interpolators for id, gm, gds.
    All interpolators take (W_um, L_um, |Vbs|, Vgs, Vds)."""
    with h5py.File(lut_path, 'r') as h5:
        W_um = h5['W_um'][:]
        L_um = h5['L_um'][:]
        Vbs = np.abs(h5['n/Vbs_V'][:])
        # Sort Vbs ascending
        order = np.argsort(Vbs)
        Vbs = Vbs[order]
        Vgs = h5['Vgs_V'][:]
        Vds = h5['Vds_V'][:]
        interp = {}
        for pol in ('n', 'p'):
            id_arr = h5[f'{pol}/id'][:]
            gm_arr = h5[f'{pol}/gm'][:]
            gds_arr = h5[f'{pol}/gds'][:]
            # Reorder Vbs axis (axis 2)
            id_arr = id_arr[:, :, order]
            gm_arr = gm_arr[:, :, order]
            gds_arr = gds_arr[:, :, order]
            interp[pol] = {
                'id': RegularGridInterpolator((W_um, L_um, Vbs, Vds, Vgs), id_arr,
                                                method='linear', bounds_error=False,
                                                fill_value=None),
                'gm': RegularGridInterpolator((W_um, L_um, Vbs, Vds, Vgs), gm_arr,
                                                method='linear', bounds_error=False,
                                                fill_value=None),
                'gds': RegularGridInterpolator((W_um, L_um, Vbs, Vds, Vgs), gds_arr,
                                                method='linear', bounds_error=False,
                                                fill_value=None),
            }
        return interp, W_um, L_um, Vbs, Vds, Vgs


def estimate_vth(interp_id_pol, W_um: float, L_um: float, Vbs_mag: float = 0.0,
                 Vds_eval: float = 0.05) -> float:
    """Estimate Vth via constant-current method: Vth where Id = (W/L) * Iref.
    Iref = 1e-7 A is a common choice. Sweep Vgs from 0..1.8."""
    Iref = (W_um / max(L_um, 1e-3)) * 1e-7  # SKY130-typical scaling
    Vgs_sweep = np.linspace(0.0, 1.8, 181)
    pts = np.column_stack([
        np.full_like(Vgs_sweep, W_um),
        np.full_like(Vgs_sweep, L_um),
        np.full_like(Vgs_sweep, Vbs_mag),
        np.full_like(Vgs_sweep, Vds_eval),
        Vgs_sweep,
    ])
    ids = interp_id_pol(pts)
    ids = np.maximum(ids, LOG_FLOOR)
    # Find first Vgs where Id >= Iref
    above = ids >= Iref
    if not above.any():
        return 0.6  # reasonable fallback
    return float(Vgs_sweep[np.argmax(above)])


def compute_descriptor(W_um, L_um, is_nmos, interp):
    """Compute 12-dim descriptor for one MOSFET covering multiple operating regimes.

    Strong saturation (Vgs=0.7, Vds=0.9, Vbs=0): main op-amp operating regime
      [0] log10(Id), [1] log10(gm), [2] log10(gds)
    Weak inversion (Vgs=0.5, Vds=0.5, Vbs=0): subthreshold/efficiency regime
      [3] log10(Id), [4] log10(gm), [5] log10(gds)
    Linear region (Vgs=0.7, Vds=0.3, Vbs=0): triode
      [6] log10(Id), [7] log10(gds)
    Strong overdrive (Vgs=1.0, Vds=0.9, Vbs=0): high-current regime
      [8] log10(Id), [9] log10(gm)
    Body effect (Vgs=0.7, Vds=0.9, Vbs=-0.4): captures Vbs-dependence
      [10] log10(gm)
    Static:
      [11] Vth at Vbs=0
    """
    pol = 'n' if is_nmos else 'p'

    # Strong saturation
    pt_sat = np.array([[W_um, L_um, 0.0, 0.9, 0.7]])
    id_sat = float(interp[pol]['id'](pt_sat))
    gm_sat = float(interp[pol]['gm'](pt_sat))
    gds_sat = float(interp[pol]['gds'](pt_sat))

    # Weak inversion
    pt_wi = np.array([[W_um, L_um, 0.0, 0.5, 0.5]])
    id_wi = float(interp[pol]['id'](pt_wi))
    gm_wi = float(interp[pol]['gm'](pt_wi))
    gds_wi = float(interp[pol]['gds'](pt_wi))

    # Linear region
    pt_lin = np.array([[W_um, L_um, 0.0, 0.3, 0.7]])
    id_lin = float(interp[pol]['id'](pt_lin))
    gds_lin = float(interp[pol]['gds'](pt_lin))

    # Strong overdrive
    pt_so = np.array([[W_um, L_um, 0.0, 0.9, 1.0]])
    id_so = float(interp[pol]['id'](pt_so))
    gm_so = float(interp[pol]['gm'](pt_so))

    # Body effect
    pt_be = np.array([[W_um, L_um, 0.4, 0.9, 0.7]])
    gm_be = float(interp[pol]['gm'](pt_be))

    # Vth
    vth = estimate_vth(interp[pol]['id'], W_um, L_um, Vbs_mag=0.0)

    def lg(x): return np.log10(max(abs(x), LOG_FLOOR))

    return np.array([
        lg(id_sat), lg(gm_sat), lg(gds_sat),         # 0,1,2: strong sat
        lg(id_wi), lg(gm_wi), lg(gds_wi),            # 3,4,5: weak inv
        lg(id_lin), lg(gds_lin),                     # 6,7: linear
        lg(id_so), lg(gm_so),                        # 8,9: strong overdrive
        lg(gm_be),                                   # 10: body effect
        vth,                                         # 11: threshold
    ], dtype=np.float32)


def patch_split(split_dir: Path, interp, norm_stats=None):
    """Patch all variant_*.pkl files in a split directory.

    If norm_stats is None, returns (mean, std) computed from this split for
    z-score normalization. If norm_stats is provided, uses those (for val to
    use train's stats).
    """
    variants = sorted(split_dir.glob('variant_*.pkl'))
    print(f'  {split_dir.name}: {len(variants)} variants')
    cache = {}  # (W, L, polarity) -> 5-dim descriptor

    # Pass 1: compute raw descriptors and stats if not provided
    all_descriptors = []
    for variant_path in variants:
        with open(variant_path, 'rb') as f:
            batches = pickle.load(f)
        for batch in batches:
            mi = batch.mosfet_info
            wl_um = batch.mosfet_wl_um if hasattr(batch, 'mosfet_wl_um') else None
            if wl_um is None:
                raise RuntimeError(f'{variant_path}: mosfet_wl_um not in batch')
            M_total = mi.shape[0]
            descriptors = np.zeros((M_total, 12), dtype=np.float32)
            for i in range(M_total):
                W = round(float(wl_um[i, 0].item()), 4)
                L = round(float(wl_um[i, 1].item()), 4)
                pol = int(mi[i, 6].item())
                key = (W, L, pol)
                if key not in cache:
                    cache[key] = compute_descriptor(W, L, pol == 1, interp)
                descriptors[i] = cache[key]
            all_descriptors.append(descriptors)

    if norm_stats is None:
        all_d = np.concatenate(all_descriptors, axis=0)  # [N_total, 5]
        mean = all_d.mean(axis=0)
        std = all_d.std(axis=0).clip(min=1e-6)
        print(f'    descriptor stats: mean={mean.tolist()}, std={std.tolist()}')
    else:
        mean, std = norm_stats

    # Pass 2: write normalized descriptors back
    di = 0
    for variant_path in variants:
        with open(variant_path, 'rb') as f:
            batches = pickle.load(f)
        for batch in batches:
            d = (all_descriptors[di] - mean) / std
            batch.mosfet_descriptor = torch.from_numpy(d.astype(np.float32))
            di += 1
        with open(variant_path, 'wb') as f:
            pickle.dump(batches, f)
    print(f'    patched {len(variants)} variants, descriptor cache size: {len(cache)}')
    return (mean, std)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-dir', required=True)
    p.add_argument('--lut-path', default=LUT_PATH)
    args = p.parse_args()

    print(f'Building LUT interpolators from {args.lut_path}...')
    interp, *_ = build_lut_interpolators(args.lut_path)
    print('Done.')

    dataset_dir = Path(args.dataset_dir)
    train_dir = dataset_dir / 'train'
    val_dir = dataset_dir / 'val'
    norm_stats = None
    if train_dir.exists():
        norm_stats = patch_split(train_dir, interp, norm_stats=None)
    if val_dir.exists():
        # Use train stats for val (don't compute val-only stats)
        patch_split(val_dir, interp, norm_stats=norm_stats)


if __name__ == '__main__':
    main()
