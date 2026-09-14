#!/usr/bin/env python
"""Add 7-dim per-MOSFET functional role one-hot to a prebatched dataset.

For openloop / fan_smc topology, MOSFET roles are:
  0: BIASCM_P  (M0-M7  — PMOS bias current mirrors)
  1: BIASCM_N  (M12-M20 — NMOS bias / cascode)
  2: GM1       (M8-M9  — input differential pair)
  3: GM2       (M10    — stage-2 PMOS gain device)
  4: GMF2      (M11    — stage-3 PMOS output)
  5: LOAD2     (M21-M22 — stage-2 NMOS load)
  6: GM3       (M23    — stage-3 NMOS output)

Writes `mosfet_role` [M, 7] per graph. Use mosfet_device_names to map device
indices to roles.

Usage:
    python scripts/patch_dataset_mosfet_role.py --dataset-dir datasets/opamp_3stage_fan_smc_openloop_5k
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch


# Device → role mapping for the fan_smc / openloop 24-MOSFET topology
ROLE_NAMES = ['BIASCM_P', 'BIASCM_N', 'GM1', 'GM2', 'GMF2', 'LOAD2', 'GM3']
ROLE_TO_IDX = {name: i for i, name in enumerate(ROLE_NAMES)}
N_ROLES = len(ROLE_NAMES)

DEVICE_TO_ROLE = {
    # PMOS bias mirrors
    **{f'M{i}': 'BIASCM_P' for i in range(0, 8)},
    # Input diff pair
    'M8': 'GM1', 'M9': 'GM1',
    # Stage 2 PMOS gain
    'M10': 'GM2',
    # Stage 3 PMOS output
    'M11': 'GMF2',
    # NMOS bias / cascode
    **{f'M{i}': 'BIASCM_N' for i in range(12, 21)},
    # Stage 2 NMOS load
    'M21': 'LOAD2', 'M22': 'LOAD2',
    # Stage 3 NMOS output
    'M23': 'GM3',
}


def device_id(name: str) -> str:
    """Match the convention used in netlist parsing: 'Xm0' -> 'M0'."""
    return name.upper().replace('X', '')


def patch_split(split_dir: Path):
    variants = sorted(split_dir.glob('variant_*.pkl'))
    print(f'  {split_dir.name}: {len(variants)} variants')
    for variant_path in variants:
        with open(variant_path, 'rb') as f:
            batches = pickle.load(f)
        for batch in batches:
            mi = batch.mosfet_info
            names = batch.mosfet_device_names  # may be flat list OR list of per-graph lists
            # Flatten if nested per-graph
            if names and isinstance(names[0], (list, tuple)):
                flat_names = [n for graph_names in names for n in graph_names]
            else:
                flat_names = list(names) if names is not None else None
            if flat_names is None or len(flat_names) != mi.shape[0]:
                raise RuntimeError(
                    f'{variant_path}: mosfet_device_names len={len(flat_names) if flat_names else 0} '
                    f'!= mosfet_info rows={mi.shape[0]}')
            roles = np.zeros((mi.shape[0], N_ROLES), dtype=np.float32)
            for i, dn in enumerate(flat_names):
                did = device_id(dn)
                role_name = DEVICE_TO_ROLE.get(did)
                if role_name is None:
                    raise KeyError(f'No role mapping for device {dn!r} (id={did})')
                roles[i, ROLE_TO_IDX[role_name]] = 1.0
            batch.mosfet_role = torch.from_numpy(roles)
        with open(variant_path, 'wb') as f:
            pickle.dump(batches, f)
    print(f'    patched {len(variants)} variants')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-dir', required=True)
    args = p.parse_args()

    dataset_dir = Path(args.dataset_dir)
    for split in ('train', 'val'):
        sd = dataset_dir / split
        if sd.exists():
            patch_split(sd)


if __name__ == '__main__':
    main()
