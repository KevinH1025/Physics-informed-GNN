#!/usr/bin/env python
"""Add per-node net_role one-hot (5 dims) to the prebatched dataset.

Classifies each NET node into one of 5 roles based on its name:
  0: VDD       — supply rail (vdd*, vcc*)
  1: GND       — ground (gnd*, vss, '0')
  2: SIG_IN    — input signal (vin*, vp, vn, vinp, vinn, vsig, inp, inn)
  3: SIG_OUT   — output signal (vout*)
  4: INTERNAL  — anything else (internal nodes, bias nets, intermediate)

Terminal nodes (MOSFET drain/gate/source/bulk, R/C/V/I terminals) get all-zero
since they don't have a "net role" — they have a "terminal role" instead
which is captured elsewhere.

Stored as `batch.net_role` shape [N, 5] per batch.

Usage:
    python scripts/patch_dataset_net_role.py --dataset-dir datasets/opamp_3stage_fan_smc_openloop_5k
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import torch


N_ROLES = 5
ROLE_VDD, ROLE_GND, ROLE_SIG_IN, ROLE_SIG_OUT, ROLE_INTERNAL = 0, 1, 2, 3, 4


def _classify(net_name: str) -> int:
    n = net_name.lower()
    if n in ('0', 'gnd', 'vss') or 'gnda' in n or n.startswith('gnd'):
        return ROLE_GND
    if 'vdd' in n or 'vcc' in n:
        return ROLE_VDD
    if 'vin' in n or n in ('vp', 'vn', 'vsig', 'inp', 'inn'):
        return ROLE_SIG_IN
    if 'vout' in n:
        return ROLE_SIG_OUT
    return ROLE_INTERNAL


def _is_net_node(node_type) -> bool:
    """node_types entries are tuples; nets are ('VNode', ...). Terminals start
    with ('M', ..) or ('V', ..) etc. — anything starting with 'VNode'."""
    if not node_type:
        return False
    return node_type[0] == 'VNode'


def _augment(batch) -> None:
    if hasattr(batch, 'net_role') and batch.net_role is not None:
        if batch.net_role.shape[1] == N_ROLES:
            return  # already patched
    node_names = batch.node_names      # list-of-lists, one per graph
    node_types = batch.node_types      # list-of-lists, one per graph
    n_total = batch.x.shape[0]
    role = torch.zeros((n_total, N_ROLES), dtype=torch.float32)
    # Each graph's nodes are contiguous in the batched tensor (PyG convention).
    # ptr gives the boundaries.
    ptr = batch.ptr.tolist()
    for gi, (gnames, gtypes) in enumerate(zip(node_names, node_types)):
        offset = ptr[gi]
        for li, (name, ntype) in enumerate(zip(gnames, gtypes)):
            if not _is_net_node(ntype):
                continue
            r = _classify(name)
            role[offset + li, r] = 1.0
    batch.net_role = role


def patch_split(split_dir: Path):
    variants = sorted(split_dir.glob('variant_*.pkl'))
    print(f'  {split_dir.name}: {len(variants)} variants')
    for variant_path in variants:
        with open(variant_path, 'rb') as f:
            batches = pickle.load(f)
        for batch in batches:
            _augment(batch)
        with open(variant_path, 'wb') as f:
            pickle.dump(batches, f)
        # Sanity: print role counts on first batch
        nr = batches[0].net_role
        print(f'    {variant_path.name}: net_role {tuple(nr.shape)}, '
              f'sums={nr.sum(0).tolist()}')


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
