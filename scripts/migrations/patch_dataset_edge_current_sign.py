#!/usr/bin/env python
"""Append per-edge current direction sign to edge_attr.

Each MOSFET terminal has a sign in `terminal_current_sign`:
  +1 = drain (current flows OUT of terminal into net)
  -1 = source (current flows INTO terminal from net)
   0 = gate / bulk / cap terminals (no DC current)

For each edge, we use the source-side terminal's sign. The reverse direction
edge gets the negated sign (current flows the other way). For non-physical
edges (terminal-to-terminal or net-to-net, neither is real), sign is 0.

Conventional layout: append at position 10 (after the 4 flags from
patch_dataset_edge_features.py). Final edge_attr shape: [E, 11].
  0-5: terminal type one-hot (gate/drain/source/bulk/p/n)
  6:   loop_flag
  7:   bound_flag
  8:   out_flag
  9:   cap_flag
  10:  current_sign  ← NEW

Usage:
    python scripts/patch_dataset_edge_current_sign.py --dataset-dir datasets/opamp_3stage_fan_smc_openloop_5k
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import torch


_BASE_DIM = 10  # expected pre-patch (after edge_features patch)
_TARGET_DIM = 11


def _augment(batch) -> None:
    ea = getattr(batch, 'edge_attr', None)
    if ea is None:
        raise RuntimeError('edge_attr missing — patch the structural features first')
    if ea.shape[1] == _TARGET_DIM:
        return  # already patched
    if ea.shape[1] != _BASE_DIM:
        raise RuntimeError(
            f'edge_attr has {ea.shape[1]} dims, expected {_BASE_DIM}')

    edge_index = batch.edge_index  # [2, E]
    sign = batch.terminal_current_sign  # [N]

    src = edge_index[0].long()
    dst = edge_index[1].long()
    # Convention: edge sign = sign at source endpoint (current flowing OUT of
    # source). For terminal→net, that's terminal_current_sign[terminal].
    # For net→terminal (the reverse edge), it's the NEGATED sign of the
    # destination terminal.
    src_sign = sign[src]
    dst_sign = sign[dst]
    # If src is a terminal (sign != 0), use src_sign. If dst is terminal but
    # src is a net, use -dst_sign. Otherwise 0.
    edge_sign = torch.where(src_sign != 0, src_sign, -dst_sign)

    new = torch.cat([ea, edge_sign.unsqueeze(1)], dim=1)
    batch.edge_attr = new


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
        ea_dim = batches[0].edge_attr.shape[1]
        print(f'    {variant_path.name}: edge_attr now {ea_dim}-dim')


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
