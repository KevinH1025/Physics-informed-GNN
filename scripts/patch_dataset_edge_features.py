#!/usr/bin/env python
"""Augment edge_attr with structural circuit features (no SPICE needed).

Existing edge_attr is a 6-dim one-hot of the source-side terminal type
(gate/drain/source/bulk/p/n). This patch concatenates 4 more dims:

  6: loop-edge flag           — destination edge is in `loop_edge_index`
  7: boundary flag            — destination node has known voltage (VDD/GND/V-src)
  8: output flag              — destination node is an output node
  9: cap-coupled flag         — destination node is a capacitor terminal (AC-coupled)

These come from existing masks in the batch — no parsing of node names.
Run AFTER the dataset is prebatched. Idempotent: only appends if edge_attr
is exactly 6-dim.

Usage:
    python scripts/patch_dataset_edge_features.py --dataset-dir datasets/opamp_3stage_fan_smc_openloop_5k
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import torch


_BASE_DIM = 6
_NEW_DIM = 4
_TARGET_DIM = _BASE_DIM + _NEW_DIM


def _augment_edges(batch) -> None:
    edge_index = batch.edge_index
    if edge_index.numel() == 0:
        return
    ea = getattr(batch, 'edge_attr', None)
    if ea is None:
        raise RuntimeError('edge_attr missing — cannot augment')
    if ea.shape[1] == _TARGET_DIM:
        return  # already patched
    if ea.shape[1] != _BASE_DIM:
        raise RuntimeError(
            f'edge_attr has {ea.shape[1]} dims, expected {_BASE_DIM}')

    num_edges = edge_index.shape[1]
    dst = edge_index[1].long()  # [E]

    # 1) Loop-edge flag: edges present in loop_edge_index (as a (src,dst) set)
    loop_flag = torch.zeros(num_edges, dtype=torch.float32)
    le = getattr(batch, 'loop_edge_index', None)
    if le is not None and le.numel() > 0:
        # Encode (src, dst) → unique int64 key, then sort+searchsorted to test
        # membership without allocating an O(N*N) flag table.
        src = edge_index[0].long()
        N = max(int(edge_index.max().item()), int(le.max().item())) + 1
        key_main = src * N + dst
        key_loop = le[0].long() * N + le[1].long()
        sorted_loop, _ = torch.sort(key_loop)
        idx = torch.searchsorted(sorted_loop, key_main)
        idx_clamped = idx.clamp(max=sorted_loop.numel() - 1)
        loop_flag = (sorted_loop[idx_clamped] == key_main).float()

    # 2) Boundary flag: destination has known_voltage_mask
    bound_flag = torch.zeros(num_edges, dtype=torch.float32)
    kv = getattr(batch, 'known_voltage_mask', None)
    if kv is not None:
        bound_flag = kv[dst].float()

    # 3) Output flag: destination is in output_node_mask
    out_flag = torch.zeros(num_edges, dtype=torch.float32)
    om = getattr(batch, 'output_node_mask', None)
    if om is not None:
        out_flag = om[dst].float()

    # 4) Cap-coupled flag: destination is a capacitor terminal (cols 0,1 of
    # capacitor_info are p/n terminal node indices in batched space)
    cap_flag = torch.zeros(num_edges, dtype=torch.float32)
    ci = getattr(batch, 'capacitor_info', None)
    if ci is not None and ci.numel() > 0:
        cap_terms = torch.cat([ci[:, 0], ci[:, 1]]).long()
        cap_set = torch.zeros(int(edge_index.max().item()) + 1, dtype=torch.bool)
        cap_set[cap_terms] = True
        cap_flag = cap_set[dst].float()

    extra = torch.stack([loop_flag, bound_flag, out_flag, cap_flag], dim=1)
    batch.edge_attr = torch.cat([ea, extra], dim=1)


def patch_split(split_dir: Path):
    variants = sorted(split_dir.glob('variant_*.pkl'))
    print(f'  {split_dir.name}: {len(variants)} variants')
    for variant_path in variants:
        with open(variant_path, 'rb') as f:
            batches = pickle.load(f)
        n_patched = 0
        for batch in batches:
            before = batch.edge_attr.shape[1] if hasattr(batch, 'edge_attr') and batch.edge_attr is not None else None
            _augment_edges(batch)
            after = batch.edge_attr.shape[1]
            if before != after:
                n_patched += 1
        with open(variant_path, 'wb') as f:
            pickle.dump(batches, f)
        print(f'    {variant_path.name}: patched {n_patched}/{len(batches)} batches '
              f'(edge_attr now {after}-dim)')


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
