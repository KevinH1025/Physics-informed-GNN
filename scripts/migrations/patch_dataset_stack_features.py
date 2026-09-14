#!/usr/bin/env python3
"""Add baseline's prediction outputs as per-node features for pass-2 training.

Stacking approach (Option A3 alternative): instead of querying the LUT, just
forward-stack the baseline's own predictions back as input features for a
fresh pass-2 model.

Per graph:
  1. Run frozen baseline → V₀ (node), I₀ (node), gm₀ (per MOSFET), gds₀ (per MOSFET).
  2. For MOSFET terminal nodes, replace V₀ with V₀ at the connected NET (because
     the baseline doesn't supervise V at terminal nodes — only at net nodes).
  3. Scatter gm₀, gds₀ to each MOSFET's four terminal nodes.
  4. Save `node_stack_features` [N, 4] = (V, I, gm, gds), all in z-score space.

Each patched graph gets `graph.node_stack_features` (shape [N, 4], float32).

Usage:
    python scripts/migrations/patch_dataset_stack_features.py \\
        --dataset-dir datasets/opamp_3stage_fan_smc_v9_5k_nofil \\
        --pass1-ckpt .../best_model.pt \\
        --pass1-config .../original_config.yaml
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from argparse import Namespace
from torch_geometric.data import Batch

from circuitgnn.data.batching import create_prebatched_dataset, _add_device_ptr_tensors
from circuitgnn.training.config import load_config, parse_training_config
from circuitgnn.training.checkpoint import create_model_from_args


def _batch_graphs(graphs, batch_size):
    for i in range(0, len(graphs), batch_size):
        chunk = graphs[i:i + batch_size]
        b = Batch.from_data_list(chunk)
        _add_device_ptr_tensors(b, chunk)
        yield chunk, b


def _stack_features_for_batch(batch, model, stats, device):
    """Forward baseline → return per-graph [N_g, 4] node_stack_features.

    Output columns: [V₀, I₀, gm₀, gds₀], all in the model's z-score space.
    """
    batch = batch.to(device)
    batch.voltage_mean = stats['voltage_mean']
    batch.voltage_std = stats['voltage_std']
    batch.current_mean = stats['current_mean']
    batch.current_std = stats['current_std']

    with torch.no_grad():
        out = model(batch)

    v_pred = out['node_voltages']        # [N], z-score
    i_pred = out.get('node_currents')    # [N], z-score (or None if not predicted)
    if i_pred is None:
        i_pred = torch.zeros_like(v_pred)
    gm_pred = out.get('mosfet_gm_pred')  # [M], z-score
    gds_pred = out.get('mosfet_gds_pred')

    num_nodes = v_pred.shape[0]
    feat = torch.zeros(num_nodes, 4, device=device, dtype=v_pred.dtype)

    # ── Column 0: V₀ ─────────────────────────────────────────────────────
    # Default = baseline V at this node.
    feat[:, 0] = v_pred
    # Override MOSFET terminal nodes to use V at the connected NET (net V is
    # supervised ~10 mV; terminal V is unsupervised ~300+ mV garbage).
    mosfet_info = batch.mosfet_info.long()
    term_idx = batch.mosfet_terminal_idx.to(device)
    M = mosfet_info.shape[0]

    from circuitgnn.training.losses import get_device_graph_idx
    num_graphs = batch.ptr.shape[0] - 1
    g_idx = get_device_graph_idx(M, num_graphs, batch.mosfet_ptr, device)
    node_offsets = batch.ptr[g_idx]

    gate_term_g   = term_idx[:, 0] + node_offsets
    drain_term_g  = term_idx[:, 1] + node_offsets
    source_term_g = term_idx[:, 2] + node_offsets
    bulk_term_g   = term_idx[:, 3].clamp_min(0) + node_offsets
    has_bulk      = term_idx[:, 3] >= 0

    gate_net_g    = mosfet_info[:, 3] + node_offsets
    drain_net_g   = mosfet_info[:, 4] + node_offsets
    source_net_g  = mosfet_info[:, 5] + node_offsets

    feat[gate_term_g,   0] = v_pred[gate_net_g]
    feat[drain_term_g,  0] = v_pred[drain_net_g]
    feat[source_term_g, 0] = v_pred[source_net_g]
    # For bulk: use GT V (bulk net V is determined by netlist topology and
    # known a priori from supplies). Read from GT to avoid reading garbage
    # baseline output at the unsupervised terminal.
    v_gt = batch.node_voltage_targets.to(device)
    v_gt_z = (v_gt - stats['voltage_mean']) / stats['voltage_std']  # match z-space
    feat[bulk_term_g[has_bulk], 0] = v_gt_z[bulk_term_g[has_bulk]]

    # ── Column 1: I₀ at this node (z-score, raw from current head) ──────
    feat[:, 1] = i_pred

    # ── Columns 2 & 3: gm₀, gds₀ scattered to all 4 MOSFET terminals ────
    if gm_pred is not None and gds_pred is not None and M > 0:
        for col in range(4):
            valid = term_idx[:, col] >= 0
            idx_g = term_idx[valid, col] + node_offsets[valid]
            feat[idx_g, 2] = gm_pred[valid]
            feat[idx_g, 3] = gds_pred[valid]

    # Split per-graph and return
    per_graph = []
    ptr = batch.ptr.tolist()
    feat_cpu = feat.cpu()
    for gi in range(num_graphs):
        per_graph.append(feat_cpu[ptr[gi]:ptr[gi + 1]].clone())
    return per_graph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-dir', required=True, type=Path)
    ap.add_argument('--pass1-ckpt', required=True, type=Path)
    ap.add_argument('--pass1-config', required=True, type=Path,
                    help='Original training YAML for the pass-1 baseline.')
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--prebatch-size', type=int, default=1024)
    ap.add_argument('--prebatch-variants', type=int, default=10)
    ap.add_argument('--prebatch-seed', type=int, default=42)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    print(f'Loading pass-1 baseline from {args.pass1_ckpt}...')
    pass1_yaml = load_config(args.pass1_config)
    pass1_args = Namespace(**parse_training_config(pass1_yaml))
    ckpt = torch.load(args.pass1_ckpt, map_location='cpu', weights_only=False)
    state_dict = ckpt['model_state_dict']
    if 'input_linear.weight' not in state_dict:
        raise RuntimeError('checkpoint missing input_linear.weight')
    pass1_input_dim = int(state_dict['input_linear.weight'].shape[1])

    model, _ = create_model_from_args(pass1_args, pass1_input_dim, device=str(device))
    model.load_state_dict(state_dict)
    model.eval()
    ckpt_stats = ckpt.get('stats', {})

    vdc_mean = float(ckpt_stats.get('vdc', {}).get('mean', 0.0))
    vdc_std = float(ckpt_stats.get('vdc', {}).get('std', 1.0))
    curr_mean = float(ckpt_stats.get('current', {}).get('mean',
                       ckpt_stats.get('curr', {}).get('mean', 0.0)))
    curr_std = float(ckpt_stats.get('current', {}).get('std',
                      ckpt_stats.get('curr', {}).get('std', 1.0)))
    print(f'  Baseline stats: vdc={vdc_mean:.3f}/{vdc_std:.3f}  '
          f'current={curr_mean:.3f}/{curr_std:.3f}')

    stats = dict(voltage_mean=vdc_mean, voltage_std=vdc_std,
                 current_mean=curr_mean, current_std=curr_std)

    for split in ('train', 'val', 'test'):
        pkl = args.dataset_dir / f'dataset_{split}.pkl'
        if not pkl.exists():
            continue
        print(f'\n=== {split} ===')
        with open(pkl, 'rb') as f:
            samples = pickle.load(f)
        graphs = [s['graph'] for s in samples]
        print(f'  {len(graphs)} graphs')

        idx = 0
        for chunk, b in _batch_graphs(graphs, args.batch_size):
            per_graph = _stack_features_for_batch(b, model, stats, device)
            for g, feat in zip(chunk, per_graph):
                g.node_stack_features = feat
            idx += len(chunk)
        print(f'  attached node_stack_features [N, 4] to {idx} graphs')

        with open(pkl, 'wb') as f:
            pickle.dump(samples, f)
        print(f'  re-saved {pkl.name}')

        split_dir = args.dataset_dir / split
        split_dir.mkdir(exist_ok=True)
        create_prebatched_dataset(
            samples=samples,
            output_dir=split_dir,
            batch_size=args.prebatch_size,
            num_variants=args.prebatch_variants,
            seed=args.prebatch_seed,
            is_validation=(split != 'train'),
        )

    full = args.dataset_dir / 'dataset.pkl'
    if full.exists():
        print(f'\n=== dataset.pkl rollup ===')
        with open(full, 'rb') as f:
            rollup = pickle.load(f)
        graphs = [s['graph'] for s in rollup]
        idx = 0
        for chunk, b in _batch_graphs(graphs, args.batch_size):
            per_graph = _stack_features_for_batch(b, model, stats, device)
            for g, feat in zip(chunk, per_graph):
                g.node_stack_features = feat
            idx += len(chunk)
        with open(full, 'wb') as f:
            pickle.dump(rollup, f)
        print(f'  attached + saved {idx} graphs')

    print('\nDone.')


if __name__ == '__main__':
    main()
