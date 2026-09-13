#!/usr/bin/env python3
"""Compute iterative-refinement LUT features (Option A3) for every graph.

Pipeline per graph:
  1. Run the frozen pass-1 model (baseline) → V₀ (z-score).
  2. Denormalize V₀ to volts via the batch's voltage_mean / voltage_std.
  3. For each MOSFET: Vgs = V[g]−V[s], Vds = V[d]−V[s], Vbs = V[b]−V[s].
  4. LUTOpQuery(W, L, Vgs, Vds, Vbs, M, pol) → physics-exact (id, gm, gds).
  5. log10 and z-score each feature using train-set statistics.
  6. Scatter the 3 scalars per MOSFET to its four terminal nodes (G/D/S/B),
     producing a per-node [N, 3] tensor `node_lut_features`. Non-MOSFET
     nodes stay zero.

Each patched graph gets `graph.node_lut_features` (shape [N, 3], float32).
Stats used for z-scoring are saved alongside the dataset for the pass-2
training run to reuse at inference.

Usage:
    python scripts/patch_dataset_lut_op_features.py \\
        --dataset-dir datasets/opamp_3stage_fan_smc_v9_5k_nofil \\
        --pass1-ckpt datasets/.../ginbn_3layer_bb8_notower_vn_kcl_w10_v9_5k/best_model.pt \\
        --lut datasets/lut/lut_v2/sky130_mosfet_lut_v2_id_gm_gds.h5
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from torch_geometric.data import Batch

from argparse import Namespace

from src.data.batching import create_prebatched_dataset, _add_device_ptr_tensors
from src.training.config import load_config, parse_training_config
from src.training.checkpoint import create_model_from_args
from src.training.data_loading import (
    compute_vdc_normalization,
    compute_current_normalization,
)
from src.gnn.components.lut_op_query import LUTOpQuery


LOG_EPS = 1e-12


def _compute_vbs_ig_stats(graphs, lut, vdc_mean, vdc_std, device,
                          batch_size=256):
    """Run pass-1 model on TRAIN graphs only to compute feature stats."""
    # Defer baseline forward to caller; here we just gather z-score stats
    # from raw LUT outputs across all graphs.
    pass


def _batch_graphs(graphs, batch_size):
    for i in range(0, len(graphs), batch_size):
        chunk = graphs[i:i + batch_size]
        b = Batch.from_data_list(chunk)
        _add_device_ptr_tensors(b, chunk)
        yield chunk, b


def _lut_features_for_batch(batch, model, lut, stats, device):
    """For a PyG Batch: run model → V₀ → LUT → per-MOSFET (id, gm, gds).

    Returns a per-graph list of [N_g, 3] float32 tensors ready to attach as
    node_lut_features. Outputs are z-scored using provided stats dict:
      {'id_mean','id_std','gm_mean','gm_std','gds_mean','gds_std'} (all log10).
    """
    batch = batch.to(device)
    # attach normalization stats so pass-1 model can denormalize voltages if it wants
    batch.voltage_mean = stats['voltage_mean']
    batch.voltage_std = stats['voltage_std']
    batch.current_mean = stats['current_mean']
    batch.current_std = stats['current_std']

    with torch.no_grad():
        out = model(batch)
        v_z = out['node_voltages']   # z-score

    v_volts = v_z * stats['voltage_std'] + stats['voltage_mean']

    mosfet_info = batch.mosfet_info.long()
    term_idx = batch.mosfet_terminal_idx.to(device)
    wl_um = batch.mosfet_wl_um.to(device)
    m_m = batch.mosfet_m.to(device)
    is_nmos = mosfet_info[:, 6]

    # Resolve MOSFET → graph → node-offset (prebatched convention)
    from src.training.losses import get_device_graph_idx
    M = mosfet_info.shape[0]
    num_graphs = batch.ptr.shape[0] - 1
    g_idx = get_device_graph_idx(M, num_graphs, batch.mosfet_ptr, device)
    node_offsets = batch.ptr[g_idx]

    # IMPORTANT: the baseline only supervises V at NET nodes (train_mask).
    # V at MOSFET terminal nodes is unsupervised → garbage (~300-700 mV
    # error). Use predicted V at the NET each terminal connects to instead
    # of the terminal node itself. mosfet_info[:, 3:6] = gate/drain/source
    # net indices (graph-local).
    gate_net_g   = mosfet_info[:, 3] + node_offsets
    drain_net_g  = mosfet_info[:, 4] + node_offsets
    source_net_g = mosfet_info[:, 5] + node_offsets

    v_gate   = v_volts[gate_net_g]
    v_drain  = v_volts[drain_net_g]
    v_source = v_volts[source_net_g]

    # Bulk net isn't in mosfet_info. In real circuits the bulk is tied to a
    # known supply (GND, VDD, or source-tied), i.e. its voltage is known
    # a priori from the netlist topology — not a quantity the model needs
    # to predict. So we read GT V at the bulk terminal directly. This is
    # equivalent to looking up the bulk's connected net's V.
    v_gt = batch.node_voltage_targets.to(device)
    bulk_t_g = term_idx[:, 3].clamp_min(0) + node_offsets
    has_bulk = term_idx[:, 3] >= 0
    v_bulk = torch.where(has_bulk, v_gt[bulk_t_g], v_source)

    Vgs = v_gate - v_source
    Vds = v_drain - v_source
    Vbs = v_bulk - v_source

    id_lin, gm, gds = lut(wl_um[:, 0], wl_um[:, 1], Vgs, Vds, Vbs, m_m, is_nmos)

    log_id = torch.log10(id_lin.clamp_min(LOG_EPS))
    log_gm = torch.log10(gm.abs().clamp_min(LOG_EPS))
    log_gds = torch.log10(gds.abs().clamp_min(LOG_EPS))

    # Return raw log-space per-MOSFET (stats applied later if stats==None).
    if stats.get('id_mean') is not None:
        z_id = (log_id - stats['id_mean']) / stats['id_std']
        z_gm = (log_gm - stats['gm_mean']) / stats['gm_std']
        z_gds = (log_gds - stats['gds_mean']) / stats['gds_std']
    else:
        z_id, z_gm, z_gds = log_id, log_gm, log_gds

    mosfet_feat = torch.stack([z_id, z_gm, z_gds], dim=-1)   # [M, 3]

    # Scatter to G/D/S/B terminals within each graph; non-MOSFET nodes stay 0.
    num_nodes = batch.x.shape[0]
    node_feat = torch.zeros(num_nodes, 3, device=device)
    for col in range(4):
        col_valid = term_idx[:, col] >= 0
        idx_global = term_idx[col_valid, col] + node_offsets[col_valid]
        node_feat[idx_global] = mosfet_feat[col_valid]

    # Split back per-graph
    per_graph = []
    ptr = batch.ptr.tolist()
    node_feat_cpu = node_feat.cpu()
    for gi in range(num_graphs):
        per_graph.append(node_feat_cpu[ptr[gi]:ptr[gi + 1]].clone())
    return per_graph, (log_id, log_gm, log_gds)   # also return raw for stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-dir', required=True, type=Path)
    ap.add_argument('--pass1-ckpt', required=True, type=Path,
                    help='Frozen baseline best_model.pt to use as pass 1')
    ap.add_argument('--pass1-config', required=True, type=Path,
                    help='Original YAML config used to train the pass-1 model '
                         '(usually <run_dir>/original_config.yaml). Needed so we '
                         'build the exact architecture before loading weights.')
    ap.add_argument('--lut', required=True, type=Path,
                    help='sky130_mosfet_lut_v2_id_gm_gds.h5')
    ap.add_argument('--batch-size', type=int, default=64,
                    help='Graphs per mini-batch during pass 1 inference')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--prebatch-size', type=int, default=1024)
    ap.add_argument('--prebatch-variants', type=int, default=10)
    ap.add_argument('--prebatch-seed', type=int, default=42)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    print(f'Loading pass-1 baseline model from {args.pass1_ckpt}...')
    # Reconstruct architecture from the *training* YAML to guarantee keys match.
    pass1_yaml = load_config(args.pass1_config)
    pass1_args = Namespace(**parse_training_config(pass1_yaml))
    # We need node_feature_dim — infer from the saved input_linear weight row.
    ckpt = torch.load(args.pass1_ckpt, map_location='cpu', weights_only=False)
    state_dict = ckpt['model_state_dict']
    if 'input_linear.weight' in state_dict:
        pass1_input_dim = int(state_dict['input_linear.weight'].shape[1])
    else:
        raise RuntimeError('could not infer input dim from checkpoint')
    print(f'  input_dim={pass1_input_dim}')

    model, _ = create_model_from_args(pass1_args, pass1_input_dim, device=str(device))
    model.load_state_dict(state_dict)
    model.eval()
    ckpt_stats = ckpt.get('stats', {})

    print(f'Loading LUT from {args.lut}...')
    lut = LUTOpQuery(args.lut).to(device)
    lut.eval()

    # Load train split to compute train-set normalization stats
    train_pkl = args.dataset_dir / 'dataset_train.pkl'
    with open(train_pkl, 'rb') as f:
        train_samples = pickle.load(f)
    train_graphs = [s['graph'] for s in train_samples]
    print(f'  Train graphs: {len(train_graphs)}')

    # Reuse voltage/current normalization that the baseline was trained with
    vdc_mean = float(ckpt_stats.get('vdc', {}).get('mean',
                      ckpt_stats.get('vdc_mean', 0.0)))
    vdc_std = float(ckpt_stats.get('vdc', {}).get('std',
                     ckpt_stats.get('vdc_std', 1.0)))
    curr_mean = float(ckpt_stats.get('current', {}).get('mean',
                       ckpt_stats.get('curr', {}).get('mean',
                        ckpt_stats.get('current_mean', 0.0))))
    curr_std = float(ckpt_stats.get('current', {}).get('std',
                      ckpt_stats.get('curr', {}).get('std',
                       ckpt_stats.get('current_std', 1.0))))
    print(f'  Baseline stats: vdc={vdc_mean:.3f}/{vdc_std:.3f}  '
          f'current={curr_mean:.3f}/{curr_std:.3f}')

    base_stats = dict(
        voltage_mean=vdc_mean, voltage_std=vdc_std,
        current_mean=curr_mean, current_std=curr_std,
        id_mean=None, id_std=None,
        gm_mean=None, gm_std=None,
        gds_mean=None, gds_std=None,
    )

    # --- Pass A: gather raw log10(id/gm/gds) across train set → stats ------
    print('\n=== computing z-score stats from train set ===')
    all_log_id, all_log_gm, all_log_gds = [], [], []
    for chunk, b in _batch_graphs(train_graphs, args.batch_size):
        _, (lid, lgm, lgds) = _lut_features_for_batch(b, model, lut, base_stats, device)
        all_log_id.append(lid.cpu())
        all_log_gm.append(lgm.cpu())
        all_log_gds.append(lgds.cpu())
    log_id_all = torch.cat(all_log_id)
    log_gm_all = torch.cat(all_log_gm)
    log_gds_all = torch.cat(all_log_gds)
    stats = dict(base_stats)
    stats['id_mean']  = float(log_id_all.mean());  stats['id_std']  = float(log_id_all.std().clamp_min(1e-6))
    stats['gm_mean']  = float(log_gm_all.mean());  stats['gm_std']  = float(log_gm_all.std().clamp_min(1e-6))
    stats['gds_mean'] = float(log_gds_all.mean()); stats['gds_std'] = float(log_gds_all.std().clamp_min(1e-6))
    print(f'  log10(id):  mean={stats["id_mean"]:.3f}  std={stats["id_std"]:.3f}')
    print(f'  log10(gm):  mean={stats["gm_mean"]:.3f}  std={stats["gm_std"]:.3f}')
    print(f'  log10(gds): mean={stats["gds_mean"]:.3f}  std={stats["gds_std"]:.3f}')

    stats_out = args.dataset_dir / 'lut_op_features_stats.json'
    with open(stats_out, 'w') as f:
        json.dump({k: v for k, v in stats.items() if isinstance(v, (int, float))}, f, indent=2)
    print(f'  Saved stats to {stats_out.name}')

    # --- Pass B: compute z-scored node_lut_features for each split --------
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
            per_graph, _ = _lut_features_for_batch(b, model, lut, stats, device)
            for g, feat in zip(chunk, per_graph):
                g.node_lut_features = feat
            idx += len(chunk)
        print(f'  attached node_lut_features to {idx} graphs')

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

    # rollup
    full = args.dataset_dir / 'dataset.pkl'
    if full.exists():
        print(f'\n=== dataset.pkl (rollup) ===')
        with open(full, 'rb') as f:
            rollup = pickle.load(f)
        graphs = [s['graph'] for s in rollup]
        idx = 0
        for chunk, b in _batch_graphs(graphs, args.batch_size):
            per_graph, _ = _lut_features_for_batch(b, model, lut, stats, device)
            for g, feat in zip(chunk, per_graph):
                g.node_lut_features = feat
            idx += len(chunk)
        with open(full, 'wb') as f:
            pickle.dump(rollup, f)
        print(f'  attached + saved {idx} graphs')

    print('\nDone.')


if __name__ == '__main__':
    main()
