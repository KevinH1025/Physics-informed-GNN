#!/usr/bin/env python
"""Evaluate a v5 checkpoint per-topology, breaking out V/I/gm/gds metrics.

Loads a saved best.pt, runs full val pass on the 5-topology corpus, and
prints per-topology breakdown of:
  - V MAE (mV)
  - V loss (z-score MSE)
  - I MAE (µA)
  - I loss (log10 z-score MSE)
  - gm MAE (log10)
  - gds MAE (log10)
  - V@5%, I@5% relative accuracy
  - KCL violation magnitude

Usage:
    python scripts/eval_per_topo.py <experiment_name_or_path>

Example:
    python scripts/eval_per_topo.py v5_5topo_joint
    python scripts/eval_per_topo.py datasets/.../experiments/v5_5topo_zeroshot_fansmc
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from circuitgnn.data.pretrain_loader import PretrainCombinedLoader
from circuitgnn.training.checkpoint import create_model_from_args
from circuitgnn.training.config import parse_training_config


_LOG_FLOOR = 1e-12


def attach_norm(batch, stats):
    device = batch.x.device
    for name, val in [
        ('vdc_mean', stats['v_mean']), ('vdc_std', stats['v_std']),
        ('current_mean', stats['i_mean']), ('current_std', stats['i_std']),
        ('ss_gm_mean', stats['gm_mean']), ('ss_gm_std', stats['gm_std']),
        ('ss_gds_mean', stats['gds_mean']), ('ss_gds_std', stats['gds_std']),
    ]:
        setattr(batch, name, torch.tensor(val, device=device))


def _drain_idx_batched(batch):
    mi = batch.mosfet_info
    mp = batch.mosfet_ptr
    ptr = batch.ptr
    num_mos = mi.shape[0]
    graph_idx = torch.bucketize(
        torch.arange(num_mos, device=mi.device),
        mp[1:].to(mi.device), right=True,
    )
    offsets = ptr[graph_idx]
    return mi[:, 1].long() + offsets


def compute_ac_norm(dataset_dir, ac_components, device, batch_size=512):
    """Scan training set once, compute (ac_mean, ac_std) for the configured components."""
    if not ac_components:
        return None, None
    ac_raw = {c: [] for c in ac_components}
    loader = PretrainCombinedLoader(
        f'{dataset_dir}/dataset_train.pkl',
        batch_size=batch_size, device=device, shuffle=False, drop_last=False,
    )
    for b in loader:
        if not (hasattr(b, 'ac_valid') and hasattr(b, 'ac_ugbw')):
            continue
        valid = b.ac_valid
        if not valid.any():
            continue
        if 'ugbw' in ac_raw:
            ac_raw['ugbw'].extend(torch.log10(b.ac_ugbw[valid].clamp(min=1.0)).cpu().tolist())
        if 'pm' in ac_raw:
            ac_raw['pm'].extend(b.ac_pm[valid].cpu().tolist())
        if 'am' in ac_raw and hasattr(b, 'ac_am'):
            ac_raw['am'].extend(b.ac_am[valid].cpu().tolist())
    import numpy as _np
    if not ac_raw[ac_components[0]]:
        return None, None
    ac_mean = torch.tensor([_np.mean(ac_raw[c]) for c in ac_components], dtype=torch.float32)
    ac_std = torch.tensor([_np.std(ac_raw[c]) for c in ac_components], dtype=torch.float32).clamp(min=1e-6)
    return ac_mean, ac_std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('exp', help='Experiment name or full path to experiment dir')
    ap.add_argument('--dataset-dir', default='datasets/opamp_3stage_pretrain_combined_5topo')
    ap.add_argument('--batch-size', type=int, default=250)
    ap.add_argument('--topology', type=str, default=None,
                    help='Restrict val to this single topology (matches per-topo training setup)')
    ap.add_argument('--auto-topology', action='store_true', default=True,
                    help='Auto-detect from experiment name: pertopo_<topo> filters to that topo')
    ap.add_argument('--json-out', type=str, default=None,
                    help='Optional path to dump per-topo metrics as JSON')
    args = ap.parse_args()

    # Resolve experiment dir
    p = Path(args.exp)
    if not p.exists():
        p = Path(args.dataset_dir) / 'experiments' / args.exp
    if not p.exists():
        raise FileNotFoundError(f'Experiment not found: {args.exp}')
    ckpt_path = p / 'best.pt'
    cfg_path = p / 'original_config.yaml'
    if not ckpt_path.exists() or not cfg_path.exists():
        raise FileNotFoundError(f'Missing best.pt or original_config.yaml in {p}')

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    print(f'Evaluating: {p.name}')
    print(f'  ckpt: {ckpt_path}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    stats = ckpt.get('norm_stats') or ckpt.get('config', {}).get('norm_stats') or {}
    if not stats:
        # Try recomputing or fall back
        print('  WARN: no norm_stats in checkpoint, using defaults')
        stats = {'v_mean': 0.9, 'v_std': 0.5, 'i_mean': -5.55, 'i_std': 1.42,
                 'gm_mean': -4.43, 'gm_std': 1.44, 'gds_mean': -5.89, 'gds_std': 1.89}
    print(f'  norm_stats: V({stats["v_mean"]:.3f}, {stats["v_std"]:.3f}) '
          f'I({stats["i_mean"]:.3f}, {stats["i_std"]:.3f})')

    # Auto-detect single-topology from experiment name (e.g. "pertopo_fan_smc",
    # "pertopo_fansmc_n100", etc. — match the topo word in the name).
    topo_filter = args.topology
    if topo_filter is None and args.auto_topology:
        name = p.name.lower()
        # Map name fragments to topo names
        for cand in ('sau_cfcc', 'peng_tcfc', 'leung_nmcf', 'leung_nmcnr', 'fan_smc', 'fansmc'):
            if f'pertopo_{cand}' in name or f'pertopo_{cand}_' in name:
                topo_filter = 'fan_smc' if cand == 'fansmc' else cand
                break
        if topo_filter:
            print(f'  auto-detected single-topology eval: {topo_filter}')

    # Choose batch_size compatible with #topos in val
    val_batch = args.batch_size
    if topo_filter is not None:
        # Single topo — any batch_size works
        val_batch = min(args.batch_size, 256)

    val_loader = PretrainCombinedLoader(
        f'{args.dataset_dir}/dataset_val.pkl',
        batch_size=val_batch, device=device, shuffle=False, drop_last=False,
        topology_filter=topo_filter,
    )
    print(f'  val loader: {val_loader.num_topos} topos, {len(val_loader)} batches, batch_size={val_batch}')

    # Build model
    import argparse as _argparse
    cli_args = _argparse.Namespace()
    parsed = parse_training_config(cfg)
    for k, v in parsed.items():
        if v is not None:
            setattr(cli_args, k, v)
    cli_args.device = str(device)
    cli_args.dataset = args.dataset_dir
    cli_args.predict_currents = True
    sample = next(iter(val_loader))
    node_dim = sample.x.shape[-1]
    type_dim = sample.type_tens.shape[-1]
    model, _ = create_model_from_args(cli_args, node_dim + type_dim, device)
    model.load_state_dict(ckpt['model_state_dict'])
    if hasattr(model, 'set_normalization_stats'):
        model.set_normalization_stats(
            vdc_mean=stats['v_mean'], vdc_std=stats['v_std'],
            ss_gm_mean=stats['gm_mean'], ss_gm_std=stats['gm_std'],
            ss_gds_mean=stats['gds_mean'], ss_gds_std=stats['gds_std'],
            current_mean=stats['i_mean'], current_std=stats['i_std'],
        )
    # Critical: set current_epoch so loop attention / warmup-gated heads are
    # in their post-warmup state (matching training behavior at the saved epoch).
    saved_epoch = ckpt.get('epoch', 99999)
    if hasattr(model, 'current_epoch'):
        model.current_epoch = saved_epoch
        print(f'  set model.current_epoch = {saved_epoch} (post-warmup state)')
    model.eval()

    # Detect AC head and compute normalization stats
    ac_components = list(getattr(model, 'ac_components', []) or [])
    ac_mean = ac_std = None
    if ac_components and getattr(model, 'predict_ac', False):
        print(f'  AC head detected with components: {ac_components} — computing norm stats...')
        ac_mean, ac_std = compute_ac_norm(args.dataset_dir, ac_components, device)
        if ac_mean is None:
            print('  WARN: no valid AC samples in training set, skipping AC metrics')
            ac_components = []
        else:
            ac_mean = ac_mean.to(device); ac_std = ac_std.to(device)
            print(f'  AC norm stats: mean={ac_mean.tolist()}, std={ac_std.tolist()}')

    # Per-topo accumulators
    metrics = {t: {'v_mae_sum': 0.0, 'v_count': 0, 'v_loss_sum': 0.0, 'v_loss_n': 0,
                   'i_abs_sum': 0.0, 'i_count': 0, 'i_loss_sum': 0.0, 'i_loss_n': 0,
                   'i_rel_5': 0, 'i_rel_n': 0,
                   'v_rel_1': 0, 'v_rel_5': 0, 'v_rel_10': 0, 'v_rel_n': 0,
                   'gm_log_sum': 0.0, 'gds_log_sum': 0.0, 'gm_count': 0,
                   'gm_loss_sum': 0.0, 'gds_loss_sum': 0.0, 'gm_loss_n': 0,
                   'ugbw_log_err_sum': 0.0, 'ugbw_count': 0,
                   'pm_err_sum': 0.0, 'pm_count': 0,
                   'am_err_sum': 0.0, 'am_count': 0,
                   } for t in val_loader.topo_names}

    with torch.no_grad():
        for batch in val_loader:
            attach_norm(batch, stats)
            out = model(batch)
            slices = batch.topo_node_slices
            # Per-topology AC metric accumulation: AC is graph-level (one value per design)
            # Need to know which graphs in the batch belong to which topology
            ac_pred_b = out.get('ac_pred') if ac_components else None
            if ac_pred_b is not None and hasattr(batch, 'ac_valid'):
                _ac_valid_b = batch.ac_valid
                if hasattr(batch, 'ac_dc_gain'):
                    _ac_valid_b = _ac_valid_b & (batch.ac_dc_gain > 0)
                # batch.batch maps each node → graph index. Need topology per graph.
                # topo_node_slices gives (start_node, end_node, n_graphs, graph_offset)
                topo_per_graph = {}
                for t, (sn, en, ng, go) in slices.items():
                    for gi in range(go, go + ng):
                        topo_per_graph[gi] = t
                if _ac_valid_b.any():
                    for gi in range(batch.num_graphs):
                        if not bool(_ac_valid_b[gi]): continue
                        t = topo_per_graph.get(gi)
                        if t is None: continue
                        m = metrics[t]
                        pred = ac_pred_b[gi]  # [n_components], normalized
                        # denorm and compare per component
                        for ci, comp in enumerate(ac_components):
                            pred_d = (pred[ci] * ac_std[ci] + ac_mean[ci]).item()
                            if comp == 'ugbw':
                                tgt_log = float(torch.log10(batch.ac_ugbw[gi].clamp(min=1.0)).item())
                                m['ugbw_log_err_sum'] += abs(pred_d - tgt_log)
                                m['ugbw_count'] += 1
                            elif comp == 'pm':
                                tgt = float(batch.ac_pm[gi].item())
                                m['pm_err_sum'] += abs(pred_d - tgt)
                                m['pm_count'] += 1
                            elif comp == 'am':
                                tgt = float(batch.ac_am[gi].item())
                                m['am_err_sum'] += abs(pred_d - tgt)
                                m['am_count'] += 1
            for t, (start, end, _n, _g) in slices.items():
                m = metrics[t]
                # V
                v_pred = out['node_voltages'][start:end]
                v_tgt = batch.node_voltage_targets[start:end]
                internal = ~batch.known_voltage_mask[start:end]
                if internal.any():
                    pred_v = v_pred[internal] * stats['v_std'] + stats['v_mean']
                    tgt_v = v_tgt[internal]
                    err = (pred_v - tgt_v).abs()
                    m['v_mae_sum'] += err.sum().item()
                    m['v_count'] += int(internal.sum().item())
                    pred_z = v_pred[internal]
                    tgt_z = (tgt_v - stats['v_mean']) / stats['v_std']
                    m['v_loss_sum'] += F.mse_loss(pred_z, tgt_z, reduction='sum').item()
                    m['v_loss_n'] += int(internal.sum().item())
                    # Relative accuracy bins (vs |tgt|+small)
                    rel = err / (tgt_v.abs() + 1e-6)
                    m['v_rel_1'] += int((rel < 0.01).sum().item())
                    m['v_rel_5'] += int((rel < 0.05).sum().item())
                    m['v_rel_10'] += int((rel < 0.10).sum().item())
                    m['v_rel_n'] += int(internal.sum().item())
                # I
                if 'node_currents' in out:
                    i_pred_z = out['node_currents'][start:end]
                    i_tgt = batch.node_current_targets[start:end]
                    i_mask = batch.has_current_mask[start:end]
                    if i_mask.any():
                        # Denorm pred to log10 then to A
                        log_pred = i_pred_z[i_mask] * stats['i_std'] + stats['i_mean']
                        i_pred_A = torch.pow(10.0, log_pred)
                        i_tgt_A = i_tgt[i_mask].abs()
                        err_A = (i_pred_A - i_tgt_A).abs()
                        m['i_abs_sum'] += err_A.sum().item()
                        m['i_count'] += int(i_mask.sum().item())
                        log_tgt = i_tgt_A.clamp_min(_LOG_FLOOR).log10()
                        log_tgt_z = (log_tgt - stats['i_mean']) / stats['i_std']
                        m['i_loss_sum'] += F.mse_loss(i_pred_z[i_mask], log_tgt_z, reduction='sum').item()
                        m['i_loss_n'] += int(i_mask.sum().item())
                        # Rel acc: above 1nA floor
                        floor = i_tgt_A.abs() > 1e-9
                        if floor.any():
                            rel = err_A[floor] / i_tgt_A[floor]
                            m['i_rel_5'] += int((rel < 0.05).sum().item())
                            m['i_rel_n'] += int(floor.sum().item())
                # gm/gds (per-MOSFET predictions)
                if 'mosfet_gm_pred' in out and 'mosfet_gds_pred' in out:
                    drain_idx = _drain_idx_batched(batch)
                    # Per-MOSFET — slice by drain belonging to this topo's range
                    mos_in_t = (drain_idx >= start) & (drain_idx < end)
                    if mos_in_t.any():
                        gm_pred = out['mosfet_gm_pred'][mos_in_t]
                        gds_pred = out['mosfet_gds_pred'][mos_in_t]
                        # Targets via node_log_gm at drain
                        gm_tgt_z = (batch.node_log_gm[drain_idx[mos_in_t]] - stats['gm_mean']) / stats['gm_std']
                        gds_tgt_z = (batch.node_log_gds[drain_idx[mos_in_t]] - stats['gds_mean']) / stats['gds_std']
                        m['gm_loss_sum'] += F.mse_loss(gm_pred, gm_tgt_z, reduction='sum').item()
                        m['gds_loss_sum'] += F.mse_loss(gds_pred, gds_tgt_z, reduction='sum').item()
                        m['gm_loss_n'] += int(mos_in_t.sum().item())
                        # Log10-domain MAE
                        gm_pred_log = gm_pred * stats['gm_std'] + stats['gm_mean']
                        gds_pred_log = gds_pred * stats['gds_std'] + stats['gds_mean']
                        gm_tgt_log = batch.node_log_gm[drain_idx[mos_in_t]]
                        gds_tgt_log = batch.node_log_gds[drain_idx[mos_in_t]]
                        m['gm_log_sum'] += (gm_pred_log - gm_tgt_log).abs().sum().item()
                        m['gds_log_sum'] += (gds_pred_log - gds_tgt_log).abs().sum().item()
                        m['gm_count'] += int(mos_in_t.sum().item())

    # Print per-topo table
    print()
    print(f'{"Topology":<14} {"V MAE":>9} {"V@5%":>7} {"I MAE":>9} {"I@5%":>7} {"gm MAE":>9} {"gds MAE":>9} {"UGBW dec":>9} {"PM deg":>8} {"AM dB":>8}')
    print('-' * 110)
    out_metrics = {}
    for t in val_loader.topo_names:
        m = metrics[t]
        v_mae = 1000.0 * m['v_mae_sum'] / max(1, m['v_count'])  # mV
        v_5 = 100.0 * m['v_rel_5'] / max(1, m['v_rel_n'])
        i_mae = 1e6 * m['i_abs_sum'] / max(1, m['i_count'])  # µA
        i_5 = 100.0 * m['i_rel_5'] / max(1, m['i_rel_n']) if m['i_rel_n'] > 0 else 0
        gm_mae = m['gm_log_sum'] / max(1, m['gm_count'])
        gds_mae = m['gds_log_sum'] / max(1, m['gm_count'])
        ugbw_mae = m['ugbw_log_err_sum'] / m['ugbw_count'] if m['ugbw_count'] else float('nan')
        pm_mae = m['pm_err_sum'] / m['pm_count'] if m['pm_count'] else float('nan')
        am_mae = m['am_err_sum'] / m['am_count'] if m['am_count'] else float('nan')
        ugbw_str = f'{ugbw_mae:>8.3f}' if m['ugbw_count'] else '       -'
        pm_str = f'{pm_mae:>7.2f}' if m['pm_count'] else '      -'
        am_str = f'{am_mae:>7.2f}' if m['am_count'] else '      -'
        print(f'{t:<14} {v_mae:>7.2f}mV {v_5:>6.1f}% {i_mae:>7.2f}µA {i_5:>6.1f}% '
              f'{gm_mae:>9.4f} {gds_mae:>9.4f} {ugbw_str} {pm_str} {am_str}')
        out_metrics[t] = {
            'v_mae_mV': v_mae, 'v_at_5pct': v_5,
            'i_mae_uA': i_mae, 'i_at_5pct': i_5,
            'gm_log_mae': gm_mae, 'gds_log_mae': gds_mae,
            'ugbw_log_mae_dec': ugbw_mae if m['ugbw_count'] else None,
            'pm_mae_deg': pm_mae if m['pm_count'] else None,
            'am_mae_db': am_mae if m['am_count'] else None,
            'n_ac_valid': max(m['ugbw_count'], m['pm_count'], m['am_count']),
        }

    if args.json_out:
        import json
        with open(args.json_out, 'w') as f:
            json.dump({'experiment': p.name, 'metrics': out_metrics}, f, indent=2)
        print(f'\nSaved metrics to {args.json_out}')


if __name__ == '__main__':
    main()
