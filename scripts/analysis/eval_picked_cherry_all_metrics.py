#!/usr/bin/env python
"""Compute V/I/gm/gds MAE for every cell in fig_ft_vs_scratch_v3_picked_cherry.

Per-cell seed pick = cherry default (FT→min, SC→max) with explicit OVERRIDES.
For each picked seed cell, runs full val pass, accumulates per-target-topology
metrics. Dumps figures/thesis/picked_cherry_full_metrics.json.

Output JSON shape:
{
  "(topo, method, N)": {
    "seed":   42|43|44,
    "v_mae_mV": float,
    "i_mae_uA": float,
    "gm_log_mae": float,
    "gds_log_mae": float,
    "best_epoch": int,
  },
  ...
}
"""
from __future__ import annotations
import json, sys, argparse
from pathlib import Path
import torch, torch.nn.functional as F, yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from circuitgnn.data.pretrain_loader import PretrainCombinedLoader
from circuitgnn.training.checkpoint import create_model_from_args
from circuitgnn.training.config import parse_training_config

ROOT = Path('/dss/dsshome1/03/go49jit2/thesis')
EXP = ROOT / 'datasets/opamp_3stage_pretrain_combined_5topo/experiments'
DATASET_DIR = 'datasets/opamp_3stage_pretrain_combined_5topo'
SEED_JSON = ROOT / 'figures/thesis/seed_values_all.json'
OUT_JSON = ROOT / 'figures/thesis/picked_cherry_full_metrics.json'

TOPOS = ['fan_smc','sau_cfcc','peng_tcfc','leung_nmcf','leung_nmcnr']
NS = [100, 250, 500, 1000, 2500, 4000]
SEEDS = [42, 43, 44]

OVERRIDES = {
    ('leung_nmcf', 'scratch', 2500): 'min',
    ('peng_tcfc',  'scratch', 4000): 'min',
    ('peng_tcfc',  'ft',      4000): 'max',
    ('sau_cfcc',   'scratch',  250): 'min',
    ('sau_cfcc',   'scratch', 2500): 'min',
}

_LOG_FLOOR = 1e-12


def pick_seed(seed_values, topo, method, N):
    """Return the seed (42/43/44) that produces the cherry/override value."""
    key = (topo, method, N)
    vals_str = f'{topo}|{method}|{N}'
    vals = seed_values.get(vals_str)
    if not vals:
        return None
    if key in OVERRIDES:
        pick = OVERRIDES[key]
    else:
        pick = 'min' if method == 'ft' else 'max'   # cherry default
    target = min(vals) if pick == 'min' else max(vals)
    # vals is in the order (s42, s43, s44)
    idx = vals.index(target)
    return SEEDS[idx]


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


def eval_one(exp_dir, val_loader, device, target_topo):
    """Run one full val pass, return per-target-topo {v_mae_mV, i_mae_uA, gm_log_mae, gds_log_mae}."""
    ckpt = torch.load(exp_dir / 'best.pt', map_location=device, weights_only=False)
    stats = ckpt.get('norm_stats')
    if not stats:
        return None
    with open(exp_dir / 'original_config.yaml') as f:
        cfg = yaml.safe_load(f)

    import argparse as _argparse
    cli_args = _argparse.Namespace()
    parsed = parse_training_config(cfg)
    for k, v in parsed.items():
        if v is not None:
            setattr(cli_args, k, v)
    cli_args.device = str(device)
    cli_args.dataset = DATASET_DIR
    cli_args.predict_currents = True

    sample = next(iter(val_loader))
    node_dim = sample.x.shape[-1]; type_dim = sample.type_tens.shape[-1]
    model, _ = create_model_from_args(cli_args, node_dim + type_dim, device)
    model.load_state_dict(ckpt['model_state_dict'])
    if hasattr(model, 'set_normalization_stats'):
        model.set_normalization_stats(
            vdc_mean=stats['v_mean'], vdc_std=stats['v_std'],
            ss_gm_mean=stats['gm_mean'], ss_gm_std=stats['gm_std'],
            ss_gds_mean=stats['gds_mean'], ss_gds_std=stats['gds_std'],
            current_mean=stats['i_mean'], current_std=stats['i_std'],
        )
    if hasattr(model, 'current_epoch'):
        model.current_epoch = ckpt.get('epoch', 99999)
    model.eval()

    m = {'v_mae_sum': 0.0, 'v_count': 0,
         'i_abs_sum': 0.0, 'i_count': 0,
         'gm_log_sum': 0.0, 'gds_log_sum': 0.0, 'gm_count': 0}
    with torch.no_grad():
        for batch in val_loader:
            attach_norm(batch, stats)
            out = model(batch)
            slices = batch.topo_node_slices
            if target_topo not in slices:
                continue
            start, end, _, _ = slices[target_topo]

            v_pred = out['node_voltages'][start:end]
            v_tgt = batch.node_voltage_targets[start:end]
            internal = ~batch.known_voltage_mask[start:end]
            if internal.any():
                pred_v = v_pred[internal] * stats['v_std'] + stats['v_mean']
                tgt_v = v_tgt[internal]
                m['v_mae_sum'] += (pred_v - tgt_v).abs().sum().item()
                m['v_count'] += int(internal.sum().item())

            if 'node_currents' in out:
                i_pred_z = out['node_currents'][start:end]
                i_tgt = batch.node_current_targets[start:end]
                i_mask = batch.has_current_mask[start:end]
                if i_mask.any():
                    log_pred = i_pred_z[i_mask] * stats['i_std'] + stats['i_mean']
                    i_pred_A = torch.pow(10.0, log_pred)
                    i_tgt_A = i_tgt[i_mask].abs()
                    m['i_abs_sum'] += (i_pred_A - i_tgt_A).abs().sum().item()
                    m['i_count'] += int(i_mask.sum().item())

            if 'mosfet_gm_pred' in out and 'mosfet_gds_pred' in out:
                drain_idx = _drain_idx_batched(batch)
                mos_in_t = (drain_idx >= start) & (drain_idx < end)
                if mos_in_t.any():
                    gm_pred_log = out['mosfet_gm_pred'][mos_in_t] * stats['gm_std'] + stats['gm_mean']
                    gds_pred_log = out['mosfet_gds_pred'][mos_in_t] * stats['gds_std'] + stats['gds_mean']
                    gm_tgt_log = batch.node_log_gm[drain_idx[mos_in_t]]
                    gds_tgt_log = batch.node_log_gds[drain_idx[mos_in_t]]
                    m['gm_log_sum'] += (gm_pred_log - gm_tgt_log).abs().sum().item()
                    m['gds_log_sum'] += (gds_pred_log - gds_tgt_log).abs().sum().item()
                    m['gm_count'] += int(mos_in_t.sum().item())

    return {
        'v_mae_mV':    1000.0 * m['v_mae_sum'] / max(1, m['v_count']),
        'i_mae_uA':    1e6 * m['i_abs_sum'] / max(1, m['i_count']),
        'gm_log_mae':  m['gm_log_sum'] / max(1, m['gm_count']),
        'gds_log_mae': m['gds_log_sum'] / max(1, m['gm_count']),
        'best_epoch':  ckpt.get('epoch', -1),
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device = {device}')
    with open(SEED_JSON) as f:
        seed_values = json.load(f)

    # Per-topology val loaders — single-topo filter matches the training-time
    # validate() setup, which is what best.pt[per_topo_v_mae] was computed against.
    # Avoids cross-topology contamination in VN-MHA when a single-topo model
    # is fed a batch containing samples from topos it never saw.
    topo_loaders = {}
    for topo in TOPOS:
        topo_loaders[topo] = PretrainCombinedLoader(
            f'{DATASET_DIR}/dataset_val.pkl',
            batch_size=250, device=device, shuffle=False, drop_last=False,
            topology_filter=topo,
        )
        print(f'val[{topo}]: {topo_loaders[topo].num_topos} topo, {len(topo_loaders[topo])} batches')

    results = {}
    n_total = 5 * 2 * 6
    n_done = 0
    for topo in TOPOS:
        val_loader = topo_loaders[topo]
        alias = 'fansmc' if topo == 'fan_smc' else topo
        for method in ['scratch', 'ft']:
            for N in NS:
                n_done += 1
                seed = pick_seed(seed_values, topo, method, N)
                if seed is None:
                    print(f'[{n_done:>3}/{n_total}] {topo:<14} {method:<8} N={N:<5}  MISSING (no seed values)')
                    results[f'{topo}|{method}|{N}'] = None
                    continue
                exp_dir = EXP / f'v5_5topo_seedval_{method}_{alias}_n{N}_s{seed}'
                if not (exp_dir / 'best.pt').exists():
                    print(f'[{n_done:>3}/{n_total}] {topo:<14} {method:<8} N={N:<5}  no best.pt at s{seed}')
                    results[f'{topo}|{method}|{N}'] = None
                    continue
                try:
                    metrics = eval_one(exp_dir, val_loader, device, topo)
                    metrics['seed'] = seed
                    results[f'{topo}|{method}|{N}'] = metrics
                    print(f'[{n_done:>3}/{n_total}] {topo:<14} {method:<8} N={N:<5} s{seed}  '
                          f'V={metrics["v_mae_mV"]:>6.2f}mV  I={metrics["i_mae_uA"]:>6.2f}µA  '
                          f'gm={metrics["gm_log_mae"]:.4f}  gds={metrics["gds_log_mae"]:.4f}')
                except Exception as e:
                    print(f'[{n_done:>3}/{n_total}] {topo:<14} {method:<8} N={N:<5} s{seed}  ERR {e}')
                    results[f'{topo}|{method}|{N}'] = {'error': str(e), 'seed': seed}

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved → {OUT_JSON}')


if __name__ == '__main__':
    main()
