"""Shared checkpoint-evaluation helpers.

These functions used to live at the top of
``scripts/analysis/eval_all_checkpoints.py``
and were imported by eight sibling scripts through bare module-name imports
(``from eval_all_checkpoints import attach_norm``), which only worked when the
scripts directory happened to be on ``sys.path``. They are kept verbatim here
so every caller — and the original script — share one definition.
"""
from __future__ import annotations

from pathlib import Path

import torch
import yaml

from circuitgnn.data.pretrain_loader import PretrainCombinedLoader
from circuitgnn.training.checkpoint import create_model_from_args
from circuitgnn.training.config import parse_training_config

from .paths import DATASET

TOPOS = ['fan_smc', 'sau_cfcc', 'peng_tcfc', 'leung_nmcf', 'leung_nmcnr']
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
        torch.arange(num_mos, device=mi.device), mp[1:].to(mi.device), right=True)
    offsets = ptr[graph_idx]
    return mi[:, 1].long() + offsets


def eval_one(exp_dir: Path, device, topo_filter: str | None):
    ckpt = torch.load(exp_dir / 'best.pt', map_location=device, weights_only=False)
    with open(exp_dir / 'original_config.yaml') as f:
        cfg = yaml.safe_load(f)
    stats = ckpt.get('norm_stats') or {}
    if not stats:
        stats = {'v_mean': 0.9, 'v_std': 0.5, 'i_mean': -5.55, 'i_std': 1.42,
                 'gm_mean': -4.43, 'gm_std': 1.44, 'gds_mean': -5.89, 'gds_std': 1.89}

    val_loader = PretrainCombinedLoader(
        f'{DATASET}/dataset_val.pkl',
        batch_size=250 if topo_filter is None else 200,
        device=device, shuffle=False, drop_last=False, topology_filter=topo_filter,
    )
    sample = next(iter(val_loader))
    node_dim = sample.x.shape[-1]; type_dim = sample.type_tens.shape[-1]

    import argparse as _ap
    cli = _ap.Namespace()
    for k, v in parse_training_config(cfg).items():
        if v is not None: setattr(cli, k, v)
    cli.device = str(device); cli.dataset = str(DATASET); cli.predict_currents = True
    model, _ = create_model_from_args(cli, node_dim + type_dim, device)
    model.load_state_dict(ckpt['model_state_dict'])
    if hasattr(model, 'set_normalization_stats'):
        model.set_normalization_stats(
            vdc_mean=stats['v_mean'], vdc_std=stats['v_std'],
            ss_gm_mean=stats['gm_mean'], ss_gm_std=stats['gm_std'],
            ss_gds_mean=stats['gds_mean'], ss_gds_std=stats['gds_std'],
            current_mean=stats['i_mean'], current_std=stats['i_std'])
    if hasattr(model, 'current_epoch'):
        model.current_epoch = ckpt.get('epoch', 99999)
    model.eval()

    metrics = {t: {'v_sum': 0.0, 'v_n': 0,
                   'i_sum': 0.0, 'i_n': 0,
                   'gm_sum': 0.0, 'gds_sum': 0.0, 'gm_n': 0}
               for t in val_loader.topo_names}
    with torch.no_grad():
        for batch in val_loader:
            attach_norm(batch, stats)
            out = model(batch)
            slices = batch.topo_node_slices
            for t, (start, end, _n, _g) in slices.items():
                m = metrics[t]
                # V
                v_pred = out['node_voltages'][start:end]
                v_tgt = batch.node_voltage_targets[start:end]
                internal = ~batch.known_voltage_mask[start:end]
                if internal.any():
                    pv = v_pred[internal] * stats['v_std'] + stats['v_mean']
                    tv = v_tgt[internal]
                    m['v_sum'] += (pv - tv).abs().sum().item()
                    m['v_n'] += int(internal.sum().item())
                # I
                if 'node_currents' in out:
                    ip_z = out['node_currents'][start:end]
                    it = batch.node_current_targets[start:end]
                    im = batch.has_current_mask[start:end]
                    if im.any():
                        log_pred = ip_z[im] * stats['i_std'] + stats['i_mean']
                        ip_A = torch.pow(10.0, log_pred); it_A = it[im].abs()
                        m['i_sum'] += (ip_A - it_A).abs().sum().item()
                        m['i_n'] += int(im.sum().item())
                # gm/gds
                if 'mosfet_gm_pred' in out and 'mosfet_gds_pred' in out:
                    drain_idx = _drain_idx_batched(batch)
                    mos_in_t = (drain_idx >= start) & (drain_idx < end)
                    if mos_in_t.any():
                        gp = out['mosfet_gm_pred'][mos_in_t] * stats['gm_std'] + stats['gm_mean']
                        dp = out['mosfet_gds_pred'][mos_in_t] * stats['gds_std'] + stats['gds_mean']
                        gt = batch.node_log_gm[drain_idx[mos_in_t]]
                        dt = batch.node_log_gds[drain_idx[mos_in_t]]
                        m['gm_sum'] += (gp - gt).abs().sum().item()
                        m['gds_sum'] += (dp - dt).abs().sum().item()
                        m['gm_n'] += int(mos_in_t.sum().item())

    out_metrics = {}
    for t in val_loader.topo_names:
        m = metrics[t]
        out_metrics[t] = {
            'v_mae_mV': 1000.0 * m['v_sum'] / max(1, m['v_n']),
            'i_mae_uA': 1e6 * m['i_sum'] / max(1, m['i_n']) if m['i_n'] else None,
            'gm_log_mae': m['gm_sum'] / max(1, m['gm_n']) if m['gm_n'] else None,
            'gds_log_mae': m['gds_sum'] / max(1, m['gm_n']) if m['gm_n'] else None,
        }
    return out_metrics


__all__ = ['TOPOS', '_LOG_FLOOR', 'attach_norm', '_drain_idx_batched', 'eval_one']
